"""The agent core's inbox (`agent_events`) and its wakes (`wakes`): `EventStore`, mixed into `Repo`.

`db.py` is large already, so these methods live here; `Repo` inherits them and they use its
plumbing (`_all`, `_one`, `_run`, `_insert`, `_now`), its clock and its `_event_lock`.

How an event lives (see `stage-2-design.md`):

* open: `handled_at IS NULL`; waiting: open and unclaimed (`wake_id IS NULL`);
* a wake **claims** the task's waiting events that are due (`claim_events`), and when it ends
  either **handles** them (`finish_wake`) or **releases** them for a retry (`release_wake`);
* a dropped event is marked handled without a wake: it will never wake the agent.

The statements that read and then write (the burst rule, re-arming a silence, a claim, a
release) hold `_event_lock`, so two coroutines on the one connection cannot interleave them:
an event is never claimed twice, and a conversation never gets two waiting silences.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from tg_agent.models import AgentEvent, EventKind, TaskStatus, Wake

if TYPE_CHECKING:
    import asyncio

    import aiosqlite

    from tg_agent.clock import Clock

# A waiting event: not handled and not claimed by a running wake.
_WAITING = "handled_at IS NULL AND wake_id IS NULL"


def _utc(moment: datetime) -> str:
    """A moment as the UTC text the database stores (the same form as `db._utc`, which this
    module cannot import: `db` imports this module)."""
    if moment.utcoffset() is None:
        raise ValueError(f"naive datetime {moment!r}: pass a timezone-aware one")
    return moment.astimezone(UTC).isoformat()


def _moment(text: str | None) -> datetime | None:
    return None if text is None else datetime.fromisoformat(text)


def _event(r: aiosqlite.Row) -> AgentEvent:
    return AgentEvent(
        id=r["id"],
        task_id=r["task_id"],
        conversation_id=r["conversation_id"],
        kind=EventKind(r["kind"]),
        payload=json.loads(r["payload_json"]),
        created_at=datetime.fromisoformat(r["created_at"]),
        due_at=datetime.fromisoformat(r["due_at"]),
        attempts=r["attempts"],
        wake_id=r["wake_id"],
        handled_at=_moment(r["handled_at"]),
    )


def _wake(r: aiosqlite.Row) -> Wake:
    return Wake(
        id=r["id"],
        task_id=r["task_id"],
        started_at=datetime.fromisoformat(r["started_at"]),
        finished_at=_moment(r["finished_at"]),
        outcome=r["outcome"],
        steps=r["steps"],
    )


def _kinds(kinds: Iterable[EventKind]) -> list[str]:
    return [EventKind(kind).value for kind in kinds]


def _marks(values: Sequence[object]) -> str:
    return ", ".join("?" for _ in values)


class EventStore:
    """The agent core's events and wakes, as methods of `Repo`."""

    if TYPE_CHECKING:  # what `Repo` provides
        _clock: Clock
        _event_lock: asyncio.Lock

        def _now(self) -> str: ...

        async def _all(
            self, sql: str, params: Sequence[object] = ()
        ) -> list[aiosqlite.Row]: ...

        async def _one(
            self, sql: str, params: Sequence[object] = ()
        ) -> aiosqlite.Row | None: ...

        async def _run(self, sql: str, params: Sequence[object] = ()) -> int: ...

        async def _insert(self, sql: str, params: Sequence[object]) -> int: ...

    # -- adding --

    async def add_event(
        self,
        task_id: int,
        kind: EventKind,
        *,
        conv_id: int | None = None,
        payload: dict | None = None,
        due_at: datetime | None = None,
    ) -> int:
        """Add an event for the task (and maybe one of its conversations); due now unless
        `due_at` says later. Return its id."""
        now = self._now()
        return await self._insert(
            "INSERT INTO agent_events (task_id, conversation_id, kind, payload_json,"
            " created_at, due_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                task_id,
                conv_id,
                EventKind(kind).value,
                json.dumps(payload or {}, ensure_ascii=False),
                now,
                now if due_at is None else _utc(due_at),
            ),
        )

    async def bump_incoming(
        self, task_id: int, conv_id: int, *, window: timedelta, max_wait: timedelta
    ) -> int:
        """A person wrote: the burst rule. Return the id of the conversation's waiting
        `incoming` event.

        While messages keep coming the event is pushed back to `window` after the latest one,
        but never past `max_wait` after the event was made: a person who writes without a
        pause is still answered. Once a wake has claimed it, a new message starts a new one.
        """
        async with self._event_lock:
            now = self._clock.now()
            row = await self._one(
                f"SELECT id, created_at FROM agent_events WHERE conversation_id = ?"
                f" AND kind = ? AND {_WAITING} ORDER BY id LIMIT 1",
                (conv_id, EventKind.INCOMING.value),
            )
            if row is None:
                return await self.add_event(
                    task_id, EventKind.INCOMING, conv_id=conv_id, due_at=now + window
                )
            latest = datetime.fromisoformat(row["created_at"]) + max_wait
            await self._run(
                "UPDATE agent_events SET due_at = ? WHERE id = ?",
                (_utc(min(now + window, latest)), row["id"]),
            )
            return row["id"]

    async def arm_silence(self, task_id: int, conv_id: int, due_at: datetime) -> int:
        """Wake the agent at `due_at` if the person stays silent until then. A conversation has
        at most one waiting `silence` event (a unique index says so too): an existing one is
        re-armed to the new time. Return its id."""
        async with self._event_lock:
            row = await self._one(
                f"SELECT id FROM agent_events WHERE conversation_id = ? AND kind = ?"
                f" AND {_WAITING}",
                (conv_id, EventKind.SILENCE.value),
            )
            if row is None:
                return await self.add_event(
                    task_id, EventKind.SILENCE, conv_id=conv_id, due_at=due_at
                )
            await self._run(
                "UPDATE agent_events SET due_at = ? WHERE id = ?",
                (_utc(due_at), row["id"]),
            )
            return row["id"]

    # -- dropping --

    async def drop_events(self, conv_id: int, kinds: Iterable[EventKind]) -> int:
        """Mark the conversation's waiting events of these kinds handled now, so they never
        wake the agent (the person answered: his silence is no news). An event a running wake
        has claimed is left to that wake. Return how many were dropped."""
        values = _kinds(kinds)
        if not values:
            return 0
        return await self._run(
            f"UPDATE agent_events SET handled_at = ? WHERE conversation_id = ?"
            f" AND kind IN ({_marks(values)}) AND {_WAITING}",
            (self._now(), conv_id, *values),
        )

    async def drop_task_events(
        self, task_id: int, *, keep: Iterable[EventKind] = ()
    ) -> int:
        """The same for every waiting event of the task (it is over), less those of the kinds
        in `keep` (`finish` keeps the owner's words: they take the task up again). Return how
        many."""
        kept = _kinds(keep)
        spare = f" AND kind NOT IN ({_marks(kept)})" if kept else ""
        return await self._run(
            f"UPDATE agent_events SET handled_at = ? WHERE task_id = ? AND {_WAITING}"
            + spare,
            (self._now(), task_id, *kept),
        )

    # -- wakes --

    async def due_task_ids(self, now: datetime) -> list[int]:
        """The ACTIVE tasks that have a waiting event due at `now`, the most overdue first.

        A task in any other status cannot be woken: its events wait (a PAUSED task sees
        them when it resumes), and listing it would start a wake that does nothing in every
        round."""
        rows = await self._all(
            f"SELECT task_id, MIN(due_at) AS first FROM agent_events"
            f" WHERE {_WAITING} AND due_at <= ?"
            " AND task_id IN (SELECT id FROM tasks WHERE status = ?)"
            " GROUP BY task_id ORDER BY first, task_id",
            (_utc(now), TaskStatus.ACTIVE.value),
        )
        return [row["task_id"] for row in rows]

    async def claim_events(
        self, task_id: int, now: datetime
    ) -> tuple[int, list[AgentEvent]]:
        """Start a wake of the task: claim its waiting events that are due at `now`.

        Return the new wake's id and the events, oldest first. With nothing due, `(0, [])`
        and no `wakes` row.
        """
        moment = _utc(now)
        due = f"task_id = ? AND {_WAITING} AND due_at <= ?"
        async with self._event_lock:
            if (
                await self._one(
                    f"SELECT 1 FROM agent_events WHERE {due} LIMIT 1", (task_id, moment)
                )
                is None
            ):
                return 0, []
            wake_id = await self._insert(
                "INSERT INTO wakes (task_id, started_at) VALUES (?, ?)",
                (task_id, moment),
            )
            claimed = await self._run(
                f"UPDATE agent_events SET wake_id = ? WHERE {due}",
                (wake_id, task_id, moment),
            )
            if not claimed:  # cannot happen under the lock; leave no empty wake behind
                await self._run("DELETE FROM wakes WHERE id = ?", (wake_id,))
                return 0, []
            rows = await self._all(
                "SELECT * FROM agent_events WHERE wake_id = ? ORDER BY created_at, id",
                (wake_id,),
            )
        return wake_id, [_event(row) for row in rows]

    async def finish_wake(self, wake_id: int, outcome: str, steps: int) -> None:
        """The wake is over: its events are handled, and the wake gets its end."""
        now = self._now()
        async with self._event_lock:
            await self._run(
                "UPDATE agent_events SET handled_at = ? WHERE wake_id = ? AND handled_at IS NULL",
                (now, wake_id),
            )
            await self._run(
                "UPDATE wakes SET finished_at = ?, outcome = ?, steps = ? WHERE id = ?",
                (now, outcome, steps, wake_id),
            )

    async def release_wake(
        self, wake_id: int, outcome: str, retry_at: datetime, *, attempt: bool = True
    ) -> None:
        """The wake failed: its events wait again, due at `retry_at`, one attempt more
        (`attempt=False`: none; the wake did not fail, it was cut off by a crash or a stop).

        A claimed `silence`, `incoming` or `edited` event whose conversation has another
        waiting one of the same kind by now is handled instead. A silence: our message went
        out while the wake ran and a new silence was armed, and two waiting silences of one
        conversation are not allowed. A message event: the person wrote again meanwhile, and
        the newer event already wakes the agent for all his unseen messages; releasing the old
        one too would wake it twice for the same messages.
        """
        now = self._now()
        kinds = _kinds((EventKind.SILENCE, EventKind.INCOMING, EventKind.EDITED))
        async with self._event_lock:
            await self._run(
                "UPDATE agent_events SET handled_at = ? WHERE wake_id = ? AND handled_at IS NULL"
                f" AND kind IN ({_marks(kinds)}) AND EXISTS (SELECT 1 FROM agent_events other"
                " WHERE other.conversation_id = agent_events.conversation_id"
                " AND other.kind = agent_events.kind AND other.id != agent_events.id"
                " AND other.handled_at IS NULL AND other.wake_id IS NULL)",
                (now, wake_id, *kinds),
            )
            await self._run(
                "UPDATE agent_events SET wake_id = NULL, attempts = attempts + ?, due_at = ?"
                " WHERE wake_id = ? AND handled_at IS NULL",
                (int(attempt), _utc(retry_at), wake_id),
            )
            await self._run(
                "UPDATE wakes SET finished_at = ?, outcome = ? WHERE id = ?",
                (now, outcome, wake_id),
            )

    async def unfinished_wakes(self) -> list[Wake]:
        """Wakes that never ended (a crash left them), oldest first."""
        rows = await self._all(
            "SELECT * FROM wakes WHERE finished_at IS NULL ORDER BY id", ()
        )
        return [_wake(row) for row in rows]

    async def last_finished_wake(
        self, task_id: int, outcomes: Sequence[str]
    ) -> Wake | None:
        """The task's newest wake that ended with one of `outcomes` (None: none did)."""
        row = await self._one(
            f"SELECT * FROM wakes WHERE task_id = ? AND outcome IN ({_marks(outcomes)})"
            " ORDER BY id DESC LIMIT 1",
            (task_id, *outcomes),
        )
        return _wake(row) if row is not None else None

    # -- reading --

    async def claimed_events(self, wake_id: int) -> list[AgentEvent]:
        """The events this wake has claimed and not yet handled, oldest first."""
        rows = await self._all(
            "SELECT * FROM agent_events WHERE wake_id = ? AND handled_at IS NULL"
            " ORDER BY created_at, id",
            (wake_id,),
        )
        return [_event(row) for row in rows]

    async def open_events(self, task_id: int) -> list[AgentEvent]:
        """The task's events that are not handled (claimed or not), oldest first."""
        rows = await self._all(
            "SELECT * FROM agent_events WHERE task_id = ? AND handled_at IS NULL"
            " ORDER BY created_at, id",
            (task_id,),
        )
        return [_event(row) for row in rows]

    async def events_seen(
        self, task_id: int, kinds: Iterable[EventKind], wake_id: int
    ) -> list[AgentEvent]:
        """The task's events of these kinds that a wake has handled or that the wake
        `wake_id` has claimed (what the agent has been shown), oldest first."""
        values = _kinds(kinds)
        if not values:
            return []
        rows = await self._all(
            f"SELECT * FROM agent_events WHERE task_id = ? AND kind IN ({_marks(values)})"
            " AND (handled_at IS NOT NULL OR wake_id = ?) ORDER BY created_at, id",
            (task_id, *values, wake_id),
        )
        return [_event(row) for row in rows]

    async def handled_events(
        self, task_id: int, kinds: Iterable[EventKind], limit: int
    ) -> list[AgentEvent]:
        """The latest `limit` handled events of these kinds, oldest of them first."""
        values = _kinds(kinds)
        if not values or limit <= 0:  # SQLite reads a negative LIMIT as "no limit"
            return []
        rows = await self._all(
            f"SELECT * FROM (SELECT * FROM agent_events WHERE task_id = ?"
            f" AND kind IN ({_marks(values)}) AND handled_at IS NOT NULL"
            f" ORDER BY created_at DESC, id DESC LIMIT ?) ORDER BY created_at, id",
            (task_id, *values, limit),
        )
        return [_event(row) for row in rows]
