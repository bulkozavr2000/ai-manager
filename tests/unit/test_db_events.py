"""The agent core's inbox and wakes in the repository (`db_events.EventStore`), on a real file."""

from __future__ import annotations

import asyncio
import sqlite3
import types
from contextlib import closing
from dataclasses import fields
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import get_args, get_type_hints

import pytest

from tg_agent.db import Repo
from tg_agent.models import AgentEvent, ConvState, EventKind, TaskStatus, Wake

K = EventKind
WINDOW = timedelta(seconds=30)
MAX_WAIT = timedelta(minutes=2)


def raw(db_path: Path, sql: str, params: tuple = ()) -> list[tuple]:
    """What is really stored, read through a plain sqlite3 connection."""
    with closing(sqlite3.connect(db_path)) as db:
        return db.execute(sql, params).fetchall()


async def make_task(
    repo: Repo, title: str = "Дрель", status: TaskStatus = TaskStatus.ACTIVE
) -> int:
    """A task of the agent core: launched (ACTIVE) unless `status` says otherwise."""
    task_id = await repo.create_task(title, "цена", {}, None, "UZS", [], [])
    await repo.set_task_status(task_id, status)
    return task_id


async def make_conv(repo: Repo, task_id: int, peer_id: int = 1001) -> int:
    contact_id = await repo.upsert_contact(peer_id, None, f"Продавец {peer_id}", False)
    return await repo.create_conversation(task_id, contact_id, state=ConvState.PLANNED)


def utc(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


def wakes_count(db_path: Path) -> int:
    return raw(db_path, "SELECT COUNT(*) FROM wakes")[0][0]


# --- the rows -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("table", "row_class"), [("agent_events", AgentEvent), ("wakes", Wake)]
)
def test_table_columns_follow_the_row_dataclass(repo, db_path, table, row_class):
    columns = raw(db_path, f"PRAGMA table_info({table})")
    assert [name.removesuffix("_json") for _, name, *_ in columns] == [
        field.name for field in fields(row_class)
    ]
    hints = get_type_hints(row_class)
    for _, name, _type, notnull, _default, pk in columns:
        column_is_nullable = not notnull and not pk
        field_is_optional = type(None) in get_args(hints[name.removesuffix("_json")])
        assert column_is_nullable == field_is_optional, f"{table}.{name}"


def assert_typed(row: object) -> None:
    """Every field holds what its annotation promises: enum members, dicts, aware UTC times."""
    for name, hint in get_type_hints(type(row)).items():
        value = getattr(row, name)
        options = get_args(hint) if isinstance(hint, types.UnionType) else (hint,)
        if value is None:
            assert type(None) in options, name
            continue
        expected = next(option for option in options if option is not type(None))
        assert type(value) is expected, f"{name}: {value!r}"
        if expected is datetime:
            assert value.utcoffset() == timedelta(0), name


# --- 1. adding ------------------------------------------------------------------------------


async def test_add_event_is_due_now_by_default_and_open(repo, fake_clock):
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id)

    event_id = await repo.add_event(task_id, K.CREATED)
    later = fake_clock.now() + timedelta(hours=1)
    other = await repo.add_event(
        task_id, K.OWNER_ANSWER, conv_id=conv_id, payload={"answer": "да"}, due_at=later
    )

    [created, answer] = await repo.open_events(task_id)
    assert (created.id, created.kind, created.conversation_id) == (
        event_id,
        K.CREATED,
        None,
    )
    assert created.due_at == created.created_at == fake_clock.now()
    assert (created.payload, created.attempts, created.wake_id, created.handled_at) == (
        {},
        0,
        None,
        None,
    )
    assert (answer.id, answer.conversation_id, answer.payload) == (
        other,
        conv_id,
        {"answer": "да"},
    )
    assert answer.due_at == later
    for event in (created, answer):
        assert_typed(event)


# --- 2. the burst rule ----------------------------------------------------------------------


async def test_bump_incoming_waits_for_the_burst_but_not_forever(repo, fake_clock):
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id)
    start = fake_clock.now()

    first = await repo.bump_incoming(task_id, conv_id, window=WINDOW, max_wait=MAX_WAIT)
    [event] = await repo.open_events(task_id)
    assert (event.id, event.kind, event.conversation_id) == (first, K.INCOMING, conv_id)
    assert event.due_at == start + WINDOW

    fake_clock.advance(timedelta(seconds=10))
    assert (
        await repo.bump_incoming(task_id, conv_id, window=WINDOW, max_wait=MAX_WAIT)
        == first
    )
    [event] = await repo.open_events(task_id)
    assert event.due_at == fake_clock.now() + WINDOW

    for _ in range(20):  # the person keeps writing every 10 s
        fake_clock.advance(timedelta(seconds=10))
        assert (
            await repo.bump_incoming(task_id, conv_id, window=WINDOW, max_wait=MAX_WAIT)
            == first
        )
        [event] = await repo.open_events(task_id)
        assert event.due_at <= start + MAX_WAIT
    assert event.due_at == start + MAX_WAIT

    other_conv = await make_conv(repo, task_id, peer_id=1002)
    other = await repo.bump_incoming(
        task_id, other_conv, window=WINDOW, max_wait=MAX_WAIT
    )
    assert other != first
    assert len(await repo.open_events(task_id)) == 2


async def test_bump_incoming_starts_a_new_event_once_the_old_one_is_claimed(
    repo, fake_clock
):
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id)
    first = await repo.bump_incoming(task_id, conv_id, window=WINDOW, max_wait=MAX_WAIT)
    fake_clock.advance(WINDOW)
    wake_id, _ = await repo.claim_events(task_id, fake_clock.now())
    assert wake_id

    second = await repo.bump_incoming(
        task_id, conv_id, window=WINDOW, max_wait=MAX_WAIT
    )
    assert second != first


# --- 3. silence -----------------------------------------------------------------------------


async def test_arm_silence_keeps_one_waiting_event_per_conversation(repo, fake_clock):
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id)
    now = fake_clock.now()

    first = await repo.arm_silence(task_id, conv_id, now + timedelta(hours=1))
    assert await repo.arm_silence(task_id, conv_id, now + timedelta(hours=5)) == first
    [event] = await repo.open_events(task_id)
    assert (event.kind, event.due_at) == (K.SILENCE, now + timedelta(hours=5))

    # handled by a wake: arming again adds a new one
    fake_clock.advance(timedelta(hours=6))
    wake_id, claimed = await repo.claim_events(task_id, fake_clock.now())
    assert [e.id for e in claimed] == [first]
    await repo.finish_wake(wake_id, "done", 1)
    second = await repo.arm_silence(
        task_id, conv_id, fake_clock.now() + timedelta(hours=1)
    )
    assert second != first

    # dropped (the person answered): arming again adds a new one too
    assert await repo.drop_events(conv_id, [K.SILENCE]) == 1
    third = await repo.arm_silence(
        task_id, conv_id, fake_clock.now() + timedelta(hours=1)
    )
    assert third not in (first, second)
    assert [e.id for e in await repo.open_events(task_id)] == [third]


# --- 4. which tasks are due -----------------------------------------------------------------


async def test_due_task_ids_lists_tasks_with_a_due_open_unclaimed_event(
    repo, fake_clock
):
    now = fake_clock.now()
    recent, overdue, future, claimed = [
        await make_task(repo, f"t{n}") for n in range(4)
    ]
    await repo.add_event(recent, K.CREATED, due_at=now - timedelta(minutes=10))
    await repo.add_event(overdue, K.CREATED, due_at=now - timedelta(hours=1))
    await repo.add_event(overdue, K.REMINDER, due_at=now + timedelta(hours=1))
    await repo.add_event(future, K.REMINDER, due_at=now + timedelta(minutes=1))
    await repo.add_event(claimed, K.CREATED, due_at=now - timedelta(hours=2))
    await repo.claim_events(claimed, now)

    assert await repo.due_task_ids(now) == [overdue, recent]
    assert await repo.due_task_ids(now + timedelta(minutes=1)) == [
        overdue,
        recent,
        future,
    ]


@pytest.mark.parametrize(
    "status",
    [TaskStatus.DRAFT, TaskStatus.PAUSED, TaskStatus.DONE, TaskStatus.CANCELLED],
)
async def test_due_task_ids_lists_only_active_tasks(repo, fake_clock, status):
    active = await make_task(repo, "active")
    other = await make_task(repo, "other", status)
    for task_id in (active, other):
        await repo.add_event(task_id, K.CREATED)

    assert await repo.due_task_ids(fake_clock.now()) == [active]


# --- 5. claiming ----------------------------------------------------------------------------


async def test_claim_events_takes_the_due_ones_oldest_first(repo, db_path, fake_clock):
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id)
    older = await repo.add_event(task_id, K.CREATED)
    fake_clock.advance(timedelta(minutes=1))
    newer = await repo.add_event(
        task_id, K.OWNER_INSTRUCTION, payload={"text": "быстрее"}
    )
    not_yet = await repo.add_event(
        task_id, K.REMINDER, due_at=fake_clock.now() + timedelta(hours=1)
    )
    other_task = await make_task(repo, "Другая")
    elsewhere = await repo.add_event(other_task, K.CREATED)
    # created later but due earlier: the order is by creation
    fake_clock.advance(timedelta(minutes=1))
    latest = await repo.add_event(
        task_id,
        K.INCOMING,
        conv_id=conv_id,
        due_at=fake_clock.now() - timedelta(hours=1),
    )
    now = fake_clock.now()

    wake_id, events = await repo.claim_events(task_id, now)

    assert wake_id > 0
    assert [e.id for e in events] == [older, newer, latest]
    assert all(e.wake_id == wake_id for e in events)
    assert events[1].payload == {"text": "быстрее"}
    open_ = {e.id: e for e in await repo.open_events(task_id)}
    assert open_[not_yet].wake_id is None  # not due yet: still waiting
    assert open_[older].wake_id == wake_id  # claimed is still open until the wake ends
    assert (await repo.open_events(other_task))[0].id == elsewhere
    assert (await repo.open_events(other_task))[0].wake_id is None
    assert raw(db_path, "SELECT task_id, started_at, finished_at FROM wakes") == [
        (task_id, utc(now), None)
    ]

    assert await repo.claim_events(task_id, now) == (0, [])
    assert wakes_count(db_path) == 1  # no row for an empty claim


async def test_claim_events_with_nothing_due_leaves_no_wake(repo, db_path, fake_clock):
    task_id = await make_task(repo)
    await repo.add_event(
        task_id, K.REMINDER, due_at=fake_clock.now() + timedelta(minutes=5)
    )

    assert await repo.claim_events(task_id, fake_clock.now()) == (0, [])
    assert wakes_count(db_path) == 0


async def test_two_claims_at_once_never_take_the_same_event(repo, db_path, fake_clock):
    task_id = await make_task(repo)
    for kind in (K.CREATED, K.OWNER_INSTRUCTION, K.REMINDER):
        await repo.add_event(task_id, kind)

    results = await asyncio.gather(
        *(repo.claim_events(task_id, fake_clock.now()) for _ in range(5))
    )

    winners = [(wake_id, events) for wake_id, events in results if wake_id]
    assert len(winners) == 1
    assert len(winners[0][1]) == 3
    assert wakes_count(db_path) == 1


# --- 6. finishing ---------------------------------------------------------------------------


async def test_finish_wake_handles_its_events_and_fills_the_wake(
    repo, db_path, fake_clock
):
    task_id = await make_task(repo)
    await repo.add_event(task_id, K.CREATED)
    await repo.add_event(task_id, K.OWNER_ANSWER, payload={"answer": "да"})
    wake_id, events = await repo.claim_events(task_id, fake_clock.now())
    fake_clock.advance(timedelta(seconds=40))

    await repo.finish_wake(wake_id, "ended_turn", 3)

    assert await repo.open_events(task_id) == []
    assert await repo.unfinished_wakes() == []
    assert raw(
        db_path,
        "SELECT finished_at, outcome, steps FROM wakes WHERE id = ?",
        (wake_id,),
    ) == [(utc(fake_clock.now()), "ended_turn", 3)]
    [handled] = await repo.handled_events(task_id, [K.OWNER_ANSWER], 10)
    assert handled.id == events[1].id
    assert handled.handled_at == fake_clock.now()
    assert handled.wake_id == wake_id
    assert_typed(handled)


# --- 7. releasing ---------------------------------------------------------------------------


async def test_release_wake_puts_its_events_back_for_later(repo, db_path, fake_clock):
    task_id = await make_task(repo)
    first = await repo.add_event(task_id, K.CREATED)
    second = await repo.add_event(task_id, K.REMINDER)
    wake_id, _ = await repo.claim_events(task_id, fake_clock.now())
    retry_at = fake_clock.now() + timedelta(minutes=5)

    await repo.release_wake(wake_id, "failed", retry_at)

    events = await repo.open_events(task_id)
    assert [e.id for e in events] == [first, second]
    assert all(
        (e.wake_id, e.attempts, e.due_at, e.handled_at) == (None, 1, retry_at, None)
        for e in events
    )
    assert raw(
        db_path, "SELECT finished_at, outcome FROM wakes WHERE id = ?", (wake_id,)
    ) == [(utc(fake_clock.now()), "failed")]
    assert await repo.due_task_ids(fake_clock.now()) == []
    assert await repo.due_task_ids(retry_at - timedelta(seconds=1)) == []
    assert await repo.due_task_ids(retry_at) == [task_id]

    # a second failure counts again
    wake_id, _ = await repo.claim_events(task_id, retry_at)
    await repo.release_wake(wake_id, "failed", retry_at + timedelta(minutes=5))
    assert [e.attempts for e in await repo.open_events(task_id)] == [2, 2]

    # a wake cut off by a crash or a stop is released without an attempt
    wake_id, _ = await repo.claim_events(task_id, retry_at + timedelta(minutes=5))
    await repo.release_wake(wake_id, "interrupted", retry_at, attempt=False)
    assert [e.attempts for e in await repo.open_events(task_id)] == [2, 2]


async def test_a_released_silence_gives_way_to_one_armed_meanwhile(repo, fake_clock):
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id)
    old = await repo.arm_silence(task_id, conv_id, fake_clock.now())
    reminder = await repo.add_event(task_id, K.REMINDER)
    wake_id, claimed = await repo.claim_events(task_id, fake_clock.now())
    assert {e.id for e in claimed} == {old, reminder}
    # our message went out while the wake ran, and a new silence was armed
    new = await repo.arm_silence(
        task_id, conv_id, fake_clock.now() + timedelta(hours=24)
    )
    assert new != old

    await repo.release_wake(wake_id, "failed", fake_clock.now() + timedelta(minutes=5))

    open_ = {e.id: e for e in await repo.open_events(task_id)}
    assert set(open_) == {new, reminder}  # the old silence is handled, not released
    assert open_[reminder].attempts == 1
    assert open_[new].due_at == fake_clock.now() + timedelta(hours=24)
    [handled] = await repo.handled_events(task_id, [K.SILENCE], 5)
    assert handled.id == old


@pytest.mark.parametrize("kind", [K.INCOMING, K.EDITED])
async def test_a_released_message_event_gives_way_to_one_added_meanwhile(
    repo, fake_clock, kind
):
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id)
    other_conv = await make_conv(repo, task_id, peer_id=1002)
    old = await repo.add_event(task_id, kind, conv_id=conv_id)
    alone = await repo.add_event(task_id, kind, conv_id=other_conv)
    wake_id, _ = await repo.claim_events(task_id, fake_clock.now())
    # the person wrote again while the wake ran: a new waiting event of the same kind
    new = await repo.add_event(task_id, kind, conv_id=conv_id)
    retry_at = fake_clock.now() + timedelta(minutes=5)

    await repo.release_wake(wake_id, "failed", retry_at)

    open_ = {e.id: e for e in await repo.open_events(task_id)}
    assert set(open_) == {new, alone}  # one waiting event per conversation
    assert (open_[alone].attempts, open_[alone].due_at) == (1, retry_at)
    [handled] = await repo.handled_events(task_id, [kind], 5)
    assert handled.id == old


async def test_a_released_silence_without_a_newer_one_is_released(repo, fake_clock):
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id)
    old = await repo.arm_silence(task_id, conv_id, fake_clock.now())
    wake_id, _ = await repo.claim_events(task_id, fake_clock.now())
    retry_at = fake_clock.now() + timedelta(minutes=5)

    await repo.release_wake(wake_id, "failed", retry_at)

    [event] = await repo.open_events(task_id)
    assert (event.id, event.wake_id, event.attempts, event.due_at) == (
        old,
        None,
        1,
        retry_at,
    )


# --- 8. dropping ----------------------------------------------------------------------------


async def test_drop_events_handles_only_the_named_kinds_of_one_conversation(
    repo, fake_clock
):
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id)
    other_conv = await make_conv(repo, task_id, peer_id=1002)
    now = fake_clock.now()
    silence = await repo.arm_silence(task_id, conv_id, now + timedelta(hours=1))
    reminder = await repo.add_event(task_id, K.REMINDER, conv_id=conv_id)
    elsewhere = await repo.arm_silence(task_id, other_conv, now + timedelta(hours=1))
    claimed = await repo.add_event(task_id, K.INCOMING, conv_id=conv_id)
    assert await repo.drop_events(conv_id, [K.REMINDER]) == 1
    assert reminder not in {e.id for e in await repo.open_events(task_id)}
    # a wake claims the incoming one (the silences are not due yet)
    wake_id, taken = await repo.claim_events(task_id, now)
    assert [e.id for e in taken] == [claimed]

    assert await repo.drop_events(conv_id, [K.SILENCE, K.INCOMING]) == 1

    open_ids = {e.id for e in await repo.open_events(task_id)}
    assert silence not in open_ids
    assert {elsewhere, claimed} <= open_ids  # another conversation; claimed by a wake
    [dropped] = await repo.handled_events(task_id, [K.SILENCE], 5)
    assert (dropped.id, dropped.handled_at, dropped.wake_id) == (silence, now, None)
    assert wake_id


async def test_drop_task_events_handles_every_open_unclaimed_event(repo, fake_clock):
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id)
    other_task = await make_task(repo, "Другая")
    claimed = await repo.add_event(task_id, K.CREATED)
    await repo.claim_events(task_id, fake_clock.now())
    await repo.add_event(
        task_id, K.REMINDER, due_at=fake_clock.now() + timedelta(hours=1)
    )
    await repo.arm_silence(task_id, conv_id, fake_clock.now() + timedelta(hours=1))
    kept = await repo.add_event(other_task, K.CREATED)

    assert await repo.drop_task_events(task_id) == 2

    assert [e.id for e in await repo.open_events(task_id)] == [claimed]
    assert [e.id for e in await repo.open_events(other_task)] == [kept]


# --- 9. crashes -----------------------------------------------------------------------------


async def test_unfinished_wakes_lists_a_wake_a_crash_left(repo, fake_clock):
    task_id = await make_task(repo)
    done_task = await make_task(repo, "Готово")
    await repo.add_event(task_id, K.CREATED)
    await repo.add_event(done_task, K.CREATED)
    crashed, _ = await repo.claim_events(task_id, fake_clock.now())
    finished, _ = await repo.claim_events(done_task, fake_clock.now())
    await repo.finish_wake(finished, "ended_turn", 1)

    [wake] = await repo.unfinished_wakes()

    assert (wake.id, wake.task_id, wake.started_at) == (
        crashed,
        task_id,
        fake_clock.now(),
    )
    assert (wake.finished_at, wake.outcome, wake.steps) == (None, None, 0)
    assert_typed(wake)


# --- 10. what was handled -------------------------------------------------------------------


async def test_handled_events_returns_the_latest_of_the_kinds_oldest_first(
    repo, fake_clock
):
    task_id = await make_task(repo)
    ids = []
    for n, kind in enumerate(
        [
            K.OWNER_INSTRUCTION,
            K.OWNER_ANSWER,
            K.REMINDER,
            K.OWNER_INSTRUCTION,
            K.CREATED,
        ]
    ):
        ids.append(await repo.add_event(task_id, kind, payload={"n": n}))
        fake_clock.advance(timedelta(minutes=1))
    open_one = await repo.add_event(task_id, K.OWNER_ANSWER)  # not handled yet
    wake_id, _ = await repo.claim_events(
        task_id, fake_clock.now() - timedelta(seconds=1)
    )
    await repo.finish_wake(wake_id, "ended_turn", 1)

    latest = await repo.handled_events(
        task_id, [K.OWNER_INSTRUCTION, K.OWNER_ANSWER], 2
    )

    assert [e.id for e in latest] == [ids[1], ids[3]]
    assert [e.payload for e in latest] == [{"n": 1}, {"n": 3}]
    assert open_one not in {
        e.id for e in await repo.handled_events(task_id, list(K), 10)
    }
    assert await repo.handled_events(task_id, [K.OWNER_ANSWER], 0) == []


async def test_drop_task_events_can_keep_some_kinds(repo, fake_clock):
    task_id = await make_task(repo)
    told = await repo.add_event(task_id, K.OWNER_INSTRUCTION, payload={"text": "ещё"})
    await repo.add_event(task_id, K.REMINDER)

    assert await repo.drop_task_events(task_id, keep=[K.OWNER_INSTRUCTION]) == 1

    assert [e.id for e in await repo.open_events(task_id)] == [told]


async def test_move_task_if_moves_only_from_the_expected_status(repo):
    task_id = await make_task(repo, status=TaskStatus.DRAFT)

    assert await repo.move_task_if(task_id, TaskStatus.ACTIVE, TaskStatus.DONE) is False
    assert (await repo.get_task(task_id)).status is TaskStatus.DRAFT
    assert await repo.move_task_if(task_id, TaskStatus.DRAFT, TaskStatus.ACTIVE) is True
    assert (await repo.get_task(task_id)).status is TaskStatus.ACTIVE
    many = (TaskStatus.DRAFT, TaskStatus.ACTIVE)
    assert await repo.move_task_if(task_id, many, TaskStatus.CANCELLED) is True
    assert await repo.move_task_if(task_id, many, TaskStatus.CANCELLED) is False
    assert (await repo.get_task(task_id)).status is TaskStatus.CANCELLED
