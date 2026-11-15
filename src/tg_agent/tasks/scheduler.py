"""The Scheduler: the loop that wakes the agent up.

Every few seconds (`run_forever`) it does, in this order:

1. `ui.flush()`: what the bot held back in the owner's quiet hours goes out once they are over
   (it comes first so that the old notices keep their place before the new ones);
2. `wakeups.tick()`: the core does whatever is due (`ports.Wakeups`: the agent core's
   `agent/runner.Runner`). The step is called "timers" in the events;
3. `sender.tick()`: the outbox;
4. the daily reminder, once a day from `schedule.digest_at` (the injected `reminder`:
   `ChatLayout.resend_newest`): a card that has waited for the owner for hours is sent again
   at the bottom, with sound (ruling R62);
5. the daily clean-up of raw messages older than the retention period (90 days), and the
   daily copy of the database from 03:00 (`data/backup/daily/`, the newest seven kept);
6. the owner's chat put right (the injected `board`: `ChatLayout.settle`): the status redrawn
   when it changes, and the card that waits for the owner, or the status, kept last.

One step failing (the bot is down, the database is busy) does not keep the others from running,
and does not end the loop: the failure is logged and recorded as a `scheduler_error` event, at
most once in ten minutes for a step that keeps failing.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, time, timedelta

from tg_agent.clock import Clock
from tg_agent.config import Settings
from tg_agent.control.base import UI
from tg_agent.db import Repo
from tg_agent.ports import Wakeups
from tg_agent.tasks.sender import Sender

# The day (Tashkent calendar, as ISO text) each daily job last ran on. The reminder keeps
# the key of the evening digest it replaced (no migration).
DIGEST_KEY = "digest_sent_on"
PURGE_KEY = "purge_on"
# The daily copy of the database (ruling R87): the day it was written, and the day the owner
# was told it failed.
COPY_KEY = "daily_copy_on"
COPY_FAILED_KEY = "daily_copy_failed_on"
COPY_AT = time(3, 0)
COPIES_KEPT = 7
COPY_FAILED = "Не получилось сделать дневную копию базы ({error}). Попробую ещё раз."
# When the last round ran (ISO); the next start-up reads it (ruling R88, `main.start_up`).
HEARTBEAT_KEY = "heartbeat"

Reminder = Callable[[], Awaitable[object]]

# A step that keeps failing is reported (logged and written to the events) this often.
REPORT_EVERY = timedelta(minutes=10)

_log = logging.getLogger(__name__)


class Scheduler:
    """Wakes the core, calls the Sender and the daily jobs. `tick` is one round of that."""

    def __init__(
        self,
        wakeups: Wakeups,
        sender: Sender,
        repo: Repo,
        clock: Clock,
        settings: Settings,
        ui: UI,
        reminder: Reminder,
        board: Callable[[], Awaitable[object]] | None = None,
    ) -> None:
        self._wakeups = wakeups
        self._sender = sender
        self._repo = repo
        self._clock = clock
        self._settings = settings
        self._ui = ui
        self._reminder = reminder
        self._board = board
        # When each step that is failing was last reported.
        self._reported: dict[str, datetime] = {}

    async def run_forever(self, interval_s: float = 5) -> None:
        """Tick every `interval_s` seconds until the task is cancelled."""
        while True:
            await self.tick()
            await asyncio.sleep(interval_s)

    async def tick(self) -> None:
        """One round: flush the bot, wake the core, send, and the daily jobs."""
        await self._step("heartbeat", self._beat)
        await self._step("flush", self._ui.flush)
        await self._step("timers", self._wakeups.tick)
        await self._step("sender", self._sender.tick)
        await self._step("reminder", self._remind)
        await self._step("purge", self._purge)
        await self._step("daily_copy", self._daily_copy)
        if self._board is not None:
            await self._step("board", self._board)

    async def _beat(self) -> None:
        await self._repo.set_kv(HEARTBEAT_KEY, self._clock.now().isoformat())

    async def _step(self, name: str, work: Callable[[], Awaitable[object]]) -> None:
        """Run one step; a failure is reported and the round goes on."""
        try:
            await work()
        except Exception as error:  # noqa: BLE001 - a step must not end the loop
            await self._report(name, error)
        else:
            self._reported.pop(name, None)  # it works again: the next failure is news

    async def _report(self, name: str, error: Exception) -> None:
        now = self._clock.now()
        last = self._reported.get(name)
        if last is not None and now - last < REPORT_EVERY:
            return  # the same failure, again: already reported
        self._reported[name] = now
        _log.error("scheduler step %r failed", name, exc_info=error)
        try:
            await self._repo.log_event(
                "scheduler_error", None, step=name, error=type(error).__name__
            )
        except Exception:  # noqa: BLE001 - the log may be what is broken
            _log.exception("could not record the failure of step %r", name)

    def _local(self) -> datetime:
        """Now, on the owner's calendar: what "a day" means for the daily jobs."""
        return self._clock.now().astimezone(self._settings.schedule.zone)

    async def _remind(self) -> None:
        """Once a day, from `digest_at` on: a program that starts later that evening reminds
        then, and one that starts the next morning does not remind of yesterday. The day
        counts as done whether a card was sent again or nothing waited long enough."""
        local = self._local()
        today = local.date().isoformat()
        if local.time() < self._settings.schedule.digest_at:
            return
        if await self._repo.get_kv(DIGEST_KEY) == today:
            return
        await self._reminder()
        # Only now: a reminder that failed is tried again on the next round.
        await self._repo.set_kv(DIGEST_KEY, today)

    async def _purge(self) -> None:
        """Once a day: raw messages past the retention period go; facts and cards stay."""
        today = self._local().date().isoformat()
        if await self._repo.get_kv(PURGE_KEY) == today:
            return
        days = self._settings.limits.raw_retention_days
        deleted = await self._repo.purge_messages_before(
            self._clock.now() - timedelta(days=days)
        )
        await self._repo.set_kv(PURGE_KEY, today)
        await self._repo.log_event("purge", None, deleted=deleted)

    async def _daily_copy(self) -> None:
        """Once a day, from 03:00: a copy of the database by SQLite's own backup call, the
        newest seven kept. A failing copy is tried again on the next round and told to the
        owner once a day."""
        local = self._local()
        today = local.date().isoformat()
        if local.time() < COPY_AT or await self._repo.get_kv(COPY_KEY) == today:
            return
        folder = self._settings.data_dir / "backup" / "daily"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            await self._repo.back_up_to(folder / f"agent-{today}.db")
        except Exception as error:
            if await self._repo.get_kv(COPY_FAILED_KEY) != today:
                await self._repo.set_kv(COPY_FAILED_KEY, today)
                await self._ui.notify(COPY_FAILED.format(error=type(error).__name__))
            raise  # logged by the round
        await self._repo.set_kv(COPY_KEY, today)
        for old in sorted(folder.glob("agent-????-??-??.db"))[:-COPIES_KEPT]:
            old.unlink(missing_ok=True)
