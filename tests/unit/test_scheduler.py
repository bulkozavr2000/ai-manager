"""The Scheduler's daily reminder (ruling R62): once a day from `schedule.digest_at`."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from tg_agent.clock import FakeClock
from tg_agent.config import Settings
from tg_agent.control.base import FakeUI
from tg_agent.db import Repo
from tg_agent.tasks.scheduler import DIGEST_KEY, HEARTBEAT_KEY, Scheduler

TZ = ZoneInfo("Asia/Tashkent")


def at(day: int, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 11, day, hour, minute, second, tzinfo=TZ)


class Idle:
    """Wakeups and a Sender with nothing to do."""

    async def tick(self) -> int:
        return 0


def scheduler(
    repo: Repo, clock: FakeClock, settings: Settings, calls: list[str], *, fail=False
) -> Scheduler:
    async def reminder() -> bool:
        calls.append("reminder")
        if fail:
            raise ConnectionError("network is down")
        return True

    return Scheduler(
        Idle(),
        Idle(),
        repo,
        clock,
        settings,
        FakeUI(),
        reminder,  # type: ignore[arg-type]
    )


async def test_the_reminder_runs_once_a_day_from_digest_at(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    calls: list[str] = []
    rounds = scheduler(repo, fake_clock, settings, calls)
    fake_clock.set(at(10, 20, 29))
    await rounds.tick()
    assert calls == []

    fake_clock.set(at(10, 20, 30))
    await rounds.tick()
    fake_clock.set(at(10, 23, 0))
    await rounds.tick()
    assert calls == ["reminder"]
    assert await repo.get_kv(DIGEST_KEY) == "2026-11-10"

    fake_clock.set(at(11, 21, 0))
    await rounds.tick()
    assert calls == ["reminder", "reminder"]


async def test_a_reminder_that_fails_is_tried_again_on_the_next_round(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    calls: list[str] = []
    fake_clock.set(at(10, 20, 31))
    failing = scheduler(repo, fake_clock, settings, calls, fail=True)

    await failing.tick()
    await failing.tick()

    assert calls == ["reminder", "reminder"]
    assert await repo.get_kv(DIGEST_KEY) is None
    assert await repo.get_kv("purge_on") == "2026-11-10"  # the other daily job ran


async def test_every_round_writes_the_heartbeat(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    # Ruling R88: the next start-up reads when the program was last alive.
    rounds = scheduler(repo, fake_clock, settings, [])
    fake_clock.set(at(10, 9, 0))
    await rounds.tick()
    assert await repo.get_kv(HEARTBEAT_KEY) == at(10, 9, 0).isoformat()

    fake_clock.set(at(10, 9, 0, 5))
    await rounds.tick()
    assert await repo.get_kv(HEARTBEAT_KEY) == at(10, 9, 0, 5).isoformat()
