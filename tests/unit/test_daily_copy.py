"""The daily copy of the database (ruling R87): from 03:00 local time, once a day, seven kept."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tg_agent.clock import FakeClock
from tg_agent.config import Settings
from tg_agent.control.base import FakeUI
from tg_agent.db import Repo
from tg_agent.tasks.scheduler import Scheduler

TZ = ZoneInfo("Asia/Tashkent")


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 11, day, hour, minute, tzinfo=TZ)


class Idle:
    async def tick(self) -> int:
        return 0


async def nothing() -> None:
    return None


def rounds(repo: Repo, clock: FakeClock, settings: Settings, ui: FakeUI) -> Scheduler:
    return Scheduler(Idle(), Idle(), repo, clock, settings, ui, nothing)  # type: ignore[arg-type]


def daily(settings: Settings) -> Path:
    return settings.data_dir / "backup" / "daily"


async def test_the_copy_is_written_once_a_day_from_three_in_the_morning(
    repo: Repo, fake_clock: FakeClock, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    await repo.set_kv("marker", "is in the copy")
    ui = FakeUI()
    scheduler = rounds(repo, fake_clock, settings, ui)
    copies: list[Path] = []
    original = repo.back_up_to

    async def counted(target: Path) -> None:
        copies.append(target)
        await original(target)

    monkeypatch.setattr(repo, "back_up_to", counted)

    fake_clock.set(at(10, 2, 59))
    await scheduler.tick()
    assert copies == []

    fake_clock.set(at(10, 3, 0))
    await scheduler.tick()
    fake_clock.set(at(10, 15, 0))
    await scheduler.tick()
    assert len(copies) == 1

    copy = daily(settings) / "agent-2026-11-10.db"
    with sqlite3.connect(copy) as db:
        row = db.execute("SELECT value FROM kv WHERE key = 'marker'").fetchone()
    assert row == ("is in the copy",)
    assert ui.notes == []

    fake_clock.set(at(11, 3, 5))
    await scheduler.tick()
    assert len(copies) == 2
    assert (daily(settings) / "agent-2026-11-11.db").is_file()


async def test_the_newest_seven_copies_are_kept(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    folder = daily(settings)
    folder.mkdir(parents=True)
    for day in range(1, 9):
        (folder / f"agent-2026-11-{day:02d}.db").write_bytes(b"old")
    (folder / "notes.txt").write_text("not a copy", encoding="utf-8")

    fake_clock.set(at(10, 4, 0))
    await rounds(repo, fake_clock, settings, FakeUI()).tick()

    kept = sorted(path.name for path in folder.glob("agent-*.db"))
    assert kept == [f"agent-2026-11-{day:02d}.db" for day in (3, 4, 5, 6, 7, 8, 10)]
    assert (folder / "notes.txt").is_file()


async def test_a_failing_copy_is_told_to_the_owner_once_a_day(
    repo: Repo, fake_clock: FakeClock, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[Path] = []

    async def broken(target: Path) -> None:
        attempts.append(target)
        raise OSError("disk full")

    monkeypatch.setattr(repo, "back_up_to", broken)
    ui = FakeUI()
    scheduler = rounds(repo, fake_clock, settings, ui)

    fake_clock.set(at(10, 3, 0))
    await scheduler.tick()
    fake_clock.set(at(10, 3, 30))
    await scheduler.tick()

    assert len(attempts) == 2  # tried again on the next round
    assert len(ui.notes) == 1
    assert "копи" in ui.notes[0]
    assert not list(daily(settings).glob("agent-*"))  # no half-written copy is left

    fake_clock.set(at(11, 3, 0))
    await scheduler.tick()
    assert len(ui.notes) == 2  # a new day: told again


async def test_a_copy_that_works_again_after_a_failure_is_written(
    repo: Repo, fake_clock: FakeClock, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = repo.back_up_to
    broken = True

    async def flaky(target: Path) -> None:
        if broken:
            raise OSError("disk full")
        await original(target)

    monkeypatch.setattr(repo, "back_up_to", flaky)
    ui = FakeUI()
    scheduler = rounds(repo, fake_clock, settings, ui)
    fake_clock.set(at(10, 3, 0))
    await scheduler.tick()
    broken = False
    fake_clock.set(at(10, 3, 1))
    await scheduler.tick()

    assert (daily(settings) / "agent-2026-11-10.db").is_file()
    assert len(ui.notes) == 1
