from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from tg_agent.clock import FakeClock, SystemClock

TASHKENT = ZoneInfo("Asia/Tashkent")


def test_fake_clock_advance(fake_clock):
    start = fake_clock.now()
    assert start == datetime(2026, 11, 10, 10, 0, tzinfo=TASHKENT)

    fake_clock.advance(timedelta(seconds=90))

    assert fake_clock.now() == start + timedelta(seconds=90)
    assert fake_clock.now() == datetime(2026, 11, 10, 10, 1, 30, tzinfo=TASHKENT)
    assert fake_clock.now().tzinfo is TASHKENT
    assert fake_clock.now().utcoffset() == timedelta(hours=5)


def test_fake_clock_set_jumps_to_the_given_moment(fake_clock):
    target = datetime(2026, 11, 11, 3, 15, tzinfo=TASHKENT)

    fake_clock.set(target)

    assert fake_clock.now() == target


def test_fake_clock_only_moves_when_told(fake_clock):
    assert fake_clock.now() == fake_clock.now()


def test_fake_clock_rejects_naive_datetimes(fake_clock):
    naive = datetime(2026, 11, 10, 10, 0)

    with pytest.raises(ValueError, match="timezone-aware"):
        FakeClock(naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        fake_clock.set(naive)


def test_system_clock_is_aware_and_in_the_given_zone():
    clock = SystemClock(TASHKENT)

    now = clock.now()

    assert now.tzinfo is TASHKENT
    assert abs(now - datetime.now(timezone.utc)) < timedelta(seconds=5)
