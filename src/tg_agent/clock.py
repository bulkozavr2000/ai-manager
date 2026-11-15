"""Time source. Production code asks a Clock for "now"; tests move a FakeClock by hand."""

from __future__ import annotations

from datetime import datetime, timedelta, tzinfo
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime:
        """The current moment as a timezone-aware datetime."""
        ...


class SystemClock:
    """The real time, expressed in the given zone."""

    def __init__(self, tz: tzinfo) -> None:
        self._tz = tz

    def now(self) -> datetime:
        return datetime.now(self._tz)


class FakeClock:
    """A clock that only moves when told to (`advance` / `set`)."""

    def __init__(self, start: datetime) -> None:
        self._now = _require_aware(start)

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta

    def set(self, dt: datetime) -> None:
        self._now = _require_aware(dt)


def _require_aware(dt: datetime) -> datetime:
    if dt.utcoffset() is None:
        raise ValueError("clock time must be timezone-aware")
    return dt
