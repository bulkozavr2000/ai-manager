"""Keeping the Telegram account safe: working hours, human-like pacing, new-contact quota, errors.

All times are judged in the schedule's zone (Asia/Tashkent), whatever zone the input carries.
"""

from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta
from enum import StrEnum

from tg_agent.config import Limits, Schedule


def _local(dt: datetime, schedule: Schedule) -> datetime:
    return dt.astimezone(schedule.zone)


def _at(day_dt: datetime, moment: time) -> datetime:
    return day_dt.replace(
        hour=moment.hour, minute=moment.minute, second=0, microsecond=0
    )


def within_working_hours(dt: datetime, schedule: Schedule) -> bool:
    """True inside [work_start, work_end): the only time the agent sends to counterparts.

    The same time for both ends means no limit: the agent sends round the clock.
    """
    if schedule.work_start == schedule.work_end:
        return True
    t = _local(dt, schedule).time()
    return schedule.work_start <= t < schedule.work_end


def next_working_time(dt: datetime, schedule: Schedule) -> datetime:
    """`dt` itself when inside working hours, otherwise the next work_start."""
    local = _local(dt, schedule)
    if within_working_hours(local, schedule):
        return local
    if local.time() < schedule.work_start:
        return _at(local, schedule.work_start)
    return _at(local + timedelta(days=1), schedule.work_start)


def first_contact_start(schedule: Schedule) -> time:
    """When a day's first messages to new people may start.

    With working hours, their start. Round the clock, `new_contacts_start`: the agent answers
    at any hour, but does not write to a stranger at night.
    """
    if schedule.work_start == schedule.work_end:
        return schedule.new_contacts_start
    return schedule.work_start


def next_first_contact_time(dt: datetime, schedule: Schedule) -> datetime:
    """`dt` itself when a first message to a new person may go then, otherwise the next such time."""
    local = _local(dt, schedule)
    start = first_contact_start(schedule)
    if within_working_hours(local, schedule) and local.time() >= start:
        return local
    if local.time() < start:
        return _at(local, start)
    return _at(local + timedelta(days=1), start)


def in_quiet_hours(dt: datetime, schedule: Schedule) -> bool:
    """True when the owner must not be disturbed; the window may span midnight (21:00-06:00)."""
    t = _local(dt, schedule).time()
    if schedule.quiet_start <= schedule.quiet_end:
        return schedule.quiet_start <= t < schedule.quiet_end
    return t >= schedule.quiet_start or t < schedule.quiet_end


def quiet_end(dt: datetime, schedule: Schedule) -> datetime:
    """When the current quiet window ends; `dt` itself outside quiet hours."""
    local = _local(dt, schedule)
    if not in_quiet_hours(local, schedule):
        return local
    end_today = _at(local, schedule.quiet_end)
    return end_today if local < end_today else end_today + timedelta(days=1)


def human_delay(rng: random.Random, limits: Limits) -> timedelta:
    """A pause before sending, drawn uniformly from the configured range."""
    return timedelta(
        seconds=rng.uniform(
            limits.send_delay_min_seconds, limits.send_delay_max_seconds
        )
    )


def typing_seconds(text: str) -> float:
    """How long to show "typing…": about 7 characters a second, between 2 and 12 seconds."""
    return min(max(len(text) / 7, 2.0), 12.0)


def new_contact_quota(day: date, live_since: date | None, limits: Limits) -> int:
    """New counterparts allowed on `day`: the ramp-up rate until `ramp_days` have passed."""
    if live_since is None or day < live_since + timedelta(days=limits.ramp_days):
        return limits.ramp_new_contacts_per_day
    return limits.new_contacts_per_day


class TgErrorKind(StrEnum):
    FLOOD_WAIT = "flood_wait"
    PEER_FLOOD = "peer_flood"
    FROZEN = "frozen"
    DEAD_SESSION = "dead_session"  # our own session is gone, or the account is banned
    PREMIUM_REQUIRED = "premium_required"
    PAYMENT_REQUIRED = "payment_required"
    BLOCKED = "blocked"
    PRIVACY = "privacy"
    OTHER = "other"


_EXACT = {
    "PEER_FLOOD": TgErrorKind.PEER_FLOOD,
    "PRIVACY_PREMIUM_REQUIRED": TgErrorKind.PREMIUM_REQUIRED,
    "BALANCE_TOO_LOW": TgErrorKind.PAYMENT_REQUIRED,
    "USER_IS_BLOCKED": TgErrorKind.BLOCKED,
    "YOU_BLOCKED_USER": TgErrorKind.BLOCKED,
    "INPUT_USER_DEACTIVATED": TgErrorKind.BLOCKED,
    "USER_PRIVACY_RESTRICTED": TgErrorKind.PRIVACY,
    # Our own session is revoked or expired, used from two places at once, or the account is
    # deactivated or banned. (A counterpart who deleted his account is INPUT_USER_DEACTIVATED.)
    "AUTH_KEY_UNREGISTERED": TgErrorKind.DEAD_SESSION,
    "AUTH_KEY_INVALID": TgErrorKind.DEAD_SESSION,
    "AUTH_KEY_DUPLICATED": TgErrorKind.DEAD_SESSION,
    "SESSION_REVOKED": TgErrorKind.DEAD_SESSION,
    "SESSION_EXPIRED": TgErrorKind.DEAD_SESSION,
    "USER_DEACTIVATED": TgErrorKind.DEAD_SESSION,
    "USER_DEACTIVATED_BAN": TgErrorKind.DEAD_SESSION,
    "PHONE_NUMBER_BANNED": TgErrorKind.DEAD_SESSION,
}
_PREFIXES = (
    ("FLOOD_WAIT", TgErrorKind.FLOOD_WAIT),
    ("FLOOD_PREMIUM_WAIT", TgErrorKind.FLOOD_WAIT),
    ("FROZEN_", TgErrorKind.FROZEN),
    ("ALLOW_PAYMENT_REQUIRED", TgErrorKind.PAYMENT_REQUIRED),
)


def classify_error(code: str) -> TgErrorKind:
    """Map a Telegram RPC error code (e.g. "FLOOD_WAIT_120") to what the agent must do about it."""
    code = code.strip().upper()
    if code in _EXACT:
        return _EXACT[code]
    for prefix, kind in _PREFIXES:
        if code.startswith(prefix):
            return kind
    return TgErrorKind.OTHER


def stops_everything(kind: TgErrorKind) -> bool:
    """Errors that mean Telegram is restricting the account, or that the session is dead or the
    account banned: all sending must stop at once."""
    return kind in (
        TgErrorKind.PEER_FLOOD,
        TgErrorKind.FROZEN,
        TgErrorKind.DEAD_SESSION,
    )
