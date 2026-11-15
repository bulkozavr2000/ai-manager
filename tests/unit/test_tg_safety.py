from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from tg_agent.config import Limits, Schedule
from tg_agent.policy.tg_safety import (
    TgErrorKind,
    classify_error,
    human_delay,
    in_quiet_hours,
    new_contact_quota,
    next_first_contact_time,
    next_working_time,
    quiet_end,
    stops_everything,
    typing_seconds,
    within_working_hours,
)

TZ = ZoneInfo("Asia/Tashkent")
SCHEDULE = Schedule()
LIMITS = Limits()


def at(hour: int, minute: int = 0, day: int = 10) -> datetime:
    return datetime(2026, 11, day, hour, minute, tzinfo=TZ)


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (at(8, 59), False),
        (at(9, 0), True),
        (at(14, 30), True),
        (at(20, 59), True),
        (at(21, 0), False),
        (at(23, 30), False),
    ],
)
def test_working_hours_boundaries(moment: datetime, expected: bool) -> None:
    assert within_working_hours(moment, SCHEDULE) is expected


def test_working_hours_use_the_schedule_zone_not_the_input_zone() -> None:
    utc_morning = datetime(
        2026, 11, 10, 5, 0, tzinfo=ZoneInfo("UTC")
    )  # 10:00 in Tashkent
    assert within_working_hours(utc_morning, SCHEDULE) is True


def test_next_working_time() -> None:
    assert next_working_time(at(10, 15), SCHEDULE) == at(10, 15)
    assert next_working_time(at(7, 0), SCHEDULE) == at(9, 0)
    assert next_working_time(at(21, 0), SCHEDULE) == at(9, 0, day=11)
    assert next_working_time(at(23, 45), SCHEDULE) == at(9, 0, day=11)


ROUND_THE_CLOCK = Schedule(work_start=time(0), work_end=time(0))


@pytest.mark.parametrize(
    "moment", [at(0, 0), at(3, 30), at(9, 0), at(21, 0), at(23, 59)]
)
def test_the_same_start_and_end_mean_round_the_clock(moment: datetime) -> None:
    # The owner lifted the 09:00-21:00 limit on 2026-11-08: one time for both ends is no limit.
    assert within_working_hours(moment, ROUND_THE_CLOCK) is True
    assert next_working_time(moment, ROUND_THE_CLOCK) == moment


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (at(0, 0), at(9, 0)),
        (at(0, 5), at(9, 0)),
        (at(3, 30), at(9, 0)),
        (at(9, 0), at(9, 0)),
        (at(21, 0), at(21, 0)),
        (at(23, 59), at(23, 59)),
    ],
)
def test_round_the_clock_strangers_are_written_to_from_nine(
    moment: datetime, expected: datetime
) -> None:
    # Row 32 (2026-11-10): round the clock, the next day's first messages went out at 00:00.
    # The agent still works all night; only a first message to a new person waits for 09:00.
    assert next_first_contact_time(moment, ROUND_THE_CLOCK) == expected


@pytest.mark.parametrize("moment", [at(7, 0), at(10, 15), at(21, 0), at(23, 45)])
def test_with_working_hours_strangers_follow_them(moment: datetime) -> None:
    assert next_first_contact_time(moment, SCHEDULE) == next_working_time(
        moment, SCHEDULE
    )


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (at(20, 59), False),
        (at(21, 0), True),
        (at(2, 0), True),
        (at(5, 59), True),
        (at(6, 0), False),
    ],
)
def test_quiet_hours_span_midnight(moment: datetime, expected: bool) -> None:
    assert in_quiet_hours(moment, SCHEDULE) is expected


def test_quiet_end() -> None:
    assert quiet_end(at(22, 0), SCHEDULE) == at(6, 0, day=11)
    assert quiet_end(at(3, 0), SCHEDULE) == at(6, 0)
    assert quiet_end(at(12, 0), SCHEDULE) == at(12, 0)


def test_human_delay_in_range() -> None:
    rng = random.Random(42)
    delays = [human_delay(rng, LIMITS) for _ in range(100)]
    assert all(timedelta(seconds=60) <= d <= timedelta(seconds=180) for d in delays)
    assert len(set(delays)) > 50  # actually random, not a constant


def test_typing_seconds_clamped() -> None:
    assert typing_seconds("ок") == 2
    assert typing_seconds("x" * 70) == pytest.approx(10)
    assert typing_seconds("x" * 1000) == 12


def test_quota_first_week_then_five() -> None:
    live = date(2026, 11, 10)
    assert new_contact_quota(date(2026, 11, 9), None, LIMITS) == 3
    assert new_contact_quota(live, live, LIMITS) == 3
    assert new_contact_quota(live + timedelta(days=6), live, LIMITS) == 3
    assert new_contact_quota(live + timedelta(days=7), live, LIMITS) == 5


@pytest.mark.parametrize(
    ("code", "kind"),
    [
        ("FLOOD_WAIT_120", TgErrorKind.FLOOD_WAIT),
        ("FLOOD_PREMIUM_WAIT_30", TgErrorKind.FLOOD_WAIT),
        ("PEER_FLOOD", TgErrorKind.PEER_FLOOD),
        ("FROZEN_METHOD_INVALID", TgErrorKind.FROZEN),
        ("FROZEN_PARTICIPANT_MISSING", TgErrorKind.FROZEN),
        ("PRIVACY_PREMIUM_REQUIRED", TgErrorKind.PREMIUM_REQUIRED),
        ("ALLOW_PAYMENT_REQUIRED_100", TgErrorKind.PAYMENT_REQUIRED),
        ("ALLOW_PAYMENT_REQUIRED", TgErrorKind.PAYMENT_REQUIRED),
        ("BALANCE_TOO_LOW", TgErrorKind.PAYMENT_REQUIRED),
        ("USER_IS_BLOCKED", TgErrorKind.BLOCKED),
        ("YOU_BLOCKED_USER", TgErrorKind.BLOCKED),
        ("INPUT_USER_DEACTIVATED", TgErrorKind.BLOCKED),
        ("USER_PRIVACY_RESTRICTED", TgErrorKind.PRIVACY),
        # our own session is gone or the account is banned (not a counterpart who is: that is
        # INPUT_USER_DEACTIVATED above)
        ("AUTH_KEY_UNREGISTERED", TgErrorKind.DEAD_SESSION),
        ("AUTH_KEY_INVALID", TgErrorKind.DEAD_SESSION),
        ("AUTH_KEY_DUPLICATED", TgErrorKind.DEAD_SESSION),
        ("SESSION_REVOKED", TgErrorKind.DEAD_SESSION),
        ("SESSION_EXPIRED", TgErrorKind.DEAD_SESSION),
        ("USER_DEACTIVATED", TgErrorKind.DEAD_SESSION),
        ("USER_DEACTIVATED_BAN", TgErrorKind.DEAD_SESSION),
        ("PHONE_NUMBER_BANNED", TgErrorKind.DEAD_SESSION),
        ("session_revoked", TgErrorKind.DEAD_SESSION),
        ("peer_flood", TgErrorKind.PEER_FLOOD),
        ("MESSAGE_TOO_LONG", TgErrorKind.OTHER),
        ("", TgErrorKind.OTHER),
    ],
)
def test_classify_errors(code: str, kind: TgErrorKind) -> None:
    assert classify_error(code) is kind


def test_a_restricted_or_dead_account_stops_everything() -> None:
    stopping = {TgErrorKind.PEER_FLOOD, TgErrorKind.FROZEN, TgErrorKind.DEAD_SESSION}
    for kind in stopping:
        assert stops_everything(kind) is True
    for kind in set(TgErrorKind) - stopping:
        assert stops_everything(kind) is False
