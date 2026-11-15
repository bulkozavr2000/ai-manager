"""First messages to new people: their slots when queued, and the Sender's check
(`tasks/first_contacts.py`, `Repo.waiting_first_contacts`, `Repo.last_first_contact_at`)."""

from __future__ import annotations

import random
from dataclasses import replace
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from tg_agent.clock import FakeClock
from tg_agent.config import Settings
from tg_agent.db import FIRST_SENT, Repo
from tg_agent.models import OutboxStatus
from tg_agent.tasks.first_contacts import (
    GAP_MINUTES,
    Hold,
    first_contact_hold,
    is_first_contact,
    next_slot,
)

TASHKENT = ZoneInfo("Asia/Tashkent")
GAP_MIN = timedelta(minutes=GAP_MINUTES[0])
GAP_MAX = timedelta(minutes=GAP_MINUTES[1])


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 11, day, hour, minute, tzinfo=TASHKENT)


TOMORROW_START = at(11, 9)


class People:
    """New people of one task, each with his own conversation."""

    def __init__(self, repo: Repo, clock: FakeClock) -> None:
        self.repo = repo
        self.clock = clock
        self.task_id: int | None = None
        self.peer = 700

    async def new(self) -> int:
        if self.task_id is None:
            self.task_id = await self.repo.create_task(
                "Дрель", "купить дрель", {}, None, "UZS", [], []
            )
        self.peer += 1
        contact_id = await self.repo.upsert_contact(
            self.peer, f"shop{self.peer}", "Shop", False
        )
        return await self.repo.create_conversation(self.task_id, contact_id)

    async def waiting(self, *moments: datetime) -> int:
        """A new person whose first message waits in the outbox (one item per moment)."""
        conv_id = await self.new()
        for moment in moments:
            await self.repo.enqueue(conv_id, "Здравствуйте!", moment)
        return conv_id

    async def written(self, moment: datetime) -> int:
        """A person whose first message went out at `moment` (as `mark_sent` records it)."""
        conv_id = await self.new()
        now = self.clock.now()
        self.clock.set(moment)
        await self.repo.log_event(FIRST_SENT, conv_id)
        self.clock.set(now)
        await self.repo.update_conversation(conv_id, first_sent=True)
        return conv_id


def rng() -> random.Random:
    return random.Random(5)


# -- the Repo --


async def test_waiting_first_contacts_are_the_earliest_item_of_each_new_person(
    repo: Repo, fake_clock: FakeClock
) -> None:
    people = People(repo, fake_clock)
    late = await people.waiting(at(10, 11), at(10, 11, 5))
    await people.waiting(at(10, 10, 30))
    known = await people.waiting(at(10, 10, 40))
    await repo.update_conversation(known, first_sent=True)  # not a first contact
    gone = await people.waiting(at(10, 10, 50))
    await repo.cancel_pending(gone)  # nothing of his waits
    sent = await people.waiting(at(10, 10, 55))
    [item] = await repo.outbox_of(sent)
    await repo.mark_outbox(item.id, OutboxStatus.SENDING)  # being sent: not waiting

    assert await repo.waiting_first_contacts() == [at(10, 10, 30), at(10, 11)]
    assert await repo.waiting_first_contacts(exclude_conv_id=late) == [at(10, 10, 30)]


async def test_a_person_written_to_in_another_conversation_is_no_first_contact(
    repo: Repo, fake_clock: FakeClock
) -> None:
    people = People(repo, fake_clock)
    conv_id = await people.waiting(at(10, 11))
    conv = await repo.get_conversation(conv_id)
    assert not await repo.was_written_to(conv.contact_id)
    assert await is_first_contact(repo, conv)
    # the same person in an earlier task, written to there
    other = await repo.create_task("Ранее", "ранее", {}, None, "UZS", [], [])
    earlier = await repo.create_conversation(other, conv.contact_id)
    await repo.update_conversation(earlier, first_sent=True)

    assert await repo.was_written_to(conv.contact_id)
    assert not await is_first_contact(repo, conv)
    assert await repo.waiting_first_contacts() == []


async def test_the_last_first_contact_is_the_newest_first_sent_event(
    repo: Repo, fake_clock: FakeClock
) -> None:
    people = People(repo, fake_clock)
    assert await repo.last_first_contact_at() is None
    await people.written(at(10, 9, 40))
    await people.written(at(10, 9, 20))
    assert await repo.last_first_contact_at() == at(10, 9, 40)


# 1. Nothing waiting


async def test_in_first_contact_hours_with_nothing_waiting_the_slot_is_now(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    now = fake_clock.now()
    assert await next_slot(repo, settings, now, rng()) == now


async def test_at_night_the_slot_is_the_morning_start(
    repo: Repo, settings: Settings
) -> None:
    assert await next_slot(repo, settings, at(10, 23), rng()) == TOMORROW_START
    assert await next_slot(repo, settings, at(10, 7), rng()) == at(10, 9)


# 2-3. The gap


async def test_one_waiting_at_t_puts_the_slot_10_to_20_minutes_after_t(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    t = at(10, 10, 30)
    await People(repo, fake_clock).waiting(t)

    slot = await next_slot(repo, settings, fake_clock.now(), rng())

    assert t + GAP_MIN <= slot <= t + GAP_MAX


async def test_a_first_message_sent_30_seconds_ago_keeps_the_gap(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    sent = fake_clock.now() - timedelta(seconds=30)
    await People(repo, fake_clock).written(sent)

    slot = await next_slot(repo, settings, fake_clock.now(), rng())

    assert sent + GAP_MIN <= slot <= sent + GAP_MAX


# 4. The quota (3 a day by default: `ramp_new_contacts_per_day`)


async def test_the_quota_used_by_sent_ones_moves_the_slot_to_tomorrow(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    people = People(repo, fake_clock)
    for minute in (10, 30, 50):
        await people.written(at(10, 9, minute))

    assert await next_slot(repo, settings, fake_clock.now(), rng()) == TOMORROW_START


async def test_the_quota_used_by_waiting_ones_moves_the_slot_to_tomorrow(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    people = People(repo, fake_clock)
    for minute in (15, 30, 45):
        await people.waiting(at(10, 10, minute))

    assert await next_slot(repo, settings, fake_clock.now(), rng()) == TOMORROW_START


async def test_the_quota_used_by_both_moves_the_slot_to_tomorrow(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    people = People(repo, fake_clock)
    await people.written(at(10, 9, 10))
    await people.written(at(10, 9, 30))
    await people.waiting(at(10, 10, 15))

    assert await next_slot(repo, settings, fake_clock.now(), rng()) == TOMORROW_START


async def test_tomorrow_full_too_the_slot_is_the_day_after(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    people = People(repo, fake_clock)
    for moment in (at(10, 10, 15), at(10, 10, 30), at(10, 10, 45)):
        await people.waiting(moment)
    for moment in (at(11, 9), at(11, 9, 15), at(11, 9, 30)):
        await people.waiting(moment)

    assert await next_slot(repo, settings, fake_clock.now(), rng()) == at(12, 9)


async def test_the_normal_rate_after_the_ramp(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    ramped = replace(settings, live_since=at(1, 9).date())  # 5 a day by now
    people = People(repo, fake_clock)
    for minute in (10, 30, 50):
        await people.written(at(10, 9, minute))

    slot = await next_slot(repo, ramped, fake_clock.now(), rng())

    assert at(10, 10) <= slot <= at(10, 10, 10)


# 5. The person's own waiting item


async def test_exclude_conv_id_leaves_his_own_waiting_item_out(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    conv_id = await People(repo, fake_clock).waiting(at(10, 10, 30))
    now = fake_clock.now()

    assert await next_slot(repo, settings, now, rng(), exclude_conv_id=conv_id) == now


# 6. The Sender's check


async def test_a_first_message_may_go_in_hours_under_quota_with_no_recent_one(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    people = People(repo, fake_clock)
    await people.written(at(10, 9, 30))
    await people.written(fake_clock.now() - GAP_MIN)

    assert await first_contact_hold(repo, settings, fake_clock.now()) is None


async def test_outside_the_hours_it_waits_for_the_start(
    repo: Repo, settings: Settings
) -> None:
    assert await first_contact_hold(repo, settings, at(10, 7)) == Hold(
        at(10, 9), "hours"
    )


async def test_over_the_quota_it_waits_for_tomorrow(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    people = People(repo, fake_clock)
    for minute in (0, 20, 40):
        await people.written(at(10, 9, minute))

    assert await first_contact_hold(repo, settings, fake_clock.now()) == Hold(
        TOMORROW_START, "quota"
    )


async def test_too_soon_after_the_last_one_it_waits_for_the_gap(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    sent = fake_clock.now() - timedelta(seconds=30)
    await People(repo, fake_clock).written(sent)

    assert await first_contact_hold(repo, settings, fake_clock.now()) == Hold(
        sent + GAP_MIN, "gap"
    )


# Fix pass C14: a waiting slot that is past counts for today.


async def test_three_left_waiting_from_yesterday_fill_today_s_quota(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    fake_clock.set(at(10, 9, 5))
    people = People(repo, fake_clock)
    for minute in (15, 30, 45):
        await people.waiting(at(9, 10, minute))  # an overnight stop held them

    assert await next_slot(repo, settings, fake_clock.now(), rng()) == TOMORROW_START


async def test_one_left_waiting_from_yesterday_waits_now_for_the_gap(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    now = at(10, 9, 5)
    fake_clock.set(now)
    await People(repo, fake_clock).waiting(at(9, 10, 15))

    slot = await next_slot(repo, settings, now, rng())

    assert now + GAP_MIN <= slot <= now + GAP_MAX


async def test_a_reply_waiting_for_a_person_who_wrote_first_is_no_first_contact(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    """Fix pass C12: he wrote to us first, so his reply takes no slot of the new people."""
    conv_id = await People(repo, fake_clock).waiting(at(10, 10, 30))
    await repo.update_conversation(conv_id, last_in_at=at(10, 9, 50))

    assert await repo.waiting_first_contacts() == []
    assert await next_slot(repo, settings, fake_clock.now(), rng()) == fake_clock.now()


def test_first_messages_to_new_people_go_one_to_two_minutes_apart() -> None:
    # Ruling R70 (2026-11-12): the quota and the hours stay; the gap is 1-2 minutes.
    assert GAP_MINUTES == (1, 2)
