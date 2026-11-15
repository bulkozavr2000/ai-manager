"""First messages to people who never heard from the account: when they may go.

They are what Telegram restricts an account for, so they are paced as the old launch paced them
(`tasks/launch.launch_times`): within each day's quota (`new_contact_quota`), 1-2 minutes
apart (ruling R70; the first of them at once), only in the first-contact hours (`next_first_contact_time`). `next_slot` gives a first
message its time when it is queued, counting the first messages already waiting in the outbox,
so that the agent can say honestly when it goes. `first_contact_hold` is the Sender's check at
send time, which holds whatever the slot said: a slot can go stale (a restart, a flood wait, a
stop), and then only this check keeps the rules.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from typing import Literal, NamedTuple

from tg_agent.config import Settings
from tg_agent.db import Repo
from tg_agent.models import Conversation
from tg_agent.policy.tg_safety import (
    first_contact_start,
    new_contact_quota,
    next_first_contact_time,
)

GAP_MINUTES = (1, 2)  # between first messages to different new people (ruling R70)
# How many days ahead `next_slot` looks for room in the quota: a quota of 0 would otherwise
# never end the search (the Sender's check keeps such a message waiting anyway).
_MAX_DAYS = 366


async def is_first_contact(repo: Repo, conv: Conversation) -> bool:
    """Is a message in this conversation a first contact, paced by the slot and counted on
    the day's quota? Only when nothing went to him here, he never wrote here, and the account
    never wrote to him in another conversation either: Telegram's risk is in cold first
    messages, not in a chat that already exists. The one rule for the slot
    (`agent/outgoing`), the Sender's guard and `tasks/outbox.mark_sent`."""
    if conv.first_sent or conv.last_in_at is not None:
        return False
    return not await repo.was_written_to(conv.contact_id)


class Hold(NamedTuple):
    """A first message may not go now: when to try again, and why."""

    until: datetime
    reason: Literal["hours", "quota", "gap"]


def _quota(day: date, settings: Settings) -> int:
    return new_contact_quota(day, settings.live_since, settings.limits)


async def next_slot(
    repo: Repo,
    settings: Settings,
    now: datetime,
    rng: random.Random,
    *,
    exclude_conv_id: int | None = None,
) -> datetime:
    """When the next first message to a new person may go.

    `launch_times` for one person, with the first messages that wait in the outbox (except
    those of `exclude_conv_id`) in place of the old LAUNCH timers. A waiting slot that is
    already past (an overnight stop, a restart) waits now: it counts on today's quota and
    for the gap from now, not on the day it was once planned for."""
    schedule = settings.schedule
    zone = schedule.zone
    waiting = sorted(
        max(slot, now) for slot in await repo.waiting_first_contacts(exclude_conv_id)
    )
    last_sent = await repo.last_first_contact_at()
    gap = timedelta(minutes=rng.uniform(*GAP_MINUTES))
    earliest = now
    for previous in (waiting[-1] if waiting else None, last_sent):
        if previous is not None:
            earliest = max(earliest, previous + gap)
    moment = next_first_contact_time(earliest, schedule)
    for _ in range(_MAX_DAYS):
        day = moment.astimezone(zone).date()
        held = sum(1 for slot in waiting if slot.astimezone(zone).date() == day)
        if await repo.new_contacts_on(day) + held < _quota(day, settings):
            break
        moment = datetime.combine(
            day + timedelta(days=1), first_contact_start(schedule), zone
        )
    return moment


async def new_people_left(repo: Repo, settings: Settings, now: datetime) -> int:
    """How many more new people may get a first message today: the day's quota, less those
    written to and those whose first message waits for a slot today."""
    zone = settings.schedule.zone
    today = now.astimezone(zone).date()
    waiting = sum(
        1
        for slot in await repo.waiting_first_contacts(None)
        if max(slot, now).astimezone(zone).date() == today
    )
    used = await repo.new_contacts_on(today) + waiting
    return max(0, _quota(today, settings) - used)


async def first_contact_hold(
    repo: Repo, settings: Settings, now: datetime
) -> Hold | None:
    """Why a first message may not go right now, and until when; None when it may. No
    randomness: the slot already drew the gap, this only keeps its least."""
    schedule = settings.schedule
    allowed = next_first_contact_time(now, schedule)
    if allowed > now:
        return Hold(allowed, "hours")
    today = now.astimezone(schedule.zone).date()
    if await repo.new_contacts_on(today) >= _quota(today, settings):
        tomorrow = datetime.combine(
            today + timedelta(days=1), first_contact_start(schedule), schedule.zone
        )
        return Hold(tomorrow, "quota")
    last_sent = await repo.last_first_contact_at()
    least = timedelta(minutes=GAP_MINUTES[0])
    if last_sent is not None and now - last_sent < least:
        return Hold(last_sent + least, "gap")
    return None
