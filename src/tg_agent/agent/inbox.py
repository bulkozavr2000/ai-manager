"""The inbox of the agent core: what Telegram reports becomes stored messages and events.

`Inbox` is the `ChatEvents` of the new core. It decides nothing and tells the owner nothing: it writes down what happened and adds the
events (`agent_events`) that wake the agent. In the new core the owner hears from the agent
only through cards, the result and emergencies, so a media message brings him no notice.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from tg_agent.agent import people
from tg_agent.agent.locks import ConvLocks
from tg_agent.agent.unclear import is_file_item, is_unclear_item
from tg_agent.clock import Clock
from tg_agent.config import Settings
from tg_agent.db import Repo
from tg_agent.gateway.base import Incoming
from tg_agent.models import (
    Conversation,
    ConvState,
    Direction,
    EventKind,
    MsgSource,
    OutboxStatus,
    Task,
    TaskStatus,
)
from tg_agent.policy.pregates import is_auto_reply
from tg_agent.ports import SendOutcomes
from tg_agent.tasks.outbox import mark_sent
from tg_agent.tasks.state import TERMINAL
from tg_agent.tasks.texts import DELETED_MARK

S = ConvState

# Earlier messages a new one is compared with to tell a canned reply.
_AUTO_REPLY_LOOKBACK = 50
# The states in which a person's messages wake the agent. While a card about him waits for the
# owner (WAITING_OWNER) or the owner has the chat (HANDED_TO_OWNER) they are only stored.
_TAKING_STATES = frozenset({S.PLANNED, S.CONTACTED, S.TALKING})
# What the owner's own message in a chat makes stale: the agent is not to answer there now.
_OVER = frozenset({TaskStatus.DONE, TaskStatus.CANCELLED})
_OWNER_WROTE_DROPS = (
    EventKind.INCOMING,
    EventKind.EDITED,
    EventKind.SILENCE,
    EventKind.OWNER_WROTE,
)


def _seconds_between(start: datetime | None, end: datetime) -> float | None:
    """Seconds from `start` to `end`; None without a start or when `end` comes first."""
    if start is None or end < start:
        return None
    return (end - start).total_seconds()


def takes_events(task: Task, conv: Conversation, now: datetime) -> bool:
    """Do this conversation's messages wake the agent now? Only in an ACTIVE task, in
    PLANNED, CONTACTED or TALKING, and outside the owner's pause. Otherwise they are only
    stored: the agent sees them when the task resumes, the owner answers or returns the chat,
    or his pause ends."""
    if conv.state not in _TAKING_STATES:
        return False
    if conv.paused_until is not None and conv.paused_until > now:
        return False
    return task.status is TaskStatus.ACTIVE


class Inbox:
    """`ChatEvents` for the new core: store what Telegram reports and add the events that
    wake the agent. It never waits for a wake: every method is a few database writes.

    The event goes in before the message is stored (or edited, or marked deleted) and once
    more after it. Before: a crash between the two then leaves an event without its message,
    and the replay at the next start reads the message again, since it is not stored; the
    other order would lose the event for good. After: an event of the burst that was already
    due may be claimed by a wake between the two writes, and that wake built its context
    without this message; the second call then adds a new event (and only moves the same one
    when nobody took it). `outcomes.on_sent` is awaited while the conversation's lock is held, so an
    implementation of it must never call back into the Inbox.
    """

    def __init__(
        self,
        repo: Repo,
        clock: Clock,
        settings: Settings,
        outcomes: SendOutcomes,
        *,
        locks: ConvLocks | None = None,
    ) -> None:
        self._repo = repo
        self._clock = clock
        self._limits = settings.limits
        self._outcomes = outcomes
        # One report of a conversation at a time: each reads and then writes (a duplicate
        # check, a state move), and two reports of one chat must not interleave. The lock is
        # shared with the send pipeline and the cards when the runner passes one.
        self._locks = locks if locks is not None else ConvLocks()

    # -- what the gateway reports (`ports.ChatEvents`) --

    async def on_incoming(self, msg: Incoming) -> None:
        """A person wrote: store it, end his silence, and (when the conversation takes
        events) push back the one `incoming` event of the burst. Nobody of a task that is
        over is heard (ruling R104 a)."""
        conv = await self._repo.conversation_by_peer(
            msg.peer_id
        ) or await self._repo.closed_conversation_by_peer(msg.peer_id)
        if conv is None:
            return
        async with self._locks.of(conv.id):
            await self._take_incoming(conv.id, msg)

    async def _take_incoming(self, conv_id: int, msg: Incoming) -> None:
        repo = self._repo
        conv = await repo.get_conversation(conv_id)
        if await repo.has_message(conv.id, msg.tg_msg_id):
            # Telegram delivers an update again after a reconnect: it is news only once.
            await repo.log_event("duplicate_incoming", conv.id, tg_msg_id=msg.tg_msg_id)
            return
        if await self._canned(conv, msg):
            # Not an answer: the silence event stays armed and the agent is not woken.
            await self._store_incoming(conv.id, msg)
            await repo.log_event("auto_reply", conv.id, tg_msg_id=msg.tg_msg_id)
            return
        if conv.state in TERMINAL and not await self._reopen(conv):
            await self._store_incoming(conv.id, msg)  # kept; nothing wakes the agent
            return
        # Only from CONTACTED as it is now: a wake may have moved the person since the read
        # above (to a card, to the owner, closed), and there he stays.
        await repo.move_if(conv.id, S.CONTACTED, S.TALKING)
        await self._bump(conv.id)
        await self._store_incoming(conv.id, msg)
        await self._bump(conv.id)  # a wake may have taken the first one meanwhile
        await repo.update_conversation(
            conv.id, last_in_at=self._clock.now(), followups_sent=0
        )
        # An answer ends the silence.
        await repo.drop_events(conv.id, [EventKind.SILENCE])

    async def on_edit(self, peer_id: int, tg_msg_id: int, text: str) -> None:
        """A person edited a message: store the new text; it is news, like a new message."""
        conv = await self._repo.conversation_by_peer(peer_id)
        if conv is None:
            return
        async with self._locks.of(conv.id):
            if not await self._repo.has_message(conv.id, tg_msg_id):
                return
            await self._bump(conv.id)
            now = self._clock.now()
            await self._repo.edit_message(conv.id, tg_msg_id, text, now)
            await self._bump(conv.id)
            # A reply queued for the old text must not go out unseen (the Sender's stale check).
            await self._repo.update_conversation(conv.id, last_in_at=now)

    async def on_delete(self, tg_msg_ids: Sequence[int]) -> None:
        """A person deleted messages: each one kept in an open conversation is marked in the
        transcript, so that the agent sees it was taken back."""
        for tg_msg_id in tg_msg_ids:
            found = await self._repo.find_incoming(tg_msg_id)
            if found is None:
                continue  # not one of ours
            conv_id = found[0]
            async with self._locks.of(conv_id):
                # Read again under the lock: the same delete may be reported twice at once.
                found = await self._repo.find_incoming(tg_msg_id)
                if found is None or found[0] != conv_id:
                    continue
                message = found[1]
                conv = await self._repo.get_conversation(conv_id)
                if conv.state in TERMINAL or message.text.startswith(DELETED_MARK):
                    continue
                await self._bump(conv_id)
                marked = f"{DELETED_MARK} {message.text}".strip()
                await self._repo.edit_message(
                    conv_id, tg_msg_id, marked, self._clock.now()
                )
                await self._bump(conv_id)
                await self._repo.log_event(
                    "incoming_deleted", conv_id, tg_msg_id=tg_msg_id
                )

    async def on_owner_manual(
        self, peer_id: int, text: str, tg_msg_id: int | None = None
    ) -> None:
        """The owner wrote in the chat himself: the agent keeps quiet there for
        `owner_pause_hours`, and an `owner_wrote` event wakes it when the pause ends.

        A message already stored (read again at start-up) changes nothing, and one that is
        the agent's own message whose sending was unclear (UNKNOWN) is written down as sent.
        In a chat handed to the owner (HANDED_TO_OWNER), or of a task that is DONE or
        CANCELLED, his message is only stored: the chat is his until he returns it by the
        button, and a finished task has nobody to wake. In a closed person's chat of a task in
        work it is stored and starts his quiet: the person's messages do not reopen the talk
        until it is over (R104 b). Nobody of a task that is over is heard (R104 a).
        """
        conv = await self._repo.conversation_by_peer(
            peer_id
        ) or await self._repo.closed_conversation_by_peer(peer_id)
        if conv is None:
            return
        async with self._locks.of(conv.id):
            # Read again under the lock: a wake may have closed him meanwhile.
            conv = await self._repo.get_conversation(conv.id)
            if tg_msg_id is not None and await self._repo.has_message(
                conv.id, tg_msg_id
            ):
                await self._repo.log_event(
                    "duplicate_outgoing", conv.id, tg_msg_id=tg_msg_id
                )
                return
            if await self._found_sent(conv, text, tg_msg_id):
                return
            # In a finished task nothing would ever claim the event: the message is stored.
            task = await self._repo.get_task(conv.task_id)
            if conv.state in TERMINAL:
                pause = timedelta(hours=self._limits.owner_pause_hours)
                await self._repo.update_conversation(
                    conv.id, paused_until=self._clock.now() + pause
                )
                await self._repo.cancel_pending(conv.id)
            elif conv.state is not S.HANDED_TO_OWNER and task.status not in _OVER:
                await self._owner_wrote(conv)
            # Stored last: a crash before it is healed by the replay at the next start.
            await self._repo.add_message(
                conv.id,
                Direction.OUT,
                MsgSource.OWNER_MANUAL,
                "text",
                text,
                tg_msg_id,
                self._clock.now(),
            )

    # -- inside --

    async def _reopen(self, conv: Conversation) -> bool:
        """A closed person of a task in work wrote (ruling R104 b): his talk opens again,
        whatever closed it, and his message reaches the agent as any incoming one. Not while
        the owner's own quiet in the chat holds, nor where `people.reopen_closed` keeps him
        shut. (One the owner excluded is not heard at all: `people.listened`.) True when it
        reopened."""
        if conv.paused_until is not None and conv.paused_until > self._clock.now():
            return False
        if not await people.reopen_closed(self._repo, conv.id):
            return False
        await self._repo.log_event("closed_heard_again", conv.id, task_id=conv.task_id)
        return True

    async def _found_sent(
        self, conv: Conversation, text: str, tg_msg_id: int | None
    ) -> bool:
        """Is this the agent's message whose sending was unclear? Then it did go out: write it
        down as sent and tell the core (`on_sent`). False if no such message matches.
        Telegram reports it as it happens (or the replay at start-up does), so it is dated
        now: the report carries no date."""
        now = self._clock.now()
        stored = await self._repo.out_tg_msg_ids(conv.id)
        unclear = [
            item
            for item in await self._repo.outbox_of(conv.id)
            if item.status is OutboxStatus.UNKNOWN
            and is_unclear_item(
                item,
                out=True,
                text=text,
                date=now,
                tg_msg_id=tg_msg_id,
                stored_out_ids=stored,
            )
        ]
        if not unclear:
            return False
        item = unclear[0]
        first = await mark_sent(
            self._repo, item, conv, tg_msg_id, now, is_file=is_file_item(item)
        )
        await self._outcomes.on_sent(item, conv, now, first=first)
        await self._repo.log_event("outbox_found_sent", conv.id, outbox_id=item.id)
        return True

    async def _owner_wrote(self, conv: Conversation) -> None:
        """The owner's pause: the agent keeps quiet in the chat until `paused_until`, and one
        `owner_wrote` event wakes it then. The caller stores his message afterwards."""
        repo = self._repo
        now = self._clock.now()
        paused_until = now + timedelta(hours=self._limits.owner_pause_hours)
        await repo.update_conversation(
            conv.id, paused_until=paused_until, last_out_at=now
        )
        cancelled = await repo.cancel_pending(conv.id)
        # He dealt with the person in the chat: the cards about him no longer hold him.
        await people.settle_cards_in_chat(repo, conv.id)
        # What was waiting is stale now; the agent looks again once the pause is over.
        await repo.drop_events(conv.id, _OWNER_WROTE_DROPS)
        await repo.add_event(
            conv.task_id, EventKind.OWNER_WROTE, conv_id=conv.id, due_at=paused_until
        )
        await repo.log_event("owner_manual", conv.id, cancelled=cancelled)

    async def _canned(self, conv: Conversation, msg: Incoming) -> bool:
        """Is the message a canned auto-reply (`is_auto_reply`)?"""
        earlier = await self._repo.last_messages(conv.id, _AUTO_REPLY_LOOKBACK)
        # A media message without a caption has no text to compare.
        return bool(msg.text.strip() or msg.offline) and is_auto_reply(
            text=msg.text,
            offline_flag=msg.offline,
            seconds_since_our_last=_seconds_between(conv.last_out_at, msg.date),
            previous_incoming=[m.text for m in earlier if m.direction is Direction.IN],
        )

    async def _store_incoming(self, conv_id: int, msg: Incoming) -> None:
        await self._repo.add_message(
            conv_id,
            Direction.IN,
            MsgSource.COUNTERPART,
            msg.kind,
            msg.text,
            msg.tg_msg_id,
            msg.date,
        )

    async def _bump(self, conv_id: int) -> None:
        """The burst rule (`Repo.bump_incoming`), when the conversation takes events."""
        # Read afresh: the state may have just moved (CONTACTED -> TALKING, or by a wake).
        conv = await self._repo.get_conversation(conv_id)
        task = await self._repo.get_task(conv.task_id)
        if not takes_events(task, conv, self._clock.now()):
            return
        await self._repo.bump_incoming(
            conv.task_id,
            conv.id,
            window=timedelta(seconds=self._limits.batch_window_seconds),
            max_wait=timedelta(seconds=self._limits.batch_max_seconds),
        )
