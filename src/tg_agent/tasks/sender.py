"""The Sender: sends what the core has queued, to the chat it was queued for.

The core decides what to say; the Sender decides whether it still goes out, and when. The
address is always the conversation's contact, never anything the model wrote. For each message
that is due, in this order:

1. the agent is stopped: the message stays queued and may go out after `/resume_all`;
2. the owner has the chat (he wrote himself), or said not to work with the person (R97):
   everything queued for the chat is dropped; so is it for a first message (as in 4) to a person who is closed, or of a task that is done
   or cancelled: nobody would read the answer, and `first_message_dropped` is logged;
3. the counterpart has written since the message was queued, and the model can still answer
   them: the reply is stale, so everything queued for the chat is dropped and the core decides
   again, now. Where nobody decides any more the message goes out as it is: the goodbye and the
   handoff words of a chat that has ended or been handed over (R22), and the stalling words
   while the owner is being asked;
4. a first message to a person never written to who never wrote to us (`conv.first_sent`
   false, `conv.last_in_at` unset, and no other conversation with him has `first_sent`:
   `tasks/first_contacts.is_first_contact`) outside the day's quota, the first-contact hours or 10
   minutes after the last first message (`tasks/first_contacts.first_contact_hold`):
   everything of the chat due sooner is moved to when it may go, `first_contact_deferred` is
   logged, and the other chats go on. A chat that already had its first message is never held
   by this;
5. outside working hours: it is moved to the next morning;
6. otherwise the message is claimed (PENDING to SENDING, so that nobody else sends or cancels
   it), `{PHONE}` and `{ADDRESS}` are filled in if the owner has approved them, "typing…" shows,
   the checks 1 to 3 are made once more (they take seconds in which anything may happen), and
   the message goes out.

What Telegram answers: a flood wait puts the message back and nothing is sent to anybody until
the wait is over; PEER_FLOOD and FROZEN stop the agent and raise the alarm; a refusal that is
final for this chat (privacy, a block, Premium, a fee, anything else) fails the message and drops
the rest of the chat's queue; the core then closes the person and lets the agent decide.

The Sender knows the core only as `ports.SendOutcomes`: whether a stale reply is to be dropped,
and what became of each message. A message the Sender fails by itself (a missing file, a
placeholder it cannot fill, "typing…" that kept failing, one it could not prepare) goes to the
core first (`on_failed`); the owner is told only when the core does not take care of it.

An error while "typing…" is not a failed send, since nothing has left: a flood wait, a stop of
the account and a final refusal are dealt with as above, and any other error puts the message
back for a minute, three times at most, after which it fails and the owner is told. Stopping
the program while it types puts the message back too.

A message whose send was started, and that is SENDING when the program dies or whose send ends
in an error nobody understands, may or may not have gone out: it becomes UNKNOWN, the owner is
asked, and it is never sent again (`recover`, at start-up). The real phone and address exist
only in the text handed to the gateway: the outbox, the transcript, the events and the
owner's notices keep the placeholders.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path

from tg_agent.clock import Clock
from tg_agent.config import Settings
from tg_agent.control.base import UI
from tg_agent.db import Repo
from tg_agent.gateway.base import Gateway, TgActionError
from tg_agent.models import (
    ApprovalKind,
    Contact,
    Conversation,
    ConvState,
    Direction,
    OutboxItem,
    OutboxStatus,
    TaskStatus,
)
from tg_agent.policy.dlp import (
    ADDRESS_PLACEHOLDER,
    PHONE_PLACEHOLDER,
    placeholders_in,
    substitute_placeholders,
)
from tg_agent.policy.tg_safety import (
    TgErrorKind,
    next_working_time,
    stops_everything,
    typing_seconds,
    within_working_hours,
)
from tg_agent.ports import SendOutcomes
from tg_agent.secrets import Key, get_secret
from tg_agent.tasks.first_contacts import Hold, first_contact_hold, is_first_contact
from tg_agent.tasks.outbox import mark_sent, unclear_text
from tg_agent.tasks.texts import ADVICE, NOT_SENT, who

_log = logging.getLogger(__name__)

# How long to wait when Telegram asks for a wait without saying for how long.
_DEFAULT_WAIT_S = 60
# "Typing…" failed, so nothing was sent: the message is tried again this much later, and this
# many times at most (when the last retry fails too, the message fails).
TYPING_RETRY_AFTER = timedelta(minutes=1)
MAX_TYPING_RETRIES = 3
_DETAIL_CHARS = 200  # how much of Telegram's own explanation goes into the outbox row

# The owner's data a message may ask for: its placeholder, the approval that allows it, where the
# real value is kept, and what it is called in a notice to the owner.
_PERSONAL = (
    (PHONE_PLACEHOLDER, ApprovalKind.SHARE_PHONE, Key.OWNER_PHONE, "телефон"),
    (ADDRESS_PLACEHOLDER, ApprovalKind.SHARE_ADDRESS, Key.OWNER_ADDRESS, "адрес"),
)

# Why a message was refused for good, for the owner (the rest: «Telegram отказал: ...»).
_REFUSALS = {
    TgErrorKind.PRIVACY: "его настройки приватности не дают написать",
    TgErrorKind.BLOCKED: "он заблокировал аккаунт или удалил свой",
    TgErrorKind.PREMIUM_REQUIRED: "ему можно писать только с Telegram Premium",
    TgErrorKind.PAYMENT_REQUIRED: (
        "Telegram просит заплатить за сообщение (Stars агент не тратит)"
    ),
}
# A task in one of these is over: nobody will read an answer to its first message.
_OVER = frozenset({TaskStatus.DONE, TaskStatus.CANCELLED})
_ALARM = (
    "ТРЕВОГА: Telegram ограничивает аккаунт ({kind}). Агент остановлен и ничего не"
    " отправляет. Разберись, и когда ограничение снято, нажми «Продолжить работу» под"
    " закреплённым статусом"
)


class _Verdict(StrEnum):
    """What the checks before sending say about a message."""

    SEND = "send"
    HOLD = "stopped"  # the agent is stopped: the message waits
    CANCEL = "paused"  # the owner has the chat: its queue is dropped
    EXCLUDED = "excluded"  # «не работай с ним» (R97): its queue is dropped
    STALE = "stale"  # the counterpart has written since: dropped, and decided again
    LATER = "later"  # outside working hours
    # A first message to a new person outside the quota, the hours or the gap: moved.
    FIRST_CONTACT = "first_contact"
    # A first message to a person nobody works with any more: dropped, never sent.
    DROPPED = "first_message_dropped"
    UNCLEAR = (
        "unclear"  # an earlier message of the chat may or may not have gone: it waits
    )


class _Step(StrEnum):
    """What came of one message, for the loop over the due ones.

    SKIPPED: not sent, but the next message may be tried. SETTLED: the chat's queue was dealt
    with, and its other messages are left alone in this tick. STOP: nothing more is sent now.
    """

    SENT = "sent"
    SKIPPED = "skipped"
    SETTLED = "settled"
    STOP = "stop"


@dataclass(frozen=True)
class _Outgoing:
    text: str  # what goes out; empty for a file
    path: str | None  # the file to send, if it is one


@dataclass(frozen=True)
class _Unsendable:
    code: str  # for the outbox row and the event log
    reason: str  # for the owner


async def _attempt[T](call: Awaitable[T]) -> tuple[T | None, Exception | None]:
    """Await a gateway call: its result, or the error to deal with.

    Only errors are taken: a crash (BaseException) or a cancelled task goes through, and the
    message stays SENDING for `recover`.
    """
    try:
        return await call, None
    except Exception as error:  # noqa: BLE001 - the caller deals with every kind of error
        return None, error


class Sender:
    """Works off the outbox. `tick` is called by the Scheduler; `recover` once at start-up."""

    def __init__(
        self,
        repo: Repo,
        gateway: Gateway,
        ui: UI,
        clock: Clock,
        settings: Settings,
        outcomes: SendOutcomes,
    ) -> None:
        self._repo = repo
        self._gateway = gateway
        self._ui = ui
        self._clock = clock
        self._settings = settings
        self._outcomes = outcomes
        # While Telegram asks to wait (FLOOD_WAIT) nothing is sent, to any chat.
        self._hold_until: datetime | None = None
        # How often "typing…" failed for each message (by outbox id); forgotten on a restart.
        self._typing_failures: dict[int, int] = {}
        # The newest incoming message marked read, per conversation; forgotten on a restart.
        self._read_up_to: dict[int, int] = {}

    async def tick(self) -> int:
        """Send what is due; return how many messages went out.

        The core's `SendOutcomes` are called from here, outside its locks.
        """
        if self._hold_until is not None and self._clock.now() < self._hold_until:
            return 0
        sent = 0
        dealt_with: set[int] = set()  # chats whose queue this tick has already settled
        for item in await self._repo.due_outbox(self._clock.now()):
            if item.conversation_id in dealt_with:
                continue
            step = await self._process(item)
            if step is _Step.SENT:
                sent += 1
            elif step is _Step.SETTLED:
                dealt_with.add(item.conversation_id)
            elif step is _Step.STOP:
                break
        return sent

    async def recover(self) -> int:
        """After a restart: what was being sent may or may not have gone out.

        Each such message becomes UNKNOWN and the owner is asked about it; none is sent again.
        Returns how many there were.
        """
        items = await self._repo.outbox_in(OutboxStatus.SENDING)
        for item in items:
            await self._repo.mark_outbox(
                item.id, OutboxStatus.UNKNOWN, error="interrupted"
            )
            await self._repo.log_event(
                "outbox_unknown", item.conversation_id, reason="interrupted"
            )
            await self._ask_unclear(item)
        return len(items)

    # -- one message --

    async def _process(self, item: OutboxItem) -> _Step:
        try:
            conv = await self._repo.get_conversation(item.conversation_id)
            verdict = await self._screen(item, conv)
            if verdict is not _Verdict.SEND:
                return await self._hold_back(item, conv, verdict)
            outgoing = await self._outgoing(item, conv)
            claimed = await self._repo.claim_outbox(item.id)
        except Exception as error:  # noqa: BLE001 - one broken message must not block the rest
            return await self._broke(item, error)
        if not claimed:
            return _Step.SKIPPED  # cancelled since the list was read
        if isinstance(outgoing, _Unsendable):
            return await self._unsendable(item, conv, outgoing)
        return await self._deliver(item, conv, outgoing)

    async def _broke(self, item: OutboxItem, error: Exception) -> _Step:
        """Something broke before the message was claimed (the keyring, the database, the
        core). A message that stays first in line and fails again every round would
        block every other chat, so the failure is dealt with here and the next message goes on.

        A message that is still waiting was never sent: it fails, and the owner is told. One that
        is gone (a decision raised after the chat's queue was dropped) has nothing left to fail.
        """
        name = type(error).__name__
        conv_id = item.conversation_id
        _log.error(
            "a message of conversation %s could not be prepared: %s",
            conv_id,
            name,
            exc_info=error,
        )
        failed = False
        try:
            if await self._repo.claim_outbox(item.id):
                await self._repo.mark_outbox(item.id, OutboxStatus.FAILED, error=name)
                failed = True
            await self._repo.log_event(
                "outbox_error", conv_id, reason=name, failed=failed
            )
            if failed:
                conv = await self._repo.get_conversation(conv_id)
                reason = f"не получилось его подготовить ({name})"
                if not await self._core_took(item, conv, name, reason):
                    contact = await self._repo.get_contact(conv.contact_id)
                    notice = NOT_SENT.format(who=who(contact), reason=reason)
                    await self._ui.notify(notice + ADVICE)
        except Exception:  # noqa: BLE001 - the other messages must still go out
            _log.exception(
                "could not deal with the failure of a message of %s", conv_id
            )
        return _Step.SKIPPED if failed else _Step.SETTLED

    async def _screen(self, item: OutboxItem, conv: Conversation) -> _Verdict:
        """The checks before a message is taken, in the order of the module docstring."""
        verdict = await self._recheck(item, conv)
        if verdict is not _Verdict.SEND:
            return verdict
        # One who wrote to us first, or one the account wrote to before, is no stranger.
        if await is_first_contact(self._repo, conv):
            hold = await first_contact_hold(
                self._repo, self._settings, self._clock.now()
            )
            if hold is not None:
                await self._defer_first_contact(conv, hold)
                return _Verdict.FIRST_CONTACT
        if not within_working_hours(self._clock.now(), self._settings.schedule):
            return _Verdict.LATER
        return _Verdict.SEND

    async def _recheck(self, item: OutboxItem, conv: Conversation) -> _Verdict:
        """The checks that are made again after "typing…", which takes seconds: the owner may
        have stopped the agent or written himself, and the counterpart may have written."""
        silenced = await self._silenced(conv)
        if silenced is not None:
            return silenced
        if await self._outcomes.excluded(conv):
            return _Verdict.EXCLUDED
        if await self._nobody_waits(conv):
            return _Verdict.DROPPED
        # A newer message makes the reply stale, but only where the model can answer it: a
        # goodbye, a handoff or a stall is all there will be, so it goes out as it is.
        newer = conv.last_in_at is not None and conv.last_in_at > item.created_at
        if newer and self._outcomes.can_reanswer(conv):
            return _Verdict.STALE
        return _Verdict.SEND

    async def _nobody_waits(self, conv: Conversation) -> bool:
        """Is this a first message (nothing went to him, he never wrote) to a person the agent
        is done with, or of a task that is over? It may have waited hours for its slot; sent
        now it would open a talk nobody listens to (review C1). The core cancels such
        messages itself; this is the belt. Any other message goes: the goodbye of a person
        the account already wrote to, a reply to one who wrote first."""
        if conv.first_sent or conv.last_in_at is not None:
            return False
        if conv.state is ConvState.CLOSED:
            return True
        task = await self._repo.get_task(conv.task_id)
        return task.status in _OVER

    async def _silenced(self, conv: Conversation) -> _Verdict | None:
        """Why nothing may go out in this chat right now, if anything is in the way."""
        if await self._repo.is_stopped():
            return _Verdict.HOLD
        # The owner wrote in the chat himself. (A paused task's queue was cancelled when it
        # was paused, and nothing is queued for a task that is not in work.)
        owner_has_it = (
            conv.paused_until is not None and conv.paused_until > self._clock.now()
        )
        if owner_has_it:
            return _Verdict.CANCEL
        if any(
            queued.status is OutboxStatus.UNKNOWN
            for queued in await self._repo.outbox_of(conv.id)
        ):
            return _Verdict.UNCLEAR
        return None

    async def _defer_first_contact(self, conv: Conversation, hold: Hold) -> None:
        """A first message to a new person may not go now: everything of the chat that is
        queued for sooner waits until `hold.until`, so that the chat's messages keep their
        order."""
        for queued in await self._repo.outbox_of(conv.id):
            if queued.status is OutboxStatus.PENDING and queued.not_before < hold.until:
                await self._repo.reschedule_outbox(queued.id, hold.until)
        await self._repo.log_event(
            "first_contact_deferred", conv.id, reason=hold.reason
        )

    async def _hold_back(
        self, item: OutboxItem, conv: Conversation, verdict: _Verdict
    ) -> _Step:
        """The message is not to go out now (it is still PENDING)."""
        if verdict is _Verdict.HOLD:
            return _Step.STOP
        if verdict is _Verdict.FIRST_CONTACT:
            # The whole chat was moved: its other due messages, read before the move, are
            # left alone in this tick (SETTLED), so the move is logged once.
            return _Step.SETTLED
        if verdict is _Verdict.UNCLEAR:
            return (
                _Step.SETTLED
            )  # it waits for the owner's word about the unclear message
        if verdict is _Verdict.LATER:
            morning = next_working_time(self._clock.now(), self._settings.schedule)
            await self._repo.reschedule_outbox(item.id, morning)
            return _Step.SKIPPED
        cancelled = await self._repo.cancel_pending(conv.id)
        await self._log_cancelled(conv, verdict, cancelled)
        if verdict is _Verdict.STALE:
            await self._outcomes.on_stale(conv.id)
        return _Step.SETTLED

    async def _log_cancelled(
        self, conv: Conversation, verdict: _Verdict, cancelled: int
    ) -> None:
        if verdict is _Verdict.DROPPED:
            await self._repo.log_event(verdict.value, conv.id, items=cancelled)
            return
        await self._repo.log_event(
            "outbox_cancelled", conv.id, reason=verdict.value, items=cancelled
        )

    async def _call_off(
        self, item: OutboxItem, conv: Conversation, verdict: _Verdict
    ) -> _Step:
        """The message is claimed, and after "typing…" it turns out it must not go out."""
        if verdict is _Verdict.HOLD:
            await self._repo.mark_outbox(item.id, OutboxStatus.PENDING)
            return _Step.STOP
        if verdict is _Verdict.UNCLEAR:
            await self._repo.mark_outbox(item.id, OutboxStatus.PENDING)
            return _Step.SETTLED
        await self._repo.mark_outbox(item.id, OutboxStatus.CANCELLED)
        cancelled = 1 + await self._repo.cancel_pending(conv.id)
        await self._log_cancelled(conv, verdict, cancelled)
        if verdict is _Verdict.STALE:
            await self._outcomes.on_stale(conv.id)
        return _Step.SETTLED

    async def _outgoing(
        self, item: OutboxItem, conv: Conversation
    ) -> _Outgoing | _Unsendable:
        """What goes out for the item, or why it cannot go out."""
        if item.attach_path and not item.text.strip():
            path = self._file(item.attach_path)
            if not path.is_file():
                return _Unsendable("file_missing", f"файла «{path.name}» нет на диске")
            return _Outgoing("", str(path))
        wanted = placeholders_in(item.text)
        if not wanted:
            return _Outgoing(item.text, None)
        granted = await self._repo.granted_kinds(conv.id)
        values: dict[str, str] = {}
        for placeholder, kind, key, name in _PERSONAL:
            if placeholder not in wanted:
                continue
            if kind not in granted:
                return _Unsendable(
                    "not_approved",
                    f"в тексте {name} владельца, а разрешения его давать нет",
                )
            value = (get_secret(key) or "").strip()
            if not value:
                return _Unsendable(
                    "no_value", f"нужен {name} владельца, но его нет в хранилище"
                )
            values[placeholder] = value
        text = substitute_placeholders(
            item.text,
            phone=values.get(PHONE_PLACEHOLDER),
            address=values.get(ADDRESS_PLACEHOLDER),
        )
        if placeholders_in(text):  # a placeholder never goes out as it stands
            return _Unsendable("placeholder_left", "в тексте осталась заглушка")
        return _Outgoing(text, None)

    def _file(self, attach_path: str) -> Path:
        """A task's file: its path is kept relative to the data folder unless it is absolute."""
        path = Path(attach_path)
        return path if path.is_absolute() else self._settings.data_dir / path

    async def _unsendable(
        self, item: OutboxItem, conv: Conversation, why: _Unsendable
    ) -> _Step:
        """The message is claimed but cannot be sent: nothing went out, so it simply fails."""
        await self._repo.mark_outbox(item.id, OutboxStatus.FAILED, error=why.code)
        await self._repo.log_event("outbox_failed", conv.id, reason=why.code)
        if not await self._core_took(item, conv, why.code, why.reason):
            contact = await self._repo.get_contact(conv.contact_id)
            await self._ui.notify(NOT_SENT.format(who=who(contact), reason=why.reason))
        return _Step.SKIPPED

    async def _core_took(
        self, item: OutboxItem, conv: Conversation, code: str, reason: str
    ) -> bool:
        """The message failed here, without Telegram's word: does the core take care of it
        (`on_failed`)? Only a plain True counts; an error means the owner is told as before."""
        try:
            return await self._outcomes.on_failed(item, conv, code, reason) is True
        except Exception:  # noqa: BLE001 - the owner must hear about it all the same
            _log.exception("conversation %s: the failure was not dealt with", conv.id)
            return False

    async def _deliver(
        self, item: OutboxItem, conv: Conversation, outgoing: _Outgoing
    ) -> _Step:
        contact = await self._repo.get_contact(conv.contact_id)
        # The chat of the conversation, and no other: the model has no say in it.
        peer = contact.peer_id
        await self._mark_read(conv, peer)
        try:
            _, error = await _attempt(
                self._gateway.typing(peer, typing_seconds(outgoing.text))
            )
        except asyncio.CancelledError:
            # The program is being stopped while it "types": nothing has been sent, so the
            # message waits for the next start, and nobody is asked about it.
            await self._repo.mark_outbox(item.id, OutboxStatus.PENDING)
            raise
        if error is not None:
            return await self._typing_failed(item, conv, contact, error)
        fresh = await self._repo.get_conversation(conv.id)  # as it is after the pause
        verdict = await self._recheck(item, fresh)
        if verdict is not _Verdict.SEND:
            return await self._call_off(item, fresh, verdict)
        send = (
            self._gateway.send_file(peer, outgoing.path)
            if outgoing.path is not None
            else self._gateway.send_text(peer, outgoing.text)
        )
        msg_id, error = await _attempt(send)
        if error is not None:
            return await self._failed(item, conv, contact, error)
        self._typing_failures.pop(item.id, None)
        await self._sent(item, conv, msg_id, is_file=outgoing.path is not None)
        return _Step.SENT

    async def _mark_read(self, conv: Conversation, peer: int) -> None:
        """Read the chat before answering, as a person would: the vendor sees his messages
        read, and the owner's app keeps no unread badge for a chat the agent handles. A
        failure is only logged: it must not hold the message back."""
        newest = await self._repo.last_tg_msg_id(conv.id, Direction.IN)
        if newest is None or self._read_up_to.get(conv.id, 0) >= newest:
            return
        try:
            await self._gateway.mark_read(peer, newest)
            self._read_up_to[conv.id] = newest
        except TgActionError as error:
            _log.warning(
                "conversation %s: not marked read (%s)", conv.id, error.kind.value
            )

    async def _sent(
        self, item: OutboxItem, conv: Conversation, msg_id: int, *, is_file: bool
    ) -> None:
        """Write down that the message went out (`mark_sent`), then tell the core."""
        at = self._clock.now()
        first = await mark_sent(self._repo, item, conv, msg_id, at, is_file=is_file)
        await self._outcomes.on_sent(item, conv, at, first=first)

    # -- when it does not go out --

    async def _typing_failed(
        self,
        item: OutboxItem,
        conv: Conversation,
        contact: Contact,
        error: Exception,
    ) -> _Step:
        """The call that shows "typing…" raised: nothing has been sent, whatever the error.

        What Telegram classifies (a flood wait, a stop of the account, a refusal that is final
        for this chat) is dealt with as it is for a send. Anything else, an error Telegram did
        not classify or one that did not come from Telegram, is tried again a minute later, three
        times at most; then the message fails and the owner is told. Never UNKNOWN: that is for
        a message whose send has been started.
        """
        if isinstance(error, TgActionError) and error.kind is not TgErrorKind.OTHER:
            return await self._refused(item, conv, contact, error)
        name = type(error).__name__
        failures = self._typing_failures.get(item.id, 0) + 1
        await self._repo.log_event(
            "typing_failed", conv.id, attempt=failures, error=name
        )
        if failures <= MAX_TYPING_RETRIES:
            self._typing_failures[item.id] = failures
            await self._put_back(item, conv, self._clock.now() + TYPING_RETRY_AFTER)
            return _Step.SETTLED
        self._typing_failures.pop(item.id, None)
        code = f"typing: {name}"
        await self._repo.mark_outbox(item.id, OutboxStatus.FAILED, error=code)
        await self._repo.log_event(
            "outbox_failed", conv.id, reason="typing", error=name
        )
        reason = f"не получилось начать отправку ({name})"
        if not await self._core_took(item, conv, code, reason):
            notice = NOT_SENT.format(who=who(contact), reason=reason)
            await self._ui.notify(notice + ADVICE)
        return _Step.SETTLED

    async def _put_back(
        self, item: OutboxItem, conv: Conversation, until: datetime
    ) -> None:
        """The claimed message goes back to the queue, to wait until `until`.

        Everything of this chat that is queued for sooner waits as long, so that the chat's
        messages keep their order.
        """
        await self._repo.mark_outbox(item.id, OutboxStatus.PENDING)
        for queued in await self._repo.outbox_of(conv.id):
            if queued.status is OutboxStatus.PENDING and queued.not_before < until:
                await self._repo.reschedule_outbox(queued.id, until)

    async def _failed(
        self,
        item: OutboxItem,
        conv: Conversation,
        contact: Contact,
        error: Exception,
    ) -> _Step:
        """The send itself raised."""
        if isinstance(error, TgActionError):
            return await self._refused(item, conv, contact, error)
        # Not Telegram's answer: nobody knows whether the message left. The text is not logged.
        name = type(error).__name__
        _log.error("sending to conversation %s ended in %s", conv.id, name)
        await self._repo.mark_outbox(item.id, OutboxStatus.UNKNOWN, error=name)
        await self._repo.log_event("outbox_unknown", conv.id, reason=name)
        await self._ask_unclear(item)
        return _Step.STOP

    async def _ask_unclear(self, item: OutboxItem) -> None:
        """The core asks the owner whether the message went (`on_unclear`); should that
        fail, he still gets the question as a plain line."""
        try:
            await self._outcomes.on_unclear(item)
        except Exception:  # noqa: BLE001 - the owner must hear about it all the same
            _log.exception(
                "conversation %s: no card for an unclear send", item.conversation_id
            )
            conv = await self._repo.get_conversation(item.conversation_id)
            contact = await self._repo.get_contact(conv.contact_id)
            await self._ui.notify(unclear_text(item, contact))

    async def _refused(
        self,
        item: OutboxItem,
        conv: Conversation,
        contact: Contact,
        error: TgActionError,
    ) -> _Step:
        kind = error.kind
        if kind is TgErrorKind.FLOOD_WAIT:
            return await self._wait(item, conv, error.wait_s)
        if stops_everything(kind):
            # Not sent: it stays queued and goes out if the owner lets the agent go on.
            await self._repo.mark_outbox(
                item.id, OutboxStatus.PENDING, error=kind.value
            )
            await self._repo.set_stopped(True, why=kind.value)  # the board says why
            await self._repo.log_event("stopped", None, reason=kind.value)
            await self._ui.notify(_ALARM.format(kind=kind.value))
            return _Step.STOP
        detail = f"{kind.value}: {error.detail}"[:_DETAIL_CHARS]
        await self._repo.mark_outbox(item.id, OutboxStatus.FAILED, error=detail)
        cancelled = await self._repo.cancel_pending(conv.id)
        reason = (
            _REFUSALS.get(kind) or f"Telegram отказал: {error.detail[:_DETAIL_CHARS]}"
        )
        try:  # the core closes the person and wakes the agent
            await self._outcomes.on_refused(item, conv, kind, reason, cancelled)
        except Exception:  # noqa: BLE001 - the owner must hear about it all the same
            _log.exception("conversation %s: the refusal was not dealt with", conv.id)
            await self._ui.notify(
                NOT_SENT.format(who=who(contact), reason=reason) + ADVICE
            )
        return _Step.SETTLED

    async def _wait(
        self, item: OutboxItem, conv: Conversation, wait_s: int | None
    ) -> _Step:
        """A flood wait: the message goes back, and nothing is sent to anyone until it is over."""
        seconds = wait_s if wait_s and wait_s > 0 else _DEFAULT_WAIT_S
        until = self._clock.now() + timedelta(seconds=seconds)
        await self._put_back(item, conv, until)
        self._hold_until = until
        await self._repo.log_event("flood_wait", conv.id, seconds=seconds)
        return _Step.STOP
