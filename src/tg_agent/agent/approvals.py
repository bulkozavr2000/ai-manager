"""Cards to the owner and his decisions, for the agent core.

`Approvals.ask` is what the agent's `ask_owner` does: it checks what is asked, sends the card,
and holds the person back (nothing goes to him while the owner decides). `owner_decision` is
the owner's press of a card's button (or his reply to it): it settles the approval and adds the
event that wakes the agent with the owner's word; a wrong nonce or a card settled already
answers «Уже решено». Permissions come only from the approval rows written here.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

from tg_agent.agent import people
from tg_agent.agent.locks import ConvLocks
from tg_agent.agent.outgoing import (
    CLOSED,
    NOT_IN_TASK,
    TASK_STOPPED,
    Outgoing,
    Queued,
    since_our_last,
)
from tg_agent.brain.context import hide_amount
from tg_agent.clock import Clock
from tg_agent.config import Settings
from tg_agent.control.base import UI, Card
from tg_agent.control.cards import approval_card
from tg_agent.db import NotFound, Repo
from tg_agent.models import (
    Approval,
    ApprovalKind,
    ApprovalStatus,
    Contact,
    Conversation,
    ConvState,
    Direction,
    EventKind,
    Message,
    OutboxStatus,
    Task,
    TaskStatus,
)
from tg_agent.tasks.state import TERMINAL
from tg_agent.tasks.texts import HANDOFF_SELF, KIND_LABELS

_log = logging.getLogger(__name__)

S = ConvState
K = ApprovalKind
_APPROVED, _REJECTED, _ANSWERED = (
    ApprovalStatus.APPROVED,
    ApprovalStatus.REJECTED,
    ApprovalStatus.ANSWERED,
)

# What the model may ask for (`ask_owner`, `kind`).
_KINDS: Mapping[str, ApprovalKind] = {
    "question": K.QUESTION,
    "approval": K.CONFIRM_ORDER,
    "phone": K.SHARE_PHONE,
    "address": K.SHARE_ADDRESS,
    "handoff": K.HANDOFF,
}
# The owner's buttons per kind and how each is recorded. Only APPROVED grants anything
# (`Repo.granted_kinds`): words on an order card never confirm the order.
_CHOICES: Mapping[ApprovalKind, Mapping[str, ApprovalStatus]] = {
    K.QUESTION: {"ok": _ANSWERED, "answer": _ANSWERED, "self": _ANSWERED},
    K.CONFIRM_ORDER: {
        "yes": _APPROVED,
        "no": _REJECTED,
        "answer": _ANSWERED,
        "self": _ANSWERED,
    },
    K.SHARE_PHONE: {"yes": _APPROVED, "no": _REJECTED, "self": _ANSWERED},
    K.SHARE_ADDRESS: {"yes": _APPROVED, "no": _REJECTED, "self": _ANSWERED},
    K.HANDOFF: {"back": _APPROVED},
}
# A question about the task as a whole has nobody to send a reply to or to hand over.
_TASK_QUESTION_CHOICES: Mapping[str, ApprovalStatus] = {"answer": _ANSWERED}
# A card about a person moves him out of these into WAITING_OWNER (a handoff: HANDED_TO_OWNER).
_TALK_STATES = frozenset({S.PLANNED, S.CONTACTED, S.TALKING})
_SHARED = {K.SHARE_PHONE: "phone", K.SHARE_ADDRESS: "address"}
# The `kind` of the event that carries the owner's words on a card.
_ANSWER_NAMES = {K.QUESTION: "question"}
_SHARED_WORD = {K.SHARE_PHONE: "телефон", K.SHARE_ADDRESS: "адрес"}

ORDER_BUTTONS = [[["Да", "yes"], ["Нет", "no"]], [["Напишу сам", "self"]]]
MAX_QUESTION = 500
MAX_SUGGESTED = 3
MAX_SUGGESTED_LENGTH = 400
# The terms of an order go into the agent's trusted block once approved: kept short.
MAX_TERMS = 300
MAX_RECEIVED = 5
MAX_QUOTE = 300

# For the model (`ask`).
SENT = (
    "Карточка отправлена владельцу. Этому человеку больше ничего не пиши, пока"
    " владелец не ответит."
)
CARD_FAILED = "Карточка не отправилась ({error}). Попробуй ещё раз позже"
SENT_TASK_QUESTION = "Вопрос отправлен владельцу."
UNKNOWN_KIND = (
    "неизвестный kind: {kind} — можно question, approval, phone, address, handoff"
)
NEED_PERSON = "укажи человека: этот вопрос о собеседнике"
EMPTY_QUESTION = "пустой вопрос"
LONG_QUESTION = "вопрос длиннее 500 символов — сократи"
BAD_SUGGESTED = "не больше 3 предложенных ответов, каждый до 400 символов"
NEED_TERMS = "для approval нужна сумма (amount) или условия (terms)"
LONG_TERMS = "условия (terms) длиннее 300 символов — сократи"
ALREADY_ASKED = "уже спрошено, ответа пока нет"
SEND_SUGGESTED = "отправь предложенный ответ"
# «Ок» went through: the agent that asked learns it, or it asks again and the reply goes twice.
# Told when the reply has really gone (ruling R98), or that it did not: the person wrote first.
SENT_SUGGESTED = "Ок: твой предложенный ответ ушёл его словами"
NOT_SENT_SUGGESTED = (
    "Владелец согласился на ответ «{text}», но он не ушёл: человек успел написать. Ответь"
    " ему с учётом нового сообщения"
)
# The approval's payload: the outbox ids of the reply «Ок» queued, and that the agent was told.
OK_ITEMS, OK_TOLD = "ok_items", "ok_told"

# For the owner (`owner_decision`): the button's toast.
ALREADY_DECIDED = "Уже решено"
UNKNOWN_BUTTON = "Неизвестная кнопка"
NEED_TEXT = "Нужен текст ответа"
NO_SUGGESTION = "Готового ответа нет — ответь на карточку своими словами"
PERSON_CLOSED = "С этим человеком уже закончили"
PASSED_ON = "Ответ передан агенту"
SENDING = "Ок, отправляю"
YOURS = "Чат за тобой"
YOURS_LATER = "Чат за тобой. Карточку для возврата пришлю позже"


def _quoted(message: Message) -> str:
    label = KIND_LABELS.get(message.kind, "вложение")
    return message.text.strip() or f"[{label}]"


def _cut(text: str) -> str:
    return text if len(text) <= MAX_QUOTE else text[: MAX_QUOTE - 1] + "…"


def _received(history: Sequence[Message]) -> list[str]:
    """The person's messages since our side last wrote: the latest few, each cut short."""
    return [_cut(_quoted(message)) for message in since_our_last(history)][
        -MAX_RECEIVED:
    ]


def _last_incoming(history: Sequence[Message]) -> str | None:
    for message in reversed(history):
        if message.direction is Direction.IN:
            return _quoted(message)
    return None


def _suggested(approval: Approval) -> list[str]:
    return [
        str(text).strip()
        for text in approval.payload.get("suggested") or []
        if str(text).strip()
    ]


@dataclass(frozen=True)
class _SendSuggested:
    """«Ок» was settled under the lock; the reply is sent once the lock is let go."""

    task: Task
    approval: Approval


class Approvals:
    """Cards to the owner (`ask`) and his decisions on them (`owner_decision`)."""

    def __init__(
        self,
        repo: Repo,
        ui: UI,
        clock: Clock,
        settings: Settings,
        outgoing: Outgoing,
        *,
        render: Callable[
            [Approval, Conversation | None, Contact | None], Card
        ] = approval_card,
        locks: ConvLocks | None = None,
        on_decided: Callable[[int], Awaitable[None]] | None = None,
    ) -> None:
        self._repo = repo
        self._locks = locks if locks is not None else ConvLocks()
        # Called with the task's id after every decision that settled a card (the Runner
        # checks that the task still has something that will wake it).
        self._on_decided = on_decided
        self._ui = ui
        self._clock = clock
        self._limits = settings.limits
        self._outgoing = outgoing
        self._render = render

    # -- the agent asks --

    async def ask(
        self,
        task: Task,
        *,
        kind: str,
        question: str,
        conv_id: int | None,
        suggested: Sequence[str],
        amount: float | None,
        terms: str,
    ) -> tuple[str, bool]:
        """Put a question or an approval to the owner; return (text for the model, is_error).

        A card about a person holds him back: nothing new is queued for him until the owner
        answers. What was queued before the card passed every gate and still goes (the
        holding words the agent writes before it asks: «Уточню и вернусь»). The card goes
        first and the person is moved only once it reached the owner, so a card that does not
        reach him holds nobody: it expires and the person stays where he was. All of it runs
        under the conversation's lock.
        """
        approval_kind = _KINDS.get(kind)
        if approval_kind is None:
            return UNKNOWN_KIND.format(kind=kind), True
        question = question.strip()
        if not question:
            return EMPTY_QUESTION, True
        if len(question) > MAX_QUESTION:
            return LONG_QUESTION, True
        if len(suggested) > MAX_SUGGESTED or any(
            len(text.strip()) > MAX_SUGGESTED_LENGTH for text in suggested
        ):
            return BAD_SUGGESTED, True
        if approval_kind is K.CONFIRM_ORDER and amount is None and not terms.strip():
            return NEED_TERMS, True
        if len(terms.strip()) > MAX_TERMS:
            return LONG_TERMS, True
        clean = [text.strip() for text in suggested if text.strip()]
        if conv_id is None:
            if approval_kind is not K.QUESTION:
                return NEED_PERSON, True
            return await self._put(
                task, None, approval_kind, question, clean, amount, terms.strip()
            )
        try:
            conv = await self._repo.get_conversation(conv_id)
        except NotFound:
            return NOT_IN_TASK, True
        if conv.task_id != task.id:
            return NOT_IN_TASK, True
        async with self._locks.of(conv_id):
            return await self._put(
                task, conv_id, approval_kind, question, clean, amount, terms.strip()
            )

    async def _put(
        self,
        task: Task,
        conv_id: int | None,
        approval_kind: ApprovalKind,
        question: str,
        suggested: list[str],
        amount: float | None,
        terms: str,
    ) -> tuple[str, bool]:
        """The card itself: checked against the conversation as it is now, sent, and only
        then the person held back."""
        conv: Conversation | None = None
        if conv_id is not None:
            conv = await self._repo.get_conversation(conv_id)
            if conv.state in TERMINAL:
                return CLOSED, True
        if await self._asked_already(task, conv_id, approval_kind, question):
            return ALREADY_ASKED, True
        history = (
            await self._repo.last_messages(conv_id, self._limits.history_messages)
            if conv_id is not None
            else []
        )
        payload = self._payload(
            task, approval_kind, question, suggested, amount, terms, history
        )
        if conv is not None and approval_kind is K.QUESTION:
            # «Ок» sends the suggested reply only to the chat as the card shows it.
            payload["seen_message_id"] = await self._repo.max_incoming_id(conv.id)
        # The owner may have paused or closed the task while the model worked: a card made
        # now would wait on a task nobody works on (a close expires only the cards it finds).
        if (await self._repo.get_task(task.id)).status is not TaskStatus.ACTIVE:
            return TASK_STOPPED, True
        approval = await self._repo.create_approval(
            task.id, conv_id, approval_kind, payload
        )
        card = await self._draw(approval, conv)
        try:
            msg_id = await self._ui.send(card)
        except Exception as exc:  # noqa: BLE001 - whatever failed, no card reached him
            return await self._card_failed(approval, exc), True
        # The card is with the owner: an error from here on is ours and is raised, never
        # taken for a card that did not go (it would expire while he holds it).
        await self._shown(approval, msg_id)
        if conv is None:
            return SENT_TASK_QUESTION, False
        await self._hold(conv, handoff=approval_kind is K.HANDOFF)
        if approval_kind is K.HANDOFF:
            # The person does not want an AI, or asks for the owner himself: no later task
            # writes to him until the owner gives the chat back (`_follow`, «Вернуть агенту»).
            await self._repo.set_do_not_contact(conv.contact_id, True)
        return SENT, False

    async def _card_failed(self, approval: Approval, exc: Exception) -> str:
        """The card did not reach the owner: it expires; the person was not moved yet, so he
        stays where he was before `ask` (a handoff too, another card of his pending or not)."""
        repo = self._repo
        await repo.resolve_approval(approval.id, approval.nonce, ApprovalStatus.EXPIRED)
        error = type(exc).__name__
        await repo.log_event(
            "approval_failed",
            approval.conversation_id,
            approval_id=approval.id,
            error=error,
        )
        return CARD_FAILED.format(error=error)

    async def _asked_already(
        self, task: Task, conv_id: int | None, kind: ApprovalKind, question: str
    ) -> bool:
        if conv_id is not None:
            pending = await self._repo.pending_approvals(conv_id)
            return any(approval.kind is kind for approval in pending)
        return any(
            approval.task_id == task.id
            and approval.conversation_id is None
            and approval.kind is kind
            and approval.payload.get("question") == question
            for approval in await self._repo.pending_approvals()
        )

    @staticmethod
    def _payload(
        task: Task,
        kind: ApprovalKind,
        question: str,
        suggested: list[str],
        amount: float | None,
        terms: str,
        history: Sequence[Message],
    ) -> dict[str, object]:
        if kind is K.QUESTION:
            return {
                "task": task.title,
                "about": "",
                "question": question,
                "received": _received(history),
                "suggested": suggested,
            }
        if kind is K.CONFIRM_ORDER:
            return {
                "text": question,
                "amount": amount,
                "terms": terms,
                "currency": task.currency,
                "last_incoming": _last_incoming(history),
                "buttons": ORDER_BUTTONS,
            }
        if kind in _SHARED:
            return {"text": question, "last_incoming": _last_incoming(history)}
        return {"text": question}

    async def _hold(self, conv: Conversation, *, handoff: bool) -> None:
        """While the owner decides, nothing new goes to this person: his conversation waits
        for the owner (a handoff gives him the chat). `Outgoing.send` refuses both."""
        target = S.HANDED_TO_OWNER if handoff else S.WAITING_OWNER
        if conv.state in _TALK_STATES or (handoff and conv.state is S.WAITING_OWNER):
            await self._repo.set_state(conv.id, target)

    async def _draw(self, approval: Approval, conv: Conversation | None) -> Card:
        """The approval's card, ready to send."""
        repo = self._repo
        contact = await repo.get_contact(conv.contact_id) if conv is not None else None
        return self._render(approval, conv, contact)

    async def _shown(self, approval: Approval, msg_id: int) -> None:
        """Remember the message that carries the approval's card."""
        repo = self._repo
        await repo.set_approval_msg(approval.id, msg_id)
        await repo.log_event(
            "approval",
            approval.conversation_id,
            approval_id=approval.id,
            kind=approval.kind.value,
        )

    # -- the owner decides --

    async def owner_decision(
        self, approval_id: int, nonce: str, choice: str, text: str | None = None
    ) -> str:
        """Carry out the owner's press (or his reply, `answer` with `text`) on a card; return
        a short line for him. A wrong nonce or a card already settled: «Уже решено».

        Its writes run under the conversation's lock; «Ок» sends the suggested reply after
        the lock is let go (the send pipeline takes it itself)."""
        try:
            approval = await self._repo.get_approval(approval_id)
        except NotFound:
            return ALREADY_DECIDED
        if approval.conversation_id is None:
            result = await self._decide(approval_id, nonce, choice, text)
        else:
            async with self._locks.of(approval.conversation_id):
                result = await self._decide(approval_id, nonce, choice, text)
        if isinstance(result, _SendSuggested):
            result = await self._send_suggested(result.task, result.approval)
        await self._decided(approval)
        return result

    async def _decided(self, before: Approval) -> None:
        """`on_decided`, when this press settled the card; a failure is only logged: the
        owner's decision is carried out already."""
        if self._on_decided is None or before.status is not ApprovalStatus.PENDING:
            return
        try:
            if (
                await self._repo.get_approval(before.id)
            ).status is ApprovalStatus.PENDING:
                return
            await self._on_decided(before.task_id)
        except Exception:  # noqa: BLE001 - the bot's button must not fail for it
            _log.exception("after the decision on approval %s", before.id)

    async def _decide(
        self, approval_id: int, nonce: str, choice: str, text: str | None
    ) -> str | _SendSuggested:
        repo = self._repo
        approval = await repo.get_approval(approval_id)  # afresh, under the lock
        if approval.status is not ApprovalStatus.PENDING or approval.nonce != nonce:
            return ALREADY_DECIDED
        if approval.conversation_id is None:
            choices = _TASK_QUESTION_CHOICES
        else:
            choices = _CHOICES.get(approval.kind, {})
        status = choices.get(choice)
        if status is None:
            return UNKNOWN_BUTTON
        conv = None
        if approval.conversation_id is not None:
            conv = await repo.get_conversation(approval.conversation_id)
            if conv.state in TERMINAL:
                # Nobody will write to him: whatever the owner chose, nothing is carried
                # out, and he is not told it was.
                return await self._void(approval, choice)
        words = (text or "").strip()
        if choice == "answer" and not words:
            return NEED_TEXT
        suggested = _suggested(approval)
        if choice == "ok" and not suggested:
            return NO_SUGGESTION
        answer = "\n".join(suggested) if choice == "ok" else (words or None)
        if conv is not None and choice == "self":
            # The chat is his before the card is settled: a crash in between leaves a person
            # the owner took with his card still pending (recovery keeps him with the
            # owner, and a second press completes the rest), never one the agent reopens.
            await repo.cancel_pending(conv.id)
            if conv.state is not S.HANDED_TO_OWNER:
                await repo.set_state(conv.id, S.HANDED_TO_OWNER)
        if not await repo.resolve_approval(approval.id, nonce, status, answer):
            return ALREADY_DECIDED
        await repo.log_event(
            "owner_decision",
            approval.conversation_id,
            approval_id=approval.id,
            kind=approval.kind.value,
            choice=choice,
        )
        task = await repo.get_task(approval.task_id)
        if conv is not None and choice == "self":
            return await self._hand_over(task, conv)
        if conv is not None and approval.kind is K.CONFIRM_ORDER and choice == "no":
            # A proposal the agent queued before the card must not go after a «Нет»; it is
            # dropped before the person is reopened, so that one whose only message it was
            # goes back to PLANNED, not CONTACTED.
            await repo.cancel_pending(conv.id)
        if conv is not None:
            await self._reopen(conv.id)
        return await self._follow(task, approval, choice, words)

    async def _void(self, approval: Approval, choice: str) -> str:
        """A card about a person who is closed by now: it expires, no event, no permission."""
        if not await self._repo.resolve_approval(
            approval.id, approval.nonce, ApprovalStatus.EXPIRED
        ):
            return ALREADY_DECIDED
        await self._repo.log_event(
            "owner_decision_void",
            approval.conversation_id,
            approval_id=approval.id,
            kind=approval.kind.value,
            choice=choice,
        )
        return PERSON_CLOSED

    async def _follow(
        self, task: Task, approval: Approval, choice: str, words: str
    ) -> str | _SendSuggested:
        """The owner's word goes to the agent (or, with «Ок», the reply goes out)."""
        base: dict[str, object] = {"approval_id": approval.id}
        kind = approval.kind
        if kind is K.HANDOFF:
            # The owner gives the chat back: the agent may write to him again, here and in
            # later tasks (a handoff the agent asked for marked him "do not contact").
            if approval.conversation_id is not None:
                conv = await self._repo.get_conversation(approval.conversation_id)
                await self._repo.set_do_not_contact(conv.contact_id, False)
            await self._event(approval, EventKind.RESUMED, base)
            return "Агент снова ведёт чат"
        if choice == "ok":
            return _SendSuggested(task, approval)
        if choice == "answer":
            name = _ANSWER_NAMES.get(kind, "approval")
            text = hide_amount(words, task.ceiling)
            event = base | {"kind": name, "text": text}
            await self._event(approval, EventKind.OWNER_ANSWER, event)
            return PASSED_ON
        decision = "approved" if choice == "yes" else "rejected"
        if kind is K.CONFIRM_ORDER:
            payload = approval.payload
            event = base | {"kind": "approval", "decision": decision}
            if choice == "yes":
                # What the owner approved: confirmations up to this amount may go out.
                await self._repo.set_approval_payload(
                    approval.id, {**payload, "approved_amount": payload.get("amount")}
                )
                event |= {
                    "amount": payload.get("amount"),
                    "terms": payload.get("terms"),
                }
            await self._event(approval, EventKind.OWNER_ANSWER, event)
            return "Заказ подтверждён" if choice == "yes" else "Ок, не заказываем"
        await self._event(
            approval,
            EventKind.OWNER_ANSWER,
            base | {"kind": _SHARED[kind], "decision": decision},
        )
        word = _SHARED_WORD[kind]
        return f"Ок, {word} дадим" if choice == "yes" else f"Ок, {word} не дадим"

    async def _send_suggested(self, task: Task, approval: Approval) -> str:
        """«Ок» under the suggested reply: it goes out through the send pipeline. If the
        pipeline refuses it, the agent is asked to write it itself."""
        repo = self._repo
        conv_id = approval.conversation_id
        assert conv_id is not None  # a task-level question takes only `answer`
        conv = await repo.get_conversation(conv_id)
        contact = await repo.get_contact(conv.contact_id)
        suggested = _suggested(approval)
        # The chat as the card showed it: a person who wrote again since makes the reply
        # stale, and the agent decides with his new message in view (the event below).
        seen = approval.payload.get("seen_message_id")
        if not isinstance(seen, int):
            seen = await repo.max_incoming_id(conv_id)
        result = await self._outgoing.send(
            task,
            conv_id,
            suggested,
            language=conv.language or contact.language or "ru",
            commitment=False,
            attach_files=False,
            seen_message_id=seen,
            relay=True,  # the owner's own reply: passed on, never a reminder
            owner_text=True,  # he read it on the card: its promises are his own word
        )
        if isinstance(result, Queued):
            # No event now: `tell_ok` tells the agent once the reply went, or did not.
            ids = [
                item.id
                for item in await repo.outbox_of(conv_id)
                if item.status is OutboxStatus.PENDING and item.wake_id is None
            ]
            payload = (await repo.get_approval(approval.id)).payload
            await repo.set_approval_payload(approval.id, payload | {OK_ITEMS: ids})
            return SENDING
        async with self._locks.of(conv_id):
            await self._event(
                approval,
                EventKind.OWNER_ANSWER,
                {
                    "approval_id": approval.id,
                    "kind": "question",
                    "text": SEND_SUGGESTED,
                    "suggested": suggested,
                },
            )
        return PASSED_ON

    async def tell_ok(self, conv_id: int, *, stale: bool = False) -> None:
        """Ruling R98: the agent that asked hears of each «Ок» once — when the whole reply
        went, or (`stale`: the Sender dropped the chat's queue, the person wrote first) that
        it did not, with the owner's text. Never takes the conversation's lock (the inbox
        calls `on_sent` holding it). An event already waiting for the approval is not added
        again: a crash between the event and its mark leaves it waiting."""
        repo = self._repo
        status = {item.id: item.status for item in await repo.outbox_of(conv_id)}
        for approval in await repo.approvals_of(conv_id):
            ids = approval.payload.get(OK_ITEMS)
            if not ids or approval.payload.get(OK_TOLD):
                continue
            went = [status.get(i) for i in ids]
            if all(s is OutboxStatus.SENT for s in went):
                text = SENT_SUGGESTED
            elif stale and OutboxStatus.CANCELLED in went:
                text = NOT_SENT_SUGGESTED.format(text=" ".join(_suggested(approval)))
            else:
                continue
            if not any(
                event.payload.get("approval_id") == approval.id
                for event in await repo.open_events(approval.task_id)
            ):
                payload = {"approval_id": approval.id, "kind": "question", "text": text}
                await repo.add_event(
                    approval.task_id,
                    EventKind.OWNER_ANSWER,
                    conv_id=conv_id,
                    payload=payload | {"suggested": _suggested(approval)},
                )
            await repo.set_approval_payload(
                approval.id, approval.payload | {OK_TOLD: True}
            )

    async def _hand_over(self, task: Task, conv: Conversation) -> str:
        """«Напишу сам»: the chat is the owner's, with the card that hands it back."""
        repo = self._repo
        await repo.cancel_pending(conv.id)
        conv = await repo.get_conversation(conv.id)
        if conv.state is not S.HANDED_TO_OWNER:
            await repo.set_state(conv.id, S.HANDED_TO_OWNER)
            conv = await repo.get_conversation(conv.id)
        if any(
            card.kind is K.HANDOFF for card in await repo.pending_approvals(conv.id)
        ):
            return (
                YOURS  # a press repeated after a crash: recovery sent the card already
            )
        # The chat is his whether the card goes or not; a card that did not go is sent by
        # the Runner's keep-alive or recovery (a chat handed over with no hand-back card).
        return YOURS if await self.send_hand_back(task.id, conv) else YOURS_LATER

    async def send_hand_back(self, task_id: int, conv: Conversation) -> bool:
        """The card «Вернуть агенту» for a chat the owner has. False when it did not reach
        him: its approval expires, so that the chat counts as one without a card."""
        approval = await self._repo.create_approval(
            task_id, conv.id, K.HANDOFF, {"text": HANDOFF_SELF}
        )
        card = await self._draw(approval, conv)
        try:
            msg_id = await self._ui.send(card)
        except Exception as exc:  # noqa: BLE001 - whatever failed, no card reached him
            await self._card_failed(approval, exc)
            return False
        await self._shown(approval, msg_id)
        return True

    async def _reopen(self, conv_id: int) -> None:
        """Back to where the talk stands, once no other card about him waits."""
        if not await self._repo.pending_approvals(conv_id):
            await people.reopen(self._repo, conv_id)

    async def _event(
        self, approval: Approval, kind: EventKind, payload: dict[str, object]
    ) -> None:
        await self._repo.add_event(
            approval.task_id,
            kind,
            conv_id=approval.conversation_id,
            payload=payload,
            due_at=self._clock.now(),
        )
