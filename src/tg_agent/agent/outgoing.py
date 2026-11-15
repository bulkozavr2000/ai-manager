"""The send pipeline of the agent core: the one way a message reaches the outbox.

`Outgoing.send` checks, in this order: who the person is and where his conversation stands,
whether the draft is stale, the texts, the reminder count, the AI introduction, the contacts of
the other people of the task, the gates, the judge, and only then the queue. The first check that fails refuses the send, in
Russian words written for the model, and nothing is queued. An answer to a person who waits
goes after a human pause (`send_delay_*`), any other message at once (ruling R70); a first
message to a new person is queued for its paced slot (`tasks/first_contacts.next_slot`).

`with_own_intro` and `must_introduce` say what a first reply looks like.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import PurePath

from tg_agent.agent import memory, people
from tg_agent.agent.locks import ConvLocks
from tg_agent.brain.context import counterpart_language, hide_amount
from tg_agent.brain.llm import LLM, LLMError
from tg_agent.brain.pricing import price
from tg_agent.clock import Clock
from tg_agent.config import Settings
from tg_agent.db import NotFound, Repo
from tg_agent.gateway.base import Gateway, TgActionError
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
    MsgSource,
    OutboxItem,
    OutboxStatus,
    Task,
    TaskStatus,
)
from tg_agent.policy.dlp import (
    ADDRESS_PLACEHOLDER,
    PHONE_PLACEHOLDER,
    normalize,
)
from tg_agent.policy.postgates import (
    check_outgoing,
    contains_ai_admission,
    contains_commitment,
    extract_amounts,
)
from tg_agent.policy.tg_safety import human_delay, next_working_time
from tg_agent.search.contacts import is_phone, seen_in
from tg_agent.tasks import templates
from tg_agent.tasks.first_contacts import is_first_contact, next_slot
from tg_agent.tasks.outbox import same_text
from tg_agent.tasks.state import TERMINAL

S = ConvState

# Ruling R70: a message nobody waits for (a first message, a reminder, a word after the
# owner's answer) goes at once — a pause of a few seconds and the Sender's «typing…»; the
# human pause of `send_delay_*` is only for answering a person who wrote and waits.
AT_ONCE_SECONDS = (3, 8)

# The words of a refusal, keyed by the rule names of `check_outgoing` and by `judge_<issue>`.
REASONS: dict[str, str] = {
    "pii": "в тексте личные данные (телефон, карта, email, паспорт) — их писать нельзя",
    "link": "в тексте ссылка — ссылки писать нельзя",
    "placeholder_phone": (
        "телефон владельца без его разрешения давать нельзя — сначала спроси:"
        " ask_owner, kind = phone"
    ),
    "placeholder_address": (
        "адрес владельца без его разрешения давать нельзя — сначала спроси:"
        " ask_owner, kind = address"
    ),
    "ai_denial": "нельзя отрицать, что ты ИИ",
    "commitment": (
        "заказ, оплату, бронь и согласие на условия без решения владельца не подтверждай"
        " — сначала спроси: ask_owner, kind = approval"
    ),
    "over_ceiling": (
        "владелец разрешил только то, что записано в блоке «Разрешения»: другую сумму не"
        " подтверждай — сначала спроси: ask_owner, kind = approval"
    ),
    "too_long": "слишком длинно: не больше 3 сообщений до 400 символов",
    "not_russian": (
        "собеседник пишет по-русски: ответь по-русски, без узбекских слов"
        " («спасибо», а не «rahmat»)"
    ),
    "not_uzbek_latin": (
        "собеседник пишет по-узбекски: ответь по-узбекски латиницей, даже если он пишет"
        " кириллицей"
    ),
    "canary": "в тексте служебная метка из твоих правил — её писать нельзя",
    "owner_secret": (
        "в тексте настоящий телефон или адрес владельца — пиши {PHONE} или {ADDRESS},"
        " и только после разрешения"
    ),
    "judge_leak": "в тексте то, что сообщать нельзя",
    "judge_invented_fact": "в тексте факт о владельце или о деле, которого нет в задаче",
    "judge_commitment": "в тексте обещание без разрешения владельца",
    "judge_claims_human": "ты выдаёшь себя за человека",
    "judge_off_policy": "текст нарушает правила",
    "judge": "контролёр не пропустил текст",
    "judge_refused": "контролёр не пропустил текст",
}

NOT_IN_TASK = "в этой задаче нет такого человека"
CLOSED = (
    "С этим человеком уже закончили. Написать ему снова можно, когда об этом попросит"
    " владелец"
)
WITH_OWNER = "этот чат у владельца: он пишет сам"
WAITING_OWNER = "по этому человеку ждём ответа владельца — пока ему ничего не пишем"
OWNER_WRITES = "владелец сам пишет в этом чате — подожди"
STALE = (
    "собеседник написал после начала этого пробуждения — ответ устарел;"
    " новое сообщение увидишь в следующем пробуждении"
)
EMPTY = "пустое сообщение"
TOO_MANY_REMINDERS = (
    "напоминаний без ответа уже {n} — закончи с этим человеком или спроси владельца"
)
OTHER_CONTACT = "не называй одному собеседнику контакты другого"
TASK_STOPPED = "задача остановлена владельцем"
EXCLUDED = "владелец запретил писать этому человеку"
OWNER_REPLY_WAITS = (
    "этому человеку ещё не ушёл ответ, который владелец отправил кнопкой «Ок»: подожди, пока"
    " он уйдёт"
)
ALREADY_SENT = "это сообщение уже ушло"
NAME_AMOUNT = "подтверждая, назови сумму, которую разрешил владелец: {amount}"

# The approval that would let a draft through, per rule; a leak or a wording rule has none.
_ASKS = {
    "commitment": ApprovalKind.CONFIRM_ORDER,
    "over_ceiling": ApprovalKind.CONFIRM_ORDER,
    "placeholder_phone": ApprovalKind.SHARE_PHONE,
    "placeholder_address": ApprovalKind.SHARE_ADDRESS,
}
_LEAKS = frozenset({"canary", "owner_secret"})
# Our message that reached the person, or may have: a reminder follows one of these. A PENDING
# one does not count: the send that is checked now replaces it before it goes.
_WENT = frozenset({OutboxStatus.SENT, OutboxStatus.SENDING, OutboxStatus.UNKNOWN})
# A file queued in one of these is not queued again.
_LIVE = frozenset(OutboxStatus) - {OutboxStatus.CANCELLED, OutboxStatus.FAILED}
# The newest messages of a chat looked at for an edit made after the model's picture of it.
_EDITS_LOOKBACK = 50
# The languages a send names that the conversation remembers («other» says nothing).
_KEPT_LANGUAGES = frozenset({"ru", "uz_latn"})
# What the Sender writes to the outbox row of a task file that is gone from the disk.
FILE_MISSING = "file_missing"


def _by_id(item: OutboxItem) -> int:
    return item.id


# The program's own sentence (`templates.INTRO`) in every language it has, as words: a send
# that already holds one of them is not introduced again (ruling R82). Compared word by word,
# so that case, a dash or an apostrophe do not hide it.
_WORDS = re.compile(r"[^\W_]+")


def _words(text: str) -> str:
    return " " + " ".join(_WORDS.findall(normalize(text))) + " "


_PROGRAM_INTROS = tuple(_words(text) for text in templates.INTRO.values())

# Words that agree to a deal: with a sum beside them, the draft commits the owner whatever
# the model's `commits_owner` says (compared after `normalize`, «ё» read as «е»).
_AGREEING = (
    "берём",
    "беру",
    "возьмём",
    "возьму",
    "давайте за",
    "согласен на",
    "согласна на",
    "согласны на",
    "договорились на",
    "договорились за",
    "оформляйте",
    "оформляем",
    "заказываю",
    "заказываем",
    "подтверждаю",
    "подтверждаем",
    "olamiz",
    "olaman",
    "kelishdik",
    "roziman",
    "tasdiqlayman",
    "buyurtma beraman",
)
_AGREEING_RE = re.compile(
    r"(?<![^\W_])(?:"
    + "|".join(re.escape(normalize(word).replace("ё", "е")) for word in _AGREEING)
    + r")(?![^\W_])"
)
# An apostrophe between digits groups thousands («590'000»): read as one number.
_DIGIT_APOSTROPHE = re.compile(r"(?<=\d)['’‘ʼʻ`´](?=\d)")


@dataclass(frozen=True)
class Queued:
    """The messages are in the outbox."""

    count: int  # messages queued (files not counted)
    send_at: datetime  # when the first of them may go
    # A paced first contact: nothing was ever sent to this person, and he never wrote to us.
    first_contact: bool
    # Task files not queued: the last try to send each found it gone from the disk.
    missing_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class Refused:
    """Nothing was queued; `reason` is in Russian, written for the model."""

    reason: str
    ask: ApprovalKind | None = None  # the owner's approval that would let it through


def reason_for(rules: Sequence[str]) -> str:
    """The words for the model for the rules a draft broke, each once, in their order."""
    return "; ".join(
        REASONS.get(rule, f"текст не прошёл проверку: {rule}")
        for rule in dict.fromkeys(rules)
    )


def _ask_for(rules: Sequence[str]) -> ApprovalKind | None:
    """The approval that would let the draft through: only when every rule it broke is
    lifted by that one approval (a leak or a link stays a refusal whatever the owner says)."""
    kinds = {_ASKS.get(rule) for rule in rules}
    if len(kinds) == 1 and None not in kinds:
        return kinds.pop()
    return None


def with_own_intro(
    messages: Sequence[str], language: str | None, *, introduce: bool
) -> list[str]:
    """The texts as they will go out, one message each, as written. With `introduce` (a first
    reply from an account whose name does not say it is an AI) the counterpart must learn it
    from the chat itself: the program's own sentence goes in right after the greeting of the
    first message, unless the send already holds that very sentence (ruling R82)."""
    texts = [text.strip() for text in messages if text.strip()]
    said = _words(" ".join(texts))
    if texts and introduce and not any(intro in said for intro in _PROGRAM_INTROS):
        intro = templates.pick(templates.INTRO, language)
        texts[0] = templates.introduce(texts[0], intro)
    return texts


# «SI» — sun'iy intellekt, the Uzbek for AI — as a word of its own, in any case: «Said SI»,
# «SI-yordamchi». Not letters inside a word («yordamchisi», «Sirojiddin»).
_NAME_SI = re.compile(r"(?<![^\W_])si(?![^\W_])", re.IGNORECASE)


def name_says_ai(name: str) -> bool:
    """Does the account's name tell that an AI writes (ruling R82)? A word about AI, as in a
    text («ИИ», «AI», «sun'iy intellekt»…), or the Uzbek abbreviation «SI»."""
    return contains_ai_admission(name) or bool(_NAME_SI.search(name))


async def must_introduce(gateway: Gateway, conv: Conversation) -> bool:
    """Must this reply say it comes from an AI? Only the first one, and only while the
    account's name does not say it already («ИИ помощник Саида», «Said SI» show on every
    message). A name that cannot be read counts as one that does not say it."""
    if conv.first_sent:
        return False
    try:
        return not name_says_ai(await gateway.own_name())
    except TgActionError:
        return True


def since_our_last(history: Sequence[Message]) -> list[Message]:
    """The person's messages since our side last wrote, oldest first (the send pipeline and
    the owner's cards read them alike)."""
    found: list[Message] = []
    for message in reversed(history):
        if message.direction is Direction.OUT:
            break
        found.append(message)
    return found[::-1]


def _written_as_contact(username: str, text: str) -> bool:
    name = re.escape(username)
    pattern = rf"(?:(?<![\w.@])@|(?<![\w.])t\.me/){name}(?![A-Za-z0-9_])"
    return re.search(pattern, text, re.IGNORECASE) is not None


def _is_reminder(conv: Conversation, items: Sequence[OutboxItem]) -> bool:
    """Has our side already written after the person's last message?"""
    went = [item for item in items if item.status in _WENT]
    if conv.last_in_at is None:
        return bool(went)
    return any((item.sent_at or item.created_at) > conv.last_in_at for item in went)


def joined_digits(text: str) -> str:
    """The text with an apostrophe between digits removed, so that «590'000» is read as
    590 000 and not as 590 and 0 (the amounts are read from this form, the text goes as
    written)."""
    return _DIGIT_APOSTROPHE.sub("", text)


def agrees_to_a_sum(texts: Sequence[str]) -> bool:
    """Do the texts agree to a deal next to a sum («Хорошо, берём за 590 000», «Согласен на
    590 000»)? Such a draft commits the owner even when the model did not flag it."""
    text = "\n".join(texts)
    folded = normalize(text).replace("ё", "е")
    return bool(_AGREEING_RE.search(folded)) and bool(extract_amounts(text))


def _amount(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


# The judge's hint about orders. The words are chosen for a small model: after the owner's
# «Да» it refused «Саид согласен на 240 000» twice, because the line read like a remark.
NO_ORDER_ALLOWED = "Заказ: владелец не разрешал подтверждать заказ, оплату и условия"
ORDER_ALLOWED = "Заказ: владелец разрешил кнопкой — подтверждать можно в этих пределах:"
# How the agent's context words the owner's «Да» on an order card («Указания»,
# `agent/context._instruction`): such a line is the code's, not his free text.
APPROVED_ORDER = "владелец разрешил подтвердить"
_CARD_QUOTE = 200  # how much of a card's question the hint quotes
# What the judge may object to in the model's text but not in a text the owner approved.
_THE_OWNERS_OWN = frozenset({"judge_commitment", "judge_invented_fact"})


def _approved_orders(approvals: Sequence[Approval]) -> list[Approval]:
    return [
        approval
        for approval in approvals
        if approval.kind is ApprovalKind.CONFIRM_ORDER
        and approval.status is ApprovalStatus.APPROVED
    ]


def approved_amounts(approvals: Sequence[Approval]) -> list[float]:
    """The amounts the owner approved an order at in this conversation, oldest first."""
    return [
        float(amount)
        for approval in _approved_orders(approvals)
        if isinstance(amount := approval.payload.get("approved_amount"), int | float)
        and not isinstance(amount, bool)
    ]


def effective_ceiling(task: Task, approvals: Sequence[Approval]) -> float | None:
    """The limit a commitment is checked against. Once the owner approved an order at an
    amount, that amount (the highest he approved here) is the limit, whatever the task's
    ceiling: a commitment above it goes back to him. An approval of terms only leaves the
    task's ceiling."""
    amounts = approved_amounts(approvals)
    return max(amounts) if amounts else task.ceiling


def names_approved_amount(texts: Sequence[str], amounts: Sequence[float]) -> bool:
    """Does a committing draft name one of the approved amounts (as `extract_amounts` reads
    numbers)? A bare «Да, подтверждаю» would confirm whatever the person last asked for."""
    named = set(extract_amounts("\n".join(texts)))
    return any(amount in named for amount in amounts)


def _gate_language(
    conv: Conversation, contact: Contact, history: Sequence[Message], declared: str
) -> str:
    """The language the reply must be in. The person's latest message settles it when its
    letters are decisive (Uzbek Cyrillic, or letters only Russian has); otherwise the language
    the agent declared for this send holds. The language we wrote in before is not asked: our
    first message goes out before the person has said a word, and he may answer in another."""
    unknown = replace(conv, language=None), replace(contact, language=None)
    return counterpart_language(*unknown, history) or declared


class Outgoing:
    """The send pipeline: checks, introduction, gates, judge, queue."""

    def __init__(
        self,
        repo: Repo,
        gateway: Gateway,
        llm: LLM,
        clock: Clock,
        settings: Settings,
        rng: random.Random,
        *,
        canary: str,
        owner_secrets: Sequence[str],
        locks: ConvLocks | None = None,
    ) -> None:
        self._repo = repo
        self._locks = locks if locks is not None else ConvLocks()
        self._gateway = gateway
        self._llm = llm
        self._clock = clock
        self._settings = settings
        self._limits = settings.limits
        self._rng = rng
        self._canary = canary
        self._owner_secrets = list(owner_secrets)
        # Held from a first contact's slot until its items are in the outbox, so that two
        # sends at once do not take the same slot. Taken only inside a conversation's lock
        # (that one first, then this), never across the gates, the judge or the network.
        self._slots = asyncio.Lock()

    @property
    def locks(self) -> ConvLocks:
        """The per-conversation locks this pipeline queues under (the tools that change a
        person's queue take the same one)."""
        return self._locks

    async def send(
        self,
        task: Task,
        conv_id: int,
        messages: Sequence[str],
        *,
        language: str,
        commitment: bool,
        attach_files: bool,
        seen_message_id: int,
        wake_id: int | None = None,
        instructions: Sequence[str] = (),
        seen_at: datetime | None = None,
        relay: bool = False,
        owner_text: bool = False,
    ) -> Queued | Refused:
        """Check the messages for the person of `conv_id` and queue them, or refuse.

        `seen_message_id` is the highest message the caller has shown the model, and
        `seen_at` when it was shown: a reply to an older picture of the chat (a newer message,
        or one the person edited since) is stale. `instructions` are the owner's, for the
        judge. A second message after ours, with no answer from the person, is a reminder
        (counted, and capped), in whatever wake it is made. `relay`: the send passes on the
        owner's word about this person (after «Уточню и вернусь»); that is not a reminder.
        `owner_text`: the owner read these very messages on a card and pressed «Ок», so what
        they promise or say about him is his own word, and the judge's `commitment` and
        `invented_fact` do not refuse them. Every gate of the code still does: an order still
        needs the order card and its amount.
        """
        repo = self._repo
        # 1. The person.
        try:
            conv = await repo.get_conversation(conv_id)
        except NotFound:
            return Refused(NOT_IN_TASK)
        if conv.task_id != task.id:
            return Refused(NOT_IN_TASK)
        # «Не работай с @x» (ruling R92): nothing goes to him, and no word opens his talk.
        if await memory.is_excluded(repo, await repo.get_contact(conv.contact_id)):
            return Refused(EXCLUDED)
        # The owner's «Ок» reply (queued by no wake) still waits its pause: a line of the
        # agent's own would replace it in the queue, and nobody would learn (review 6b, D4).
        if wake_id is not None and any(
            item.status is OutboxStatus.PENDING and item.wake_id is None
            for item in await repo.outbox_of(conv_id)
        ):
            return Refused(OWNER_REPLY_WAITS)
        if (
            conv.state in TERMINAL
            and wake_id is not None
            and await self._owner_spoke(wake_id)
            and await people.reopen_closed(repo, conv_id)
        ):
            # The owner's word in this wake opens the talk again (ruling R72).
            conv = await repo.get_conversation(conv_id)
        if refusal := self._closed_to_us(conv):
            return refusal
        # 2. Stale.
        if await self._stale(conv_id, seen_message_id, seen_at):
            return Refused(STALE)
        # 3. The texts.
        texts = [text.strip() for text in messages if text.strip()]
        if not texts:
            return Refused(EMPTY)
        # 4. Reminders.
        reminder = not relay and _is_reminder(conv, await repo.outbox_of(conv_id))
        if reminder and conv.followups_sent >= self._limits.max_followups:
            return Refused(TOO_MANY_REMINDERS.format(n=conv.followups_sent))
        # 5. Introduction.
        texts = with_own_intro(
            texts, language, introduce=await must_introduce(self._gateway, conv)
        )
        # 6. Other people of the task.
        if await self._names_another(conv, "\n".join(texts)):
            return Refused(OTHER_CONTACT)
        # 7. The gates.
        contact = await repo.get_contact(conv.contact_id)
        history = await repo.last_messages(conv_id, self._limits.history_messages)
        approvals = await repo.approvals_of(conv_id)
        granted = await repo.granted_kinds(conv_id)
        # The amounts are read from the texts with digit groups joined; agreeing to a sum
        # commits the owner whatever the model's flag says.
        checked = [joined_digits(text) for text in texts]
        commitment = commitment or agrees_to_a_sum(checked)
        gate = check_outgoing(
            checked,
            commitment=commitment,
            granted=granted,
            canaries=[self._canary],
            owner_secrets=self._owner_secrets,
            ceiling=effective_ceiling(task, approvals),
            language=_gate_language(conv, contact, history, language),
        )
        if not gate.ok:
            leaks = [rule for rule in gate.reasons if rule in _LEAKS]
            if leaks:
                await repo.log_event(
                    "leak_blocked", conv_id, task_id=task.id, reasons=gate.reasons
                )
            return Refused(reason_for(gate.reasons), _ask_for(gate.reasons))
        # A confirmation is tied to the amount the owner approved: it must say it.
        amounts = approved_amounts(approvals)
        committed = commitment or contains_commitment("\n".join(checked))
        if committed and amounts and not names_approved_amount(checked, amounts):
            return Refused(NAME_AMOUNT.format(amount=_amount(max(amounts))))
        # 8. The judge.
        hint = self._judge_hint(task, granted, approvals, instructions)
        issues = await self._judge(task, conv_id, texts, hint, wake_id)
        if owner_text:
            issues = [issue for issue in issues if issue not in _THE_OWNERS_OWN]
        if issues:
            return Refused(reason_for(issues))
        # A second look: the judge (and the account's name) took seconds of network time, in
        # which a card about him, the owner taking the chat or his pause, or a newer message
        # may have come, or the owner paused or closed the task. No network wait comes
        # between this look and the queue, and the conversation's lock keeps the inbox and the
        # cards out until the queue is written.
        async with self._locks.of(conv_id):
            if (await repo.get_task(task.id)).status is not TaskStatus.ACTIVE:
                return Refused(TASK_STOPPED)
            conv = await repo.get_conversation(conv_id)
            if refusal := self._closed_to_us(conv):
                return refusal
            if await self._stale(conv_id, seen_message_id, seen_at):
                return Refused(STALE)
            # What the Sender sent while the model thought was queued in the model's picture:
            # the same text again would reach the person twice.
            texts = await self._not_sent_since(conv_id, texts, seen_at)
            if not texts:
                return Refused(ALREADY_SENT)
            # 9-10. The queue.
            return await self._queue(
                task, conv_id, texts, attach_files, reminder, wake_id, language
            )

    async def _stale(
        self, conv_id: int, seen_message_id: int, seen_at: datetime | None
    ) -> bool:
        """Did the person write, or change or take back a message, after the caller's
        picture of the chat?"""
        if await self._repo.max_incoming_id(conv_id) > seen_message_id:
            return True
        if seen_at is None:
            return False
        return any(
            message.direction is Direction.IN
            and message.edited_at is not None
            and message.edited_at > seen_at
            for message in await self._repo.last_messages(conv_id, _EDITS_LOOKBACK)
        )

    async def _not_sent_since(
        self, conv_id: int, texts: list[str], seen_at: datetime | None
    ) -> list[str]:
        """The texts less those identical to an agent message stored after `seen_at`."""
        if seen_at is None:
            return texts
        went = [
            message.text
            for message in await self._repo.last_messages(conv_id, _EDITS_LOOKBACK)
            if message.direction is Direction.OUT
            and message.source is MsgSource.AGENT
            and message.created_at > seen_at
        ]
        return [text for text in texts if not any(same_text(text, w) for w in went)]

    async def _owner_spoke(self, wake_id: int) -> bool:
        """Does this wake carry the owner's instruction or answer for the task?"""
        return any(
            event.kind in (EventKind.OWNER_INSTRUCTION, EventKind.OWNER_ANSWER)
            for event in await self._repo.claimed_events(wake_id)
        )

    def _closed_to_us(self, conv: Conversation) -> Refused | None:
        """Why nothing may go to this person now, if something holds it back."""
        if conv.state in TERMINAL:
            return Refused(CLOSED)
        if conv.state is S.HANDED_TO_OWNER:
            return Refused(WITH_OWNER)
        if conv.state is S.WAITING_OWNER:
            return Refused(WAITING_OWNER)
        if conv.paused_until is not None and conv.paused_until > self._clock.now():
            return Refused(OWNER_WRITES)
        return None

    async def _names_another(self, conv: Conversation, text: str) -> bool:
        """Do the texts give the contact of another person of the task: «@username» or
        «t.me/username» (any letter case), or his phone? A bare word is no contact: a person
        named `tashkent` must not block «in Tashkent»."""
        for other in await self._repo.task_conversations(conv.task_id):
            if other.id == conv.id:
                continue
            contact = await self._repo.get_contact(other.contact_id)
            if contact.username and _written_as_contact(contact.username, text):
                return True
            phone = contact.card.get("contact")
            if isinstance(phone, str) and is_phone(phone) and seen_in(phone, text):
                return True
        # A candidate the search found is another person too (fix pass E1.6, M12).
        mine = await self._repo.get_contact(conv.contact_id)
        own = {
            value.casefold()
            for value in (mine.username, mine.card.get("contact"))
            if isinstance(value, str) and value
        }
        for candidate in await self._repo.task_candidates(conv.task_id):
            found = candidate.contact
            if found.casefold() in own:
                continue
            if is_phone(found) and seen_in(found, text):
                return True
            if not is_phone(found) and _written_as_contact(found, text):
                return True
        return False

    async def _judge(
        self,
        task: Task,
        conv_id: int,
        texts: list[str],
        hint: str,
        wake_id: int | None,
    ) -> list[str]:
        """The judge model's objections (empty: none). Any issue at all is an objection, and
        a judge that cannot answer lets nothing through."""
        try:
            result = await self._llm.judge(texts, self._shareable(task), hint)
        except LLMError as exc:  # a refusal too
            await self._repo.log_event(
                "judge_failed", conv_id, task_id=task.id, error=type(exc).__name__
            )
            return ["judge_refused"]
        await self._repo.log_llm_call(
            conv_id,
            "judge",
            result.model,
            result.usage,
            price(result.model, result.usage),
            result.stop_reason,
            task_id=task.id,
            wake_id=wake_id,
        )
        verdict = result.value
        if verdict.ok and not verdict.issues:
            return []
        issues = [f"judge_{issue}" for issue in dict.fromkeys(verdict.issues)]
        return issues or ["judge"]

    def _shareable(self, task: Task) -> list[str]:
        """What the task lets the agent say about the owner."""
        owner = self._settings.owner
        items = []
        if "name" in task.allowed:
            items.append(f"имя — {owner.full_name}")
        if "city" in task.allowed and owner.city:
            items.append(f"город — {owner.city}")
        return items

    def _judge_hint(
        self,
        task: Task,
        granted: set[ApprovalKind],
        approvals: Sequence[Approval],
        instructions: Sequence[str],
    ) -> str:
        """What the judge may take as known. Never the price ceiling: the judge is a model too,
        so it is masked in the goal and in the owner's words."""
        ceiling = task.ceiling
        phone = "разрешён" if ApprovalKind.SHARE_PHONE in granted else "не разрешён"
        address = "разрешён" if ApprovalKind.SHARE_ADDRESS in granted else "не разрешён"
        lines = [
            f"Задача: {hide_amount(task.goal, ceiling)}",
            f"Телефон владельца ({PHONE_PLACEHOLDER}): {phone}",
            f"Адрес владельца ({ADDRESS_PLACEHOLDER}): {address}",
            *self._order_lines(approvals),
        ]
        # A line the code wrote from an approved order card names what the owner approved
        # by button: no secret from the judge, even when it equals the ceiling (E1.3).
        lines += [
            "Указание владельца: "
            + (text if text.startswith(APPROVED_ORDER) else hide_amount(text, ceiling))
            for text in instructions
            if text.strip()
        ]
        return "\n".join(lines)

    @staticmethod
    def _order_lines(approvals: Sequence[Approval]) -> list[str]:
        """What the owner allowed about an order, in words the judge cannot misread: every
        order card he approved, oldest first (a later card about a prepayment must not narrow
        an earlier one about the order), each with its amount, its terms and what the card
        asked him."""
        orders = _approved_orders(approvals)
        if not orders:
            return [NO_ORDER_ALLOWED]
        lines = [ORDER_ALLOWED]
        for order in orders:
            payload = order.payload
            parts = []
            amount = payload.get("approved_amount")
            if isinstance(amount, int | float) and not isinstance(amount, bool):
                parts.append(f"сумма до {_amount(amount)}")
            terms = " ".join(str(payload.get("terms") or "").split())
            if terms:
                parts.append(f"условия: {terms}")
            asked = " ".join(str(payload.get("text") or "").split())
            if asked:
                parts.append(f"на карточке было: «{asked[:_CARD_QUOTE]}»")
            lines.append("• " + ("; ".join(parts) or "без суммы и условий"))
        return lines

    async def _queue(
        self,
        task: Task,
        conv_id: int,
        texts: list[str],
        attach_files: bool,
        reminder: bool,
        wake_id: int | None,
        language: str,
    ) -> Queued:
        repo = self._repo
        now = self._clock.now()
        before = await repo.get_conversation(conv_id)
        # He waits for an answer when his newest message is newer than our newest one that
        # went out (one still queued is replaced by this very send).
        waiting = before.last_in_at is not None and (
            before.last_out_at is None or before.last_in_at > before.last_out_at
        )
        if waiting:
            pause = human_delay(self._rng, self._limits)
        else:
            pause = timedelta(seconds=self._rng.uniform(*AT_ONCE_SECONDS))
        send_at = next_working_time(now + pause, self._settings.schedule)
        # A first contact: nothing went to him, he never wrote to us, and the account never
        # wrote to him before. One who wrote first, or one written to in another task, is no
        # stranger to the account, and his message is not paced.
        first_contact = await is_first_contact(repo, before)
        # The slot and the enqueue are one step (`_slots`), for first contacts only.
        async with self._slots if first_contact else contextlib.nullcontext():
            kept = None
            if first_contact:
                # A re-send keeps his place: the slot of his first message that still waits.
                kept = min(
                    (
                        item.not_before
                        for item in await repo.outbox_of(conv_id)
                        if item.status is OutboxStatus.PENDING and item.not_before > now
                    ),
                    default=None,
                )
            # A newer reply replaces one that has not gone yet.
            await repo.cancel_pending(conv_id)
            if kept is not None:
                send_at = kept
            elif first_contact:
                # A first message to a new person waits for its paced slot. Counted after
                # his own older items are cancelled, so that he does not queue behind
                # himself.
                slot = await next_slot(
                    repo, self._settings, now, self._rng, exclude_conv_id=conv_id
                )
                send_at = max(send_at, slot)
            missing = await self._enqueue(
                task, conv_id, texts, attach_files, send_at, wake_id
            )
        conv = await repo.get_conversation(conv_id)
        fields: dict[str, object] = {"agent_messages": conv.agent_messages + len(texts)}
        if reminder:
            fields["followups_sent"] = conv.followups_sent + 1
        if language in _KEPT_LANGUAGES:
            # What the person is written in: the goodbye and the «Ок» button read it.
            fields["language"] = language
        await repo.update_conversation(conv_id, **fields)
        if conv.state is S.PLANNED:
            await repo.set_state(conv_id, S.CONTACTED)
            if conv.last_in_at is not None:
                # He wrote first: our message answers him, the talk is on.
                await repo.set_state(conv_id, S.TALKING)
        return Queued(
            len(texts),
            send_at,
            first_contact=first_contact,
            missing_files=tuple(missing),
        )

    async def _enqueue(
        self,
        task: Task,
        conv_id: int,
        texts: list[str],
        attach_files: bool,
        send_at: datetime,
        wake_id: int | None,
    ) -> list[str]:
        """The texts, then the task files not sent yet; the names of the files gone from the
        disk."""
        repo = self._repo
        for text in texts:
            await repo.enqueue(conv_id, text, send_at, wake_id=wake_id)
        missing: list[str] = []
        if attach_files:
            items = await repo.outbox_of(conv_id)
            taken = {item.attach_path for item in items if item.status in _LIVE}
            # The newest item of each file: one that failed for a missing file would fail
            # again, and every try writes the text to the person once more.
            latest = {item.attach_path: item for item in sorted(items, key=_by_id)}
            for path in task.attachments:
                if path in taken:
                    continue
                last = latest.get(path)
                if (
                    last is not None
                    and last.status is OutboxStatus.FAILED
                    and last.error == FILE_MISSING
                ):
                    missing.append(PurePath(path).name)
                    continue
                await repo.enqueue(
                    conv_id, "", send_at, attach_path=path, wake_id=wake_id
                )
        return missing
