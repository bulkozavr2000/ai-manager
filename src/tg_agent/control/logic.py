"""What the control bot does with the owner's messages, commands and button presses.

Only the owner's Telegram ids are obeyed; anyone else gets nothing and changes nothing. The
owner talks to the bot in words: his free text is read by the secretary (`agent/secretary`,
ruling R57), which makes and rewrites drafts, launches them, passes instructions and answers
questions. The fast paths need no model (ruling R58): buttons (they carry one-time nonces), a
reply to an approval's card (his answer), a reply to the card of a task in work or finished
(an instruction for its agent), and the commands `/stop`, `/help`, `/start`.

The bot works on the agent core: what the owner does to a task goes through `agent/ops.TaskOps`,
his decisions on cards through the runner's `agent/approvals.Approvals`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Protocol

from tg_agent.agent.approvals import ALREADY_DECIDED, UNKNOWN_BUTTON, Approvals
from tg_agent.agent.ops import TaskOps
from tg_agent.agent.people import Person, task_people_names, task_targets
from tg_agent.agent.secretary import (
    ANSWERED_BY_WORDS,
    OwnerMessage,
    Secretary,
    SecretaryReply,
    draft_missing,
    draft_nonce_key,
    raise_month_limit,
    take_person_into,
)
from tg_agent.agent.tools import (
    allowance_key,
    money_card_key,
    money_limit,
    spend_cap,
    task_allowance,
)
from tg_agent.brain.llm import LLM
from tg_agent.clock import Clock
from tg_agent.config import Settings
from tg_agent.control.base import UI, Card, CardGone
from tg_agent.control.cards import (
    MONEY_MORE,
    PICK_FROM_CONTACTS,
    PICKER_DATA,
    RESUME_DATA,
    TAKES_PERSON,
    WAITING_MARK,
    board_card,
    cannot_write_first,
    draft_card,
    draft_closed_card,
    help_text,
    launched_card,
    nobody_card,
    person_label,
    pick_person_card,
    pick_person_key,
    picked_card,
    result_waits_key,
)
from tg_agent.control.layout import (
    ChatLayout,
    approval_card_key,
    task_card_key,
)
from tg_agent.db import NotFound, Repo
from tg_agent.gateway.base import Gateway
from tg_agent.models import (
    Approval,
    ApprovalStatus,
    Contact,
    Conversation,
    Task,
    TaskStatus,
)
from tg_agent.search.contacts import normalize_contact, normalize_phone

_log = logging.getLogger(__name__)

TASK_CLOSED = "Эта задача уже закрыта. Новую — просто напиши боту."
NOT_CLOSING = "Задача #{id}: не закрываю"
# A draft's card that shows what the draft held before it was rewritten (review M3).
CARD_OUTDATED = "Карточка устарела — внизу текущая"
# What the board shows: the tasks in work, not drafts and not finished or closed ones.
_ON_BOARD = (TaskStatus.ACTIVE, TaskStatus.PAUSED)
# A reply to a task's card in these is an instruction for its agent (`TaskOps.instruct`).
_INSTRUCTED = (TaskStatus.ACTIVE, TaskStatus.PAUSED, TaskStatus.DONE)
BUTTONS_ONLY = "На эту карточку отвечают кнопками"
FILE_NOT_INSTRUCTED = "Файл к указанию не прикладывается: пришли его новой задачей"
FILE_NEEDS_WORDS = "Файл к указанию не прикладывается: напиши указание словами"
FILE_SAVED = "Файл сохранён. Пришли текст задачи — приложу его к ней."
NOT_LAUNCHED = "Никому не написал — почему, в сообщении ниже"
LAUNCHED = "Запущено"
# The «запущена» card of a task with nobody in it: its agent finds people itself (R26).
FINDS_PEOPLE = "Кому писать — ищу сам. Ход виден в закреплённом статусе."
# The line the «запущена» card gains when the agent will not start on it now (ruling R42).
LAUNCHED_STOPPED = (
    "Агент сейчас остановлен: нажми «Продолжить работу» под закреплённым статусом"
    " — и он начнёт."
)
LAUNCHED_STOPPED_BY_TELEGRAM = (
    "Агент сейчас остановлен: Telegram ограничил аккаунт. Когда ограничение снято,"
    " нажми «Продолжить работу» под закреплённым статусом."
)
LAUNCHED_MONTH_SPENT = (
    "Лимит денег на месяц исчерпан: агент начнёт, когда лимит поднимут"
    " или наступит новый месяц."
)
LAUNCHED_DAY_SPENT = "Лимит денег на сегодня исчерпан: агент начнёт завтра."
STOPPED = "Всё остановлено. Продолжить — кнопка под закреплённым статусом"
RESUMED = "Работаю дальше"
ALREADY_WORKING = "Агент уже работает"
PENDING_FILES_KEY = "pending_files"
NO_COMMANDS = (
    "Команды больше не нужны — напиши словами: «что по задачам?»,"
    " «поставь задачу 3 на паузу»"
)
NOT_TEXT = (
    "Пока понимаю только текст. Напиши словами — или пришли файл вместе с задачей."
)
# The owner's messages that wait to be read together (ruling R71): kept in kv, so that a
# stop of the program loses none (`handle_waiting_burst` at start-up).
BURST_KEY = "owner_burst"
# A turn the secretary reads now; one found at start-up was cut by the stop (review m1).
READING_KEY = "owner_burst_reading"
TYPING_EVERY = 4.0  # seconds: Telegram shows "typing…" for five
QUOTED_CHARS = 60  # how much of each message a line quotes
NOT_FINISHED = (
    "Не успел обработать перед перезапуском: {quoted}. Напиши, пожалуйста, ещё раз"
)
BURST_BROKEN = (
    "Что-то сломалось ({error}), когда я читал твои последние сообщения. Посмотри, что"
    " успело сделаться, и напиши ещё раз, если нужно"
)
DROPPED = "Последние сообщения не выполнил: {quoted}"
BURST_MAX = timedelta(seconds=30)  # no burst waits longer, from its first message
EDITS_NOT_READ = "Исправления я не читаю — напиши поправку новым сообщением"
# A person by name (Task 5.4).
NOBODY_FOUND = (
    "Не нашёл такого человека среди контактов рабочего аккаунта. Выбери его в списке"
    " Telegram — кнопка ниже — или пришли его контакт."
)
NOBODY_PICKED = (
    "Тогда выбери его в списке Telegram — кнопка ниже — или пришли его контакт."
)
NOBODY_TOAST = "Никого не беру"
# What a contact card is in the burst and the chat log (the person rides beside it).
PERSON_SENT = "владелец прислал контакт"
PICKED_TOAST = "Выбран: {name}"

# Runs `job` after `delay` seconds: the bot's event loop in production, a test's list.
Timer = Callable[[float, Callable[[], Awaitable[None]]], None]


class SiteButtons(Protocol):
    """The buttons of a sign-in card (`agent/site_logins.SiteLogins.button`)."""

    async def button(self, number: int, action: str, nonce: str) -> str: ...


def _quoted(texts: Sequence[str]) -> str:
    """The owner's messages quoted on one line, each cut."""

    def cut(text: str) -> str:
        line = " ".join(text.split())
        return line if len(line) <= QUOTED_CHARS else line[: QUOTED_CHARS - 1] + "…"

    return ", ".join(f"«{cut(text)}»" for text in texts)


def _nonce_key(task_id: int) -> str:
    return draft_nonce_key(task_id)


def _finish_key(task_id: int) -> str:
    """The one-time code of the «Закрыть задачу?» card."""
    return f"finish_nonce:{task_id}"


class ControlLogic:
    def __init__(
        self,
        repo: Repo,
        ops: TaskOps,
        approvals: Approvals,
        llm: LLM,
        settings: Settings,
        clock: Clock,
        ui: UI,
        *,
        gateway: Gateway | None = None,
        site_logins: SiteButtons | None = None,
    ) -> None:
        self.repo = repo
        # «Войти» and «Не сейчас» of a sign-in card (stage 4б); None: those buttons are
        # unknown.
        self.site_logins = site_logins
        self.ops = ops
        self.approvals = approvals
        self.llm = llm
        self.settings = settings
        self.clock = clock
        # The work account: its address book for a person by name, its link for one who
        # cannot be written to first (Task 5.4). None: only earlier people are found.
        self.gateway = gateway
        # Everything the logic shows goes through the chat's layout (stage 5, R52); a bare
        # `UI` (tests) gets one of its own. `ui` stays what was given.
        self.ui = ui
        self.layout = (
            ui if isinstance(ui, ChatLayout) else ChatLayout(repo, ui, clock=clock)
        )
        if self.layout.status is None:
            self.layout.status = self.status_card
        # The one reader of the owner's free text (ruling R57). Its callbacks are this
        # logic's own, so that a draft's card, a launch by words and a close by words look
        # exactly as the buttons make them.
        self.secretary = Secretary(
            repo, llm, ops, approvals, clock, settings,
            show_draft=self.show_draft, launch=self.launch,
            confirm_close=self.confirm_close, gateway=gateway,
            ask_which=self.ask_which,
        )  # fmt: skip
        # Without a timer (tests, the evals) a free text is read at once; the running bot
        # sets one (`main.run`), and the owner's messages of a burst are read together.
        self.timer: Timer | None = None
        self._burst_lock = asyncio.Lock()
        # Every read-modify-write of the burst and of the waiting files (review I1).
        self._kv_lock = asyncio.Lock()
        self._burst_round = 0

    def is_owner(self, user_id: int) -> bool:
        return user_id in self.settings.owner.control_user_ids

    # -- text and files --

    async def on_text(
        self,
        user_id: int,
        text: str,
        reply_to_msg_id: int | None,
        attachments: Sequence[str],
        *,
        forwarded: bool = False,
    ) -> list[Card]:
        """The fast paths first (ruling R58), with no model: a reply to an approval's card is
        the owner's answer; a reply to the card of a task in work, paused or finished is an
        instruction for its agent. A reply to a draft's card, and any other text, is the
        secretary's to read (ruling R57)."""
        if not self.is_owner(user_id) or not text.strip():
            return []
        if reply_to_msg_id is not None:
            approval = await self._replied_approval(reply_to_msg_id)
            if approval is not None:
                if approval.status is not ApprovalStatus.PENDING:
                    return [Card(ALREADY_DECIDED)]
                if approval.kind not in ANSWERED_BY_WORDS:
                    return [Card(BUTTONS_ONLY)]
                return [
                    Card(
                        await self.approvals.owner_decision(
                            approval.id, approval.nonce, "answer", text.strip()
                        )
                    )
                ]
            replied = await self.repo.get_kv(task_card_key(reply_to_msg_id))
            if replied:
                task = await self.repo.get_task(int(replied))
                if task.status is TaskStatus.DRAFT:
                    return await self._to_secretary(text, task.id, attachments)
                return await self._to_task(task.id, text, attachments)
        return await self._to_secretary(text, None, attachments, forwarded=forwarded)

    async def on_person(
        self,
        user_id: int,
        name: str,
        username: str | None,
        phone: str | None,
        reply_to_msg_id: int | None = None,
    ) -> list[Card]:
        """A contact card, or a person he picked in Telegram's list (Task 5.4): he chose
        him himself, so no card asks «этому?». The person joins the burst as a part of the
        turn — with a text of the same turn it is one task; alone, the secretary asks what
        to do with him. One the work account cannot open (no @username, no phone it can
        see) is said so plainly, with the account's link (ruling R61); no model."""
        if not self.is_owner(user_id):
            return []
        target = normalize_contact(username) if username else None
        if target is None and phone:
            # Telegram gives a contact's number without «+»; a local one is Uzbek.
            target = normalize_phone(phone) or normalize_phone(f"+{phone.lstrip('+')}")
        if target is None:
            return [Card(cannot_write_first(name, await self.secretary.own_username()))]
        reply_to_draft = None
        if reply_to_msg_id is not None:
            replied = await self.repo.get_kv(task_card_key(reply_to_msg_id))
            if (
                replied
                and (await self.repo.get_task(int(replied))).status is TaskStatus.DRAFT
            ):
                reply_to_draft = int(replied)
        return await self._to_secretary(
            PERSON_SENT, reply_to_draft, [], person=(name.strip(), target)
        )

    async def on_edit(self, user_id: int) -> list[Card]:
        """The owner edited a text or a caption: an edit is no correction (the
        simplification pass), and he is told so. No model is asked."""
        return [Card(EDITS_NOT_READ)] if self.is_owner(user_id) else []

    async def _replied_approval(self, msg_id: int) -> Approval | None:
        """The approval whose card is message `msg_id` — now, or before the layout moved
        the card (the owner replied to it as he saw it, or to the line left in its place)."""
        approval = await self.repo.approval_by_control_msg(msg_id)
        if approval is not None:
            return approval
        moved = await self.repo.get_kv(approval_card_key(msg_id))
        if not moved:
            return None
        try:
            return await self.repo.get_approval(int(moved))
        except NotFound:
            return None

    async def _to_secretary(
        self,
        text: str,
        reply_to_draft: int | None,
        attachments: Sequence[str],
        *,
        person: tuple[str, str] | None = None,
        forwarded: bool = False,
    ) -> list[Card]:
        """The message joins the owner's burst; the secretary reads the burst once it ends
        (ruling R71) — at once when there is no timer. Its files wait from now on."""
        if attachments:
            await self._keep_files(attachments)
        first = await self._add_to_burst(
            OwnerMessage(text, reply_to_draft, person, forwarded)
        )
        if self.timer is None:
            return await self.read_burst()
        await self._typing()  # "typing…" while the bot waits, and again while it reads
        self._restart_timer(first)
        return []

    def _restart_timer(self, first: datetime) -> None:
        """The burst's window starts again: a newer message makes the older timer a no-op."""
        assert self.timer is not None
        self._burst_round += 1
        round_ = self._burst_round
        window = timedelta(seconds=self.settings.limits.owner_burst_seconds)
        delay = min(window, first + BURST_MAX - self.clock.now())
        self.timer(max(delay.total_seconds(), 0.0), lambda: self._burst_due(round_))

    async def _typing(self) -> None:
        try:
            await self.layout.typing()
        except Exception as error:  # noqa: BLE001 - only a courtesy
            _log.info("could not show typing: %s", error)

    async def _add_to_burst(self, message: OwnerMessage) -> datetime:
        """Keep the message with the burst that waits; returns when the burst began. Under
        the kv lock: two messages handled at once both stay (review I1)."""
        async with self._kv_lock:
            raw = await self.repo.get_kv(BURST_KEY)
            burst = json.loads(raw) if raw else {"first": self.clock.now().isoformat()}
            entry = {"text": message.text, "reply_to_draft": message.reply_to_draft}
            if message.person is not None:
                entry["person"] = list(message.person)
            if message.forwarded:
                entry["forwarded"] = True
            burst.setdefault("messages", []).append(entry)
            await self.repo.set_kv(BURST_KEY, json.dumps(burst, ensure_ascii=False))
            return datetime.fromisoformat(burst["first"])

    async def _waiting_burst(self) -> datetime | None:
        """When the burst that waits began, or None."""
        raw = await self.repo.get_kv(BURST_KEY)
        return datetime.fromisoformat(json.loads(raw)["first"]) if raw else None

    async def _burst_due(self, round_: int) -> None:
        """A burst's timer: a newer message of the same burst set a later one. A failure is
        logged, and the owner hears of it (review m1)."""
        if round_ != self._burst_round:
            return
        await self.read_waiting_burst()

    async def read_waiting_burst(self) -> None:
        """`handle_waiting_burst` that never raises: its timer, and the start-up (review
        M4: a refused send must not stop the program)."""
        try:
            await self.handle_waiting_burst()
        except Exception as error:  # noqa: BLE001 - the owner must hear of it
            _log.exception("the owner's burst could not be read")
            try:
                await self.show(Card(BURST_BROKEN.format(error=type(error).__name__)))
            except Exception:  # noqa: BLE001 - nothing more can be done
                _log.exception("could not tell the owner about the failure")

    async def handle_waiting_burst(self) -> None:
        """Read the burst that waits, show what came of it, put the chat right: when its
        timer fires, and at start-up for one a stop of the program left waiting. A turn a
        stop cut in the middle is not run again: the owner is told which messages to send
        once more (review m1). With nothing waiting nothing happens."""
        cut_short = await self._take_cut_turn()
        if cut_short:
            await self.show(Card(NOT_FINISHED.format(quoted=_quoted(cut_short))))
        if await self.repo.get_kv(BURST_KEY):
            for card in await self.read_burst():
                await self.show(card)
        elif not cut_short:
            return
        await self.layout.settle()

    async def _take_cut_turn(self) -> list[str]:
        """The messages of a turn that was being read when the program stopped (start-up
        only: no turn is being read then), forgotten once taken."""
        if self._burst_lock.locked():
            return []
        async with self._kv_lock:
            raw = await self.repo.get_kv(READING_KEY)
            if not raw:
                return []
            await self.repo.set_kv(READING_KEY, "")
        return [m["text"] for m in json.loads(raw)["messages"]]

    async def drop_burst(self) -> list[str]:
        """`/stop`: the waiting messages are not carried out; their texts (ruling R71b)."""
        async with self._kv_lock:
            raw = await self.repo.get_kv(BURST_KEY)
            await self.repo.set_kv(BURST_KEY, "")
        self._burst_round += 1  # its timer does nothing now
        return [m["text"] for m in json.loads(raw)["messages"]] if raw else []

    async def read_burst(self) -> list[Card]:
        """The secretary reads the waiting messages as one turn, with the files that came
        with them and those that wait. A file stops waiting when a draft holds it after the
        turn: a turn that made no draft (small talk, or the model failed) loses none. The
        turn is kept as "being read" until it ends, so that a stop in the middle loses it
        neither silently nor runs it twice (review m1)."""
        async with self._burst_lock:
            async with self._kv_lock:
                raw = await self.repo.get_kv(BURST_KEY)
                if not raw:
                    return []
                await self.repo.set_kv(READING_KEY, raw)
                await self.repo.set_kv(BURST_KEY, "")
            messages = [
                OwnerMessage(
                    m["text"],
                    m.get("reply_to_draft"),
                    tuple(m["person"]) if m.get("person") else None,
                    bool(m.get("forwarded")),
                )
                for m in json.loads(raw)["messages"]
            ]
            try:
                pending = await self._pending_files()
                async with self._typing_meanwhile():
                    reply = await self.secretary.handle_messages(
                        messages, files=pending
                    )
                cards = await self._after_turn(reply, pending)
            except Exception:
                # A failure is not a stop: the owner hears of it now, not at the next turn
                # as «перед перезапуском» (review M3). A stop of the program (a cancel)
                # leaves the mark for the next start (review m1).
                await self.repo.set_kv(READING_KEY, "")
                raise
            await self.repo.set_kv(READING_KEY, "")
            return cards

    @contextlib.asynccontextmanager
    async def _typing_meanwhile(self) -> AsyncIterator[None]:
        """Show "typing…" again every few seconds while the secretary reads: Telegram shows it
        for five seconds only (review m7)."""

        async def keep() -> None:
            while True:
                await self._typing()
                await asyncio.sleep(TYPING_EVERY)

        keeper = asyncio.create_task(keep())
        try:
            yield
        finally:
            keeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await keeper

    async def _after_turn(
        self, reply: SecretaryReply, pending: list[str]
    ) -> list[Card]:
        """Every draft the turn touched shows what the database holds; the files a draft
        holds wait no more; the secretary's words are the reply."""
        held: set[str] = set()
        for task_id in reply.drafts:
            task = await self.repo.get_task(task_id)
            held.update(task.attachments)
            if task.status is TaskStatus.DRAFT:
                await self._redraw_draft(task_id)
        await self._drop_pending_files([path for path in pending if path in held])
        if reply.picker:
            # The search by name found nobody: Telegram's own list, one tap away.
            picker = [[(PICK_FROM_CONTACTS, PICKER_DATA)]]
            return [Card(reply.text or NOBODY_FOUND, picker)]
        return [Card(reply.text, reply.buttons)] if reply.text else []

    async def _draft_card(self, task_id: int) -> Card:
        """A draft's card drawn from the database; the draft keeps its one-time code, so
        the buttons of a card drawn again still work."""
        task = await self.repo.get_task(task_id)
        nonce = await self.repo.get_kv(_nonce_key(task_id))
        if not nonce:
            nonce = secrets.token_hex(3)
            await self.repo.set_kv(_nonce_key(task_id), nonce)
        targets = await task_targets(self.repo, task_id)
        missing = await draft_missing(self.repo, task_id)
        names = await task_people_names(self.repo, task_id)
        return draft_card(task, targets, missing, nonce, names)

    async def show_draft(self, task_id: int) -> int:
        """The secretary's callback: the draft's card as the database holds it. A draft
        with no card on screen gets one; a draft whose card is on screen has it replaced by
        the new one, at the bottom: what the owner just changed is in front of him, and he
        never sees two cards of one draft. A draft without contacts starts nothing before
        «Запустить»: after it, the task's agent finds people itself (ruling R26)."""
        card = await self._draft_card(task_id)
        msg_id = await self.layout.replace("task", task_id, card)
        if msg_id is None:
            msg_id = await self.show(card)
        return msg_id

    async def _redraw_draft(self, task_id: int) -> None:
        """Once the message is done, a draft's card equals the database: a later draft of
        the same message may have taken the files an earlier card showed. In place."""
        await self.layout.update("task", task_id, await self._draft_card(task_id))

    async def launch(self, task_id: int) -> tuple[bool, list[str]]:
        """The secretary's callback: what «Запустить» does (the draft's card turns into
        «запущена» in place), and whether the task started, with the notes."""
        nonce = await self.repo.get_kv(_nonce_key(task_id)) or ""
        card, launched, notes = await self._start(task_id, nonce)
        if card is not None:
            await self.show(card)
        return launched, notes

    async def confirm_close(self, task_id: int) -> str | None:
        """The secretary's callback for «закрой». A draft is closed at once — nobody was
        written to — and its card settles as «закрыта»; the line is returned. A task in work
        gets «Закрыть задачу?» with its buttons: closing it may write goodbyes, so the owner
        closes it with the button, never by a word alone."""
        task = await self.repo.get_task(task_id)
        if task.status is TaskStatus.DRAFT:
            await self.repo.set_kv(_nonce_key(task_id), "")
            line = await self.ops.close(task_id)
            await self.layout.release(
                "task", task_id, draft_closed_card(task_id, task.title)
            )
            return line
        await self.layout.send(await self._finish_ask(task_id))
        return None

    # -- a person by name (Task 5.4, rulings R60, R61) --

    async def ask_which(self, task_id: int, persons: Sequence[Person]) -> int:
        """The secretary's callback when the search by name found several: one card
        «❓ Кого ты имел в виду?» with a button each and «Никто из них». The matches it
        offers are kept until he presses (they are forgotten then): nothing else of the
        address book is."""
        task = await self.repo.get_task(task_id)
        nonce = secrets.token_hex(3)
        record = {"nonce": nonce, "people": [asdict(person) for person in persons]}
        await self.repo.set_kv(
            pick_person_key(task_id), json.dumps(record, ensure_ascii=False)
        )
        labels = [person_label(person.name, person.target) for person in persons]
        return await self.layout.send(
            pick_person_card(task_id, task.title, labels, nonce)
        )

    async def _pick_button(self, task_id: int, choice: str, nonce: str) -> str:
        """A press on «Кого ты имел в виду?»: that person is taken (no second card — he
        chose him); «Никто из них» takes nobody, and the bot asks for his contact."""
        key = pick_person_key(task_id)
        raw = await self.repo.get_kv(key)
        record = json.loads(raw) if raw else None
        if record is None or not secrets.compare_digest(record["nonce"], nonce):
            return ALREADY_DECIDED
        persons = [Person(**data) for data in record["people"]]
        if choice != "none" and not (choice.isdigit() and int(choice) < len(persons)):
            return UNKNOWN_BUTTON
        await self.repo.set_kv(key, "")
        task = await self.repo.get_task(task_id)
        if (
            task.status not in TAKES_PERSON
        ):  # closed meanwhile: its card settles by itself
            return ALREADY_DECIDED
        if choice == "none":
            await self.layout.release("pick_person", task_id, nobody_card())
            await self.show(Card(NOBODY_PICKED, [[(PICK_FROM_CONTACTS, PICKER_DATA)]]))
            return NOBODY_TOAST
        person = persons[int(choice)]
        await self.layout.release("pick_person", task_id, picked_card(person.name))
        toast = PICKED_TOAST.format(name=person.name)
        if person.target is None:
            await self.show(
                Card(
                    cannot_write_first(person.name, await self.secretary.own_username())
                )
            )
            return toast
        line, refused = await take_person_into(self.repo, self.ops, task, person)
        if task.status is TaskStatus.DRAFT:
            await self.show_draft(task_id)  # with him on it, at the bottom
            return toast
        return line if refused else f"{toast}. {line}"

    async def _to_task(
        self, task_id: int, text: str, attachments: Sequence[str]
    ) -> list[Card]:
        """The owner's reply to the card of a task in work, paused or finished: an
        instruction for its agent (a finished task is taken up again)."""
        task = await self.repo.get_task(task_id)
        if task.status not in _INSTRUCTED:
            return [Card(TASK_CLOSED)]
        line = await self.ops.instruct(task_id, text)
        if attachments:
            # The agent sends only the task's own files; a file would go nowhere.
            line += "\n" + FILE_NOT_INSTRUCTED
        return [Card(line)]

    async def show(self, card: Card) -> int:
        """Send a card to the owner; a task's card is remembered, so a reply to it corrects
        its draft or instructs its agent."""
        msg_id = await self.layout.send(card)
        if card.task_id is not None:
            await self.repo.set_kv(task_card_key(msg_id), str(card.task_id))
        return msg_id

    async def on_file(
        self,
        user_id: int,
        path: str,
        caption: str,
        reply_to_msg_id: int | None = None,
        *,
        forwarded: bool = False,
    ) -> list[Card]:
        """A file with a caption is the secretary's to read: the caption is the text, the
        files are this one and those that wait. Without a caption the file waits for the
        text. A file sent as a reply to a draft's card belongs to that draft: with a caption
        the secretary reads it as a correction; without one it joins the draft's files, no
        model asked. For a task in work the caption is an instruction and the file goes
        nowhere (it is not kept for the next task either)."""
        if not self.is_owner(user_id):
            return []
        if reply_to_msg_id is not None:
            replied = await self.repo.get_kv(task_card_key(reply_to_msg_id))
            if replied:
                return await self._file_to_task(int(replied), path, caption)
        if caption.strip():
            return await self._to_secretary(caption, None, [path], forwarded=forwarded)
        await self._keep_files([path])
        first = await self._waiting_burst()
        if first is not None and self.timer is not None:
            # Sent right after the text (a photo of the part): it goes with that turn, and
            # the window starts again (review I3).
            await self._typing()
            self._restart_timer(first)
            return []
        return [Card(FILE_SAVED)]

    async def _file_to_task(self, task_id: int, path: str, caption: str) -> list[Card]:
        task = await self.repo.get_task(task_id)
        if task.status is TaskStatus.DRAFT and caption.strip():
            return await self._to_secretary(caption, task_id, [path])
        if task.status is TaskStatus.DRAFT:
            return await self._file_to_draft(task, path)
        if task.status not in _INSTRUCTED:
            return [Card(TASK_CLOSED)]
        if not caption.strip():
            return [Card(FILE_NEEDS_WORDS)]
        return await self._to_task(task_id, caption, [path])

    async def _file_to_draft(self, task: Task, path: str) -> list[Card]:
        """A file alone, sent as a reply to a draft's card: it joins the draft's files and
        the card is drawn again; nothing else of the draft changes."""
        files = [*task.attachments, *([path] if path not in task.attachments else [])]
        if not await self.repo.rewrite_draft(
            task.id, title=task.title, goal=task.goal, ceiling=task.ceiling,
            currency=task.currency, allowed=task.allowed, attachments=files,
            source_text=task.source_text,
        ):  # fmt: skip
            # Launched or closed in the meantime: the file waits for a task instead.
            await self._keep_files([path])
            return [Card(FILE_SAVED)]
        await self.show_draft(task.id)
        return []

    async def _pending_files(self) -> list[str]:
        """The files the owner sent before his text, still waiting for a task."""
        return list(json.loads(await self.repo.get_kv(PENDING_FILES_KEY) or "[]"))

    async def _keep_files(self, paths: Sequence[str]) -> None:
        """These files wait for a task from now on (until a draft holds them). Under the kv
        lock: the photos of one album arrive together (review I1)."""
        async with self._kv_lock:
            pending = await self._pending_files()
            pending += [path for path in paths if path not in pending]
            await self.repo.set_kv(
                PENDING_FILES_KEY, json.dumps(pending, ensure_ascii=False)
            )

    async def _drop_pending_files(self, taken: Sequence[str]) -> None:
        """A draft took these files: they wait no more (one sent meanwhile still does)."""
        if not taken:
            return
        async with self._kv_lock:
            left = [path for path in await self._pending_files() if path not in taken]
            await self.repo.set_kv(
                PENDING_FILES_KEY, json.dumps(left, ensure_ascii=False)
            )

    # -- commands --

    async def on_command(self, user_id: int, name: str, args: str) -> list[Card]:
        """Three commands stay (ruling R64): /stop stops everything at once, /help and
        /start say how to talk to the bot. Every other one says to write in words."""
        if not self.is_owner(user_id):
            return []
        name = name.lstrip("/").split("@", 1)[0].lower()
        if name in ("start", "help"):
            return [
                Card(
                    help_text(
                        self.settings.schedule.new_contacts_start,
                        self.settings.limits.new_contacts_per_day,
                    )
                )
            ]
        if name == "stop":
            # At once, before any model: the waiting messages are not carried out, and he
            # is told so in the same line (ruling R71b).
            await self.ops.stop_all()
            dropped = await self.drop_burst()
            await self.layout.settle()  # the status's line and button say so at once
            if dropped:
                return [Card(f"{STOPPED}. {DROPPED.format(quoted=_quoted(dropped))}")]
            return [Card(STOPPED)]
        return [Card(NO_COMMANDS)]

    async def on_other(self, user_id: int) -> list[Card]:
        """A voice message, a video, a sticker, a location, a poll… (ruling R63): the bot
        reads only text, and says so. No model is asked."""
        if not self.is_owner(user_id):
            return []
        return [Card(NOT_TEXT)]

    async def _with_contacts(
        self, task_id: int
    ) -> list[tuple[Conversation, Contact | None]]:
        convs = await self.repo.task_conversations(task_id)
        return [(conv, await self.repo.get_contact(conv.contact_id)) for conv in convs]

    def _local_now(self) -> datetime:
        """Now, on the owner's calendar."""
        return self.clock.now().astimezone(self.settings.schedule.zone)

    async def _goes_at(
        self, convs: Sequence[tuple[Conversation, Contact | None]]
    ) -> dict[int, datetime]:
        """When the first message of each person shown goes, for those it has not gone to
        yet: the earliest slot of his queued messages (Task 3.1 paces first messages)."""
        waiting = [conv.id for conv, _ in convs if not conv.first_sent]
        return await self.repo.first_message_slots(waiting) if waiting else {}

    async def status_card(self) -> Card:
        """The status of every task in work: the chat's layout keeps it current and, when
        nothing waits for the owner, last (`ChatLayout.settle`)."""
        tasks = [t for t in await self.repo.all_tasks() if t.status in _ON_BOARD]
        convs = {t.id: await self._with_contacts(t.id) for t in tasks}
        shown = [pair for pairs in convs.values() for pair in pairs]
        return board_card(
            tasks,
            convs,
            self._local_now(),
            await self._goes_at(shown),
            stopped=await self.repo.is_stopped(),
            stopped_why=await self.repo.stopped_why(),
            ready=[t.id for t in tasks if await self.repo.get_kv(result_waits_key(t.id))],
        )

    # -- buttons --

    async def on_callback(self, user_id: int, data: str) -> str:
        """A button press; returns the short text Telegram shows to the owner."""
        if not self.is_owner(user_id):
            return ""
        if data == RESUME_DATA:  # «Продолжить работу» under the status
            return await self._resume_button()
        parts = data.split(":")
        if len(parts) != 4 or not parts[1].isdigit():
            return UNKNOWN_BUTTON
        kind, number, action, nonce = parts
        if kind == "ap":
            return await self.approvals.owner_decision(int(number), nonce, action)
        if kind == "tk":
            return await self._task_button(int(number), action, nonce)
        if kind == "fn":
            return await self._finish_button(int(number), action, nonce)
        if kind == "pp":  # «Кого ты имел в виду?» (Task 5.4)
            return await self._pick_button(int(number), action, nonce)
        if kind == "mn":  # «Продолжить» of a task that spent its money (R76)
            return await self._money_button(int(number), action, nonce)
        if kind == "ml" and action == "more":  # the month's limit + $10 (R90)
            line = await raise_month_limit(self.repo, self.settings, nonce)
            if line:  # it stays in the chat: the button's own answer is gone in seconds
                await self.show(Card(line))
            return line or ALREADY_DECIDED
        if kind == "sl" and self.site_logins is not None:  # a sign-in card (stage 4б)
            return await self.site_logins.button(int(number), action, nonce)
        return UNKNOWN_BUTTON

    async def _money_button(self, task_id: int, action: str, code: str) -> str:
        """The one way the task's money grows: + `task_usd`; its waiting events fire at
        the next round. A second press or an old code: «Уже решено»."""
        stored = (await self.repo.get_kv(money_card_key(task_id)) or "").split(" ")
        if action != "more" or len(stored) != 3 or not secrets.compare_digest(
            stored[1], code
        ):
            return ALREADY_DECIDED
        await self.repo.set_kv(money_card_key(task_id), "")
        usd = await money_limit(self.repo, self.settings, "task")
        allowance = await task_allowance(self.repo, self.settings, task_id) + usd
        await self.repo.set_kv(allowance_key(task_id), str(allowance))
        line = MONEY_MORE.format(id=task_id, usd=usd)
        await self.layout.release("task_money", task_id, Card(line))
        return line

    async def _resume_button(self) -> str:
        """The agent works again (the word «продолжай» does the same through the
        secretary); a second press finds the agent working already."""
        if not await self.repo.is_stopped():
            return ALREADY_WORKING
        await self.ops.resume_all()
        await self.layout.settle()
        return RESUMED

    # -- finishing a task by hand --

    async def _finish_ask(self, task_id: int) -> Card:
        """«Закрыть задачу?» with a one-time yes and no: closing may write a goodbye. A task
        the agent finished can be closed too; only a closed one cannot. The card waits for
        the owner (kind `finish_ask`, ruling R53) until he answers it."""
        try:
            task = await self.repo.get_task(task_id)
        except NotFound:
            return Card(f"Задачи #{task_id} нет")
        if task.status is TaskStatus.CANCELLED:
            return Card(f"Задача #{task_id} уже закрыта")
        nonce = secrets.token_hex(3)
        await self.repo.set_kv(_finish_key(task_id), nonce)
        # What closing will really do (`TaskOps.close`): a goodbye only where a talk is cut.
        if task.status is TaskStatus.DONE or await self.repo.get_kv(
            result_waits_key(task_id)
        ):
            what = "Агент её уже закончил — никому ничего не напишу."
        elif task.status is TaskStatus.DRAFT:
            what = "Она ещё не запущена — никому ничего не напишу."
        else:
            what = (
                "Тем, с кем ещё идёт разговор, уйдёт «Спасибо, пока не нужно»."
                " Остальным агент ничего не напишет."
            )
        # Two lines: the first is what is left of it if nobody answers («— не понадобилось»).
        return Card(
            f"{WAITING_MARK} Закрыть задачу #{task_id} «{task.title}»?\n{what}",
            [
                [
                    ("Да, закрыть", f"fn:{task_id}:yes:{nonce}"),
                    ("Нет", f"fn:{task_id}:no:{nonce}"),
                ]
            ],
            kind="finish_ask",
            key=str(task_id),
        )

    async def _finish_button(self, task_id: int, action: str, nonce: str) -> str:
        if action == "ask":  # «Закрыть задачу» on a task card or the status
            await self.layout.send(await self._finish_ask(task_id))
            return ""
        if action == "ack":  # «Завершить» under a result (ruling R56)
            return await self.ops.acknowledge(task_id)
        stored = await self.repo.get_kv(_finish_key(task_id))
        if not stored or not secrets.compare_digest(stored, nonce):
            return ALREADY_DECIDED
        await self.repo.set_kv(_finish_key(task_id), "")
        if action != "yes":
            line = NOT_CLOSING.format(id=task_id)
            await self.layout.release("finish_ask", task_id, Card(line))
            return "Не закрываю"
        line = await self.ops.close(task_id)
        # The confirmation becomes the line: no separate notice (ruling R55).
        await self.layout.release("finish_ask", task_id, Card(line))
        return line

    async def _task_button(self, task_id: int, action: str, nonce: str) -> str:
        stored = await self.repo.get_kv(_nonce_key(task_id))
        if not stored or not secrets.compare_digest(stored, nonce):
            if (
                stored
                and (await self.repo.get_task(task_id)).status is TaskStatus.DRAFT
            ):
                await self.show_draft(
                    task_id
                )  # the draft was rewritten since this card
                return CARD_OUTDATED
            return ALREADY_DECIDED
        if action not in ("launch", "cancel"):
            return UNKNOWN_BUTTON
        if action == "cancel":  # «Закрыть задачу» on a draft: nobody was written to
            await self.repo.set_kv(_nonce_key(task_id), "")
            line = await self.ops.close(task_id)
            task = await self.repo.get_task(task_id)
            await self.layout.release(
                "task", task_id, draft_closed_card(task_id, task.title)
            )
            return line
        card, toast = await self._launch(task_id, nonce)
        if card is not None:
            await self.show(card)
        return toast

    async def _launch(self, task_id: int, nonce: str) -> tuple[Card | None, str]:
        """«Запустить»: what `_start` does, and the short reply for the button."""
        card, launched, _ = await self._start(task_id, nonce)
        return card, LAUNCHED if launched else NOT_LAUNCHED

    async def _start(
        self, task_id: int, nonce: str
    ) -> tuple[Card | None, bool, list[str]]:
        """Hand the task to the agent now: the draft's card is edited into «запущена», with
        «Закрыть задачу» (a reply to it is an instruction: it is the task's card). Returns a
        card to send only when there is no draft's card to edit, or for the notes of a
        launch that reached nobody (the draft and its card keep working then); whether the
        task started; the notes."""
        await self.repo.set_kv(_nonce_key(task_id), "")
        try:
            launched, notes = await self.ops.launch(task_id)
        except BaseException:
            # The card's button must still work after an unexpected error.
            await self.repo.set_kv(_nonce_key(task_id), nonce)
            raise
        if not launched:
            await self.repo.set_kv(_nonce_key(task_id), nonce)  # its card still works
            return Card("\n".join(notes)), False, notes
        task = await self.repo.get_task(task_id)
        lines = []
        if not await self.repo.task_conversations(task_id):
            lines.append(FINDS_PEOPLE)  # the owner named nobody (ruling R26)
        lines += notes
        if waits := await self._why_it_waits():
            lines.append(waits)
        card = launched_card(task_id, task.title, lines)
        if await self.layout.release("task", task_id, card):
            return None, True, notes
        # A settle in between may have settled the draft's card already, without the
        # notes: it is edited again, so that one «запущена» shows them.
        if (shown := await self.layout.last_message("task", task_id)) is not None:
            try:
                await self.layout.edit(shown, card)
            except CardGone:
                return card, True, notes
            return None, True, notes
        return card, True, notes

    async def _why_it_waits(self) -> str:
        """Why the agent will not start on a launched task now, or "" when it will: a stop
        (the owner's to undo, so it comes first), then the month's money, then the day's.
        Said on the card, since the runner's own notice of a spent limit goes once a day
        (review I1, M6)."""
        if await self.repo.is_stopped():
            by_telegram = await self.repo.stopped_why()
            return LAUNCHED_STOPPED_BY_TELEGRAM if by_telegram else LAUNCHED_STOPPED
        cap = await spend_cap(self.repo, self.clock, self.settings)
        if cap == "month":
            return LAUNCHED_MONTH_SPENT
        if cap == "day":
            return LAUNCHED_DAY_SPENT
        return ""
