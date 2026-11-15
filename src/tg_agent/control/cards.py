"""What the owner sees in the control bot: cards with buttons, built from plain data.

Button data: `ap:<approval id>:<choice>:<nonce>` for approvals, `tk:<task id>:<action>:<nonce>`
for a task draft, `pp:<task id>:<number|none>:<nonce>` for «Кого ты имел в виду?», `url:<link>`
for a link button, `users:<request id>` for the reply button that opens Telegram's own list of
people. All of it fits Telegram's 64-byte limit.
Counterpart text is shown to the owner as a quote, never as an instruction.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, time
from pathlib import PurePath

from tg_agent.control.base import Bold, Card, compose
from tg_agent.models import (
    Approval,
    ApprovalKind,
    ApprovalStatus,
    Contact,
    Conversation,
    ConvState,
    Task,
    TaskStatus,
)
from tg_agent.search.contacts import show
from tg_agent.tasks.texts import who

# The kinds of card the agent core makes (`agent/approvals.py`).
_TITLES = {
    ApprovalKind.SHARE_PHONE: "Дать твой телефон?",
    ApprovalKind.SHARE_ADDRESS: "Дать твой адрес?",
    ApprovalKind.CONFIRM_ORDER: "Подтвердить заказ?",
    ApprovalKind.QUESTION: "Вопрос агента",
    ApprovalKind.HANDOFF: "Напиши сам",
}
_BUTTONS: Mapping[ApprovalKind, list[list[tuple[str, str]]]] = {
    ApprovalKind.SHARE_PHONE: [[("Дать", "yes"), ("Не давать", "no")]],
    ApprovalKind.SHARE_ADDRESS: [[("Дать", "yes"), ("Не давать", "no")]],
    # An order card carries its buttons in its payload; these are for one that does not.
    ApprovalKind.CONFIRM_ORDER: [
        [("Да", "yes"), ("Нет", "no")],
        [("Напишу сам", "self")],
    ],
    ApprovalKind.QUESTION: [[("Напишу сам", "self")]],
    ApprovalKind.HANDOFF: [[("Вернуть агенту", "back")]],
}
# Cards about one chat that the owner may want to look at himself.
_CHAT_LINK = frozenset({ApprovalKind.HANDOFF})
_HINTS = {
    ApprovalKind.QUESTION: "Ответь на эту карточку текстом — агент передаст ответ. Или «Напишу сам».",
    ApprovalKind.CONFIRM_ORDER: (
        "Можно ответить на эту карточку своими словами — агент получит их как указание."
    ),
}
_TASK_STATUS = {
    TaskStatus.DRAFT: "черновик",
    TaskStatus.ACTIVE: "в работе",
    TaskStatus.PAUSED: "на паузе",
    TaskStatus.DONE: "завершена",
    TaskStatus.CANCELLED: "отменена",
}
READY = "готова"  # a task in work whose result waits for «Завершить» (R109)


# The first character of a card tells its kind before the owner reads it (ruling R54): the
# status, a card that waits for him, a finished task. A card that stopped waiting loses its
# marker; nothing else carries one.
STATUS_MARK = "📋"
WAITING_MARK = "❓"
RESULT_MARK = "✅"

# A task's result waits for «Завершить» (ruling R56). Its text is kept so that its card can
# be drawn again (until migration 3 gives `tasks` a column for it); the flag is "1" while the
# result waits for the owner: «Завершить» or a take-up clears it.
ACK = "Завершить"


def result_key(task_id: int) -> str:
    return f"task_result:{task_id}"


def result_waits_key(task_id: int) -> str:
    return f"task_result_waits:{task_id}"


def result_wake_key(task_id: int) -> str:
    """The wake that gave the newest result (its id; empty when none is known): what it
    queued still goes after «Завершить», what a later wake queued does not (R109)."""
    return f"task_result_wake:{task_id}"


def result_card(task_id: int, title: str, result: str) -> Card:
    """«✅ Готово — <title>», the result, and «Завершить»."""
    card = compose(
        [f"{RESULT_MARK} ", Bold(f"Готово — {title}"), "\n", result],
        [[(ACK, f"fn:{task_id}:ack:-")]],
    )
    card.kind, card.task_id = "result", task_id
    return card


def marked(card: Card, mark: str) -> Card:
    """The card with `mark` and a space before its text; its bold ranges follow."""
    shift = len(mark) + 1
    return replace(
        card,
        text=f"{mark} {card.text}",
        bold=[(start + shift, end + shift) for start, end in card.bold],
        buttons=[list(row) for row in card.buttons],
    )


def unmarked(card: Card) -> Card:
    """The card without the marker of a waiting card, and without buttons: what is left
    of a card that stopped waiting (ruling R54). A result keeps its «✅»: the task is done."""
    text, bold = card.text, list(card.bold)
    prefix = f"{WAITING_MARK} "
    if text.startswith(prefix):
        shift = len(prefix)
        text = text[shift:]
        bold = [(max(s - shift, 0), e - shift) for s, e in bold if e > shift]
    return replace(card, text=text, bold=bold, buttons=[])


# What is left of a card nobody needed any more: its first line and these words.
NOT_NEEDED = "— не понадобилось"
# What an old message of the bot becomes when Telegram no longer lets it be deleted (it is
# older than 48 hours) and it must be shown again at the bottom (ruling R52).
MOVED = "⬇ перенесено ниже"
SETTLED_CHARS = 120  # how much of a question and of its answer a settled card keeps


def not_needed(card: Card) -> Card:
    """«<its first line> — не понадобилось», no marker, no buttons."""
    first = unmarked(card).text.split("\n", 1)[0].strip()
    return Card(f"{first} {NOT_NEEDED}", kind=card.kind, task_id=card.task_id)


# A task that spent its money (R76): one button, `ControlLogic` raises the allowance.
MONEY_MORE = "Задача #{id}: продолжаю, ещё ${usd:g}"


def task_money_card(task_id: int, title: str, spent: float, usd: float, code: str) -> Card:
    """«Продолжить ещё на $3?»: a waiting card with no row of its own (`UNOWNED`)."""
    return Card(
        f"{WAITING_MARK} Задача #{task_id} «{title}» потратила ${spent:.2f}."
        f" Продолжить ещё на ${usd:g}?",
        [[("Продолжить", f"mn:{task_id}:more:{code}")]],
        task_id=task_id,
        kind="task_money",
        key=str(task_id),
    )


def launched_card(task_id: int, title: str, lines: Sequence[str] = ()) -> Card:
    """What a draft's card becomes on «Запустить» (ruling R55): a reply to it is an
    instruction for the task's agent, and the task can be closed from it."""
    text = "\n".join([f"Задача #{task_id} «{title}» запущена", *lines])
    return Card(
        text, [[(CLOSE_TASK, f"fn:{task_id}:ask:-")]], task_id=task_id, kind="task"
    )


def draft_closed_card(task_id: int, title: str) -> Card:
    return Card(f"Задача #{task_id} «{title}» закрыта", task_id=task_id, kind="task")


def _settled_title(approval: Approval, contact: Contact | None) -> str:
    """The first line of an approval's card, without its marker."""
    payload = approval.payload
    if approval.kind is ApprovalKind.QUESTION and (
        "received" in payload or "question" in payload
    ):
        task = _flat(str(payload.get("task") or ""))
        return f"Вопрос — {task}" if task else "Вопрос"
    title = _TITLES[approval.kind]
    return f"{title} · {who(contact)}" if contact is not None else title


def settled_approval_card(approval: Approval, contact: Contact | None) -> Card:
    """An approval's card once it waits no more (ruling R55): one or two lines, what was
    asked and what the owner answered, no marker and no buttons."""
    payload, status = approval.payload, approval.status
    answer = (approval.answer or "").strip()
    about = f" · {who(contact)}" if contact is not None else ""
    if status is ApprovalStatus.EXPIRED or status is ApprovalStatus.PENDING:
        text = f"{_settled_title(approval, contact)} {NOT_NEEDED}"
    elif approval.kind is ApprovalKind.QUESTION:
        question = _flat(str(payload.get("question") or payload.get("text") or ""))
        head = _settled_title(approval, contact)
        if question:
            head += f": «{_short(question, SETTLED_CHARS)}»"
        said = f"«{_short(_flat(answer), SETTLED_CHARS)}»" if answer else "чат у тебя"
        text = f"{head}\nОтвет: {said}"
    elif approval.kind is ApprovalKind.CONFIRM_ORDER:
        amount = payload.get("amount")
        if isinstance(amount, int | float):
            what = money(amount, str(payload.get("currency") or ""))
        else:
            what = _short(_flat(str(payload.get("terms") or "")), SETTLED_CHARS)
        outcome = {
            ApprovalStatus.APPROVED: "подтверждён",
            ApprovalStatus.REJECTED: "отклонён",
        }.get(status, "ответил словами" if answer else "чат у тебя")
        text = f"Заказ{about}: {what} — {outcome}"
    elif approval.kind in (ApprovalKind.SHARE_PHONE, ApprovalKind.SHARE_ADDRESS):
        word = "Телефон" if approval.kind is ApprovalKind.SHARE_PHONE else "Адрес"
        outcome = {
            ApprovalStatus.APPROVED: "дали",
            ApprovalStatus.REJECTED: "не дали",
        }.get(status, "чат у тебя")
        text = f"{word}{about}: {outcome}"
    else:  # a hand-back card is never tracked; said plainly all the same
        text = f"{_settled_title(approval, contact)} — решено"
    return Card(text, kind="approval", approval_id=approval.id)


def _ap(approval: Approval, choice: str) -> str:
    return f"ap:{approval.id}:{choice}:{approval.nonce}"


_CURRENCY = {"UZS": "сум"}


def money(value: float, currency: str = "") -> str:
    text = f"{value:,.0f}".replace(",", " ")
    return f"{text} {_CURRENCY.get(currency, currency)}".strip()


def approval_card(
    approval: Approval, conv: Conversation | None, contact: Contact | None
) -> Card:
    """The card that puts an approval to the owner, with its buttons."""
    payload = approval.payload
    if approval.kind is ApprovalKind.QUESTION and (
        "received" in payload or "question" in payload
    ):
        card = marked(question_card(approval, contact), WAITING_MARK)
        card.kind, card.approval_id = "approval", approval.id
        return card
    title = _TITLES[approval.kind]
    if contact is not None:
        title += f" · {who(contact)}"
    # A hand-back card does not wait for the owner: the chat is his (ruling R53).
    waits = approval.kind is not ApprovalKind.HANDOFF
    lines = [f"{WAITING_MARK} {title}" if waits else title]
    if text := approval.payload.get("text"):
        lines.append(str(text))
    yes_label = None
    if approval.kind is ApprovalKind.CONFIRM_ORDER:
        # What «Да» approves is on the card.
        amount, terms = payload.get("amount"), str(payload.get("terms") or "").strip()
        if isinstance(amount, int | float):
            shown = money(amount, str(payload.get("currency") or ""))
            lines.append(f"Сумма: {shown}")
            yes_label = f"Да — {shown}"
        if terms:
            lines.append(f"Условия: {terms}")
    if last := approval.payload.get("last_incoming"):
        lines.append(f"От собеседника: «{last}»")
    if hint := _HINTS.get(approval.kind):
        lines.append(hint)

    buttons: list[list[tuple[str, str]]] = []
    if approval.kind in _CHAT_LINK and contact is not None and contact.username:
        buttons.append([("Открыть чат", f"url:https://t.me/{contact.username}")])
    # A card may carry its own buttons (the order card's «Да» / «Нет» / «Напишу сам»).
    for row in approval.payload.get("buttons") or _BUTTONS[approval.kind]:
        buttons.append(
            [
                (
                    yes_label if yes_label and choice == "yes" else label,
                    _ap(approval, choice),
                )
                for label, choice in row
            ]
        )
    return Card("\n".join(lines), buttons, kind="approval", approval_id=approval.id)


def _quoted(texts: Sequence[object]) -> str:
    return "\n".join(f"• «{text}»" for text in texts if str(text).strip()) or "—"


def question_card(approval: Approval, contact: Contact | None) -> Card:
    """The agent's question in Саид's format (2026-11-09): «Вопрос» and the task, who asks and
    who they are, what came in, the reply suggested ("—" where there is nothing). «Ок» sends
    that reply; any other answer is a reply to the card.

    The agent core (stage 2) puts its own `question` in the payload: the card then shows it
    («❓ …») under the person, and «Получено» only when something came in. A question about
    the task as a whole (no contact) has no «👤» line. A payload without `question` (a card of
    the old engine, expired by migration 2) reads as before."""
    payload = approval.payload
    suggested = [
        str(text) for text in payload.get("suggested") or [] if str(text).strip()
    ]
    pieces: list[str] = [Bold("Вопрос")]
    if task := str(payload.get("task") or "").strip():
        pieces.append(f" — {task}")
    if contact is not None:
        person = f"👤 {who(contact)}"
        if about := str(payload.get("about") or "").strip():
            person += f" — {about}"
        pieces.append(f"\n{person}")
    received = payload.get("received") or []
    question = str(payload.get("question") or "").strip()
    if question:
        pieces.append(f"\n❓ {question}")
    if received or not question:
        pieces.append(f"\n✉️ Получено\n{_quoted(received)}")
    pieces.append(f"\n💬 Предлагаемый ответ\n{_quoted(suggested)}")
    buttons = [[("Ок", _ap(approval, "ok"))]] if suggested else []
    return compose(pieces, buttons)


def _numbered(sections: Sequence[tuple[str, str]]) -> list[str]:
    """Bold numbered headings, each with its text below, a blank line between sections."""
    pieces: list[str] = []
    for number, (heading, body) in enumerate(sections, 1):
        if pieces:
            pieces.append("\n\n")
        pieces += [Bold(f"{number}. {heading}"), f"\n{body}"]
    return pieces


def _description(goal: str, ceiling: float | None, currency: str) -> str:
    if ceiling is None:
        return goal
    return f"{goal}\nБюджет до {money(ceiling, currency)} — агенту не виден"


def _shared(allowed: Sequence[str]) -> str:
    names = {"name": "имя", "city": "город"}
    return ", ".join(names[x] for x in allowed if x in names) or "ничего"


# The «Контакты» of a draft whose owner named nobody: the agent finds people (R26).
FIND_MYSELF = "найду сам"


def _planned(targets: Sequence[str], names: Mapping[str, str] | None = None) -> str:
    """Whom the draft writes to: a contact the owner typed as he typed it; a person taken by
    his name (`names`) by that name and his contact masked."""
    folded = {target.casefold(): name for target, name in (names or {}).items()}

    def shown(target: str) -> str:
        name = folded.get(target.casefold())
        return person_label(name, target) if name else show(target)

    return (
        "\n".join(f"• {shown(target)} — напишу после запуска" for target in targets)
        or FIND_MYSELF
    )


# -- a person by name (Task 5.4, rulings R60, R61) --

NAME_CHARS = 40  # a name from an address book, on a card or a button


def masked(contact: str) -> str:
    """A contact as the owner and the secretary's model see a person of the address book:
    «@name», or the phone with its middle hidden («+998 90 ••• 12 34», ruling R60)."""
    if not contact.startswith("+"):
        return f"@{contact}"
    digits = contact[1:]
    if len(digits) == 12 and digits.startswith("998"):
        d = digits[3:]
        return f"+998 {d[:2]} ••• {d[5:7]} {d[7:]}"
    if len(digits) < 8:
        return "+" + "•" * len(digits)
    return f"+{digits[:-7]} ••• {digits[-4:-2]} {digits[-2:]}"


def person_label(name: str, contact: str | None) -> str:
    """«Алишер Каримов, +998 90 ••• 12 34»: a name someone chose (flattened and cut) and the
    contact masked; either alone when the other is missing."""
    shown = _short(_flat(name), NAME_CHARS)
    parts = [part for part in (shown, masked(contact) if contact else "") if part]
    return ", ".join(parts)


PICK_TITLE = "Кого ты имел в виду?"
NOBODY_OF_THEM = "Никто из них"
PICKED = "{name} — выбран"
# The reply button under which Telegram opens its own list of people (`control/bot`).
PICK_FROM_CONTACTS = "Выбрать из контактов"
PICKER_DATA = "users:1"


# The tasks a person can be taken into: a draft, or one in work, paused or finished.
TAKES_PERSON = (TaskStatus.DRAFT, TaskStatus.ACTIVE, TaskStatus.PAUSED, TaskStatus.DONE)


def pick_person_key(task_id: int) -> str:
    """The matches a card «Кого ты имел в виду?» offers, until the owner presses one."""
    return f"pick_person:{task_id}"


def pick_person_card(
    task_id: int, title: str, labels: Sequence[str], nonce: str
) -> Card:
    """«❓ Кого ты имел в виду?» — the search by name found several: a button each, and
    «Никто из них». It waits for the owner (kind `pick_person`, its key the task)."""
    text = (
        f"{WAITING_MARK} {PICK_TITLE}\nЗадача #{task_id} «{title}»."
        " Нажми на человека, которому писать."
    )
    buttons = [
        [(label, f"pp:{task_id}:{number}:{nonce}")]
        for number, label in enumerate(labels)
    ]
    buttons.append([(NOBODY_OF_THEM, f"pp:{task_id}:none:{nonce}")])
    return Card(text, buttons, task_id=task_id, kind="pick_person", key=str(task_id))


def cannot_write_first(name: str, own_username: str | None) -> str:
    """Ruling R61: a person with no @username and no phone the work account can see cannot
    be written to first; he can write first, and the link says where."""
    who_ = _short(_flat(name), NAME_CHARS) or "Этот человек"
    where = f": t.me/{own_username}" if own_username else ""
    return (
        f"{who_}: первым написать ему не получится — у него нет @username, а его номер"
        f" рабочему аккаунту не виден. Пусть он сам напишет рабочему аккаунту{where}."
        " Или пришли мне его номер телефона."
    )


def picked_card(name: str) -> Card:
    """What the card «Кого ты имел в виду?» becomes when he pressed a person."""
    return Card(PICKED.format(name=_short(_flat(name), NAME_CHARS)), kind="pick_person")


def nobody_card() -> Card:
    """What the card «Кого ты имел в виду?» becomes on «Никто из них»."""
    return Card(NOBODY_OF_THEM, kind="pick_person")


# Саид, 2026-11-10: «Кнопка закрыть задачу для каждой задачи».
CLOSE_TASK = "Закрыть задачу"
CORRECT_IN_WORDS = (
    "Поправить — напиши, что изменить: «добавь @username», «нужно 10 штук»."
)
# The bot saves a file as «<its message id>_<name>»: the owner knows it by the name.
_SAVED_PREFIX = re.compile(r"^\d+_")


def _file_names(paths: Sequence[str]) -> str:
    return ", ".join(_SAVED_PREFIX.sub("", PurePath(path).name) for path in paths)


def draft_card(
    task: Task,
    targets: Sequence[str],
    missing: Sequence[str],
    nonce: str,
    names: Mapping[str, str] | None = None,
) -> Card:
    """A draft as it is stored: the task, what to do (and the files that go with it), what
    the agent may tell about the owner, whom it will write to, and what is missing ("—"
    where there is nothing). A task whose owner named nobody launches too: its agent finds
    people itself (ruling R26). A person taken by his name (`names`: target → name) shows
    his name and his contact masked: the card is the owner's check of whom he meant."""
    task_id = task.id
    description = _description(task.goal, task.ceiling, task.currency)
    if task.attachments:
        description += f"\nФайлы: {_file_names(task.attachments)}"
    pieces = [f"{WAITING_MARK} Новая задача #{task_id} — проверь и запусти\n\n"]
    pieces += _numbered(
        [
            ("Задача", task.title),
            ("Описание задачи", description),
            ("Делюсь", _shared(task.allowed)),
            ("Контакты", _planned(targets, names)),
            ("Нужно уточнить", "\n".join(f"• {item}" for item in missing) or "—"),
        ]
    )
    buttons = [
        [("Запустить", f"tk:{task_id}:launch:{nonce}")],
        [(CLOSE_TASK, f"tk:{task_id}:cancel:{nonce}")],
    ]
    pieces.append("\n\n" + CORRECT_IN_WORDS)
    card = compose(pieces, buttons)
    card.task_id, card.kind = task_id, "task"
    return card


WAITING = "жду ответа"
NOBODY_YET = "• людей пока нет"  # a task its agent has not found anybody for yet
ANSWERED = "ответил"
NO_TASKS = f"{STATUS_MARK} Задач в работе нет"
STOPPED_LINE = "⏸ Агент остановлен — ничего не пишет и не отвечает"
# Stopped by the Sender's alarm: one tap on the button would send into a restricted account.
STOPPED_BY_TELEGRAM = (
    "⏸ Агент остановлен: Telegram ограничил аккаунт ({why})."
    " Продолжай, только когда ограничение снято"
)
# The board's button that does what /resume_all does (`ControlLogic.on_callback`).
RESUME = "Продолжить работу"
RESUME_DATA = "rs:0:go:-"
_IN_TALK = (ConvState.CONTACTED, ConvState.TALKING)
_STAGES = {
    ConvState.PLANNED: "ещё не написали",
    ConvState.WAITING_OWNER: "ждёт твоего решения",
    ConvState.HANDED_TO_OWNER: "чат у тебя",
    ConvState.CLOSED: "закончили",
}
# How much of the agent's own words a card shows: its line under a task, a person's note
# (and the note on a board that had to be shortened to fit one message). The first two are
# the limits the agent's tools already put on what it writes (`agent/tools.py`
# MAX_HEADLINE, MAX_PERSON_NOTE): keep them in step.
HEADLINE_CHARS = 200
NOTE_CHARS = 150
SHORT_NOTE_CHARS = 60
# A Telegram message holds 4 096 characters; the board keeps a margin under it. On a board
# that is still too long once shortened, one task takes at most TASK_BLOCK_CHARS of it.
BOARD_CHARS = 3800
TASK_BLOCK_CHARS = 1200


def _flat(text: str) -> str:
    """The agent's words on one line: it writes them after reading counterparts' messages,
    so a line break in them must never start a line of the owner's status of its own."""
    return " ".join(text.split())


def _answered(conv: Conversation) -> bool:
    """Has the counterpart written since our side's last message?"""
    return conv.last_in_at is not None and (
        conv.last_out_at is None or conv.last_in_at > conv.last_out_at
    )


def conv_stage(conv: Conversation) -> str:
    """Where a conversation stands, in a few words: for the board and a task's card."""
    if conv.state in _IN_TALK:
        return ANSWERED if _answered(conv) else WAITING
    if conv.state is ConvState.CLOSED and (reason := _flat(conv.close_reason or "")):
        return f"{_STAGES[conv.state]}: {reason}"
    return _STAGES[conv.state]


def _when(moment: datetime, now: datetime) -> str:
    local = moment.astimezone(now.tzinfo)
    return f"{local:%H:%M}" if local.date() == now.date() else f"{local:%d.%m %H:%M}"


def _goes(moment: datetime, now: datetime) -> str:
    """When a queued first message goes: «напишу в 14:20», «… завтра в 09:05», «… 12.11 в …».
    A slot already past (the Sender is held back) goes as soon as the Sender may send."""
    if moment <= now:
        return "напишу в ближайшее время"
    local = moment.astimezone(now.tzinfo)
    days = (local.date() - now.date()).days
    day = "" if days == 0 else "завтра " if days == 1 else f"{local:%d.%m} "
    return f"напишу {day}в {local:%H:%M}"


PERSON_NAME_CHARS = 40  # a name a person chose himself, on the board


def _person(contact: Contact) -> str:
    """How the board names a person: his @username; one added by phone has none, so his
    name (his own words: flattened and cut) and the phone he was added by (E1.6, M10)."""
    if contact.username:
        return who(contact)
    name = _short(_flat(contact.display_name), PERSON_NAME_CHARS)
    added_by = contact.card.get("contact")
    if isinstance(added_by, str) and added_by:
        return f"{name} ({show(added_by)})"
    return name


def _conv_line(
    conv: Conversation,
    contact: Contact | None,
    now: datetime,
    goes_at: Mapping[int, datetime],
    note_chars: int,
) -> str:
    stage = conv_stage(conv)
    if stage == WAITING and not conv.first_sent and conv.id in goes_at:
        # Nothing went to him yet: his first message waits for its paced slot (Task 3.1).
        stage = _goes(goes_at[conv.id], now)
    elif stage == ANSWERED and conv.last_in_at is not None:
        stage += f" в {_when(conv.last_in_at, now)}"
    elif stage == WAITING and conv.last_out_at is not None:
        stage += f" с {_when(conv.last_out_at, now)}"
    line = f"• {_person(contact) if contact else f'чат {conv.id}'} — {stage}"
    # The agent's own words about him, not rewritten: only put on one line and cut to length.
    if note := _flat(conv.note):
        line += " · " + _short(note, note_chars)
    return line


def _task_lines(
    task: Task,
    convs: Sequence[tuple[Conversation, Contact | None]],
    now: datetime,
    goes_at: Mapping[int, datetime] | None = None,
    *,
    compact: bool = False,
    ready: bool = False,
) -> tuple[list[str], list[tuple[str, int]]]:
    """A task's head lines (the head and the agent's line about it, when it has one) and its
    people's lines, each with how many people it stands for. `compact` is for a board that
    does not fit one message: the people who are done collapse into one last line and the
    notes are cut shorter."""
    head = [f"#{task.id} {task.title} — {READY if ready else _TASK_STATUS[task.status]}"]
    if headline := _flat(task.headline):
        head.append(_short(headline, HEADLINE_CHARS))
    note_chars = SHORT_NOTE_CHARS if compact else NOTE_CHARS
    closed = [conv for conv, _ in convs if conv.state is ConvState.CLOSED]
    people = [
        (_conv_line(conv, contact, now, goes_at or {}, note_chars), 1)
        for conv, contact in convs
        if not (compact and conv.state is ConvState.CLOSED)
    ]
    if compact and closed:
        people.append((f"• закончили с {len(closed)}", len(closed)))
    return head, people or [(NOBODY_YET, 0)]


# One task as the board shows it: the task, its head lines, its people's lines.
_Block = tuple[Task, list[str], list[str]]


def board_card(
    tasks: Sequence[Task],
    convs_by_task: Mapping[int, Sequence[tuple[Conversation, Contact | None]]],
    now: datetime,
    goes_at: Mapping[int, datetime] | None = None,
    *,
    stopped: bool = False,
    stopped_why: str = "",
    ready: Collection[int] = (),
) -> Card:
    """Every task in progress and where each counterpart stands: the pinned board.

    `stopped`: the agent is stopped (/stop, or the Sender's alarm). The board then says so on
    its first line and offers «Продолжить работу» first, empty or not: otherwise a stopped
    agent looks like a broken one (review I1). `stopped_why`: what stopped it when that was
    Telegram restricting the account; the line then warns before the button. `ready`: the
    tasks whose result waits for «Завершить»: they show as «готова».

    `now` is on the owner's calendar: times of today show as 21:40, older ones as 09.11 21:40.
    `goes_at`: conversation id → when its earliest queued message goes, for the people whose
    first message still waits. The board must fit one Telegram message: when it is too long,
    the people who are done collapse and notes shrink; if it is still too long, a big task
    shows only the people that fit in TASK_BLOCK_CHARS, tasks that do not fit are skipped
    (later, smaller ones may still be shown), and the last line names the skipped ones so
    that the owner can ask about each in words.
    """
    card = _tasks_board(tasks, convs_by_task, now, goes_at, ready)
    return _stopped_board(card, stopped_why) if stopped else card


def _stopped_board(card: Card, why: str = "") -> Card:
    """The board with the stopped line second, under its title (the status's marker stays
    its first character), and the resume button first."""
    line = STOPPED_BY_TELEGRAM.format(why=why) if why else STOPPED_LINE
    title, _, rest = card.text.partition("\n")
    after, shift = len(title), len(line) + 1
    return Card(
        f"{title}\n{line}" + (f"\n{rest}" if rest else ""),
        [[(RESUME, RESUME_DATA)], *card.buttons],
        [
            (start + shift, end + shift) if start > after else (start, end)
            for start, end in card.bold
        ],
        kind=card.kind,
    )


def _tasks_board(
    tasks: Sequence[Task],
    convs_by_task: Mapping[int, Sequence[tuple[Conversation, Contact | None]]],
    now: datetime,
    goes_at: Mapping[int, datetime] | None,
    ready: Collection[int] = (),
) -> Card:
    if not tasks:
        return Card(NO_TASKS, kind="board")

    def lines(compact: bool) -> list[tuple[Task, list[str], list[tuple[str, int]]]]:
        return [
            (task, *_task_lines(task, convs_by_task.get(task.id, []), now, goes_at,
                                compact=compact, ready=task.id in ready))
            for task in tasks
        ]  # fmt: skip

    for compact in (False, True):
        whole = [
            (task, head, [line for line, _ in people])
            for task, head, people in lines(compact)
        ]
        card = _board(whole, [])
        if len(card.text) <= BOARD_CHARS:
            return card
    shown: list[_Block] = []
    skipped: list[Task] = []
    for task, head, people in lines(True):
        block = _cut_block(task, head, people)
        if len(_board([*shown, block], skipped).text) <= BOARD_CHARS:
            shown.append(block)
        else:
            skipped.append(task)
    # The last line grew with each skipped task: drop from the end until it all fits.
    while shown and len(_board(shown, skipped).text) > BOARD_CHARS:
        skipped.append(shown.pop()[0])
    order = {task.id: n for n, task in enumerate(tasks)}
    return _board(shown, sorted(skipped, key=lambda task: order[task.id]))


def _cut_block(task: Task, head: list[str], people: list[tuple[str, int]]) -> _Block:
    """A task whose block is longer than TASK_BLOCK_CHARS keeps its head lines and the
    people that fit, then «• …и ещё N человек: спроси «что по задаче <id>?»»."""
    lines = [line for line, _ in people]
    if len("\n".join(head + lines)) <= TASK_BLOCK_CHARS:
        return task, head, lines
    total = sum(count for _, count in people)
    room = TASK_BLOCK_CHARS - len("\n".join(head)) - len(_more(task, total)) - 1
    kept: list[str] = []
    for line in lines:
        if len(line) + 1 > room:
            break
        kept.append(line)
        room -= len(line) + 1
    rest = sum(count for _, count in people[len(kept) :])
    return task, head, [*kept, _more(task, rest)]


def _more(task: Task, count: int) -> str:
    return f"• …и ещё {count} человек: спроси «что по задаче {task.id}?»"


def _board(shown: Sequence[_Block], skipped: Sequence[Task]) -> Card:
    pieces: list[str] = [f"{STATUS_MARK} ", Bold("Статус задач")]
    for _, head, people in shown:
        pieces += ["\n\n", Bold(head[0]), *(f"\n{line}" for line in head[1:] + people)]
    if skipped:
        numbers = ", ".join(f"#{task.id}" for task in skipped)
        pieces.append(
            f"\n\nНе поместились: {numbers}. Спроси меня, например:"
            f" «что по задаче {skipped[0].id}?»"
        )
    # A close button for every task shown, right under the pinned board (Саид, 2026-11-10).
    buttons = [
        [(f"Закрыть #{task.id} {_short(task.title)}", f"fn:{task.id}:ask:-")]
        for task, _, _ in shown
    ]
    card = compose(pieces, buttons)
    card.kind = "board"
    return card


def _short(title: str, limit: int = 30) -> str:
    return title if len(title) <= limit else title[: limit - 1].rstrip() + "…"


_HELP = """\
Пиши мне обычными словами — я пойму, что нужно сделать.

Новая задача: «Узнай цену на печать 5 корпусов у @print_lab». Я пришлю карточку задачи: \
проверь её и нажми «Запустить». Если ты не дал контактов, агент после запуска сам найдёт, \
кому написать.
Поправить задачу до запуска можно словами: «искать можно и на OLX», «добавь @username».
Указание задаче в работе — ответь на её карточку или напиши: «по задаче 3: спроси ещё \
про доставку».
Спросить можно так: «что по задачам?», «что ответил @username?».
Поставить на паузу или закрыть задачу можно словами («поставь задачу 3 на паузу») или \
кнопкой «Закрыть #N» под статусом.
На вопрос агента ответь на его карточку словами или нажми «Ок».
Файл к задаче можно прислать до текста или вместе с ним.

Внизу чата всегда то, что важно сейчас. 📋 — статус всех задач. ❓ — карточка, которая ждёт \
твоего ответа. ✅ — готовая задача: прочитай итог и нажми «Завершить».
Новым людям агент пишет первым с {start}, и не больше {per_day} новых людей в день.

/stop — сразу остановить всё. Продолжить — кнопка «Продолжить работу» под статусом или \
слово «продолжай»."""


def help_text(new_contacts_start: time, new_contacts_per_day: int) -> str:
    """What /help and /start say: how to talk to the bot in words (ruling R64)."""
    return _HELP.format(
        start=f"{new_contacts_start:%H:%M}", per_day=new_contacts_per_day
    )
