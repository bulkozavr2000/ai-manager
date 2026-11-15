"""The cases of the secretary's evals: the owner's words and the state he says them in.

One TOML file a case, in `evals/secretary_cases/`, named after its `id`:

    id = "launch_anyway"
    about = "one line: what the case proves"
    stopped = false                  # the agent is stopped (/stop)
    messages = ["неважно, запускай"] # the owner's turn: one message, or several in a row;
                                     # a message may be { text, forwarded = true } (someone
                                     # else's, forwarded) or { contact, name } (a contact card)
    reply_to_draft = 1               # the first message replies to this draft's card

    [[drafts]]    id, title, brief; people, budget, currency, missing, words (his own words)
    [[tasks]]     id, title, brief, status = "active" | "paused"; headline, budget
      [[tasks.people]]  username, state, name; messages = [{ from = "agent" | "counterpart", text }]
    [[finished]]  id, title, brief; headline, budget, people
    [[cards]]     id (1, 2, … in order), task, kind, question; about (a person of that task;
                  none: the task as a whole)
    [[chat]]      from = "owner" | "bot", text — the tail of his chat with the bot
    [[contacts]]  name; username, phone — the work account's address book (`find_person`)
    memory = ["живёт в Ташкенте"]  # facts already in the memory about him, numbered 1, 2, …
    [expect]      see `evals/secretary/checks.py`

Task numbers are the numbers the owner and the model see (gaps are filled with closed tasks
that show nowhere), so «по задаче 3» in a message means the task with `id = 3`.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from tg_agent.evals.secretary.checks import SecretaryExpect
from tg_agent.search.contacts import normalize_contact

MAX_MESSAGES = 10
_USERNAME = re.compile(r"[A-Za-z0-9_]{5,32}")
TASK_STATUSES = ("active", "paused")
PERSON_STATES = ("contacted", "talking", "waiting_owner", "closed")
SPEAKERS = ("agent", "counterpart")
CARD_KINDS = (
    "question",
    "confirm_order",
    "share_phone",
    "share_address",
)
# A card about one person of the task; the others may be about the task as a whole.
_PERSON_CARDS = ("confirm_order", "share_phone", "share_address")

_TOP_KEYS = (
    "id", "about", "stopped", "messages", "reply_to_draft",
    "drafts", "tasks", "finished", "cards", "chat", "contacts", "memory", "expect",
)  # fmt: skip
# What a contact card is in the owner's turn (`control/logic.PERSON_SENT`).
PERSON_SENT = "владелец прислал контакт"
_DRAFT_KEYS = (
    "id",
    "title",
    "brief",
    "people",
    "budget",
    "currency",
    "missing",
    "words",
)
_TASK_KEYS = ("id", "title", "brief", "status", "headline", "budget", "people")
_FINISHED_KEYS = ("id", "title", "brief", "headline", "budget", "people")
_PERSON_KEYS = ("username", "state", "messages", "name")
_CARD_KEYS = ("id", "task", "kind", "about", "question")


class CaseError(ValueError):
    """A case file is wrong; the message names the file and the key at fault."""


@dataclass(frozen=True)
class DraftSpec:
    id: int
    title: str
    brief: str
    people: tuple[str, ...] = ()
    budget: float | None = None
    currency: str = "UZS"
    missing: tuple[str, ...] = ()
    words: str = ""  # the owner's words the draft was made from


@dataclass(frozen=True)
class PersonSpec:
    username: str
    state: str = "talking"
    messages: tuple[tuple[str, str], ...] = ()  # (speaker, text), oldest first
    name: str | None = None  # his Telegram name (`find_person` matches it)


@dataclass(frozen=True)
class MessagePart:
    """One message of the owner's turn: his words, someone else's he forwarded, or a
    contact card (`person` = name, contact)."""

    text: str
    forwarded: bool = False
    person: tuple[str, str] | None = None


@dataclass(frozen=True)
class BookEntry:
    """A person in the work account's address book."""

    name: str
    username: str | None = None
    phone: str | None = None


@dataclass(frozen=True)
class TaskSpec:
    """A task in work, paused, or finished (`status` "done")."""

    id: int
    title: str
    brief: str
    status: str
    headline: str = ""
    budget: float | None = None
    people: tuple[PersonSpec, ...] = ()


@dataclass(frozen=True)
class CardSpec:
    id: int
    task: int
    kind: str
    question: str
    about: str | None = None


@dataclass(frozen=True)
class ChatRow:
    speaker: str  # "owner" or "bot"
    text: str


@dataclass(frozen=True)
class SecretaryCase:
    id: str
    about: str
    parts: tuple[MessagePart, ...]
    expect: SecretaryExpect
    stopped: bool = False
    reply_to_draft: int | None = None
    drafts: tuple[DraftSpec, ...] = ()
    tasks: tuple[TaskSpec, ...] = ()  # in work, paused and finished
    cards: tuple[CardSpec, ...] = ()
    chat: tuple[ChatRow, ...] = ()
    contacts: tuple[BookEntry, ...] = ()
    memory: tuple[str, ...] = ()  # the facts about him already remembered (ruling R77)

    @property
    def messages(self) -> tuple[str, ...]:
        return tuple(part.text for part in self.parts)


def parse_case(text: str, source: str = "кейс") -> SecretaryCase:
    """A case from the text of its TOML file; `source` names the file in the errors."""
    try:
        return _case(tomllib.loads(text))
    except tomllib.TOMLDecodeError as error:
        raise CaseError(f"{source}: это не TOML ({error})") from None
    except (CaseError, ValueError) as error:
        raise CaseError(f"{source}: {error}") from None


def load_cases(only: Sequence[str] = ()) -> list[SecretaryCase]:
    """The cases that ship with the program, by file name; or just those named, in order."""
    found: dict[str, SecretaryCase] = {}
    folder = files("tg_agent.evals") / "secretary_cases"
    for entry in sorted(folder.iterdir(), key=lambda item: item.name):
        if not entry.name.endswith(".toml"):
            continue
        case = parse_case(entry.read_text(encoding="utf-8"), entry.name)
        if case.id != entry.name.removesuffix(".toml"):
            raise CaseError(f"{entry.name}: id «{case.id}» не совпадает с именем файла")
        found[case.id] = case
    if not only:
        return list(found.values())
    unknown = [name for name in dict.fromkeys(only) if name not in found]
    if unknown:
        raise CaseError(f"нет кейса: {', '.join(unknown)}. Есть: {', '.join(found)}")
    return [found[name] for name in dict.fromkeys(only)]


# -- reading --


def _case(raw: Mapping[str, Any]) -> SecretaryCase:
    _only(raw, _TOP_KEYS, "")
    expect_table = raw.get("expect")
    if not isinstance(expect_table, dict):
        raise CaseError("нет таблицы [expect]")
    expect = SecretaryExpect.from_toml(expect_table)
    if not expect.wanted():
        raise CaseError("[expect]: ни одной проверки")
    messages = raw.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_MESSAGES:
        raise CaseError(
            f"messages: нужен список непустых строк, от 1 до {MAX_MESSAGES}"
        )
    parts = tuple(_part(m, f"messages №{n}") for n, m in enumerate(messages, 1))
    stopped = raw.get("stopped", False)
    if not isinstance(stopped, bool):
        raise CaseError("stopped: нужно true или false")
    drafts = tuple(
        _draft(table, f"[[drafts]] №{n}") for n, table in _tables(raw, "drafts")
    )
    tasks = tuple(
        _task(table, f"[[tasks]] №{n}", finished=False)
        for n, table in _tables(raw, "tasks")
    ) + tuple(
        _task(table, f"[[finished]] №{n}", finished=True)
        for n, table in _tables(raw, "finished")
    )
    ids = [spec.id for spec in (*drafts, *tasks)]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise CaseError(f"номер задачи повторяется: {', '.join(map(str, repeated))}")
    usernames = [p.username.casefold() for task in tasks for p in task.people]
    if len(set(usernames)) != len(usernames):
        raise CaseError("[[tasks.people]] username: один человек — в одной задаче")
    reply_to = raw.get("reply_to_draft")
    if reply_to is not None and reply_to not in {draft.id for draft in drafts}:
        raise CaseError(f"reply_to_draft: черновика #{reply_to} в кейсе нет")
    by_id = {task.id: task for task in tasks}
    cards = tuple(
        _card(table, f"[[cards]] №{n}", n, by_id) for n, table in _tables(raw, "cards")
    )
    chat = tuple(_row(table, f"[[chat]] №{n}") for n, table in _tables(raw, "chat"))
    book = tuple(
        _book(table, f"[[contacts]] №{n}") for n, table in _tables(raw, "contacts")
    )
    return SecretaryCase(
        id=_text(raw, "id", ""),
        about=_text(raw, "about", ""),
        parts=parts,
        expect=expect,
        stopped=stopped,
        reply_to_draft=reply_to,
        drafts=drafts,
        tasks=tasks,
        cards=cards,
        chat=chat,
        contacts=book,
        memory=_strings(raw, "memory", ""),
    )


def _part(raw: Any, where: str) -> MessagePart:
    """A message of the turn: a string, {text, forwarded = true} or {contact, name}."""
    if isinstance(raw, str) and raw.strip():
        return MessagePart(raw.strip())
    if isinstance(raw, dict) and "contact" in raw:
        _only(raw, ("contact", "name"), where)
        contact = normalize_contact(_text(raw, "contact", where))
        if contact is None:
            raise CaseError(f"{where} contact: нужен @username или телефон")
        return MessagePart(PERSON_SENT, person=(_text(raw, "name", where), contact))
    if isinstance(raw, dict) and raw.get("forwarded") is True:
        _only(raw, ("text", "forwarded"), where)
        return MessagePart(_text(raw, "text", where), forwarded=True)
    raise CaseError(
        f"{where}: нужна строка, {{text, forwarded = true}} или {{contact, name}}"
    )


def _book(table: Mapping[str, Any], where: str) -> BookEntry:
    _only(table, ("name", "username", "phone"), where)
    username = _text(table, "username", where) if "username" in table else None
    phone = _text(table, "phone", where) if "phone" in table else None
    return BookEntry(_text(table, "name", where), username, phone)


def _draft(table: Mapping[str, Any], where: str) -> DraftSpec:
    _only(table, _DRAFT_KEYS, where)
    brief = _text(table, "brief", where)
    return DraftSpec(
        id=_number(table, "id", where),
        title=_text(table, "title", where),
        brief=brief,
        people=_strings(table, "people", where),
        budget=_budget(table, where),
        currency=_text(table, "currency", where) if "currency" in table else "UZS",
        missing=_strings(table, "missing", where),
        words=_text(table, "words", where) if "words" in table else brief,
    )


def _task(table: Mapping[str, Any], where: str, *, finished: bool) -> TaskSpec:
    _only(table, _FINISHED_KEYS if finished else _TASK_KEYS, where)
    if finished:
        status = "done"
    else:
        status = table.get("status")
        if status not in TASK_STATUSES:
            raise CaseError(
                f"{where} status: нужно одно из: {', '.join(TASK_STATUSES)}"
            )
    people = table.get("people", [])
    if not isinstance(people, list) or not all(isinstance(p, dict) for p in people):
        raise CaseError(f"{where} people: нужен список таблиц")
    return TaskSpec(
        id=_number(table, "id", where),
        title=_text(table, "title", where),
        brief=_text(table, "brief", where),
        status=status,
        headline=_text(table, "headline", where) if "headline" in table else "",
        budget=_budget(table, where),
        people=tuple(
            _person(person, f"{where} people №{n}")
            for n, person in enumerate(people, 1)
        ),
    )


def _person(table: Mapping[str, Any], where: str) -> PersonSpec:
    _only(table, _PERSON_KEYS, where)
    username = _text(table, "username", where)
    if not _USERNAME.fullmatch(username):
        raise CaseError(f"{where} username: нужно 5–32 символа: буквы, цифры, «_»")
    state = table.get("state", "talking")
    if state not in PERSON_STATES:
        raise CaseError(f"{where} state: нужно одно из: {', '.join(PERSON_STATES)}")
    rows = table.get("messages", [])
    if not isinstance(rows, list):
        raise CaseError(f"{where} messages: нужен список таблиц {{from, text}}")
    messages = []
    for n, row in enumerate(rows, 1):
        place = f"{where} messages №{n}"
        if not isinstance(row, dict):
            raise CaseError(f"{place}: нужна таблица {{from, text}}")
        _only(row, ("from", "text"), place)
        if row.get("from") not in SPEAKERS:
            raise CaseError(f"{place} from: нужно одно из: {', '.join(SPEAKERS)}")
        messages.append((row["from"], _text(row, "text", place)))
    name = _text(table, "name", where) if "name" in table else None
    return PersonSpec(username, state, tuple(messages), name)


def _card(
    table: Mapping[str, Any], where: str, number: int, tasks: Mapping[int, TaskSpec]
) -> CardSpec:
    _only(table, _CARD_KEYS, where)
    card_id = _number(table, "id", where)
    if card_id != number:
        raise CaseError(
            f"{where} id: карточки нумеруются по порядку с 1 (нужно {number})"
        )
    kind = table.get("kind")
    if kind not in CARD_KINDS:
        raise CaseError(f"{where} kind: нужно одно из: {', '.join(CARD_KINDS)}")
    task_id = _number(table, "task", where)
    task = tasks.get(task_id)
    if task is None:
        raise CaseError(f"{where} task: задачи #{task_id} в работе или завершённой нет")
    about = _text(table, "about", where) if "about" in table else None
    if about is not None:
        if about.removeprefix("@").casefold() not in {
            p.username.casefold() for p in task.people
        }:
            raise CaseError(f"{where} about: «{about}» — не человек задачи #{task_id}")
        about = about.removeprefix("@")
    elif kind in _PERSON_CARDS:
        raise CaseError(f"{where} about: карточка «{kind}» — о человеке задачи")
    return CardSpec(card_id, task_id, kind, _text(table, "question", where), about)


def _row(table: Mapping[str, Any], where: str) -> ChatRow:
    _only(table, ("from", "text"), where)
    speaker = table.get("from")
    if speaker not in ("owner", "bot"):
        raise CaseError(f"{where} from: нужно owner или bot")
    return ChatRow(speaker, _text(table, "text", where))


# -- small readers --


def _tables(raw: Mapping[str, Any], name: str) -> list[tuple[int, Mapping[str, Any]]]:
    rows = raw.get(name, [])
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise CaseError(f"[[{name}]]: нужен список таблиц")
    return list(enumerate(rows, 1))


def _only(table: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        place = f"{where}: " if where else ""
        raise CaseError(f"{place}неизвестные ключи: {', '.join(unknown)}")


def _text(table: Mapping[str, Any], key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CaseError(f"{where} {key}: нужна непустая строка".strip())
    return value.strip()


def _number(table: Mapping[str, Any], key: str, where: str) -> int:
    value = table.get(key)
    if type(value) is not int or value < 1:
        raise CaseError(f"{where} {key}: нужно целое число больше нуля")
    return value


def _budget(table: Mapping[str, Any], where: str) -> float | None:
    if "budget" not in table:
        return None
    value = table["budget"]
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise CaseError(f"{where} budget: нужно число больше нуля")
    return float(value)


def _strings(table: Mapping[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = table.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise CaseError(f"{where} {key}: нужен список непустых строк")
    return tuple(item.strip() for item in value)
