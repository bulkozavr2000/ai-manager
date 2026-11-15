"""The memory about the owner (ruling R77): what he said about himself, kept across tasks.

kv `owner_memory` holds a JSON list of `{id, fact, contact, at}`; the ids come from kv
`owner_memory_next`. The secretary writes it from his own words of the current turn (its tools
`remember` and `forget`, behind the quote rule R59); the secretary and every task agent read
it. A fact's `contact` is someone he said not to work with: no task attaches him
(`people.attach`), the search does not bring him and a draft does not take him.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime

from tg_agent.brain.context import escape_untrusted, hide_money
from tg_agent.control.cards import masked
from tg_agent.db import Repo
from tg_agent.models import Contact
from tg_agent.policy.dlp import plain, prepare, scan
from tg_agent.secrets import load_owner_secrets

KEY = "owner_memory"
NEXT_KEY = "owner_memory_next"
MAX_FACTS = 40
MAX_CHARS = 200
HEAD = "# Что владелец сказал о себе (его слова)"
_PRIVATE = frozenset({"phone", "card", "email"})

# For the model, and through it for the owner.
PRIVATE = "Телефон, карту и почту в память не пишу"
OWNER_DATA = (
    "Адрес и телефон я не запоминаю: их даёт только кнопка „Дать адрес?“ / „Дать телефон?“"
)
EMPTY = "Пустой факт: запоминать нечего"
TOO_LONG = f"Факт длиннее {MAX_CHARS} символов: сократи"
FULL = f"В памяти уже {MAX_FACTS} фактов: сначала пусть владелец скажет, что забыть"
STORED = "Запомнил: {fact}"
NO_FACT = "В памяти нет факта №{id}"
FORGOTTEN = "Забыл: {fact}"
# `people.attach`, the draft and the person taken by name refuse him with it.
NOT_WITH = "ты просил с ним не работать — скажи «забудь …», если передумал"
# Why his talk in a task was closed when the owner said not to work with him (ruling R92):
# the owner sees it in the pinned status, the model in «Люди».
EXCLUDED = "владелец просил с ним не работать"


@dataclass(frozen=True)
class Fact:
    id: int
    fact: str
    contact: str | None  # a username without "@" or a phone "+…": not to work with him
    at: str  # when it was said, ISO


async def load(repo: Repo) -> list[Fact]:
    raw = await repo.get_kv(KEY)
    return [Fact(**item) for item in json.loads(raw)] if raw else []


async def _save(repo: Repo, facts: list[Fact]) -> None:
    await repo.set_kv(
        KEY, json.dumps([asdict(f) for f in facts], ensure_ascii=False)
    )


async def remember(
    repo: Repo, fact: str, contact: str | None, now: datetime
) -> tuple[str, bool]:
    """Keep `fact` (and the contact not to work with); the line for the owner and whether
    it was kept. A phone, a card or an e-mail is refused, and so is the owner's own stored
    address or phone in any spelling (ruling R91); a sum of money is masked. The phone of the
    contact not to work with is no reason to refuse: it is masked in the fact (R102)."""
    fact = " ".join(fact.split())
    if not fact:
        return EMPTY, False
    owners = [plain(secret) for secret in load_owner_secrets()]
    if any(secret and secret in plain(fact) for secret in owners):
        return OWNER_DATA, False
    findings = scan(fact)
    own = {f.match for f in findings if f.kind == "phone" and _is_contact(f.match, contact)}
    if any(f.kind in _PRIVATE and f.match not in own for f in findings):
        return PRIVATE, False
    if own:  # his own number is no reason to refuse; it is kept masked (ruling R102)
        fact = prepare(fact)
        for text in own:
            fact = fact.replace(text, masked(contact or ""))
    fact = hide_money(fact)
    if len(fact) > MAX_CHARS:
        return TOO_LONG, False
    facts = await load(repo)
    if len(facts) >= MAX_FACTS:
        return FULL, False
    fact_id = int(await repo.get_kv(NEXT_KEY) or 1)
    await repo.set_kv(NEXT_KEY, str(fact_id + 1))
    kept = Fact(fact_id, fact, contact, now.isoformat())
    await _save(repo, [*facts, kept])
    return STORED.format(fact=_shown(kept)), True


async def forget(repo: Repo, fact_id: int) -> tuple[str, bool]:
    """Drop the fact with this id; the line for the owner and whether it was there."""
    facts = await load(repo)
    gone = [f for f in facts if f.id == fact_id]
    if not gone:
        return NO_FACT.format(id=fact_id), False
    await _save(repo, [f for f in facts if f.id != fact_id])
    return FORGOTTEN.format(fact=_shown(gone[0])), True


async def excluded(repo: Repo) -> set[str]:
    """The contacts he said not to work with, casefolded."""
    return {f.contact.casefold() for f in await load(repo) if f.contact}


def identifiers(contact: Contact) -> set[str]:
    """Every contact his stored record knows him by, casefolded: his @username (without
    "@") and the contact he was added by (`people.CONTACT_KEY`: a phone, or a username)."""
    values = (contact.username, contact.card.get("contact"))
    return {value.casefold() for value in values if isinstance(value, str) and value}


async def is_excluded(repo: Repo, contact: Contact) -> bool:
    """Did the owner say not to work with him, by any contact his record knows (ruling R92)?
    Two contacts of one person never stored together cannot be compared."""
    return bool(identifiers(contact) & await excluded(repo))


def _is_contact(number: str, contact: str | None) -> bool:
    """Is `number` (a phone found in the fact) the phone `contact`, with or without its
    country code?"""
    digits = "".join(char for char in number if char.isdecimal())
    return bool(contact and contact.startswith("+") and contact[1:].endswith(digits))


def _shown(fact: Fact) -> str:
    line = escape_untrusted(fact.fact)
    return f"{line} (не работать: {masked(fact.contact)})" if fact.contact else line


def render(facts: list[Fact]) -> list[str]:
    """The facts as numbered lines: the number is the id `forget` takes."""
    return [f"{fact.id}. {_shown(fact)}" for fact in facts]
