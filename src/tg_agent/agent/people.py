"""People of a task: the contacts and sites the owner named for a draft, attaching a person,
and moving his conversation between the core's states.

The agent core attaches people one at a time, at the launch and when the agent asks for it: a
person who cannot be found, a bot, one who asked not to be written to, or one busy in another
task is not attached, and the reason comes back in Russian, for the owner and for the model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from tg_agent.agent import memory
from tg_agent.db import Repo
from tg_agent.gateway.base import Gateway, TgActionError, name_matches
from tg_agent.models import (
    ApprovalKind,
    ApprovalStatus,
    Contact,
    ConvState,
    EventKind,
    OutboxItem,
    OutboxStatus,
    TaskStatus,
)
from tg_agent.policy.tg_safety import TgErrorKind
from tg_agent.search.contacts import find_phones, is_phone, show
from tg_agent.search.sites import domains_in
from tg_agent.tasks.state import InvalidTransition

S = ConvState

# The states of a conversation the core still works with (or waits on): not closed.
OPEN_STATES = frozenset(
    {S.PLANNED, S.CONTACTED, S.TALKING, S.WAITING_OWNER, S.HANDED_TO_OWNER}
)

# The states in which the conversation waits for the owner; `reopen` takes it back from them.
_WAITING = frozenset({S.WAITING_OWNER, S.HANDED_TO_OWNER})

# An outbox item in one of these has gone, or is on its way, or may have gone.
_ON_ITS_WAY = frozenset(
    {OutboxStatus.PENDING, OutboxStatus.SENDING, OutboxStatus.UNKNOWN}
)

# The answer recorded on a question card the owner answered in the chat himself.
ANSWERED_IN_CHAT = "владелец ответил сам в чате"

# Why `attach` did not attach a person who is in the task already.
IN_TASK_ALREADY = "уже в задаче"


# -- the contacts and sites the owner named for a draft --
#
# The model may misread or invent a username; only names written in the owner's own text become
# targets. A target is a Telegram username (without "@") or a phone ("+998..."). Targets are
# kept until the launch, when they are attached as people.

# More drafts than this from one message are not made: the owner sees the cards that came.
MAX_TASKS = 5

# @username (not inside an e-mail address) or a t.me / telegram.me link; 5-32 characters.
_TARGET = re.compile(
    r"(?:(?<![\w@.])@|(?<![\w/.])(?:https?://)?(?:t|telegram)\.me/)"
    r"([A-Za-z0-9_]{5,32})(?![A-Za-z0-9_])"
)
# Telegram's own addresses are where people are, not a site to search on.
_NOT_SITES = frozenset({"t.me", "telegram.me"})


def _targets_key(task_id: int) -> str:
    return f"task_targets:{task_id}"


def _sites_key(task_id: int) -> str:
    return f"task_sites:{task_id}"


def targets_with_places(text: str) -> list[tuple[int, str]]:
    """Every contact the text names, with where it starts, in the order written (duplicates
    kept): usernames without "@", phones in one form."""
    found = [(m.start(), m.group(1)) for m in _TARGET.finditer(text)]
    found += find_phones(text)
    return sorted(found)


def extract_targets(text: str) -> list[str]:
    """The contacts the text names, in the order written, no duplicates: Telegram usernames
    without "@" (first spelling kept) and phones in one form ("+998901234567")."""
    seen: dict[str, str] = {}
    for _, target in targets_with_places(text):
        seen.setdefault(target.casefold(), target)
    return list(seen.values())


async def save_targets(repo: Repo, task_id: int, targets: Sequence[str]) -> None:
    await repo.set_kv(_targets_key(task_id), json.dumps(list(targets)))


async def task_targets(repo: Repo, task_id: int) -> list[str]:
    """The contacts a draft will write to after its launch."""
    raw = await repo.get_kv(_targets_key(task_id))
    return list(json.loads(raw)) if raw else []


def sites_of(text: str, goal: str) -> list[str]:
    """The sites the owner named for one task: those in his message that the task's brief
    names too (a message may hold several tasks)."""
    return sorted((domains_in(text) & domains_in(goal)) - _NOT_SITES)


async def save_sites(repo: Repo, task_id: int, sites: Sequence[str]) -> None:
    await repo.set_kv(_sites_key(task_id), json.dumps(list(sites)))


async def task_sites(repo: Repo, task_id: int) -> list[str]:
    """The sites the owner named for this task, kept when its draft was made."""
    raw = await repo.get_kv(_sites_key(task_id))
    return list(json.loads(raw)) if raw else []


def _marks_key(task_id: int) -> str:
    return f"task_owner_marks:{task_id}"


async def save_marks(repo: Repo, task_id: int, excluded: Sequence[str]) -> None:
    await repo.set_kv(_marks_key(task_id), json.dumps({"excluded": list(excluded)}))


async def task_marks(repo: Repo, task_id: int) -> list[str]:
    """The contacts the owner's messages excluded for this task, kept when its draft was
    made (fix pass F3, I-A (d)): a message that made several drafts keeps no text (ruling
    R38), and his «не пиши @x» must not be lost."""
    raw = await repo.get_kv(_marks_key(task_id))
    return list(json.loads(raw).get("excluded", [])) if raw else []


def marks_of(text: str) -> list[str]:
    """What the owner's message excludes: for every task of the message, the safe side."""
    # Imported here: `owner_contacts` reads contacts with this module's own reader.
    from tg_agent.agent.owner_contacts import EXCLUDED, owner_words

    return list(
        dict.fromkeys(c for c, verdict in owner_words(text) if verdict == EXCLUDED)
    )


# The key of a contact's card that keeps the contact he was added by (`add_person`): a phone
# is known to Telegram only as the person it resolved to, and the search must skip it.
CONTACT_KEY = "contact"


# -- a person by name (Task 5.4, rulings R60, R61) --
#
# The owner says «свяжись с Алишером»: the secretary looks among the people of earlier tasks
# and in the work account's address book. The book is searched on demand and never copied:
# what is kept is the person the owner chose — his contact among the draft's targets and his
# name beside it, for the draft's card.

MAX_MATCHES = 5


@dataclass(frozen=True)
class Person:
    """Someone found by name or sent by the owner. `name` is the name someone chose (the
    owner in his book, or the person himself): data, never words of the owner."""

    name: str
    username: str | None
    phone: str | None  # "+<digits>"
    peer_id: int | None = None
    # Where he was found: "" (the address book or the owner's pick) or «уже писали (задача
    # #N)» for a person of an earlier task.
    known: str = ""

    @property
    def target(self) -> str | None:
        """The contact the work account can open him by: his username, else his phone; None
        when it has neither (ruling R61: he cannot be written to first)."""
        return self.username or self.phone


def _names_key(task_id: int) -> str:
    return f"task_people_names:{task_id}"


async def task_people_names(repo: Repo, task_id: int) -> dict[str, str]:
    """The names of the draft's people taken by name or sent by the owner: target → name."""
    raw = await repo.get_kv(_names_key(task_id))
    return dict(json.loads(raw)) if raw else {}


async def save_person_name(repo: Repo, task_id: int, target: str, name: str) -> None:
    names = await task_people_names(repo, task_id)
    for known in [k for k in names if k.casefold() == target.casefold()]:
        del names[known]
    names[target] = " ".join(name.split())
    await repo.set_kv(_names_key(task_id), json.dumps(names, ensure_ascii=False))


async def take_into_draft(repo: Repo, task_id: int, person: Person) -> None:
    """The owner chose this person for the draft: his contact joins its targets (once), his
    name is kept for its card, and an older «не пиши» about him no longer holds — choosing
    him is the newer word."""
    target = person.target
    assert target is not None, "a person without a contact cannot be taken"
    targets = await task_targets(repo, task_id)
    if target.casefold() not in {t.casefold() for t in targets}:
        await save_targets(repo, task_id, [*targets, target])
    await save_person_name(repo, task_id, target, person.name)
    key = target.casefold()
    excluded = await task_marks(repo, task_id)
    await save_marks(repo, task_id, [c for c in excluded if c.casefold() != key])


# What may stay of a name in the owner's word to a task's agent: letters and the marks of a
# name. Nothing that reads as a contact («@x», a link, a phone) survives, so a name a person
# chose cannot hand the agent someone else (`owner_contacts.owner_words`).
_NOT_IN_NAME = re.compile(r"[^\w\s.'\-]|[\d_]")


def owner_added_text(person: Person) -> str:
    """The owner's word to the agent of a task in work when he chose a person by name: it
    names his contact in a clause that tells to write to him, so `add_person` takes him as
    given by the owner (ruling R28; Task 5.4)."""
    name = " ".join(_NOT_IN_NAME.sub(" ", person.name).split())[:40].strip()
    contact = show(person.target or "")
    about = f" ({name})" if name else ""
    return f"владелец добавил человека — добавь его в задачу и напиши ему: {contact}{about}"


async def earlier_people(repo: Repo, query: str) -> list[Person]:
    """The people of earlier tasks whose name or username matches `query`, newest task
    first, each once; one who asked not to be written to, and a bot, are left out."""
    found: dict[int, Person] = {}
    tasks = sorted(await repo.all_tasks(), key=lambda task: task.id, reverse=True)
    for task in tasks:
        for conv in await repo.task_conversations(task.id):
            contact = await repo.get_contact(conv.contact_id)
            if contact.peer_id in found or contact.is_bot or contact.do_not_contact:
                continue
            if not name_matches(query, contact.display_name, contact.username or ""):
                continue
            added_by = contact.card.get(CONTACT_KEY)
            phone = (
                added_by if isinstance(added_by, str) and is_phone(added_by) else None
            )
            found[contact.peer_id] = Person(
                contact.display_name, contact.username, phone, contact.peer_id,
                known=f"уже писали (задача #{task.id})",
            )  # fmt: skip
    return list(found.values())


async def task_contacts(repo: Repo, task_id: int) -> list[str]:
    """The contacts of everyone attached to the task, as far as they are known: each one's
    Telegram username and the contact he was added by (`CONTACT_KEY`). A username without
    "@", a phone as "+<digits>"; no duplicates (compared case-insensitively)."""
    seen: dict[str, str] = {}
    for conv in await repo.task_conversations(task_id):
        contact = await repo.get_contact(conv.contact_id)
        for value in (contact.username, contact.card.get(CONTACT_KEY)):
            if isinstance(value, str) and value:
                seen.setdefault(value.casefold(), value)
    return list(seen.values())


@dataclass(frozen=True)
class Attached:
    """The person is a part of the task now: his conversation, in PLANNED."""

    conv_id: int
    contact: Contact


@dataclass(frozen=True)
class NotAttached:
    """The person was not attached; `reason` is in Russian, for the owner and for the model."""

    reason: str


async def attach(
    repo: Repo, gateway: Gateway, task_id: int, target: str, *, found: str | None = None
) -> Attached | NotAttached:
    """Make the person `target` (a username or a phone) a part of the task.

    `found` says where the search found him; it goes into his card, where the agent sees it.
    """
    try:
        peer = await gateway.resolve(target)
    except TgActionError as exc:
        if exc.kind is not TgErrorKind.OTHER:
            return NotAttached(f"не удалось открыть ({exc.kind.value})")
        if is_phone(target):
            return NotAttached("нет в Telegram или он скрыт по номеру — позвони сам")
        return NotAttached("не найден — проверь имя")
    if peer.is_bot:
        return NotAttached("это бот: агент с ботами не переписывается")
    contact_id = await repo.upsert_contact(
        peer.peer_id, peer.username, peer.display_name, peer.is_bot
    )
    contact = await repo.get_contact(contact_id)
    # The owner's «не работай с @x» (rulings R77, R92): every person of every task passes
    # here, compared by every contact known of him — the target, his @username, and the
    # contact his record kept from an earlier task (a phone, say).
    banned = await memory.excluded(repo)
    if ({target.casefold()} | memory.identifiers(contact)) & banned:
        return NotAttached(memory.NOT_WITH)
    if contact.do_not_contact:
        return NotAttached("просил не писать")
    if any(
        conv.contact_id == contact_id for conv in await repo.task_conversations(task_id)
    ):
        return NotAttached(IN_TASK_ALREADY)
    # One live conversation per person: his replies are read in the newest one, so a second
    # would take what he writes about the other task for this one.
    busy = await repo.blocking_conversation(peer.peer_id)
    if busy is not None:
        return NotAttached(f"уже в задаче #{busy.task_id} — пока она не закончится")
    if found is not None:
        await repo.update_contact_card(
            contact_id, {**contact.card, "found": found}, contact.language
        )
        contact = await repo.get_contact(contact_id)
    conv_id = await repo.create_conversation(task_id, contact_id, state=S.PLANNED)
    # Without it the gateway would drop the replies of this person.
    await listen(repo, gateway)
    return Attached(conv_id, contact)


async def listened(repo: Repo) -> set[int]:
    """Telegram ids the gateway listens to: every person of a task in work but one the
    owner said not to work with (rulings R104, R92). Nobody else's message reaches the inbox."""
    excluded = await memory.excluded(repo)
    return {
        contact.peer_id
        for contact in await repo.contacts_in_work()
        if not memory.identifiers(contact) & excluded
    }


async def listen(repo: Repo, gateway: Gateway) -> None:
    """Called after every change of who the gateway hears (`listened`)."""
    await gateway.set_allowed(await listened(repo))


async def reopen_closed(repo: Repo, conv_id: int) -> bool:
    """The owner asked to write again to a person the agent had finished with (ruling R72),
    or the person wrote himself (R104 b): his conversation goes back to where the talk stood
    — TALKING when he ever wrote, CONTACTED otherwise. True when it reopened. (The gateway
    hears him already: every person of a task in work is listened to.)

    Not reopened: one who asked not to be written to or wanted the owner himself (the
    do-not-contact mark), one whose chat was with the owner when the talk closed (a
    hand-over card not given back), one now busy in another task, a person of a task that is not in work. In a
    PAUSED task he reopens too: nothing wakes while it is paused, and the wake after
    «Продолжить» has his message. A card about him that ended stays ended."""
    conv = await repo.get_conversation(conv_id)
    if conv.state is not S.CLOSED:
        return False
    status = (await repo.get_task(conv.task_id)).status
    if status not in (TaskStatus.ACTIVE, TaskStatus.PAUSED):
        return False
    contact = await repo.get_contact(conv.contact_id)
    if contact.do_not_contact:
        return False
    # The chat is the owner's while his newest hand-over card was not given back («Вернуть
    # агенту» approves it): closing the person expires a card still pending, so «not
    # approved» is «still with the owner when the talk closed».
    handoffs = [
        a for a in await repo.approvals_of(conv_id) if a.kind is ApprovalKind.HANDOFF
    ]
    if handoffs and max(handoffs, key=lambda a: a.id).status is not (
        ApprovalStatus.APPROVED
    ):
        return False
    if await repo.blocking_conversation(contact.peer_id) is not None:
        return False
    new = S.TALKING if conv.last_in_at is not None else S.CONTACTED
    return await repo.move_if(conv_id, S.CLOSED, new)


async def close(repo: Repo, gateway: Gateway, conv_id: int, reason: str) -> bool:
    """Done with this person: CLOSED with `reason`. False when the state machine does not
    allow the move (or the conversation is closed already); nothing changes then."""
    try:
        await repo.set_state(conv_id, S.CLOSED)
    except InvalidTransition:
        return False
    await repo.update_conversation(conv_id, close_reason=reason)
    # Nothing a card about him decides can be carried out now: «Да» must not answer
    # «Заказ подтверждён» for a person nobody will write to.
    for approval in await repo.pending_approvals(conv_id):
        await repo.resolve_approval(approval.id, approval.nonce, ApprovalStatus.EXPIRED)
    # His silence is no news any more.
    await repo.drop_events(conv_id, [EventKind.SILENCE])
    await listen(repo, gateway)
    return True


def failed_in_a_row(items: Sequence[OutboxItem]) -> int:
    """How many of the conversation's outbox items FAILED since the newest one that was
    SENT: the sends that went wrong one after another for this person."""
    count = 0
    for item in sorted(items, key=lambda item: item.id, reverse=True):
        if item.status is OutboxStatus.SENT:
            break
        if item.status is OutboxStatus.FAILED:
            count += 1
    return count


async def settle_cards_in_chat(repo: Repo, conv_id: int) -> None:
    """The owner wrote to this person himself while cards about him waited: he dealt with it
    in the chat. A question is answered by that («владелец ответил сам в чате»), every other
    card expires (no permission is granted by it), and the person goes back to where the talk
    stands. A hand-back card is left alone: the chat must stay returnable."""
    for approval in await repo.pending_approvals(conv_id):
        if approval.kind is ApprovalKind.HANDOFF:
            continue
        if approval.kind is ApprovalKind.QUESTION:
            await repo.resolve_approval(
                approval.id, approval.nonce, ApprovalStatus.ANSWERED, ANSWERED_IN_CHAT
            )
        else:
            await repo.resolve_approval(
                approval.id, approval.nonce, ApprovalStatus.EXPIRED
            )
    if not await repo.pending_approvals(conv_id):
        await reopen(repo, conv_id)


async def reopen(repo: Repo, conv_id: int) -> ConvState:
    """A conversation that waited for the owner goes back to where the talk stands; return
    its state. One in any other state is left alone."""
    conv = await repo.get_conversation(conv_id)
    if conv.state not in _WAITING:
        return conv.state
    if conv.last_in_at is not None:
        new = S.TALKING
    elif conv.first_sent or any(
        item.status in _ON_ITS_WAY for item in await repo.outbox_of(conv_id)
    ):
        new = S.CONTACTED
    else:
        new = S.PLANNED
    # Only from the state read above: a conversation a wake moved meanwhile (closed, say)
    # stays where it is.
    if await repo.move_if(conv_id, conv.state, new):
        return new
    return (await repo.get_conversation(conv_id)).state
