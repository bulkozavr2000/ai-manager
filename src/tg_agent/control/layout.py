"""The layout of the owner's chat with the bot (stage 5, rulings R52–R56).

The owner opens the chat and sees at once what needs him: the last message is the newest card
that waits for him (a draft, a question, a result, a confirmation), or else the status of all
tasks. `ChatLayout` wraps the program's `UI`, so everything the bot shows goes through it: it
knows the chat's tail (the newest message of either side), tracks the cards that wait, and
`settle()` puts the chat right after every handler of the bot and every round of the
Scheduler — a card that waits no more is edited into its settled form, the status is redrawn
in place, and the message that must be last and is not is deleted and sent again silently.

What it keeps lives in kv, so that a restart finds the chat as it was: the tail, the status
message (`board_msg_id`, `board_text`) and the list of waiting cards, each with the card as
it was sent, so that it can be sent again as it is.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from tg_agent.clock import Clock
from tg_agent.control.base import UI, Card, CardGone
from tg_agent.control.cards import (
    MOVED,
    NO_TASKS,
    TAKES_PERSON,
    draft_closed_card,
    launched_card,
    not_needed,
    pick_person_key,
    result_waits_key,
    settled_approval_card,
    unmarked,
)
from tg_agent.db import NotFound, Repo
from tg_agent.models import ApprovalKind, ApprovalStatus, TaskStatus

_log = logging.getLogger(__name__)

# The newest message of the chat, of either side.
TAIL_KEY = "chat_tail"
# The cards that wait for the owner, oldest first (a JSON list of entries, see `_Entry`).
CARDS_KEY = "chat_cards"
# The status message and what it shows now (text and buttons).
BOARD_MSG_KEY = "board_msg_id"
BOARD_TEXT_KEY = "board_text"
# Kinds of waiting card that have no database row of their own: the code that sent one
# releases it (`release`); one nobody released is settled after a day.
UNOWNED = frozenset({"finish_ask", "site_login", "task_money"})
UNOWNED_WAIT = timedelta(hours=24)
# A card that waited longer than this is sent again by the evening reminder (ruling R62).
REMIND_AFTER = timedelta(hours=3)


def task_card_key(msg_id: int) -> str:
    """A task's card by its message id: a reply to it corrects the draft or instructs the
    task's agent (`ControlLogic`). Every id the card ever had keeps pointing at its task."""
    return f"task_card:{msg_id}"


def approval_card_key(msg_id: int) -> str:
    """An approval by an old message id of its card (the card moved since): a reply to the
    card as the owner saw it still answers it (`ControlLogic`)."""
    return f"approval_card:{msg_id}"


def card_msg_key(kind: str, key: int | str) -> str:
    """The message a tracked card was last shown in."""
    return f"card_msg:{kind}:{key}"


def _card_data(card: Card) -> dict[str, Any]:
    return {
        "text": card.text,
        "buttons": card.buttons,
        "bold": card.bold,
        "task_id": card.task_id,
        "kind": card.kind,
        "approval_id": card.approval_id,
        "key": card.key,
    }


def _card(data: dict[str, Any]) -> Card:
    return Card(
        data["text"],
        [[(label, value) for label, value in row] for row in data["buttons"]],
        [(start, end) for start, end in data["bold"]],
        task_id=data.get("task_id"),
        kind=data.get("kind", ""),
        approval_id=data.get("approval_id"),
        key=data.get("key", ""),
    )


def _shown(card: Card) -> str:
    """What a status shows, buttons included: a change of either is a change."""
    return card.text + "\n" + json.dumps(card.buttons, ensure_ascii=False)


def _same(one: dict[str, Any], other: dict[str, Any]) -> bool:
    """Two stored cards are the same (tuples and lists alike, as JSON keeps them)."""
    return json.dumps(one, ensure_ascii=False) == json.dumps(other, ensure_ascii=False)


class ChatLayout:
    """Implements `UI` around the bot's own: keeps the right message last (ruling R52).

    `status` draws the status card (`ControlLogic.status_card`); without it there is no
    status. Every public call that changes what is kept takes one lock, so that `settle`,
    the sends of a wake and a handler's `release` never interleave their writes."""

    def __init__(
        self,
        repo: Repo,
        ui: UI,
        *,
        clock: Clock | None = None,
        status: Callable[[], Awaitable[Card]] | None = None,
    ) -> None:
        self.repo = repo
        self.inner = ui
        self.status = status
        self._clock = clock
        self._lock = asyncio.Lock()

    def _now(self) -> datetime:
        return self._clock.now() if self._clock is not None else datetime.now(UTC)

    # -- UI --

    async def send(self, card: Card, *, silent: bool = False) -> int:
        """Show a card; one that waits for the owner is tracked from now on."""
        async with self._lock:
            msg_id = await self.inner.send(card, silent=silent)
            await self._tail(msg_id)
            try:
                await self._track(msg_id, card)
            except Exception:  # noqa: BLE001 - the card went: the sender must hear so
                _log.exception("could not track the card %s", msg_id)
            return msg_id

    async def notify(self, text: str) -> int:
        async with self._lock:
            msg_id = await self.inner.notify(text)
            await self._tail(msg_id)
            return msg_id

    async def edit(self, msg_id: int, card: Card) -> None:
        await self.inner.edit(msg_id, card)

    async def delete(self, msg_id: int) -> None:
        await self.inner.delete(msg_id)

    async def pin(self, msg_id: int) -> None:
        await self.inner.pin(msg_id)

    async def unpin(self, msg_id: int) -> None:
        await self.inner.unpin(msg_id)

    async def flush(self) -> int:
        return await self.inner.flush()

    async def typing(self) -> None:
        await self.inner.typing()

    # -- what the bot's handlers report --

    async def saw(self, msg_id: int) -> None:
        """A message of the owner: it is the chat's tail now."""
        async with self._lock:
            await self._tail(msg_id)

    async def release(
        self, kind: str, key: int | str, settled: Card | None = None
    ) -> bool:
        """The code that owns a waiting card says it waits no more: it is edited into
        `settled` (None: the same card without its marker and buttons) and is not tracked
        any more. False when no such card is tracked."""
        async with self._lock:
            entries = await self._entries()
            found = [
                e
                for e in entries
                if e["kind"] == kind and e["key"] == str(key) and e["settled"] is None
            ]
            if not found:
                return False
            for entry in found:
                card = (
                    settled if settled is not None else unmarked(_card(entry["card"]))
                )
                entry["settled"] = _card_data(card)
            await self._apply_settled(entries)
            return True

    def _waiting(
        self, entries: list[dict[str, Any]], kind: str, key: int | str
    ) -> dict[str, Any] | None:
        found = [
            e
            for e in entries
            if e["kind"] == kind and e["key"] == str(key) and e["settled"] is None
        ]
        return found[-1] if found else None

    async def replace(self, kind: str, key: int | str, card: Card) -> int | None:
        """A waiting card whose thing changed (a draft rewritten in words): the old message
        goes and `card` is sent as the newest waiting card, with sound — what the owner just
        changed is in front of him, and he never sees two cards of one thing. Every record
        of the message id follows. Returns the new id, or None when no such card waits."""
        async with self._lock:
            entries = await self._entries()
            entry = self._waiting(entries, kind, key)
            if entry is None:
                return None
            entries.remove(entry)
            entries.append(entry)  # the newest now: `settle` keeps it last
            entry["card"] = _card_data(card)
            entry["since"] = self._now().isoformat()
            await self._move_card(entry, entries, silent=False)
            return int(entry["msg"])

    async def resend_newest(self, waited: timedelta = REMIND_AFTER) -> bool:
        """The daily reminder (ruling R62): when some card has waited longer than `waited`,
        the newest waiting card is deleted and sent again at the bottom, with sound; every
        record of its id follows. False when nothing waited that long. A failure is raised:
        the Scheduler tries again on its next round."""
        async with self._lock:
            entries = await self._entries()
            for entry in entries:
                if entry["settled"] is None:
                    entry["settled"] = await self._settled_form(entry)
            entries = await self._apply_settled(entries)
            waiting = [e for e in entries if e["settled"] is None]
            now = self._now()
            if not any(
                now - datetime.fromisoformat(entry["since"]) > waited
                for entry in waiting
            ):
                return False
            await self._move_card(waiting[-1], entries, silent=False)
            return True

    async def update(self, kind: str, key: int | str, card: Card) -> bool:
        """A waiting card redrawn in place when what it shows changed; it keeps its message
        and its place. False when no such card waits."""
        async with self._lock:
            entries = await self._entries()
            entry = self._waiting(entries, kind, key)
            if entry is None:
                return False
            data = _card_data(card)
            if _same(entry["card"], data):
                return True
            await self.inner.edit(entry["msg"], card)
            entry["card"] = data
            await self._save(entries)
            return True

    # -- settle --

    async def settle(self) -> None:
        """Reconcile the waiting cards with the database, redraw the status, restore the
        bottom of the chat. Never raises: a failure is logged, and the next call tries
        again."""
        async with self._lock:
            try:
                await self._settle()
            except Exception:  # noqa: BLE001 - the next call tries again
                _log.exception("the chat's layout could not be settled")

    async def _settle(self) -> None:
        entries = await self._entries()
        for entry in entries:
            if entry["settled"] is None:
                entry["settled"] = await self._settled_form(entry)
        # What is kept from here on: a move saves this list, and must not bring back the
        # cards that were just settled.
        entries = await self._apply_settled(entries)
        waiting = [e for e in entries if e["settled"] is None]
        card = await self.status() if self.status is not None else None
        board = await self.repo.get_kv(BOARD_MSG_KEY)
        if card is not None and not board and card.text == NO_TASKS:
            card = None  # nothing to show yet
        tail = int(await self.repo.get_kv(TAIL_KEY) or 0)
        if waiting:
            if card is not None:
                await self._redraw_status(card)
                # A status sent anew just now (there was none, or the owner deleted it)
                # is the tail: the waiting card must still come below it.
                tail = int(await self.repo.get_kv(TAIL_KEY) or 0)
            wanted = waiting[-1]
            if wanted["msg"] != tail:
                await self._move_card(wanted, entries)
        elif card is not None:
            if board and int(board) == tail:
                await self._redraw_status(card)
            else:
                await self._move_status(card, int(board) if board else None)

    # -- the waiting cards --

    async def _entries(self) -> list[dict[str, Any]]:
        return list(json.loads(await self.repo.get_kv(CARDS_KEY) or "[]"))

    async def _save(self, entries: list[dict[str, Any]]) -> None:
        await self.repo.set_kv(CARDS_KEY, json.dumps(entries, ensure_ascii=False))

    async def _tail(self, msg_id: int) -> None:
        tail = int(await self.repo.get_kv(TAIL_KEY) or 0)
        await self.repo.set_kv(TAIL_KEY, str(max(tail, msg_id)))

    async def _waits(self, card: Card) -> tuple[str, str] | None:
        """(kind, key) of a card that waits for the owner (ruling R53), or None."""
        if card.kind == "approval" and card.approval_id is not None:
            approval = await self.repo.get_approval(card.approval_id)
            if approval.kind is ApprovalKind.HANDOFF:
                return None  # the owner took that chat himself
            return "approval", str(card.approval_id)
        if card.kind == "task" and card.task_id is not None:
            task = await self.repo.get_task(card.task_id)
            return (
                ("task", str(card.task_id)) if task.status is TaskStatus.DRAFT else None
            )
        if card.kind == "result" and card.task_id is not None:
            return "result", str(card.task_id)
        if card.kind == "pick_person" and card.key:
            return "pick_person", card.key
        if card.kind in UNOWNED and card.key:
            return card.kind, card.key
        return None

    async def _track(self, msg_id: int, card: Card) -> None:
        waits = await self._waits(card)
        if waits is None:
            return
        kind, key = waits
        entries = await self._entries()
        # A newer card of the same kind and key replaces the older one.
        for entry in entries:
            if (
                entry["kind"] == kind
                and entry["key"] == key
                and entry["settled"] is None
            ):
                old = _card(entry["card"])
                gone = not_needed(old) if kind in UNOWNED else unmarked(old)
                entry["settled"] = _card_data(gone)
        entries.append(
            {
                "msg": msg_id,
                "kind": kind,
                "key": key,
                "since": self._now().isoformat(),
                "card": _card_data(card),
                "settled": None,
            }
        )
        await self.repo.set_kv(card_msg_key(kind, key), str(msg_id))
        if kind == "approval":
            # Its reply finds it by this id; `Approvals` writes the same right after.
            await self.repo.set_approval_msg(int(key), msg_id)
        await self._apply_settled(entries)

    async def last_message(self, kind: str, key: int | str) -> int | None:
        """The message a card of this kind and key was last shown in, waiting or settled
        since (a settle may have settled it a moment before its owner's `release`)."""
        msg = await self.repo.get_kv(card_msg_key(kind, key))
        return int(msg) if msg else None

    async def _settled_form(self, entry: dict[str, Any]) -> dict[str, Any] | None:
        """The card's settled form when its thing waits no more, else None (it waits; the
        records of its message id are put right on the way)."""
        kind, key, msg = entry["kind"], entry["key"], entry["msg"]
        card = _card(entry["card"])
        if kind == "approval":
            try:
                approval = await self.repo.get_approval(int(key))
            except NotFound:
                return _card_data(unmarked(card))
            if approval.status is ApprovalStatus.PENDING:
                if approval.control_msg_id != msg:
                    await self.repo.set_approval_msg(approval.id, msg)
                return None
            contact = None
            if approval.conversation_id is not None:
                conv = await self.repo.get_conversation(approval.conversation_id)
                contact = await self.repo.get_contact(conv.contact_id)
            return _card_data(settled_approval_card(approval, contact))
        if kind == "task":
            task = await self.repo.get_task(int(key))
            if task.status is TaskStatus.DRAFT:
                await self._task_card_follows(task.id, msg)
                return None
            if task.status is TaskStatus.CANCELLED:
                return _card_data(draft_closed_card(task.id, task.title))
            return _card_data(launched_card(task.id, task.title))
        if kind == "result":
            task = await self.repo.get_task(int(key))
            waits = await self.repo.get_kv(result_waits_key(task.id)) == "1"
            # A result waits while its task is in work (R109: only «Завершить» ends it);
            # a task closed meanwhile has nothing for «Завершить» to do: the card settles.
            if waits and task.status is not TaskStatus.CANCELLED:
                await self._task_card_follows(task.id, msg)
                return None
            return _card_data(unmarked(card))
        if kind == "pick_person":
            # «Кого ты имел в виду?» waits while its matches are kept and its task can still
            # take a person; a press releases it (`ControlLogic._pick_button`).
            task = await self.repo.get_task(int(key))
            if (
                await self.repo.get_kv(pick_person_key(task.id))
                and task.status in TAKES_PERSON
            ):
                return None
            return _card_data(not_needed(card))
        since = datetime.fromisoformat(entry["since"])
        if self._now() - since >= UNOWNED_WAIT:
            return _card_data(not_needed(card))
        return None

    async def _task_card_follows(self, task_id: int, msg_id: int) -> None:
        if await self.repo.get_kv(task_card_key(msg_id)) != str(task_id):
            await self.repo.set_kv(task_card_key(msg_id), str(task_id))

    async def _apply_settled(
        self, entries: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Edit every card that waits no more into its settled form and stop tracking it.
        One that is gone is just dropped; one whose edit failed otherwise is kept, with its
        settled form, for the next call. Returns the entries kept (and saved)."""
        kept = []
        for entry in entries:
            if entry["settled"] is None:
                kept.append(entry)
                continue
            try:
                await self.inner.edit(entry["msg"], _card(entry["settled"]))
            except CardGone as gone:
                _log.info("a settled card is gone (%s): dropped", gone)
            except Exception:  # noqa: BLE001 - tried again by the next call
                _log.exception("could not settle the card %s", entry["msg"])
                kept.append(entry)
        await self._save(kept)
        return kept

    async def _move_card(
        self,
        entry: dict[str, Any],
        entries: list[dict[str, Any]],
        *,
        silent: bool = True,
    ) -> None:
        """A waiting card to the bottom; every record of its id follows."""
        old = entry["msg"]
        new = await self._resend(old, _card(entry["card"]), silent=silent)
        entry["msg"] = new
        await self._save(entries)
        kind, key = entry["kind"], entry["key"]
        # The old id is never forgotten: the owner may reply to the card as he saw it a
        # moment ago, or to the «⬇ перенесено ниже» line left in its place.
        await self.repo.set_kv(card_msg_key(kind, key), str(new))
        if kind == "approval":
            await self.repo.set_approval_msg(int(key), new)
            await self.repo.set_kv(approval_card_key(old), key)
        elif kind in ("task", "result"):
            await self.repo.set_kv(task_card_key(new), key)

    # -- the status --

    async def _redraw_status(self, card: Card) -> None:
        """The status edited in place when what it shows changed; one that cannot be edited
        (the owner deleted it) is sent and pinned again."""
        board = await self.repo.get_kv(BOARD_MSG_KEY)
        if not board:
            await self._move_status(card, None)
            return
        if _shown(card) == await self.repo.get_kv(BOARD_TEXT_KEY):
            return
        try:
            await self.inner.edit(int(board), card)
        except CardGone as gone:
            _log.info("the status cannot be edited (%s): sending it anew", gone)
            await self._move_status(card, None)
            return
        await self.repo.set_kv(BOARD_TEXT_KEY, _shown(card))

    async def _move_status(self, card: Card, old: int | None) -> None:
        """The status sent at the bottom (the old one deleted) and pinned."""
        new = await self._resend(old, card, pinned=True)
        await self.repo.set_kv(BOARD_MSG_KEY, str(new))
        await self.repo.set_kv(BOARD_TEXT_KEY, _shown(card))
        try:
            await self.inner.pin(new)
        except Exception as error:  # noqa: BLE001 - an unpinned status still works
            _log.warning("could not pin the status: %s", error)

    async def _resend(
        self, old: int | None, card: Card, *, silent: bool = True, pinned: bool = False
    ) -> int:
        """Delete `old` and send `card` again, without sound unless `silent` is False;
        returns the new id. A delete Telegram refuses (a message older than 48 hours) leaves
        «⬇ перенесено ниже» in its place, unpinned when it was the `pinned` status (else the
        old pins pile up beside the new one); an edit or unpin refused too is only logged.
        Any other failure of the delete is raised: nothing was changed, and the next call
        tries again."""
        if old is not None:
            try:
                await self.inner.delete(old)
            except CardGone as refused:
                _log.info("message %s cannot be deleted (%s)", old, refused)
                try:
                    await self.inner.edit(old, Card(MOVED))
                except Exception as error:  # noqa: BLE001 - the new one goes all the same
                    _log.info("message %s cannot be edited either: %s", old, error)
                if pinned:
                    try:
                        await self.inner.unpin(old)
                    except Exception as error:  # noqa: BLE001 - a second pin is harmless
                        _log.info("message %s cannot be unpinned: %s", old, error)
        new = await self.inner.send(card, silent=silent)
        await self._tail(new)
        return new
