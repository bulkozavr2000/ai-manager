"""What the program shows the owner: cards with buttons and plain notifications.

`UI` is implemented by the control bot and, in tests, by `FakeUI`. The bot holds cards back in
the owner's quiet hours; `flush` sends what was held back.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from tg_agent.models import ApprovalKind


@dataclass
class Card:
    """A message to the owner. `buttons` are rows of (label, callback data)."""

    text: str
    buttons: list[list[tuple[str, str]]] = field(default_factory=list)
    # Character ranges [start, end) of `text` shown in bold. The text itself is never parsed
    # as markup, so nothing a counterpart wrote can turn into formatting.
    bold: list[tuple[int, int]] = field(default_factory=list)
    # The task this card shows, when it is a task card: a reply to it corrects that task.
    task_id: int | None = None
    # What the card is, for the owner's chat log and the chat's layout: "task", "approval",
    # "result", "finish_ask", "site_login", "task_money" or "board" (the status, which is
    # never logged); empty for the rest.
    kind: str = ""
    approval_id: int | None = None
    # What a card that waits for the owner without a database row of its own is about
    # (`control/layout.ChatLayout`): the site of a sign-in card, say.
    key: str = ""


class Bold(str):
    """A piece of a card's text shown in bold (see `compose`)."""


def compose(
    pieces: Iterable[str], buttons: list[list[tuple[str, str]]] | None = None
) -> Card:
    """A card from plain and `Bold` pieces; the bold ones are kept as character ranges."""
    text, bold = "", []
    for piece in pieces:
        if isinstance(piece, Bold) and piece:
            bold.append((len(text), len(text) + len(piece)))
        text += str(piece)
    return Card(text, buttons or [], bold)


class CardGone(Exception):
    """A card that can no longer be edited: the owner deleted it, or Telegram refuses."""


class UI(Protocol):
    async def send(self, card: Card, *, silent: bool = False) -> int:
        """Show a card; return its message id (to edit it in place later). `silent`: no
        sound or vibration (a card moved to the bottom of the chat is no news)."""
        ...

    async def edit(self, msg_id: int, card: Card) -> None:
        """Replace a card shown earlier; `CardGone` when it cannot be edited any more."""
        ...

    async def delete(self, msg_id: int) -> None:
        """Delete a message of the bot; `CardGone` when Telegram refuses (a message older
        than 48 hours, or one that is gone already)."""
        ...

    async def pin(self, msg_id: int) -> None:
        """Pin a card to the top of the chat, without a notification."""
        ...

    async def unpin(self, msg_id: int) -> None:
        """Unpin a message pinned earlier (an old status left behind, ruling R52)."""
        ...

    async def notify(self, text: str) -> int:
        """Tell the owner something that needs no answer; return the message id."""
        ...

    async def flush(self) -> int:
        """Send what quiet hours held back; return how many messages went."""
        ...

    async def typing(self) -> None:
        """Show "typing…" in the chat while the bot reads the owner's message."""
        ...


class FakeUI:
    """Implements `UI` in memory: `sent` cards, `edits` (msg_id, card), `pins` and `notes`.

    Like Telegram, it gives every message, card or note, a unique and increasing id;
    `sent_ids` and `note_ids` run parallel to `sent` and `notes`. It also keeps what is on
    screen (`chat`): every message in the order of the chat, as it reads now — the owner's
    messages too (`owner_says`, which takes its id from the same count). `silent` holds the
    ids sent without sound, `deleted` the ids deleted; ids in `undeletable` refuse a delete
    (a message older than 48 hours), and a deleted message refuses an edit, as Telegram does.
    `unpins` holds the ids unpinned, `typings` counts the "typing…" shown.
    """

    def __init__(self) -> None:
        self.sent: list[Card] = []
        self.sent_ids: list[int] = []
        self.edits: list[tuple[int, Card]] = []
        self.pins: list[int] = []
        self.unpins: list[int] = []
        self.typings = 0
        self.notes: list[str] = []
        self.note_ids: list[int] = []
        self.silent: list[int] = []
        self.deleted: list[int] = []
        self.undeletable: set[int] = set()
        self.screen: dict[int, Card] = {}
        self._last_id = 0

    def _next_id(self) -> int:
        self._last_id += 1
        return self._last_id

    def owner_says(self, text: str) -> int:
        """A message of the owner in the chat; returns its id."""
        self.screen[msg_id := self._next_id()] = Card(text, kind="owner")
        return msg_id

    def chat(self) -> list[tuple[int, Card]]:
        """What is on screen now, oldest first."""
        return sorted(self.screen.items())

    def last(self) -> Card:
        """The last message of the chat."""
        return self.chat()[-1][1]

    async def send(self, card: Card, *, silent: bool = False) -> int:
        self.sent.append(card)
        self.sent_ids.append(msg_id := self._next_id())
        self.screen[msg_id] = card
        if silent:
            self.silent.append(msg_id)
        return msg_id

    async def edit(self, msg_id: int, card: Card) -> None:
        if msg_id in self.deleted:
            raise CardGone("Bad Request: message to edit not found")
        self.edits.append((msg_id, card))
        if msg_id in self.screen:
            self.screen[msg_id] = card

    async def delete(self, msg_id: int) -> None:
        if msg_id in self.undeletable or msg_id in self.deleted:
            raise CardGone("Bad Request: message can't be deleted")
        self.deleted.append(msg_id)
        self.screen.pop(msg_id, None)

    async def pin(self, msg_id: int) -> None:
        self.pins.append(msg_id)

    async def unpin(self, msg_id: int) -> None:
        self.unpins.append(msg_id)

    async def notify(self, text: str) -> int:
        self.notes.append(text)
        self.note_ids.append(msg_id := self._next_id())
        self.screen[msg_id] = Card(text, kind="notice")
        return msg_id

    async def flush(self) -> int:
        return 0

    async def typing(self) -> None:
        self.typings += 1


_TITLES = {
    ApprovalKind.SHARE_PHONE: "Дать твой телефон?",
    ApprovalKind.SHARE_ADDRESS: "Дать твой адрес?",
    ApprovalKind.CONFIRM_ORDER: "Подтвердить заказ?",
    ApprovalKind.QUESTION: "Вопрос агента",
    ApprovalKind.HANDOFF: "Напиши сам",
}
