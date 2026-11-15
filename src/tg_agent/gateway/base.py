"""What the rest of the program sees of Telegram: incoming messages, actions, errors.

`Gateway` is implemented by the Telethon userbot and, in tests, by `FakeGateway`. A Telegram
error reaches the caller as `TgActionError`, already classified into what to do about it.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from tg_agent.policy.tg_safety import TgErrorKind

MessageKind = Literal[
    "text",
    "photo",
    "voice",
    "video",
    "document",
    "sticker",
    "location",
    "contact",
    "poll",
    "other",
]

# How many chats an ordinary (not Premium) account may keep in one chat folder (ruling R106).
FOLDER_LIMIT = 100


@dataclass(frozen=True)
class PeerInfo:
    """A Telegram user as `Gateway.resolve` or `Gateway.search_contacts` finds it."""

    peer_id: int
    username: str | None  # not every account has one
    display_name: str
    is_bot: bool
    # "+<digits>": the number of a contact of the account, when Telegram shows it to the
    # account; None otherwise (and always for `resolve`).
    phone: str | None = None


_WORD = re.compile(r"\w+")
MIN_NAME_WORD = 3  # a shorter word of a name tells nobody apart


def _words(text: str) -> list[str]:
    return [w.replace("ё", "е") for w in _WORD.findall(text.casefold())]


def name_matches(query: str, *names: str) -> bool:
    """Does every word of `query` stand at the start of a word of `names`? A Russian case
    ending of the query is forgiven («у Алишера», «с Алишером» find «Алишер»), and «ё» is
    «е». A word shorter than three letters matches nothing: it would match half the book."""
    wanted = _words(query)
    have = [word for name in names for word in _words(name)]
    if not wanted or any(len(word) < MIN_NAME_WORD for word in wanted):
        return False

    def found(word: str) -> bool:
        stems = [word]
        if len(word) >= 4:
            stems.append(word[:-1])
        if len(word) >= 5:
            stems.append(word[:-2])
        return any(name.startswith(stem) for name in have for stem in stems)

    return all(found(word) for word in wanted)


@dataclass(frozen=True)
class PublicPeer:
    """An account, channel or group that Telegram's public search finds."""

    username: str
    title: str  # the name it shows
    kind: Literal["user", "channel", "group"]
    about: str = ""  # a channel's or group's description, where contacts often are


@dataclass(frozen=True)
class Incoming:
    """A message a counterpart sent to the account."""

    peer_id: int
    tg_msg_id: int
    text: str  # the text, or the caption of a media message; "" when there is none
    kind: MessageKind
    date: datetime  # when Telegram says it was sent; timezone-aware
    # Telegram marks it as sent on the sender's behalf (a business greeting or away message).
    offline: bool = False
    file_name: str | None = None  # the name of a document


@dataclass(frozen=True)
class ChatMessage:
    """One message of a chat, as `Gateway.recent_messages` reads it."""

    tg_msg_id: int
    out: bool  # the account sent it (the agent or the owner himself)
    text: str  # "" for media without a caption
    date: datetime  # timezone-aware


class TgActionError(Exception):
    """A Telegram action failed. `wait_s` is how long Telegram asks to wait (FLOOD_WAIT)."""

    def __init__(
        self, kind: TgErrorKind, detail: str, wait_s: int | None = None
    ) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail
        self.wait_s = wait_s


class TgSendUncertain(Exception):
    """A message may or may not have reached the chat.

    Raised by `send_text` and `send_file` when the call itself failed in a way that does not
    say whether Telegram took the message: the connection broke, Telethon gave up after its
    retries, nothing came back. It is not a `TgActionError` on purpose: a `TgActionError` is
    Telegram's answer (or a refusal before anything was asked), and "refused" means "nothing
    was sent". Here nobody knows. The text is the name of the underlying error's class, never
    a word of any text.
    """


class Gateway(Protocol):
    """The account the agent writes from.

    Every action may raise `TgActionError`; a send may also raise `TgSendUncertain`.
    """

    async def resolve(self, target: str) -> PeerInfo:
        """Find a user by @username (the "@" is optional) or by phone ("+998901234567").

        A phone finds only a person whose privacy settings let strangers find him by it.
        """
        ...

    async def search_public(self, query: str, limit: int) -> list[PublicPeer]:
        """Telegram's global search: public accounts, channels and groups matching `query`.

        Reads only; nobody is written to and nothing is joined.
        """
        ...

    async def search_contacts(self, query: str, limit: int) -> list[PeerInfo]:
        """The account's own contacts (its address book) whose name or username matches
        `query` (`name_matches`), at most `limit` of them, bots left out.

        Reads only: nothing is written, nobody is written to, and Telegram's global search is
        not asked. The caller keeps nothing of the book but the person the owner chooses."""
        ...

    async def own_name(self) -> str:
        """The account's own name, as every counterpart sees it on its messages."""
        ...

    async def own_username(self) -> str | None:
        """The account's own @username (without "@"), or None when it has none: the link a
        person who cannot be written to first is given (ruling R61)."""
        ...

    async def send_text(self, peer_id: int, text: str) -> int:
        """Send a text message; return its Telegram id."""
        ...

    async def send_file(self, peer_id: int, path: str) -> int:
        """Send a file from disk; return the message's Telegram id."""
        ...

    async def typing(self, peer_id: int, seconds: float) -> None:
        """Show "typing…" in the chat for about this long."""
        ...

    async def mark_read(self, peer_id: int, max_id: int) -> None:
        """Mark the chat's messages up to `max_id` as read."""
        ...

    async def download(self, peer_id: int, tg_msg_id: int) -> bytes:
        """The media (a photo) of one message."""
        ...

    async def set_allowed(self, peer_ids: set[int]) -> None:
        """Listen only to these chats from now on: every other chat is ignored."""
        ...

    async def replay(
        self, peer_id: int, after_msg_id: int, known_out_ids: set[int]
    ) -> int:
        """Hand over what a chat got while the program was off; return how many messages.

        The messages newer than `after_msg_id` (the last one stored for the chat) go oldest
        first, the way live ones do: a counterpart's to the incoming callback, and one the
        owner wrote himself to the owner callback, with its Telegram id. An outgoing message
        whose id is in `known_out_ids` (the agent's own, stored) or that this gateway sent is
        skipped. Only a chat that is allowed (`set_allowed`) is read; for any other the answer
        is 0 and nothing is asked of Telegram, and so it is for a chat with no stored message
        (`after_msg_id` below 1: it has nothing it could have missed). At most a hundred
        messages come per call, the oldest of those newer than `after_msg_id`, so the rest is
        for the next call.
        """
        ...

    async def recent_messages(
        self, peer_id: int, after_msg_id: int, limit: int
    ) -> list[ChatMessage]:
        """The chat's messages newer than `after_msg_id`, oldest first; when there are more than
        `limit`, the newest `limit` of them. Reads only: nothing is marked read and no callback
        is called. The caller names the chat: pass only the peer of a conversation in the
        database (a person the agent has finished with can still be read: a goodbye whose
        send broke off must be looked for)."""
        ...

    # -- the chats themselves: their sound and the folder (rulings R105, R106) --

    async def is_muted(self, peer_id: int) -> bool:
        """Is the chat with this user muted now (its notifications off until a later
        moment)? Reads only."""
        ...

    async def set_muted(self, peer_id: int, muted: bool) -> None:
        """Mute the chat with this user for good, or give it its sound back."""
        ...

    async def folder_peers(self, title: str) -> list[int] | None:
        """The chats in the account's chat folder with this title, its pinned ones too (a
        user's id, or another number for a chat that is no user), or None when there is no
        such folder. Reads only."""
        ...

    async def add_to_folder(
        self, title: str, peer_id: int, remove: Collection[int] = ()
    ) -> None:
        """Put the chat with this user into the folder with this title, made with this one
        chat when there is none; the chats `remove` leave it first. Nothing else of the
        folder and no other folder changes. At most `FOLDER_LIMIT` chats fit in a folder."""
        ...
