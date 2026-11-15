"""The Telegram account the agent writes from, on Telethon (a user account, not a bot).

`TelethonGateway` implements `Gateway`. Only private chats with the users it was told about
(`set_allowed`) are listened to; groups, channels and every other chat are dropped before the
message is looked at. What it reports to the core:

- a counterpart's message in such a chat: `on_incoming(Incoming)`;
- an edit of one of their messages: `on_edit(peer_id, tg_msg_id, text)`;
- a message the owner wrote himself in the chat, from another device:
  `on_owner_manual(peer_id, text, tg_msg_id)`. The ids that `send_text` and `send_file` return
  are remembered, and an outgoing message with one of them is the agent's own, not the owner's.

`replay` hands over what a chat got while the program was off (a session string keeps no update
state, so Telegram does not send it again) through the same two callbacks. `wait_disconnected`
lets the app watch the link, which is retried for as long as it takes.

Every Telegram failure leaves a call as `TgActionError`, classified by `classify_error`; its
detail is a Telegram error code or the name of a Python exception class, never any text. The
one exception is a send whose outcome is unknown: only Telegram's own answer (an `RPCError`) or
a failure before anything was asked (no such peer, no such file, not started, not connected) is
a `TgActionError` there. A connection that breaks in the middle of `send_text` / `send_file`,
Telethon giving up after its retries, or a result with no message may all have come after the
message left, and raise `TgSendUncertain` (see `_put_in_chat`).

Things about Telethon 1.45 that the code relies on (read from its source, not its docs):

- A Telethon error class for a known code does not keep the code in `message`; see `_rpc_code`.
- `Message.photo`, `.video` and `.document` also return what a link preview shows; see
  `message_kind`.
- A message the client sends itself is not given to its own event handlers (`_updates/
  messagebox.py`), so the remembered ids only guard against an echo that gets through anyway.
- A `StringSession` keeps no entities between runs, so a user is unknown to the client after a
  restart until he writes or the dialogs are read; see `_peer`.
- `send_message` reads Markdown in the text unless `parse_mode=None`, and `send_file` takes a
  `str` that starts with http for a link to fetch.
- `get_input_entity` reads a bare id as a user, a chat or a channel, whichever it finds, and a
  name also as a display name; its in-memory cache is keyed by the bare number. See `_peer` and
  `resolve`.
- `is_connected()` stays True through a reconnect (it is only the wish to be connected), and
  `connection_retries=-1` retries forever, the first connection included.
- `iter_messages(min_id=0, reverse=True)` starts at the chat's first message. See `replay`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import re
import traceback
from collections import deque
from collections.abc import Awaitable, Callable, Collection
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from telethon import TelegramClient, events, functions, types, utils
from telethon.errors import RPCError
from telethon.errors.rpcerrorlist import rpc_errors_dict, rpc_errors_re
from telethon.sessions import StringSession

from tg_agent.gateway.base import (
    ChatMessage,
    Incoming,
    MessageKind,
    PeerInfo,
    PublicPeer,
    TgActionError,
    TgSendUncertain,
    name_matches,
)
from tg_agent.policy.tg_safety import TgErrorKind, classify_error
from tg_agent.search.contacts import normalize_phone

# Channels whose description is read on a public search: one more request each.
_ABOUTS_PER_SEARCH = 5

_log = logging.getLogger(__name__)

# Telegram's own waits up to this many seconds are slept off inside Telethon and the call is
# repeated; a longer one reaches the caller as FLOOD_WAIT.
FLOOD_SLEEP_THRESHOLD = 60
# Telethon's default is five attempts at reconnecting, after which the client stays disconnected
# and deaf for good. A PC that wakes before its network is up is the normal case here, so a lost
# link is retried for as long as it takes (the same goes for the first connection in `start()`).
CONNECTION_RETRIES = -1
RETRY_DELAY = 2  # seconds between two attempts
SENT_IDS_KEPT = 500  # how many of the ids of our own messages are remembered
REPLAY_LIMIT = (
    100  # how many messages one `replay` hands over, the oldest of the new ones
)
# A chat muted "for good" is muted until the largest moment Telegram's 32-bit date holds,
# which its apps show as muted forever; 0 gives the sound back (R105).
MUTED_FOR_GOOD = 2**31 - 1
# Folder ids below 2 are Telegram's own (all chats, the archive): a new folder takes the
# smallest free id from here (R106).
FIRST_FOLDER_ID = 2
_LOGIN_HINT = (
    "The Telegram session is not authorised. Run: uv run python -m tg_agent.setup login"
)


class SessionNotAuthorized(RuntimeError):
    """The saved Telegram session is not (or no longer) logged in."""


class NotStarted(RuntimeError):
    """The gateway was used before `start()` or after `stop()`: a mistake in the program."""


# -- reading a Telethon message --

# What an attachment is called, strongest first. Telethon calls a voice note a document, a video
# note a video, and a video sticker both a video and a sticker: the order tells them apart.
_ATTACHMENTS: tuple[tuple[MessageKind, str], ...] = (
    ("voice", "voice"),
    ("sticker", "sticker"),
    ("video", "video"),
    ("video", "video_note"),
    ("photo", "photo"),
    ("document", "document"),
)


def _text_or_other(msg: Any) -> MessageKind:
    return "text" if getattr(msg, "message", None) else "other"


def message_kind(msg: Any) -> MessageKind:
    """What a Telethon message is: voice, video, photo, sticker, document, text, or other.

    A location, a contact and a poll have their own kinds (2.22); other is what this program
    does not read, or nothing at all.
    A link preview is part of a text message, not an attachment, although `Message.photo`,
    `.video` and `.document` also return what the preview shows.
    """
    media = getattr(msg, "media", None)
    if isinstance(media, types.MessageMediaWebPage):
        return _text_or_other(msg)
    for kind, attribute in _ATTACHMENTS:
        if getattr(msg, attribute, None):
            return kind
    if isinstance(
        media,
        types.MessageMediaGeo | types.MessageMediaGeoLive | types.MessageMediaVenue,
    ):
        return "location"
    if isinstance(media, types.MessageMediaContact):
        return "contact"
    if isinstance(media, types.MessageMediaPoll):
        return "poll"
    if media is not None:
        return "other"
    return _text_or_other(msg)


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def to_incoming(msg: Any) -> Incoming:
    """A message a counterpart sent, as the core takes it.

    The text of a media message is its caption. `offline` is Telegram's mark of a message sent
    on someone's behalf (a business away or greeting message, or a scheduled one); a message
    Telethon built itself leaves it None.
    """
    kind = message_kind(msg)
    name = None
    if kind == "document":
        name = getattr(getattr(msg, "file", None), "name", None)
    return Incoming(
        peer_id=msg.chat_id,
        tg_msg_id=msg.id,
        text=getattr(msg, "message", None) or _spelled(kind, msg),
        kind=kind,
        date=_aware(msg.date),
        offline=bool(getattr(msg, "offline", False)),
        file_name=name or None,
    )


def _spelled(kind: MessageKind, msg: Any) -> str:
    """The words of a message that has no text of its own (2.22): a sticker's emoji, a
    location's place and coordinates, a contact's name and phone, a poll's question and
    answers. Whatever cannot be read is left out rather than guessed."""
    media = getattr(msg, "media", None)
    if kind == "sticker":
        return str(getattr(getattr(msg, "file", None), "emoji", None) or "")
    if kind == "location":
        point = getattr(media, "geo", None)
        lat, long = getattr(point, "lat", None), getattr(point, "long", None)
        where = f"{lat:.5f}, {long:.5f}" if lat is not None and long is not None else ""
        place = ", ".join(
            part
            for part in (getattr(media, "title", ""), getattr(media, "address", ""))
            if part
        )
        if place and where:
            return f"{place} ({where})"
        return place or where
    if kind == "contact":
        name = " ".join(
            part
            for part in (
                getattr(media, "first_name", ""),
                getattr(media, "last_name", ""),
            )
            if part
        )
        return " ".join(
            part for part in (name, getattr(media, "phone_number", "")) if part
        )
    if kind == "poll":
        poll = getattr(media, "poll", None)
        question = _plain(getattr(poll, "question", None))
        answers = [
            _plain(getattr(a, "text", None)) for a in getattr(poll, "answers", [])
        ]
        return " — ".join(part for part in (question, " / ".join(answers)) if part)
    return ""


def _plain(value: Any) -> str:
    """A text that newer Telegram layers wrap with its formatting (TextWithEntities)."""
    return str(getattr(value, "text", value) or "")


def is_relevant(event: Any, allowed: Collection[int]) -> bool:
    """Only a private chat that someone opened a conversation in is read at all.

    Groups, channels and every other private chat (a stranger, a bot, Saved Messages) are
    dropped before the message is looked at.
    """
    return bool(event.is_private) and event.chat_id in allowed


# -- errors --

_NUMBER_IN_PATTERN = re.compile(r"_?\(\\d\+\)")
_CODE_SHAPE = re.compile(r"[A-Za-z0-9_]{1,64}")


def _known_codes() -> dict[type, str]:
    """Telegram's code for each error class Telethon has, read from Telethon's own tables."""
    codes: dict[type, str] = {cls: code for code, cls in rpc_errors_dict.items()}
    for pattern, cls in rpc_errors_re:
        # 'FLOOD_WAIT_(\d+)': the number is not in the class's name; `seconds` holds it
        codes.setdefault(cls, _NUMBER_IN_PATTERN.sub("", pattern))
    return codes


_KNOWN_CODES = _known_codes()


def _rpc_code(error: Exception) -> str | None:
    """Telegram's error code, if the error carries one.

    Telethon has a class for most codes and keeps the code nowhere else: `message` of such an
    error is a constant of the class's family ('BAD_REQUEST', 'FLOOD'), never the code, so the
    class is looked up first. A code Telethon does not know comes as a plain family error whose
    `message` is the code. Anything that does not look like a code is not one: it must never
    end up in a detail.
    """
    for cls in type(error).__mro__:
        if cls in _KNOWN_CODES:
            return _KNOWN_CODES[cls]
    message = getattr(error, "message", None)
    if isinstance(message, str) and _CODE_SHAPE.fullmatch(message):
        return message
    return None


def _flood_wait_seconds(error: Exception, code: str) -> int | None:
    seconds = getattr(error, "seconds", None)
    if isinstance(seconds, int):
        return seconds
    tail = re.search(r"_(\d+)$", code)
    return int(tail.group(1)) if tail else None


def error_to_action(error: Exception) -> TgActionError:
    """What a failed Telegram call means for the program.

    A Telegram error is classified by its code (`classify_error`); a flood wait says how long.
    Anything else (a dropped connection, a timeout, a bad argument) is OTHER, named by its
    type. The detail is only ever a Telegram code or a class name, never a word of any text,
    since it is shown to the owner and written to the outbox.
    """
    if isinstance(error, TgActionError):
        return error
    code = _rpc_code(error)
    if code is None:
        return TgActionError(TgErrorKind.OTHER, type(error).__name__)
    kind = classify_error(code)
    wait = _flood_wait_seconds(error, code) if kind is TgErrorKind.FLOOD_WAIT else None
    return TgActionError(kind, code, wait)


def _action[**P, R](method: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Whatever goes wrong in a Telegram action leaves it as a `TgActionError`.

    The exceptions are a send of unknown outcome (`TgSendUncertain`, see `_put_in_chat`) and a
    gateway that was never started (`NotStarted`, a mistake in the program).
    """

    @functools.wraps(method)
    async def guarded(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await method(*args, **kwargs)
        except (TgActionError, TgSendUncertain, NotStarted):
            raise
        except Exception as error:  # noqa: BLE001 - Telethon and the network raise anything
            # `from None`: the original error's text must not be printed along with ours
            raise error_to_action(error) from None

    return guarded


async def _put_in_chat(call: Awaitable[Any]) -> Any:
    """Await the client call that puts a message into a chat; return the message it made.

    Only Telegram's own answer says that nothing was sent, and that is an `RPCError`. Whatever
    else goes wrong with the call itself may have happened after the message left: a
    connection that broke, Telethon giving up after its retries, a result with no message in
    it. That is `TgSendUncertain`, which the Sender answers by asking the owner instead of
    pausing the chat on a guess. Everything before the call (the peer, the file, the link) is
    not this: if it fails, nothing has left.
    """
    try:
        message = await call
    except RPCError as error:
        raise error_to_action(error) from None
    except Exception as error:  # noqa: BLE001 - whatever it is, the message may have left
        raise TgSendUncertain(type(error).__name__) from None
    if message is None or not isinstance(getattr(message, "id", None), int):
        raise TgSendUncertain("NoMessage")
    return message


def _contained[**P](
    handler: Callable[P, Awaitable[None]],
) -> Callable[P, Awaitable[None]]:
    """An event handler of the gateway never raises.

    Telethon would log the exception with its text, and the core's failure may quote a
    message. The stack of calls is logged instead: it names the line, not the words.
    """

    @functools.wraps(handler)
    async def contained(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            await handler(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - the core's failure ends here
            _log_failure(handler.__name__, error)

    return contained


def _log_failure(where: str, error: Exception) -> None:
    """Log that the core's callback failed: the line, not the words (see `_contained`)."""
    _log.error(
        "%s failed with %s\n%s",
        where,
        type(error).__name__,
        "".join(traceback.format_tb(error.__traceback__)),
    )


def _peer_key(peer: Any) -> int:
    """The number a folder's entry stands for: a user's id, a group's or a channel's marked
    id; 0 for what has none (the account's own «Saved Messages»)."""
    try:
        return utils.get_peer_id(peer)
    except TypeError:
        return 0


def _folder(folders: list[Any], title: str) -> Any:
    """The account's own folder with this title (`DialogFilter`), or None. A shared folder
    (`DialogFilterChatlist`) and «All chats» are never it."""
    for folder in folders:
        if isinstance(folder, types.DialogFilter) and folder.title.text == title:
            return folder
    return None


def _username(target: str) -> str:
    """The username a target names: "name", "@name" or a t.me link to it."""
    name, is_invite = utils.parse_username(target)
    if not name or is_invite:
        raise TgActionError(TgErrorKind.OTHER, "not a username")
    return name


def _peer_info(user: Any) -> PeerInfo:
    names = (user.first_name, user.last_name)
    return PeerInfo(
        peer_id=user.id,
        username=user.username,
        display_name=" ".join(name for name in names if name),
        is_bot=bool(user.bot),
    )


def _usernames(user: Any) -> set[str]:
    """Every username a user answers to, lower-cased: his main one and his active extra ones."""
    names = [user.username] if user.username else []
    names += [
        extra.username for extra in user.usernames or () if extra.active is not False
    ]
    return {name.lower() for name in names}


# -- the gateway --

IncomingHandler = Callable[[Incoming], Awaitable[None]]
EditHandler = Callable[[int, int, str], Awaitable[None]]
OwnerHandler = Callable[[int, str, int], Awaitable[None]]  # peer, text, Telegram id
DeleteHandler = Callable[[list[int]], Awaitable[None]]  # the deleted Telegram ids


class TelethonGateway:
    """Implements `Gateway` on a Telethon client.

    `client` is for tests (and any other way of building one); without it `start()` builds the
    real client from the session string. The session string is never logged or printed.
    """

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        session: str,
        on_incoming: IncomingHandler,
        on_edit: EditHandler,
        on_owner_manual: OwnerHandler,
        *,
        on_delete: DeleteHandler | None = None,
        client: Any | None = None,
    ) -> None:
        self._api_id = api_id
        self._api_hash = api_hash
        self._session = session
        self._on_incoming = on_incoming
        self._on_edit = on_edit
        self._on_owner_manual = on_owner_manual
        self._on_delete = on_delete
        self._client = client
        self._started = False
        self._allowed: set[int] = set()
        self._sent_ids: deque[int] = deque(maxlen=SENT_IDS_KEPT)
        self._handlers = (
            (self._handle_incoming, events.NewMessage(incoming=True)),
            (self._handle_own, events.NewMessage(outgoing=True)),
            (self._handle_edit, events.MessageEdited(incoming=True)),
            (self._handle_delete, events.MessageDeleted()),
        )

    # -- start and stop --

    async def start(self) -> None:
        """Connect, check the login and start listening.

        Raises `SessionNotAuthorized` when the session is not logged in; nothing stays
        connected or listening then, nor after any other failure to start. With no network the
        first connection is retried like every other (see `CONNECTION_RETRIES`): `start()`
        waits for the network instead of failing, and can be cancelled.
        """
        if self._started:
            return
        if self._client is None:
            self._client = TelegramClient(
                StringSession(self._session),
                self._api_id,
                self._api_hash,
                flood_sleep_threshold=FLOOD_SLEEP_THRESHOLD,
                connection_retries=CONNECTION_RETRIES,
                retry_delay=RETRY_DELAY,
                auto_reconnect=True,
                # Each update in its own task: a handler waits for its chat's lock, which a
                # wake may hold while the model thinks, and one at a time every other chat
                # (the owner's own messages too) would wait behind it. The core's
                # per-chat lock keeps one chat's messages in turn.
                sequential_updates=False,
            )
        client = self._client
        # Listening starts before the connection: Telegram may send at once
        for callback, builder in self._handlers:
            client.add_event_handler(callback, builder)
        try:
            await client.connect()
            if not await client.is_user_authorized():
                raise SessionNotAuthorized(_LOGIN_HINT)
        except BaseException:  # a cancelled start must not leave handlers behind either
            await self._release(client)
            raise
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        await self._release(self._client)

    async def wait_disconnected(self) -> None:
        """Wait until the connection is gone for good, for the app to react to.

        Returns when `stop()` ended it, or at once if nothing is started; raises what ended it
        otherwise (Telethon's `disconnected`). With `CONNECTION_RETRIES` forever, a lost link
        does not end it: only an error Telethon cannot get past does, such as the session
        being revoked.
        """
        if not self._started or self._client is None:
            return
        await self._client.disconnected

    async def _release(self, client: Any) -> None:
        for callback, _ in self._handlers:
            client.remove_event_handler(callback)
        await client.disconnect()

    def _connected(self) -> Any:
        if not self._started:
            raise NotStarted("The Telegram gateway is not started")
        return self._client

    def _for_sending(self) -> Any:
        """The client, if a message can be handed to it at all.

        A send that stops here has asked Telegram nothing: it is refused, not in doubt.
        """
        try:
            client = self._connected()
        except NotStarted:
            raise TgActionError(TgErrorKind.OTHER, "not started") from None
        if not client.is_connected():
            raise TgActionError(TgErrorKind.OTHER, "not connected")
        return client

    # -- Gateway --

    @_action
    async def resolve(self, target: str) -> PeerInfo:
        client = self._connected()
        phone = normalize_phone(target) if target.strip().startswith("+") else None
        if phone is not None:
            # A phone the vendor search found. Telegram answers only for a person whose
            # privacy lets strangers find him by it; anyone else is PHONE_NOT_OCCUPIED.
            found = await client(
                functions.contacts.ResolvePhoneRequest(phone=phone[1:])
            )
            user_id = getattr(found.peer, "user_id", None)
            user = next((u for u in found.users if u.id == user_id), None)
            if user_id is None or user is None:
                raise TgActionError(TgErrorKind.OTHER, "not a user")
            return _peer_info(user)
        username = _username(target)
        try:
            # The session's cache first: asking Telegram for a name is what it limits. The
            # entity is then fetched by what the cache gave, not by the name once more.
            peer = await client.get_input_entity(username)
            entity = await client.get_entity(peer)
        except ValueError:  # Telethon: "no user has this username"
            raise TgActionError(TgErrorKind.OTHER, "no such user") from None
        if not isinstance(entity, types.User):
            raise TgActionError(TgErrorKind.OTHER, "not a user")
        # The cache also finds a user by his display name, so the one that came back is only
        # the one asked for if the name asked for is really one of his usernames
        if username not in _usernames(entity):
            raise TgActionError(TgErrorKind.OTHER, "not found")
        return _peer_info(entity)

    @_action
    async def search_public(self, query: str, limit: int) -> list[PublicPeer]:
        """Telegram's global search, for the vendor search: people, channels and groups with a
        username (one without has no way to be pointed to). A channel's or group's description
        is read too, for the first few: contacts for orders are often there. Reads only."""
        client = self._connected()
        found = await client(functions.contacts.SearchRequest(q=query, limit=limit))
        users = {user.id: user for user in found.users}
        chats = {chat.id: chat for chat in found.chats}
        peers: list[PublicPeer] = []
        abouts = 0
        for result in found.results:
            if isinstance(result, types.PeerUser):
                user = users.get(result.user_id)
                if user is None or user.bot or not user.username:
                    continue
                info = _peer_info(user)
                peers.append(PublicPeer(user.username, info.display_name, "user"))
            elif isinstance(result, types.PeerChannel):
                chat = chats.get(result.channel_id)
                if not isinstance(chat, types.Channel) or not chat.username:
                    continue
                about = ""
                if abouts < _ABOUTS_PER_SEARCH:
                    abouts += 1
                    full = await client(
                        functions.channels.GetFullChannelRequest(
                            types.InputChannel(chat.id, chat.access_hash or 0)
                        )
                    )
                    about = getattr(full.full_chat, "about", "") or ""
                kind = "group" if chat.megagroup else "channel"
                peers.append(PublicPeer(chat.username, chat.title, kind, about))
        return peers

    @_action
    async def search_contacts(self, query: str, limit: int) -> list[PeerInfo]:
        """The account's address book, read in one request and matched by name here: the
        book stays in Telegram and nothing of it is kept (ruling R60). Not
        `contacts.Search`, which asks the global search as well. A bot, a deleted account
        and, with `hash=0`, nothing else is left out."""
        client = self._connected()
        book = await client(functions.contacts.GetContactsRequest(hash=0))
        found: list[PeerInfo] = []
        for user in getattr(book, "users", None) or []:  # ContactsNotModified: none
            if not isinstance(user, types.User) or user.bot or user.deleted:
                continue
            info = _peer_info(user)
            if not name_matches(query, info.display_name, user.username or ""):
                continue
            phone = f"+{user.phone}" if user.phone else None
            found.append(dataclasses.replace(info, phone=phone))
            if len(found) >= limit:
                break
        return found

    @_action
    async def own_name(self) -> str:
        me = await self._connected().get_me()
        return " ".join(name for name in (me.first_name, me.last_name) if name)

    @_action
    async def own_username(self) -> str | None:
        me = await self._connected().get_me()
        return getattr(me, "username", None) or None

    @_action
    async def send_text(self, peer_id: int, text: str) -> int:
        client = self._for_sending()
        peer = await self._peer(peer_id)
        # parse_mode=None: the text goes out as written; Telethon would read Markdown in it
        message = await _put_in_chat(client.send_message(peer, text, parse_mode=None))
        return self._remember(message.id)

    @_action
    async def send_file(self, peer_id: int, path: str) -> int:
        client = self._for_sending()
        file = Path(path)
        if not file.is_file():
            raise TgActionError(TgErrorKind.OTHER, "file not found")
        peer = await self._peer(peer_id)
        # A Path, not the str: Telethon takes a str that starts with http for a link to fetch
        message = await _put_in_chat(client.send_file(peer, file))
        return self._remember(message.id)

    @_action
    async def typing(self, peer_id: int, seconds: float) -> None:
        client = self._connected()
        peer = await self._peer(peer_id)
        # Telethon repeats the "typing" request while the block is open; an error of that
        # request is raised when the block ends
        async with client.action(peer, "typing"):
            await asyncio.sleep(max(seconds, 0))

    @_action
    async def mark_read(self, peer_id: int, max_id: int) -> None:
        client = self._connected()
        await client.send_read_acknowledge(await self._peer(peer_id), max_id=max_id)

    @_action
    async def download(self, peer_id: int, tg_msg_id: int) -> bytes:
        client = self._connected()
        peer = await self._peer(peer_id)
        message = await client.get_messages(peer, ids=tg_msg_id)
        media = getattr(message, "media", None)
        # A link preview is not media; Telethon would download its picture all the same
        if media is None or isinstance(media, types.MessageMediaWebPage):
            raise TgActionError(TgErrorKind.OTHER, "no media in the message")
        data = await client.download_media(message, file=bytes)
        if not data:
            raise TgActionError(TgErrorKind.OTHER, "no media in the message")
        return data

    @_action
    async def replay(
        self, peer_id: int, after_msg_id: int, known_out_ids: set[int]
    ) -> int:
        client = self._connected()
        # No anchor, nothing to catch up on: Telethon would take min_id=0 for "from the very
        # first message", and a chat's whole old history would come as news
        if peer_id not in self._allowed or after_msg_id < 1:
            return 0
        peer = await self._peer(peer_id)
        dispatched = 0
        async for message in client.iter_messages(
            peer, limit=REPLAY_LIMIT, min_id=after_msg_id, reverse=True
        ):
            # a service message (a call, a cleared history) is no news, live or not
            if isinstance(message, types.Message) and await self._replayed(
                message, peer_id, known_out_ids
            ):
                dispatched += 1
        return dispatched

    @_action
    async def recent_messages(
        self, peer_id: int, after_msg_id: int, limit: int
    ) -> list[ChatMessage]:
        # No allowed-chat check: the caller names a chat of the database, and a person the
        # agent has finished with (no longer listened to) must still be readable.
        if limit < 1:
            return []
        client = self._connected()
        peer = await self._peer(peer_id)
        newest_first = [
            message
            async for message in client.iter_messages(
                peer, limit=limit, min_id=max(after_msg_id, 0)
            )
            # a service message (a call, a cleared history) is not a message of the chat
            if isinstance(message, types.Message)
        ]
        return [
            ChatMessage(
                tg_msg_id=message.id,
                out=bool(message.out),
                text=getattr(message, "message", None) or "",
                date=_aware(message.date),
            )
            for message in reversed(newest_first)
        ]

    @_action
    async def is_muted(self, peer_id: int) -> bool:
        client = self._connected()
        peer = await self._peer(peer_id)
        settings = await client(
            functions.account.GetNotifySettingsRequest(peer=types.InputNotifyPeer(peer))
        )
        until = getattr(settings, "mute_until", None)
        return until is not None and _aware(until) > datetime.now(UTC)

    @_action
    async def set_muted(self, peer_id: int, muted: bool) -> None:
        client = self._connected()
        peer = await self._peer(peer_id)
        settings = types.InputPeerNotifySettings(mute_until=MUTED_FOR_GOOD if muted else 0)
        await client(
            functions.account.UpdateNotifySettingsRequest(
                peer=types.InputNotifyPeer(peer), settings=settings
            )
        )

    @_action
    async def folder_peers(self, title: str) -> list[int] | None:
        found = _folder(await self._folders(), title)
        if found is None:
            return None
        return [_peer_key(peer) for peer in [*found.pinned_peers, *found.include_peers]]

    @_action
    async def add_to_folder(
        self, title: str, peer_id: int, remove: Collection[int] = ()
    ) -> None:
        client = self._connected()
        peer = await self._peer(peer_id)
        folders = await self._folders()
        found = _folder(folders, title)
        if found is None:
            taken = {getattr(folder, "id", None) for folder in folders}
            free = next(i for i in range(FIRST_FOLDER_ID, 256) if i not in taken)
            found = types.DialogFilter(
                id=free,
                title=types.TextWithEntities(text=title, entities=[]),
                pinned_peers=[],
                include_peers=[],
                exclude_peers=[],
            )
        found.include_peers = [p for p in found.include_peers if _peer_key(p) not in remove]
        if peer_id not in {_peer_key(p) for p in found.pinned_peers + found.include_peers}:
            found.include_peers.append(peer)
        await client(functions.messages.UpdateDialogFilterRequest(id=found.id, filter=found))

    async def _folders(self) -> list[Any]:
        """The account's chat folders (`messages.DialogFilters`, a bare list before layer
        160 or so)."""
        result = await self._connected()(functions.messages.GetDialogFiltersRequest())
        return list(getattr(result, "filters", result))

    async def _replayed(
        self, message: Any, peer_id: int, known_out_ids: set[int]
    ) -> bool:
        """Report one old message the way a live one is; False if it is skipped or fails."""
        try:
            if message.out:
                return await self._report_own(message, peer_id, known_out_ids)
            await self._report_incoming(message)
            return True
        except Exception as error:  # noqa: BLE001 - the others must still be handed over
            _log_failure("replay", error)
            return False

    async def set_allowed(self, peer_ids: set[int]) -> None:
        self._allowed = set(peer_ids)

    # -- helpers --

    def _remember(self, msg_id: int) -> int:
        """Write down the id of a message we sent, before the caller can see it."""
        self._sent_ids.append(msg_id)
        return msg_id

    async def _peer(self, peer_id: int) -> Any:
        """The client's handle on a user, which Telegram needs besides his id.

        He is looked up as a user (`PeerUser`), never as a bare number: Telethon reads a bare
        id as a user, a chat or a channel, whichever it finds, and its in-memory cache is keyed
        by the number alone, so what comes back is checked to be a user as well. Anything else
        is not used: the message would go to the wrong chat.

        A session string carries no entities, so after a restart the client knows nobody until
        he writes. Reading the dialogs (newest first, as far as needed) teaches it who is who.
        """
        client = self._connected()
        try:
            peer = await client.get_input_entity(types.PeerUser(peer_id))
        except ValueError:
            peer = None
        if isinstance(peer, types.InputPeerUser):
            return peer
        async for dialog in client.iter_dialogs():
            if dialog.id == peer_id and isinstance(
                dialog.input_entity, types.InputPeerUser
            ):
                return dialog.input_entity
        raise TgActionError(TgErrorKind.OTHER, "unknown peer")

    # -- events (called by Telethon) --

    @_contained
    async def _handle_incoming(self, event: Any) -> None:
        if is_relevant(event, self._allowed):
            await self._report_incoming(event.message)

    @_contained
    async def _handle_own(self, event: Any) -> None:
        if is_relevant(event, self._allowed):
            await self._report_own(event.message, event.chat_id, ())

    async def _report_incoming(self, message: Any) -> None:
        await self._on_incoming(to_incoming(message))

    async def _report_own(
        self, message: Any, peer_id: int, known_out_ids: Collection[int]
    ) -> bool:
        """An outgoing message: the owner's own words, unless we sent it ourselves.

        True if it was reported. Live events and `replay` come through here alike, so the
        owner is told apart from the agent in one place.
        """
        if message.id in self._sent_ids or message.id in known_out_ids:
            return False
        await self._on_owner_manual(peer_id, message.message or "", message.id)
        return True

    @_contained
    async def _handle_edit(self, event: Any) -> None:
        if is_relevant(event, self._allowed):
            message = event.message
            await self._on_edit(event.chat_id, message.id, message.message or "")

    @_contained
    async def _handle_delete(self, event: Any) -> None:
        """2.24: messages were deleted. For a private chat Telegram does not say which chat,
        so the ids go to the core, which looks them up among the messages it keeps; ids of
        another chat that it can tell are dropped here."""
        if self._on_delete is None:
            return
        chat_id = getattr(event, "chat_id", None)
        if chat_id is not None and chat_id not in self._allowed:
            return
        await self._on_delete(list(getattr(event, "deleted_ids", []) or []))
