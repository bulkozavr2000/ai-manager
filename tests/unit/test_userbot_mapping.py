"""The Telethon userbot without a network: how a message, an error and a chat are told apart.

Telethon objects are real wherever the mapping reads them (messages, events, errors), so these
tests also pin down what Telethon 1.45 really does. Stubs stand in only for what needs a server:
the client.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import islice
from types import SimpleNamespace
from typing import Self

import pytest
from telethon import errors, events, types
from telethon.crypto import AuthKey
from telethon.helpers import retry_range
from telethon.sessions import StringSession

from tg_agent.gateway import userbot
from tg_agent.gateway.base import (
    ChatMessage,
    Gateway,
    Incoming,
    PeerInfo,
    TgActionError,
    TgSendUncertain,
)
from tg_agent.policy.tg_safety import TgErrorKind

NOW = datetime(2026, 11, 5, 10, 0, tzinfo=UTC)
PEER = 42  # the counterpart's Telegram id


# -- Telethon objects as the server would send them --


def _document(*attributes: object, mime: str = "application/octet-stream"):
    return types.Document(
        id=1,
        access_hash=2,
        file_reference=b"",
        date=NOW,
        mime_type=mime,
        size=10,
        dc_id=2,
        attributes=list(attributes),
    )


def _photo():
    sizes = [types.PhotoSize(type="x", w=10, h=10, size=5)]
    return types.Photo(
        id=1, access_hash=2, file_reference=b"", date=NOW, sizes=sizes, dc_id=2
    )


def _message(
    media=None, text: str = "", *, peer: int = PEER, msg_id: int = 7, **fields
):
    return types.Message(
        id=msg_id,
        peer_id=types.PeerUser(peer),
        date=NOW,
        message=text,
        media=media,
        **fields,
    )


def _sticker_attribute():
    return types.DocumentAttributeSticker(
        alt="x", stickerset=types.InputStickerSetEmpty()
    )


def _preview(*, photo=None, document=None):
    page = types.WebPage(
        id=1, url="u", display_url="u", hash=0, photo=photo, document=document
    )
    return types.MessageMediaWebPage(webpage=page)


# -- message_kind --


def stub(**attributes: object) -> SimpleNamespace:
    """A message as `message_kind` reads it: no media and no text unless told otherwise."""
    fields = {
        "voice": None,
        "video": None,
        "video_note": None,
        "sticker": None,
        "photo": None,
        "document": None,
        "media": None,
        "message": "",
    }
    return SimpleNamespace(**{**fields, **attributes})


@pytest.mark.parametrize(
    ("attributes", "kind"),
    [
        # Telethon also calls a voice note a document, and a video note a video
        ({"voice": object(), "document": object()}, "voice"),
        ({"video": object(), "document": object()}, "video"),
        ({"video_note": object(), "video": object(), "document": object()}, "video"),
        ({"photo": object()}, "photo"),
        # a video sticker is a video too, and must still be told as a sticker
        ({"sticker": object(), "video": object(), "document": object()}, "sticker"),
        ({"document": object()}, "document"),
        ({"message": "Здравствуйте"}, "text"),
        # media this program does not know
        ({"media": object()}, "other"),
        ({}, "other"),
    ],
)
def test_message_kind_mapping(attributes: dict, kind: str) -> None:
    assert userbot.message_kind(stub(**attributes)) == kind


def test_a_link_preview_is_not_an_attachment() -> None:
    # Message.photo, .video and .document also return what a link preview shows
    preview = types.MessageMediaWebPage(webpage=types.WebPageEmpty(id=1))
    message = stub(media=preview, photo=object(), message="https://example.uz/catalog")
    assert userbot.message_kind(message) == "text"


@pytest.mark.parametrize(
    ("build", "kind"),
    [
        pytest.param(lambda: _message(text="Здравствуйте"), "text", id="text"),
        pytest.param(
            lambda: _message(
                types.MessageMediaDocument(
                    document=_document(
                        types.DocumentAttributeAudio(duration=3, voice=True),
                        mime="audio/ogg",
                    )
                )
            ),
            "voice",
            id="voice",
        ),
        pytest.param(
            lambda: _message(
                types.MessageMediaDocument(
                    document=_document(
                        types.DocumentAttributeAudio(duration=3),
                        types.DocumentAttributeFilename("song.mp3"),
                        mime="audio/mpeg",
                    )
                )
            ),
            "document",
            id="music-file",
        ),
        pytest.param(
            lambda: _message(
                types.MessageMediaDocument(
                    document=_document(
                        types.DocumentAttributeVideo(duration=3, w=10, h=10),
                        mime="video/mp4",
                    )
                )
            ),
            "video",
            id="video",
        ),
        pytest.param(
            lambda: _message(
                types.MessageMediaDocument(
                    document=_document(
                        types.DocumentAttributeVideo(
                            duration=3, w=10, h=10, round_message=True
                        ),
                        mime="video/mp4",
                    )
                )
            ),
            "video",
            id="round-video",
        ),
        pytest.param(
            lambda: _message(
                types.MessageMediaDocument(
                    document=_document(
                        types.DocumentAttributeVideo(duration=3, w=10, h=10),
                        _sticker_attribute(),
                        mime="video/webm",
                    )
                )
            ),
            "sticker",
            id="video-sticker",
        ),
        pytest.param(
            lambda: _message(
                types.MessageMediaDocument(
                    document=_document(_sticker_attribute(), mime="image/webp")
                )
            ),
            "sticker",
            id="sticker",
        ),
        pytest.param(
            lambda: _message(types.MessageMediaPhoto(photo=_photo()), "Вот фото"),
            "photo",
            id="photo-with-caption",
        ),
        pytest.param(
            lambda: _message(
                types.MessageMediaDocument(
                    document=_document(types.DocumentAttributeFilename("part.stl"))
                )
            ),
            "document",
            id="document",
        ),
        pytest.param(
            lambda: _message(types.MessageMediaPoll(poll=None, results=None)),
            "poll",
            id="poll",
        ),
        pytest.param(lambda: _message(_GEO), "location", id="location"),
        pytest.param(lambda: _message(_CONTACT), "contact", id="contact"),
        pytest.param(
            lambda: _message(_preview(photo=_photo()), "Смотрите https://example.uz"),
            "text",
            id="link-with-a-picture",
        ),
        pytest.param(
            lambda: _message(
                _preview(
                    document=_document(
                        types.DocumentAttributeVideo(duration=3, w=1, h=1)
                    )
                ),
                "Смотрите https://example.uz",
            ),
            "text",
            id="link-with-a-video",
        ),
    ],
)
def test_message_kind_on_real_telethon_messages(build, kind: str) -> None:
    assert userbot.message_kind(build()) == kind


_POINT = types.GeoPoint(long=69.24, lat=41.31, access_hash=0)
_GEO = types.MessageMediaGeo(geo=_POINT)
_CONTACT = types.MessageMediaContact(
    phone_number="+998901112233",
    first_name="Иван",
    last_name="Петров",
    vcard="",
    user_id=0,
)


def test_to_incoming_spells_out_a_location_a_place_and_a_contact() -> None:
    # 2.22: these used to reach the engine as an empty "other": the address or the phone was lost.
    venue = types.MessageMediaVenue(
        geo=_POINT,
        title="3D Lab",
        address="ул. Навои, 5",
        provider="",
        venue_id="",
        venue_type="",
    )
    spelled = [
        (incoming.kind, incoming.text)
        for incoming in (
            userbot.to_incoming(_message(_GEO)),
            userbot.to_incoming(_message(venue)),
            userbot.to_incoming(_message(_CONTACT)),
        )
    ]
    assert spelled == [
        ("location", "41.31000, 69.24000"),
        ("location", "3D Lab, ул. Навои, 5 (41.31000, 69.24000)"),
        ("contact", "Иван Петров +998901112233"),
    ]


# -- to_incoming --


def _incoming_stub(**attributes: object) -> SimpleNamespace:
    fields = {
        "chat_id": PEER,
        "id": 7,
        "message": "Здравствуйте",
        "date": NOW,
        "file": None,
    }
    return stub(**{**fields, **attributes})


def test_to_incoming_of_a_text_message() -> None:
    assert userbot.to_incoming(_incoming_stub()) == Incoming(
        peer_id=PEER,
        tg_msg_id=7,
        text="Здравствуйте",
        kind="text",
        date=NOW,
        offline=False,
        file_name=None,
    )


def test_to_incoming_takes_the_caption_of_a_media_message() -> None:
    incoming = userbot.to_incoming(_incoming_stub(photo=object(), message="Вот фото"))
    assert (incoming.kind, incoming.text) == ("photo", "Вот фото")
    incoming = userbot.to_incoming(_incoming_stub(voice=object(), message=None))
    assert (incoming.kind, incoming.text) == ("voice", "")


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        ({"offline": True}, True),
        ({"offline": False}, False),
        (
            {"offline": None},
            False,
        ),  # Telethon leaves it None on a message it built itself
        ({}, False),  # no such attribute
    ],
)
def test_to_incoming_offline_flag(attributes: dict, expected: bool) -> None:
    assert userbot.to_incoming(_incoming_stub(**attributes)).offline is expected


def test_to_incoming_names_a_document_and_only_a_document() -> None:
    named = SimpleNamespace(name="part.stl")
    document = userbot.to_incoming(_incoming_stub(document=object(), file=named))
    assert (document.kind, document.file_name) == ("document", "part.stl")
    nameless = userbot.to_incoming(
        _incoming_stub(document=object(), file=SimpleNamespace(name=None))
    )
    assert nameless.file_name is None
    photo = userbot.to_incoming(_incoming_stub(photo=object(), file=named))
    assert photo.file_name is None


def test_to_incoming_dates_are_timezone_aware() -> None:
    naive = datetime(2026, 11, 5, 10, 0)
    assert userbot.to_incoming(_incoming_stub(date=naive)).date == NOW


def test_to_incoming_of_a_real_telethon_message() -> None:
    message = _message(
        types.MessageMediaDocument(
            document=_document(types.DocumentAttributeFilename("part.stl"))
        ),
        "Файл готов",
        msg_id=91,
        peer=555,
    )
    assert userbot.to_incoming(message) == Incoming(
        peer_id=555,
        tg_msg_id=91,
        text="Файл готов",
        kind="document",
        date=NOW,
        offline=False,
        file_name="part.stl",
    )


# -- error_to_action --


def _rpc_error(code: str, http: int):
    """The error exactly as Telethon's network layer builds it from Telegram's answer."""
    answer = types.RpcError(error_code=http, error_message=code)
    return errors.rpc_message_to_error(answer, request=None)


def test_error_mapping_peer_flood_and_payment() -> None:
    # PEER_FLOOD is the alarm that stops the whole agent. Telethon has a class for it, and
    # that class does not keep the code in `.message` (it says BAD_REQUEST), so the code has
    # to come from the class.
    flood = _rpc_error("PEER_FLOOD", 400)
    assert flood.message == "BAD_REQUEST"
    action = userbot.error_to_action(flood)
    assert (action.kind, action.detail, action.wait_s) == (
        TgErrorKind.PEER_FLOOD,
        "PEER_FLOOD",
        None,
    )
    # Telethon 1.45 has no class for the payment errors: the code is in `.message`
    for code in (
        "ALLOW_PAYMENT_REQUIRED_100",
        "ALLOW_PAYMENT_REQUIRED",
        "BALANCE_TOO_LOW",
    ):
        action = userbot.error_to_action(_rpc_error(code, 403))
        assert (action.kind, action.detail) == (TgErrorKind.PAYMENT_REQUIRED, code)


@pytest.mark.parametrize(
    ("code", "http", "kind"),
    [
        ("PEER_FLOOD", 400, TgErrorKind.PEER_FLOOD),
        ("FLOOD_WAIT_120", 420, TgErrorKind.FLOOD_WAIT),
        ("FLOOD_PREMIUM_WAIT_30", 420, TgErrorKind.FLOOD_WAIT),
        ("FROZEN_METHOD_INVALID", 420, TgErrorKind.FROZEN),
        ("FROZEN_PARTICIPANT_MISSING", 400, TgErrorKind.FROZEN),
        ("PRIVACY_PREMIUM_REQUIRED", 403, TgErrorKind.PREMIUM_REQUIRED),
        ("ALLOW_PAYMENT_REQUIRED_100", 403, TgErrorKind.PAYMENT_REQUIRED),
        ("BALANCE_TOO_LOW", 400, TgErrorKind.PAYMENT_REQUIRED),
        ("USER_IS_BLOCKED", 400, TgErrorKind.BLOCKED),
        ("YOU_BLOCKED_USER", 400, TgErrorKind.BLOCKED),
        ("INPUT_USER_DEACTIVATED", 400, TgErrorKind.BLOCKED),
        ("USER_PRIVACY_RESTRICTED", 403, TgErrorKind.PRIVACY),
        ("AUTH_KEY_UNREGISTERED", 401, TgErrorKind.DEAD_SESSION),
        ("AUTH_KEY_DUPLICATED", 406, TgErrorKind.DEAD_SESSION),
        ("SESSION_REVOKED", 401, TgErrorKind.DEAD_SESSION),
        ("USER_DEACTIVATED_BAN", 401, TgErrorKind.DEAD_SESSION),
        ("PHONE_NUMBER_BANNED", 400, TgErrorKind.DEAD_SESSION),
        ("MESSAGE_TOO_LONG", 400, TgErrorKind.OTHER),
        ("SOMETHING_TELEGRAM_ADDS_NEXT_YEAR", 400, TgErrorKind.OTHER),
    ],
)
def test_every_policy_code_is_recognised_on_the_errors_telethon_raises(
    code: str, http: int, kind: TgErrorKind
) -> None:
    assert userbot.error_to_action(_rpc_error(code, http)).kind is kind


@pytest.mark.parametrize(
    ("code", "seconds"),
    [("FLOOD_WAIT_120", 120), ("FLOOD_PREMIUM_WAIT_30", 30)],
)
def test_flood_wait_carries_how_long_to_wait(code: str, seconds: int) -> None:
    action = userbot.error_to_action(_rpc_error(code, 420))
    assert (action.kind, action.wait_s) == (TgErrorKind.FLOOD_WAIT, seconds)


def test_other_errors_do_not_ask_for_a_wait() -> None:
    assert userbot.error_to_action(_rpc_error("USER_IS_BLOCKED", 400)).wait_s is None


def test_a_non_telegram_error_is_other_and_named_by_its_type() -> None:
    action = userbot.error_to_action(ConnectionError("Привет, цена 500 сум"))
    assert (action.kind, action.detail, action.wait_s) == (
        TgErrorKind.OTHER,
        "ConnectionError",
        None,
    )


def test_error_detail_never_carries_text() -> None:
    # What a counterpart wrote may sit in an error's text; the detail is shown to the owner
    # and written to the outbox, so it is only ever a Telegram code or a class name.
    secret = "Привет, это личное сообщение"
    for error in (
        ValueError(secret),
        OSError(secret),
        errors.RPCError(request=None, message=secret, code=400),
        SimpleNamespace(message=secret),
    ):
        action = userbot.error_to_action(error)
        assert secret not in action.detail and secret not in str(action)
        assert action.kind is TgErrorKind.OTHER
    unknown = errors.RPCError(request=None, message="SOME_CODE", code=400)
    assert userbot.error_to_action(unknown).detail == "SOME_CODE"


def test_an_object_that_only_carries_a_code_in_message_is_classified() -> None:
    action = userbot.error_to_action(SimpleNamespace(message="FLOOD_WAIT_45"))
    assert (action.kind, action.wait_s) == (TgErrorKind.FLOOD_WAIT, 45)

    class Limited(Exception):
        message = "PEER_FLOOD"

    assert userbot.error_to_action(Limited()).kind is TgErrorKind.PEER_FLOOD


def test_a_gateway_error_stays_as_it_is() -> None:
    error = TgActionError(TgErrorKind.PRIVACY, "USER_PRIVACY_RESTRICTED")
    assert userbot.error_to_action(error) is error


# -- is_relevant --


def _event(chat_id: int | None, *, private: bool | None = True) -> SimpleNamespace:
    return SimpleNamespace(is_private=private, chat_id=chat_id)


def test_allowed_filter_blocks_other_chats() -> None:
    allowed = {10, 20}
    assert userbot.is_relevant(_event(10), allowed) is True
    assert userbot.is_relevant(_event(20), allowed) is True
    # a private chat nobody has opened a conversation in (a stranger, a bot, Saved Messages)
    assert userbot.is_relevant(_event(30), allowed) is False
    assert userbot.is_relevant(_event(10), set()) is False
    # a group or a channel is never read, even with its id in the list
    assert userbot.is_relevant(_event(10, private=False), allowed) is False
    assert userbot.is_relevant(_event(-5, private=False), {-5}) is False
    # not enough information: not relevant
    assert userbot.is_relevant(_event(None, private=None), allowed) is False


def test_is_relevant_on_real_telethon_events() -> None:
    allowed = {PEER, -5}
    for peer, expected in (
        (types.PeerUser(PEER), True),
        (types.PeerUser(PEER + 1), False),
        (types.PeerChat(5), False),
        (types.PeerChannel(5), False),
    ):
        message = types.Message(id=1, peer_id=peer, date=NOW, message="hi")
        assert (
            userbot.is_relevant(events.NewMessage.Event(message), allowed) is expected
        )
        assert (
            userbot.is_relevant(events.MessageEdited.Event(message), allowed)
            is expected
        )


# -- the gateway, on a stub client --

INPUT = types.InputPeerUser(PEER, 7)  # what the client hands back for PEER


class _Typing:
    """What `client.action(peer, "typing")` returns: "typing…" is shown while it is open."""

    def __init__(self, client: StubClient) -> None:
        self.client = client

    async def __aenter__(self) -> Self:
        self.client.record("typing_on")
        return self

    async def __aexit__(self, *exc: object) -> None:
        # Telethon re-raises here what its background "typing" request raised
        self.client.record("typing_off")


class StubClient:
    """The part of TelegramClient the gateway uses.

    It records every call as (name, args, kwargs) and raises what `failures` holds for a name
    (once). Telegram is not reachable from a test.
    """

    def __init__(self) -> None:
        self.handlers: list[tuple[Callable, object]] = []
        self.calls: list[tuple[str, tuple, dict]] = []
        self.failures: dict[str, Exception] = {}
        self.authorized = True
        self.input_peers: dict[object, object] = {PEER: INPUT}
        # entities found by a bare id, which Telethon reads as a user, a chat or a channel
        self.bare_ids: dict[int, object] = {}
        self.entities: dict[object, object] = {}
        self.dialogs: list[SimpleNamespace] = []
        self.messages: dict[int, object] = {}
        self.history: list[object] = []  # the messages of the chat, for iter_messages
        self.media = b""
        self.last_id = 100
        self.linked = True  # what the client's own is_connected() says
        self.transport_up = True  # what Telethon's sender says of the connection itself
        self._sender = SimpleNamespace(_transport_connected=lambda: self.transport_up)
        self.disconnected = None  # the future Telethon resolves when the link is gone
        self.sends_nothing = False  # a send that comes back with no message
        # Raw requests (client(SomeRequest(...))): the answer, by the request's class name.
        self.answers: dict[str, object] = {}

    def record(self, name: str, *args: object, **kwargs: object) -> None:
        self.calls.append((name, args, kwargs))
        failure = self.failures.pop(name, None)
        if failure is not None:
            raise failure

    def names(self) -> list[str]:
        return [name for name, _, _ in self.calls]

    def last(self, name: str) -> tuple[tuple, dict]:
        for called, args, kwargs in reversed(self.calls):
            if called == name:
                return args, kwargs
        raise AssertionError(f"{name} was never called")

    def _sent(self) -> SimpleNamespace | None:
        if self.sends_nothing:
            return None
        self.last_id += 1
        return SimpleNamespace(id=self.last_id)

    def is_connected(self) -> bool:
        return self.linked

    async def __call__(self, request: object) -> object:
        name = type(request).__name__
        self.record(name, request)
        answer = self.answers[name]
        return answer(request) if callable(answer) else answer

    def add_event_handler(self, callback: Callable, event: object) -> None:
        self.handlers.append((callback, event))

    def remove_event_handler(self, callback: Callable, event: object = None) -> int:
        kept = [(c, e) for c, e in self.handlers if c != callback]
        removed, self.handlers = len(self.handlers) - len(kept), kept
        return removed

    async def connect(self) -> None:
        self.record("connect")

    async def disconnect(self) -> None:
        self.record("disconnect")

    async def is_user_authorized(self) -> bool:
        self.record("is_user_authorized")
        return self.authorized

    async def get_input_entity(self, peer: object) -> object:
        self.record("get_input_entity", peer)
        if isinstance(
            peer, types.PeerUser
        ):  # exact: a user, never another kind of entity
            found = self.input_peers.get(peer.user_id)
        elif isinstance(peer, int):
            # a bare id is read as a user, a chat or a channel: whichever Telethon finds
            found = self.bare_ids.get(peer) or self.input_peers.get(peer)
        else:  # a username
            found = self.input_peers.get(peer)
        if found is None:  # what Telethon says of an id it has not seen
            raise ValueError("Could not find the input entity")
        return found

    async def get_entity(self, peer: object) -> object:
        self.record("get_entity", peer)
        return self.entities[peer]

    async def get_me(self) -> object:
        self.record("get_me")
        return SimpleNamespace(first_name="ИИ помощник", last_name="Саида")

    async def iter_dialogs(self):
        self.record("iter_dialogs")
        for dialog in self.dialogs:
            yield dialog

    async def iter_messages(self, peer: object, **kwargs: object):
        self.record("iter_messages", peer, **kwargs)
        # what Telethon does with min_id, reverse and limit
        newer = [m for m in self.history if m.id > kwargs.get("min_id", 0)]
        newer.sort(key=lambda m: m.id, reverse=not kwargs.get("reverse", False))
        for message in newer[: kwargs.get("limit")]:
            yield message

    async def send_message(self, peer: object, text: str, **kwargs: object) -> object:
        self.record("send_message", peer, text, **kwargs)
        return self._sent()

    async def send_file(self, peer: object, file: object, **kwargs: object) -> object:
        self.record("send_file", peer, file, **kwargs)
        return self._sent()

    def action(self, peer: object, kind: str) -> _Typing:
        self.record("action", peer, kind)
        return _Typing(self)

    async def send_read_acknowledge(self, peer: object, **kwargs: object) -> None:
        self.record("send_read_acknowledge", peer, **kwargs)

    async def get_messages(self, peer: object, **kwargs: object) -> object:
        self.record("get_messages", peer, **kwargs)
        return self.messages.get(kwargs["ids"])

    async def download_media(self, message: object, **kwargs: object) -> bytes:
        self.record("download_media", message, **kwargs)
        return self.media


class Engine:
    """The three callbacks the gateway reports to, recorded."""

    def __init__(self) -> None:
        self.incoming: list[Incoming] = []
        self.edits: list[tuple[int, int, str]] = []
        self.deleted: list[list[int]] = []
        self.manual: list[tuple[int, str, int]] = []
        # ("in" or "own", the Telegram id of the message) in the order the callbacks ran
        self.order: list[tuple[str, int]] = []

    async def on_incoming(self, msg: Incoming) -> None:
        self.incoming.append(msg)
        self.order.append(("in", msg.tg_msg_id))

    async def on_edit(self, peer_id: int, tg_msg_id: int, text: str) -> None:
        self.edits.append((peer_id, tg_msg_id, text))

    async def on_owner_manual(self, peer_id: int, text: str, tg_msg_id: int) -> None:
        self.manual.append((peer_id, text, tg_msg_id))
        self.order.append(("own", tg_msg_id))

    async def on_delete(self, tg_msg_ids: list[int]) -> None:
        self.deleted.append(list(tg_msg_ids))


def _build(client: StubClient, engine: Engine, session: str = "session-string"):
    return userbot.TelethonGateway(
        12345,
        "api-hash",
        session,
        engine.on_incoming,
        engine.on_edit,
        engine.on_owner_manual,
        on_delete=engine.on_delete,
        client=client,
    )


@pytest.fixture
def client() -> StubClient:
    return StubClient()


@pytest.fixture
def engine() -> Engine:
    return Engine()


@pytest.fixture
async def gateway(client: StubClient, engine: Engine):
    """A started gateway that listens to the chat with PEER."""
    started = _build(client, engine)
    await started.start()
    await started.set_allowed({PEER})
    return started


async def deliver(client: StubClient, event: object) -> None:
    """Hand an event to the handlers Telethon would: those of its kind whose own filter passes."""
    for callback, builder in client.handlers:
        if type(event) is type(builder).Event:
            builder.resolved = True  # what Telethon does before the first filtering
            if builder.filter(event):
                await callback(event)


def incoming_event(
    text: str = "Здравствуйте", *, peer: int = PEER, msg_id: int = 7, edited=False
):
    message = _message(text=text, peer=peer, msg_id=msg_id, out=False)
    return (events.MessageEdited if edited else events.NewMessage).Event(message)


def own_event(text: str, *, msg_id: int, peer: int = PEER, edited=False):
    message = _message(text=text, peer=peer, msg_id=msg_id, out=True)
    return (events.MessageEdited if edited else events.NewMessage).Event(message)


def _session_string() -> str:
    session = StringSession()
    session.set_dc(2, "149.154.167.51", 443)
    session.auth_key = AuthKey(bytes(range(256)))
    return session.save()


def test_the_gateway_has_every_method_of_the_gateway_protocol() -> None:
    wanted = [name for name in vars(Gateway) if not name.startswith("_")]
    assert "send_text" in wanted and "set_allowed" in wanted
    for name in wanted:
        method = getattr(userbot.TelethonGateway, name)
        assert inspect.iscoroutinefunction(method), name


# -- start and stop --


async def test_start_listens_to_incoming_own_and_edited_messages(
    gateway, client: StubClient
) -> None:
    listening = [
        (type(b).__name__, getattr(b, "incoming", None), getattr(b, "outgoing", None))
        for _, b in client.handlers
    ]
    assert listening == [
        ("NewMessage", True, False),
        ("NewMessage", False, True),
        ("MessageEdited", True, False),
        ("MessageDeleted", None, None),  # 2.24: deletions are listened to as well
    ]
    assert client.names() == ["connect", "is_user_authorized"]


async def test_start_builds_the_telethon_client_from_the_session(
    monkeypatch: pytest.MonkeyPatch, client: StubClient, engine: Engine
) -> None:
    string = _session_string()
    real_class = userbot.TelegramClient
    built: list[object] = []

    def factory(*args, **options):
        # the real constructor takes these arguments (it connects to nothing); the stub
        # stands in for it afterwards
        built.append(real_class(*args, **options))
        return client

    monkeypatch.setattr(userbot, "TelegramClient", factory)
    gateway = userbot.TelethonGateway(
        12345,
        "api-hash",
        string,
        engine.on_incoming,
        engine.on_edit,
        engine.on_owner_manual,
    )
    await gateway.start()
    (made,) = built
    assert isinstance(made.session, StringSession)
    assert made.session.save() == string
    assert (made.api_id, made.api_hash) == (12345, "api-hash")
    assert made.flood_sleep_threshold == 60


async def test_start_builds_a_client_that_never_gives_up_reconnecting(
    monkeypatch: pytest.MonkeyPatch, client: StubClient, engine: Engine
) -> None:
    # Telethon's default is five attempts, after which the client stays disconnected and
    # deaf for good; a PC that wakes before its network is up is the normal case here.
    real_class = userbot.TelegramClient
    asked: dict[str, object] = {}
    built: list[object] = []

    def factory(*args, **options):
        asked.update(options)
        built.append(real_class(*args, **options))
        return client

    monkeypatch.setattr(userbot, "TelegramClient", factory)
    gateway = userbot.TelethonGateway(
        12345,
        "api-hash",
        _session_string(),
        engine.on_incoming,
        engine.on_edit,
        engine.on_owner_manual,
    )
    await gateway.start()
    assert (
        asked
        == {
            "flood_sleep_threshold": 60,
            "connection_retries": -1,  # a link that is down is retried for as long as it takes
            "retry_delay": 2,
            "auto_reconnect": True,
            # A decision holds its chat's lock while the model thinks: handled one at a time,
            # every other chat's updates (the owner's own messages too) would wait behind it
            "sequential_updates": False,
        }
    )
    # Telethon keeps them in private attributes: this is what its reconnect loop reads
    (real,) = built
    sender = real._sender
    assert (sender._retries, sender._delay, sender._auto_reconnect) == (-1, 2, True)
    assert real._sequential_updates is False
    assert len(list(islice(retry_range(sender._retries), 1000))) == 1000  # no end


async def test_wait_disconnected_waits_until_telethon_lets_go(
    gateway, client: StubClient
) -> None:
    client.disconnected = asyncio.get_running_loop().create_future()
    waiting = asyncio.ensure_future(gateway.wait_disconnected())
    await asyncio.sleep(0)
    assert not waiting.done()  # the link is up, or being brought back
    client.disconnected.set_result(None)
    assert await waiting is None


async def test_wait_disconnected_raises_what_ended_the_connection(
    gateway, client: StubClient
) -> None:
    client.disconnected = asyncio.get_running_loop().create_future()
    client.disconnected.set_exception(ConnectionError("link lost for good"))
    with pytest.raises(ConnectionError):
        await gateway.wait_disconnected()


async def test_wait_disconnected_returns_at_once_when_nothing_is_started(
    client: StubClient, engine: Engine
) -> None:
    await _build(client, engine).wait_disconnected()  # nothing to wait for


async def test_stop_ends_a_wait_for_the_disconnection(
    gateway, client: StubClient
) -> None:
    client.disconnected = asyncio.get_running_loop().create_future()
    waiting = asyncio.ensure_future(gateway.wait_disconnected())
    await asyncio.sleep(0)
    await gateway.stop()
    client.disconnected.set_result(None)  # what Telethon's disconnect() does
    assert await waiting is None


async def test_start_without_a_valid_login_says_how_to_log_in(
    client: StubClient, engine: Engine
) -> None:
    client.authorized = False
    with pytest.raises(userbot.SessionNotAuthorized) as caught:
        await _build(client, engine, session="the-secret-session").start()
    assert "uv run python -m tg_agent.setup login" in str(caught.value)
    assert "the-secret-session" not in str(caught.value)
    assert "disconnect" in client.names()  # the connection is not left open
    assert client.handlers == []  # and nothing listens


async def test_a_failed_connection_leaves_nothing_listening(
    client: StubClient, engine: Engine
) -> None:
    client.failures["connect"] = OSError("network is down")
    failing = _build(client, engine)
    with pytest.raises(OSError):
        await failing.start()
    assert client.handlers == []
    await failing.start()  # and it can be tried again
    assert len(client.handlers) == 4


async def test_start_twice_listens_once(gateway, client: StubClient) -> None:
    await gateway.start()
    assert len(client.handlers) == 4 and client.names().count("connect") == 1


async def test_stop_stops_listening_and_disconnects(
    gateway, client: StubClient
) -> None:
    await gateway.stop()
    assert client.handlers == [] and client.names()[-1] == "disconnect"
    await gateway.stop()  # a second time does nothing
    assert client.names().count("disconnect") == 1


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda g: g.resolve("@ali_print"), id="resolve"),
        pytest.param(lambda g: g.typing(PEER, 0), id="typing"),
        pytest.param(lambda g: g.mark_read(PEER, 1), id="mark_read"),
        pytest.param(lambda g: g.download(PEER, 1), id="download"),
        pytest.param(lambda g: g.replay(PEER, 1, set()), id="replay"),
        pytest.param(lambda g: g.recent_messages(PEER, 0, 5), id="recent_messages"),
    ],
)
async def test_a_read_on_a_gateway_that_was_not_started_is_not_usable(
    client: StubClient, engine: Engine, call
) -> None:
    with pytest.raises(userbot.NotStarted):
        await call(_build(client, engine))


async def _send(gateway, how: str, tmp_path) -> int:
    """One send, as a text or as a file."""
    if how == "file":
        path = tmp_path / "part.stl"
        path.write_bytes(b"solid")
        return await gateway.send_file(PEER, str(path))
    return await gateway.send_text(PEER, "Привет, как дела")


# What the client's call is named when it is the one that puts the message into the chat
SEND_CALL = {"text": "send_message", "file": "send_file"}


@pytest.mark.parametrize("how", ["text", "file"])
async def test_a_send_on_a_gateway_that_was_not_started_is_refused_not_uncertain(
    client: StubClient, engine: Engine, tmp_path, how: str
) -> None:
    # Nothing was asked of Telegram, so nothing can have left: a refusal, not a doubt
    with pytest.raises(TgActionError) as caught:
        await _send(_build(client, engine), how, tmp_path)
    assert (caught.value.kind, caught.value.detail) == (
        TgErrorKind.OTHER,
        "not started",
    )
    assert client.calls == []


@pytest.mark.parametrize("how", ["text", "file"])
async def test_a_send_with_the_client_disconnected_is_refused_not_uncertain(
    gateway, client: StubClient, tmp_path, how: str
) -> None:
    client.linked = False  # Telethon would only say "cannot send while disconnected"
    with pytest.raises(TgActionError) as caught:
        await _send(gateway, how, tmp_path)
    assert (caught.value.kind, caught.value.detail) == (
        TgErrorKind.OTHER,
        "not connected",
    )
    assert SEND_CALL[how] not in client.names()


# -- what reaches the engine --


async def test_an_incoming_message_reaches_the_engine(
    gateway, client: StubClient, engine: Engine
) -> None:
    await deliver(client, incoming_event("Цена 120 000 сум", msg_id=12))
    assert engine.incoming == [
        Incoming(
            peer_id=PEER, tg_msg_id=12, text="Цена 120 000 сум", kind="text", date=NOW
        )
    ]
    assert engine.edits == [] and engine.manual == []


async def test_an_edit_reaches_the_engine_with_the_new_text(
    gateway, client: StubClient, engine: Engine
) -> None:
    await deliver(client, incoming_event("Цена 130 000 сум", msg_id=12, edited=True))
    assert engine.edits == [(PEER, 12, "Цена 130 000 сум")]
    assert engine.incoming == []


async def test_deleted_messages_reach_the_engine(
    gateway, client: StubClient, engine: Engine
) -> None:
    # 2.24: deletions were not listened to. Telegram does not say the chat of a private one,
    # so the engine looks the ids up among the messages it keeps.
    await deliver(client, events.MessageDeleted.Event(deleted_ids=[12, 13], peer=None))
    assert engine.deleted == [[12, 13]]


async def test_an_edit_of_the_owners_own_message_is_not_news(
    gateway, client: StubClient, engine: Engine
) -> None:
    await deliver(client, own_event("Поправил себя", msg_id=50, edited=True))
    assert engine.edits == [] and engine.manual == []


async def test_events_from_other_chats_never_reach_the_engine(
    gateway, client: StubClient, engine: Engine
) -> None:
    await gateway.set_allowed({PEER, -5})
    stranger = PEER + 1
    group = types.Message(id=1, peer_id=types.PeerChat(5), date=NOW, message="hi")
    for event in (
        incoming_event(peer=stranger),
        incoming_event(peer=stranger, edited=True),
        own_event("Написал не тому", msg_id=60, peer=stranger),
        events.NewMessage.Event(group),  # a group, even though -5 is in the list
        events.MessageEdited.Event(group),
    ):
        await deliver(client, event)
    assert engine.incoming == [] and engine.edits == [] and engine.manual == []


async def test_nobody_is_listened_to_until_the_list_is_set(
    client: StubClient, engine: Engine
) -> None:
    fresh = _build(client, engine)
    await fresh.start()
    await deliver(client, incoming_event())
    assert engine.incoming == []
    await fresh.set_allowed({PEER})
    await deliver(client, incoming_event())
    assert len(engine.incoming) == 1


async def test_set_allowed_replaces_the_list(
    gateway, client: StubClient, engine: Engine
) -> None:
    await gateway.set_allowed({PEER + 1})
    await deliver(client, incoming_event())
    assert engine.incoming == []
    await deliver(client, incoming_event(peer=PEER + 1))
    assert [m.peer_id for m in engine.incoming] == [PEER + 1]


# -- our own messages and the owner's --


async def test_our_own_message_is_not_the_owners_but_his_own_is(
    gateway, client: StubClient, engine: Engine
) -> None:
    sent = await gateway.send_text(PEER, "Здравствуйте")
    await deliver(
        client, own_event("Здравствуйте", msg_id=sent)
    )  # the echo of our send
    assert engine.manual == []
    await deliver(
        client, own_event("Сейчас отвечу сам", msg_id=sent + 1)
    )  # from the phone
    assert engine.manual == [(PEER, "Сейчас отвечу сам", sent + 1)]


async def test_a_file_we_sent_is_ours_too(
    gateway, client: StubClient, engine: Engine, tmp_path
) -> None:
    path = tmp_path / "part.stl"
    path.write_bytes(b"solid")
    sent = await gateway.send_file(PEER, str(path))
    await deliver(client, own_event("", msg_id=sent))
    assert engine.manual == []


async def test_only_the_last_500_sent_ids_are_remembered(
    gateway, client: StubClient, engine: Engine
) -> None:
    ids = [await gateway.send_text(PEER, "Ещё вопрос") for _ in range(501)]
    for sent in (ids[0], ids[1], ids[-1]):
        await deliver(client, own_event("с телефона", msg_id=sent))
    assert engine.manual == [
        (PEER, "с телефона", ids[0])
    ]  # only the first id was forgotten


async def test_a_failing_engine_is_contained_and_no_text_is_logged(
    client: StubClient, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "Привет, " + "это текст сообщения"

    async def broken(msg: Incoming) -> None:
        raise RuntimeError(secret)

    engine = Engine()
    failing = userbot.TelethonGateway(
        1,
        "hash",
        "session",
        broken,
        engine.on_edit,
        engine.on_owner_manual,
        client=client,
    )
    await failing.start()
    await failing.set_allowed({PEER})
    with caplog.at_level(logging.DEBUG):
        await deliver(client, incoming_event(secret))  # must not raise
    assert "RuntimeError" in caplog.text
    assert secret not in caplog.text and "Привет" not in caplog.text


# -- replay: what arrived while the program was off --


def _service(msg_id: int):
    """A cleared history, a call: a message in the chat that is no news."""
    return types.MessageService(
        id=msg_id,
        peer_id=types.PeerUser(PEER),
        date=NOW,
        action=types.MessageActionHistoryClear(),
    )


async def test_replay_hands_over_what_was_missed_the_way_live_events_are(
    gateway, client: StubClient, engine: Engine
) -> None:
    sent = await gateway.send_text(PEER, "Здравствуйте")  # ours, in this run: 101
    client.history = [
        _message(text="Давно было", msg_id=40),  # older than the anchor
        _message(text="Цена 120 000", msg_id=51),
        # sent by the agent before the program stopped: the caller knows its id
        _message(text="Здравствуйте", msg_id=52, out=True),
        _message(text="Сам отвечу", msg_id=53, out=True),  # the owner, from his phone
        _message(types.MessageMediaPhoto(photo=_photo()), "Вот фото", msg_id=54),
        _message(
            text="Здравствуйте", msg_id=sent, out=True
        ),  # ours, sent by this gateway
        _service(56),
    ]

    assert await gateway.replay(PEER, 50, {52}) == 3

    # oldest first, a counterpart's messages and the owner's in the order they were written
    assert engine.order == [("in", 51), ("own", 53), ("in", 54)]
    assert engine.incoming == [
        Incoming(PEER, 51, "Цена 120 000", "text", NOW),
        Incoming(PEER, 54, "Вот фото", "photo", NOW),
    ]
    assert engine.manual == [(PEER, "Сам отвечу", 53)]  # with its Telegram id
    # newer than the anchor, oldest first, bounded
    assert client.last("iter_messages") == (
        (INPUT,),
        {"limit": 100, "min_id": 50, "reverse": True},
    )


async def test_read_marks_are_never_listened_to_nor_asked_for(
    gateway, client: StubClient, engine: Engine
) -> None:
    # Ruling R79, "read before the bot": the owner reading a chat on the work account
    # changes nothing. The userbot has exactly the handlers it has today, none for a read...
    assert [type(builder) for _, builder in client.handlers] == [
        events.NewMessage,
        events.NewMessage,
        events.MessageEdited,
        events.MessageDeleted,
    ]
    assert not any(
        isinstance(builder, events.MessageRead) for _, builder in client.handlers
    )
    # ...and its replay asks for messages past a minimum id only: no read mark, no dialog's
    # unread count, no upper bound.
    client.history = [_message(text="Цена 120 000", msg_id=51)]
    before = len(client.calls)

    assert await gateway.replay(PEER, 50, set()) == 1

    replayed = client.calls[before:]
    assert [name for name, _, _ in replayed] == ["get_input_entity", "iter_messages"]
    _, kwargs = client.last("iter_messages")
    assert kwargs == {"limit": userbot.REPLAY_LIMIT, "min_id": 50, "reverse": True}


async def test_replay_is_bounded_and_takes_the_oldest_first(
    gateway, client: StubClient, engine: Engine
) -> None:
    client.history = [_message(text=f"№{n}", msg_id=n) for n in range(1001, 1131)]
    assert await gateway.replay(PEER, 1000, set()) == userbot.REPLAY_LIMIT == 100
    # no gap behind the anchor: the rest is for the next start
    assert [m.tg_msg_id for m in engine.incoming] == list(range(1001, 1101))


async def test_replay_reads_nothing_for_a_chat_that_is_not_allowed(
    gateway, client: StubClient, engine: Engine
) -> None:
    stranger = PEER + 1
    client.history = [_message(text="Привет", msg_id=51, peer=stranger)]
    assert await gateway.replay(stranger, 50, set()) == 0
    assert "iter_messages" not in client.names()
    assert "get_input_entity" not in client.names()
    assert engine.incoming == [] and engine.manual == []


@pytest.mark.parametrize("anchor", [0, -1])
async def test_replay_without_an_anchor_reads_nothing(
    gateway, client: StubClient, engine: Engine, anchor: int
) -> None:
    # Telethon takes min_id=0 for "from the first message ever": a chat's whole old history
    # would be handed over as news. A chat with nothing stored has nothing it could have missed.
    client.history = [_message(text="Давно", msg_id=5)]
    assert await gateway.replay(PEER, anchor, set()) == 0
    assert "iter_messages" not in client.names()
    assert engine.incoming == []


async def test_replay_goes_on_when_the_engine_fails_on_a_message_and_logs_no_text(
    client: StubClient, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "Привет, " + "это текст сообщения"
    seen: list[int] = []

    async def flaky(msg: Incoming) -> None:
        if msg.tg_msg_id == 52:
            raise RuntimeError(secret)
        seen.append(msg.tg_msg_id)

    engine = Engine()
    flaky_gateway = userbot.TelethonGateway(
        1,
        "hash",
        "session",
        flaky,
        engine.on_edit,
        engine.on_owner_manual,
        client=client,
    )
    await flaky_gateway.start()
    await flaky_gateway.set_allowed({PEER})
    client.history = [
        _message(text=secret if n == 52 else "ок", msg_id=n) for n in (51, 52, 53)
    ]
    with caplog.at_level(logging.DEBUG):
        assert (
            await flaky_gateway.replay(PEER, 50, set()) == 2
        )  # the failed one is not counted
    assert seen == [51, 53]
    assert "RuntimeError" in caplog.text
    assert secret not in caplog.text and "Привет" not in caplog.text


async def test_replay_errors_follow_the_read_contract(
    gateway, client: StubClient
) -> None:
    client.history = [_message(text="x", msg_id=51)]
    client.failures["iter_messages"] = errors.FloodWaitError(request=None, capture=120)
    with pytest.raises(TgActionError) as caught:
        await gateway.replay(PEER, 50, set())
    assert (caught.value.kind, caught.value.wait_s) == (TgErrorKind.FLOOD_WAIT, 120)

    client.failures["iter_messages"] = ConnectionError("Привет, как дела")
    with pytest.raises(TgActionError) as caught:
        await gateway.replay(PEER, 50, set())
    assert (caught.value.kind, caught.value.detail) == (
        TgErrorKind.OTHER,
        "ConnectionError",
    )


async def test_replay_of_a_user_nobody_knows_is_an_error_and_reads_nothing(
    gateway, client: StubClient
) -> None:
    del client.input_peers[PEER]
    with pytest.raises(TgActionError) as caught:
        await gateway.replay(PEER, 50, set())
    assert caught.value.kind is TgErrorKind.OTHER
    assert "iter_messages" not in client.names()


# -- sending --


async def test_send_text_returns_the_telegram_id_and_sends_the_text_as_typed(
    gateway, client: StubClient
) -> None:
    sent = await gateway.send_text(PEER, "Нужно 5 штук __PETG__ [тут](http://x.y)")
    assert sent == 101
    args, kwargs = client.last("send_message")
    assert args == (INPUT, "Нужно 5 штук __PETG__ [тут](http://x.y)")
    assert kwargs["parse_mode"] is None  # Telethon would read it as Markdown otherwise
    assert "iter_dialogs" not in client.names()  # the peer was in the session's cache


async def test_send_file_sends_a_path_and_returns_the_telegram_id(
    gateway, client: StubClient, tmp_path
) -> None:
    path = tmp_path / "part.stl"
    path.write_bytes(b"solid")
    assert await gateway.send_file(PEER, str(path)) == 101
    args, _ = client.last("send_file")
    assert args == (
        INPUT,
        path,
    )  # a Path: Telethon would fetch a str that starts with http


@pytest.mark.parametrize("name", ["nope.stl", "https://example.uz/part.stl"])
async def test_send_file_of_something_that_is_not_a_file_fails_before_telegram(
    gateway, client: StubClient, tmp_path, name: str
) -> None:
    target = str(tmp_path / name) if name == "nope.stl" else name
    with pytest.raises(TgActionError) as caught:
        await gateway.send_file(PEER, target)
    assert caught.value.kind is TgErrorKind.OTHER
    assert "send_file" not in client.names()


@pytest.mark.parametrize(
    ("raised", "kind", "wait"),
    [
        (errors.PeerFloodError(request=None), TgErrorKind.PEER_FLOOD, None),
        (errors.FloodWaitError(request=None, capture=90), TgErrorKind.FLOOD_WAIT, 90),
        (_rpc_error("USER_IS_BLOCKED", 400), TgErrorKind.BLOCKED, None),
        (
            _rpc_error("ALLOW_PAYMENT_REQUIRED_5", 403),
            TgErrorKind.PAYMENT_REQUIRED,
            None,
        ),
    ],
)
async def test_a_send_that_telegram_refuses_is_a_tg_action_error_without_the_text(
    gateway, client: StubClient, raised: Exception, kind: TgErrorKind, wait: int | None
) -> None:
    client.failures["send_message"] = raised
    with pytest.raises(TgActionError) as caught:
        await gateway.send_text(PEER, "Привет, как дела")
    assert (caught.value.kind, caught.value.wait_s) == (kind, wait)
    assert "Привет" not in str(caught.value) and "Привет" not in caught.value.detail
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


async def test_a_file_that_telegram_refuses_is_a_tg_action_error_too(
    gateway, client: StubClient, tmp_path
) -> None:
    client.failures["send_file"] = errors.FloodWaitError(request=None, capture=90)
    with pytest.raises(TgActionError) as caught:
        await _send(gateway, "file", tmp_path)
    assert (caught.value.kind, caught.value.wait_s) == (TgErrorKind.FLOOD_WAIT, 90)


@pytest.mark.parametrize("how", ["text", "file"])
@pytest.mark.parametrize(
    "raised",
    [
        ConnectionError("Привет, как дела"),
        OSError("Привет, как дела"),
        ValueError("Request was unsuccessful 5 time(s)"),  # Telethon out of retries
        TimeoutError("Привет, как дела"),
    ],
    ids=lambda error: type(error).__name__,
)
async def test_a_send_that_breaks_midway_may_have_left_and_says_so(
    gateway, client: StubClient, tmp_path, how: str, raised: Exception
) -> None:
    # Only Telegram's answer says nothing was sent. A broken connection or Telethon giving up
    # may well have come after the message left, and a "refused" would pause the chat on a
    # guess: the Sender asks the owner instead.
    client.failures[SEND_CALL[how]] = raised
    with pytest.raises(TgSendUncertain) as caught:
        await _send(gateway, how, tmp_path)
    assert not isinstance(caught.value, TgActionError)
    assert str(caught.value) == type(raised).__name__  # the class, never the words
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


@pytest.mark.parametrize("how", ["text", "file"])
async def test_a_send_that_comes_back_with_no_message_may_have_left_too(
    gateway, client: StubClient, tmp_path, how: str
) -> None:
    client.sends_nothing = True
    with pytest.raises(TgSendUncertain):
        await _send(gateway, how, tmp_path)
    assert SEND_CALL[how] in client.names()  # it was asked: that is the doubt


async def test_a_failure_before_the_send_is_a_refusal_even_when_it_is_not_telegrams(
    gateway, client: StubClient
) -> None:
    # Looking the peer up is not the send: if that breaks, nothing has left
    client.failures["get_input_entity"] = ConnectionError("Привет, как дела")
    with pytest.raises(TgActionError) as caught:
        await gateway.send_text(PEER, "Здравствуйте")
    assert (caught.value.kind, caught.value.detail) == (
        TgErrorKind.OTHER,
        "ConnectionError",
    )
    assert "send_message" not in client.names()


async def test_a_failed_send_leaves_no_id_behind(
    gateway, client: StubClient, engine: Engine
) -> None:
    client.failures["send_message"] = errors.PeerFloodError(request=None)
    with pytest.raises(TgActionError):
        await gateway.send_text(PEER, "Здравствуйте")
    await deliver(client, own_event("с телефона", msg_id=101))
    assert engine.manual == [(PEER, "с телефона", 101)]


async def test_a_peer_the_session_does_not_know_is_found_among_the_dialogs(
    gateway, client: StubClient
) -> None:
    # A session string carries no entities: after a restart nobody is known until he writes
    del client.input_peers[PEER]
    client.dialogs = [
        SimpleNamespace(id=9, input_entity=types.InputPeerUser(9, 1)),
        SimpleNamespace(id=PEER, input_entity=INPUT),
    ]
    await gateway.send_text(PEER, "Добрый день")
    args, _ = client.last("send_message")
    assert args[0] == INPUT


async def test_a_user_is_looked_up_as_a_user_never_as_a_bare_id(
    gateway, client: StubClient
) -> None:
    # Telethon reads a bare positive id as a user, a chat or a channel, whichever it finds
    # first, so a channel that has the user's number could get his message
    client.bare_ids[PEER] = types.InputPeerChannel(PEER, 99)
    await gateway.send_text(PEER, "Здравствуйте")
    asked = [args[0] for name, args, _ in client.calls if name == "get_input_entity"]
    assert asked == [types.PeerUser(PEER)]
    args, _ = client.last("send_message")
    assert args[0] == INPUT


@pytest.mark.parametrize(
    "wrong",
    [
        types.InputPeerChannel(PEER, 99),
        types.InputPeerChat(PEER),
        types.InputPeerSelf(),
    ],
    ids=lambda peer: type(peer).__name__,
)
async def test_a_peer_that_is_not_a_user_is_never_used(
    gateway, client: StubClient, wrong: object
) -> None:
    # Telethon's in-memory cache is keyed by the bare number, whatever the kind of entity
    client.input_peers[PEER] = wrong
    with pytest.raises(TgActionError) as caught:
        await gateway.send_text(PEER, "Здравствуйте")
    assert (caught.value.kind, caught.value.detail) == (
        TgErrorKind.OTHER,
        "unknown peer",
    )
    assert "send_message" not in client.names()


async def test_when_the_cache_answers_with_something_else_the_dialogs_find_the_user(
    gateway, client: StubClient
) -> None:
    client.input_peers[PEER] = types.InputPeerChannel(PEER, 99)
    client.dialogs = [SimpleNamespace(id=PEER, input_entity=INPUT)]
    await gateway.send_text(PEER, "Здравствуйте")
    args, _ = client.last("send_message")
    assert args[0] == INPUT


async def test_a_peer_nobody_knows_is_an_error_and_nothing_is_sent(
    gateway, client: StubClient
) -> None:
    del client.input_peers[PEER]
    client.dialogs = [SimpleNamespace(id=9, input_entity=types.InputPeerUser(9, 1))]
    with pytest.raises(TgActionError) as caught:
        await gateway.send_text(PEER, "Добрый день")
    assert caught.value.kind is TgErrorKind.OTHER
    assert "send_message" not in client.names()


# -- typing, reading, downloading --


async def test_typing_shows_typing_for_that_long(
    gateway, client: StubClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def sleep(seconds: float) -> None:
        client.record("sleep", seconds)

    monkeypatch.setattr(userbot.asyncio, "sleep", sleep)
    await gateway.typing(PEER, 7.5)
    assert client.names()[-5:] == [
        "get_input_entity",
        "action",
        "typing_on",
        "sleep",  # the time passes while "typing…" is on
        "typing_off",
    ]
    assert client.last("action")[0] == (INPUT, "typing")
    assert client.last("sleep")[0] == (7.5,)


async def test_a_typing_error_comes_out_as_a_tg_action_error(
    gateway, client: StubClient
) -> None:
    client.failures["typing_off"] = errors.FloodWaitError(request=None, capture=30)
    with pytest.raises(TgActionError) as caught:
        await gateway.typing(PEER, 0)
    assert (caught.value.kind, caught.value.wait_s) == (TgErrorKind.FLOOD_WAIT, 30)


async def test_mark_read_acknowledges_up_to_that_message(
    gateway, client: StubClient
) -> None:
    await gateway.mark_read(PEER, 55)
    assert client.last("send_read_acknowledge") == ((INPUT,), {"max_id": 55})


async def test_download_returns_the_bytes_of_the_media(
    gateway, client: StubClient
) -> None:
    client.messages[31] = _message(types.MessageMediaPhoto(photo=_photo()), msg_id=31)
    client.media = b"\xff\xd8photo"
    assert await gateway.download(PEER, 31) == b"\xff\xd8photo"
    assert client.last("get_messages") == ((INPUT,), {"ids": 31})
    assert client.last("download_media")[1] == {"file": bytes}  # in memory, not on disk


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(None, id="no-such-message"),
        pytest.param(_message(text="Только текст", msg_id=31), id="text-only"),
        pytest.param(
            _message(
                _preview(photo=_photo()), "Смотрите https://example.uz", msg_id=31
            ),
            id="link-preview",
        ),
    ],
)
async def test_download_of_a_message_without_media_is_an_error(
    gateway, client: StubClient, message
) -> None:
    client.messages[31] = message
    with pytest.raises(TgActionError) as caught:
        await gateway.download(PEER, 31)
    assert caught.value.kind is TgErrorKind.OTHER
    assert "download_media" not in client.names()


# -- resolve --


def _user(**fields) -> types.User:
    return types.User(id=77, username="ali_print", first_name="Али", **fields)


def _know(client: StubClient, entity: object, name: str = "ali_print") -> None:
    client.input_peers[name] = ("input-peer", 77)
    client.entities[("input-peer", 77)] = entity


async def test_own_name_is_the_first_and_the_last_name_of_the_account(
    gateway, client: StubClient
) -> None:
    # What a counterpart sees on every message: the engine reads it for «ИИ».
    assert await gateway.own_name() == "ИИ помощник Саида"
    assert client.names()[-1] == "get_me"


async def test_resolve_asks_the_cache_for_the_name_and_then_for_the_entity(
    gateway, client: StubClient
) -> None:
    _know(client, _user(last_name="Каримов", bot=False))
    info = await gateway.resolve("@Ali_Print")
    assert info == PeerInfo(77, "ali_print", "Али Каримов", False)
    assert client.names()[-2:] == ["get_input_entity", "get_entity"]
    # the entity is asked for by its input peer: no second lookup of the name
    assert client.last("get_entity")[0] == (("input-peer", 77),)


@pytest.mark.parametrize(
    "target",
    [
        "ali_print",
        "@ali_print",
        "@Ali_Print",
        "  @ali_print ",
        "t.me/ali_print",
        "https://t.me/ali_print",
        "https://t.me/Ali_Print/",
        "http://telegram.me/ali_print",
    ],
)
async def test_resolve_takes_a_name_a_handle_or_a_link(
    gateway, client: StubClient, target: str
) -> None:
    _know(client, _user())
    assert (await gateway.resolve(target)).username == "ali_print"
    assert client.last("get_input_entity")[0] == ("ali_print",)


async def test_resolve_tells_a_bot_and_a_user_without_a_last_name(
    gateway, client: StubClient
) -> None:
    _know(client, _user(bot=True))
    info = await gateway.resolve("ali_print")
    assert (info.is_bot, info.display_name) == (True, "Али")


async def test_resolve_of_a_channel_is_not_a_user(gateway, client: StubClient) -> None:
    channel = types.Channel(
        id=1,
        title="Новости",
        photo=types.ChatPhotoEmpty(),
        date=NOW,
        username="news_uz",
    )
    _know(client, channel, name="news_uz")
    with pytest.raises(TgActionError) as caught:
        await gateway.resolve("@news_uz")
    assert (caught.value.kind, caught.value.detail) == (TgErrorKind.OTHER, "not a user")


async def test_resolve_refuses_a_user_found_by_his_name_instead_of_his_username(
    gateway, client: StubClient
) -> None:
    # Telethon's cache looks display names up as well: get_input_entity("kamol") can answer
    # with the user whose NAME is "kamol" and whose username is something else. Writing to
    # him would be writing to the wrong person.
    impostor = types.User(id=77, first_name="kamol", username="someone_else")
    _know(client, impostor, name="kamol")
    with pytest.raises(TgActionError) as caught:
        await gateway.resolve("@kamol")
    assert (caught.value.kind, caught.value.detail) == (TgErrorKind.OTHER, "not found")


async def test_resolve_accepts_any_active_username_of_the_user(
    gateway, client: StubClient
) -> None:
    user = types.User(
        id=77,
        first_name="Али",
        username="main_name",
        usernames=[types.Username(username="Kamol_Print", editable=False, active=True)],
    )
    _know(client, user, name="kamol_print")
    info = await gateway.resolve("@KAMOL_PRINT")
    assert (info.peer_id, info.username) == (77, "main_name")


async def test_resolve_does_not_take_a_username_that_is_switched_off(
    gateway, client: StubClient
) -> None:
    user = types.User(
        id=77,
        first_name="Али",
        username="main_name",
        usernames=[types.Username(username="old_name", editable=False, active=False)],
    )
    _know(client, user, name="old_name")
    with pytest.raises(TgActionError) as caught:
        await gateway.resolve("@old_name")
    assert caught.value.detail == "not found"


async def test_resolve_of_an_unknown_name_is_an_error_without_the_name(
    gateway, client: StubClient
) -> None:
    client.failures["get_input_entity"] = ValueError(
        'No user has "ghost_shop" as username'
    )
    with pytest.raises(TgActionError) as caught:
        await gateway.resolve("@ghost_shop")
    assert caught.value.kind is TgErrorKind.OTHER
    assert "ghost_shop" not in caught.value.detail


@pytest.mark.parametrize("target", ["12345", "a", "+123", "https://t.me/+AbCdEfGh", ""])
async def test_resolve_of_something_that_is_not_a_username_asks_nobody(
    gateway, client: StubClient, target: str
) -> None:
    with pytest.raises(TgActionError) as caught:
        await gateway.resolve(target)
    assert caught.value.kind is TgErrorKind.OTHER
    assert "get_input_entity" not in client.names()


async def test_resolve_during_a_flood_wait(gateway, client: StubClient) -> None:
    client.failures["get_input_entity"] = errors.FloodWaitError(
        request=None, capture=300
    )
    with pytest.raises(TgActionError) as caught:
        await gateway.resolve("@ali_print")
    assert (caught.value.kind, caught.value.wait_s) == (TgErrorKind.FLOOD_WAIT, 300)


# -- a phone found by the vendor search --


async def test_resolve_finds_a_person_by_his_phone(gateway, client: StubClient) -> None:
    client.answers["ResolvePhoneRequest"] = types.contacts.ResolvedPeer(
        peer=types.PeerUser(77), chats=[], users=[_user(last_name="Каримов")]
    )
    info = await gateway.resolve("+998901234567")
    assert info == PeerInfo(77, "ali_print", "Али Каримов", False)
    [request] = client.last("ResolvePhoneRequest")[0]
    assert request.phone == "998901234567"  # Telegram's form: digits, no "+"
    assert "get_input_entity" not in client.names()


async def test_resolve_of_a_phone_nobody_can_be_found_by(
    gateway, client: StubClient
) -> None:
    client.failures["ResolvePhoneRequest"] = errors.PhoneNotOccupiedError(request=None)
    with pytest.raises(TgActionError) as caught:
        await gateway.resolve("+998901234567")
    assert caught.value.kind is TgErrorKind.OTHER
    assert "901234567" not in caught.value.detail


async def test_resolve_of_a_phone_that_answers_with_no_user(
    gateway, client: StubClient
) -> None:
    client.answers["ResolvePhoneRequest"] = types.contacts.ResolvedPeer(
        peer=types.PeerChannel(5), chats=[], users=[]
    )
    with pytest.raises(TgActionError) as caught:
        await gateway.resolve("+998901234567")
    assert caught.value.detail == "not a user"


# -- Telegram's public search, for the vendor search --


def _channel(
    channel_id: int, username: str | None, title: str, **kind
) -> types.Channel:
    return types.Channel(
        id=channel_id,
        title=title,
        photo=types.ChatPhotoEmpty(),
        date=NOW,
        username=username,
        access_hash=9,
        **kind,
    )


async def test_search_public_lists_people_channels_and_groups(
    gateway, client: StubClient
) -> None:
    from tg_agent.gateway.base import PublicPeer

    client.answers["SearchRequest"] = types.contacts.Found(
        my_results=[],
        results=[
            types.PeerUser(77),
            types.PeerChannel(5),
            types.PeerChannel(6),
            types.PeerUser(78),
            types.PeerChannel(7),
        ],
        chats=[
            _channel(5, "print3d_tash", "3D печать Ташкент", broadcast=True),
            _channel(6, "makers_uz", "Мейкеры", megagroup=True),
            _channel(7, None, "Без имени", broadcast=True),  # nobody can be pointed to
        ],
        users=[
            _user(),
            types.User(id=78, first_name="Бот", username="some_bot", bot=True),
        ],
    )
    abouts = {5: "Заказы: @ali_3d", 6: ""}
    client.answers["GetFullChannelRequest"] = lambda request: SimpleNamespace(
        full_chat=SimpleNamespace(about=abouts[request.channel.channel_id])
    )

    found = await gateway.search_public("3D печать", 10)

    assert found == [
        PublicPeer("ali_print", "Али", "user"),
        PublicPeer("print3d_tash", "3D печать Ташкент", "channel", "Заказы: @ali_3d"),
        PublicPeer("makers_uz", "Мейкеры", "group"),
    ]
    [request] = client.last("SearchRequest")[0]
    assert (request.q, request.limit) == ("3D печать", 10)


async def test_search_public_during_a_flood_wait(gateway, client: StubClient) -> None:
    client.failures["SearchRequest"] = errors.FloodWaitError(request=None, capture=60)
    with pytest.raises(TgActionError) as caught:
        await gateway.search_public("3D печать", 10)
    assert caught.value.kind is TgErrorKind.FLOOD_WAIT


# -- the work account's own contacts, by name (Task 5.4, ruling R60) --


async def test_search_contacts_reads_the_address_book_and_keeps_the_names_that_match(
    gateway, client: StubClient
) -> None:
    client.answers["GetContactsRequest"] = types.contacts.Contacts(
        contacts=[],
        saved_count=0,
        users=[
            types.User(id=71, first_name="Алишер", last_name="Каримов",
                       username="alisher_k", phone="998901231234", contact=True),
            types.User(id=72, first_name="Алишер", phone="998935550011", contact=True),
            types.User(id=73, first_name="Дилшод", phone="998977770000", contact=True),
            types.User(id=74, first_name="Алишер", last_name="бот", bot=True,
                       username="alisher_bot", contact=True),
            types.User(id=75, first_name="Алишер", deleted=True, contact=True),
            types.User(id=76, first_name="Алишер", last_name="Без номера",
                       contact=True),
        ],
    )  # fmt: skip

    found = await gateway.search_contacts("Алишера", 5)

    assert found == [
        PeerInfo(71, "alisher_k", "Алишер Каримов", False, "+998901231234"),
        PeerInfo(72, None, "Алишер", False, "+998935550011"),
        PeerInfo(76, None, "Алишер Без номера", False, None),
    ]
    # The book is read, never Telegram's global search, and nothing is written.
    assert "SearchRequest" not in client.names()
    [request] = client.last("GetContactsRequest")[0]
    assert request.hash == 0
    assert (await gateway.search_contacts("Алишер", 2))[-1].peer_id == 72


async def test_search_contacts_when_the_book_did_not_change_finds_nobody(
    gateway, client: StubClient
) -> None:
    client.answers["GetContactsRequest"] = types.contacts.ContactsNotModified()
    assert await gateway.search_contacts("Алишер", 5) == []


async def test_search_contacts_during_a_flood_wait(gateway, client: StubClient) -> None:
    client.failures["GetContactsRequest"] = errors.FloodWaitError(
        request=None, capture=60
    )
    with pytest.raises(TgActionError) as caught:
        await gateway.search_contacts("Алишер", 5)
    assert caught.value.kind is TgErrorKind.FLOOD_WAIT


async def test_own_username_is_the_accounts_username(
    gateway, client: StubClient
) -> None:
    assert await gateway.own_username() is None  # the stub's account has none

    async def me() -> object:
        return SimpleNamespace(first_name="ИИ", last_name=None, username="work_acc")

    client.get_me = me  # type: ignore[method-assign]
    assert await gateway.own_username() == "work_acc"


# -- recent_messages: the agent core reads a chat --


async def test_recent_messages_maps_telethon_messages_oldest_first(
    gateway, client: StubClient, engine: Engine
) -> None:
    naive = datetime(
        2026, 11, 5, 9, 30
    )  # Telethon gives aware dates; a naive one is UTC
    client.history = [
        _message(text="Давно было", msg_id=40),  # not newer than the anchor
        _message(text="Цена 120 000", msg_id=51),
        _message(text="Здравствуйте", msg_id=52, out=True),  # the agent or the owner
        _message(types.MessageMediaPhoto(photo=_photo()), msg_id=53),  # no caption
        _message(types.MessageMediaPhoto(photo=_photo()), "Вот фото", msg_id=54),
        _service(55),  # a cleared history is not a message of the chat
        types.Message(id=56, peer_id=types.PeerUser(PEER), date=naive, message="Ок"),
    ]

    read = await gateway.recent_messages(PEER, 50, 10)

    assert read == [
        ChatMessage(51, False, "Цена 120 000", NOW),
        ChatMessage(52, True, "Здравствуйте", NOW),
        ChatMessage(53, False, "", NOW),
        ChatMessage(54, False, "Вот фото", NOW),
        ChatMessage(56, False, "Ок", naive.replace(tzinfo=UTC)),
    ]
    assert all(m.date.utcoffset() is not None for m in read)
    # newest first from Telegram, at most `limit`, newer than the anchor
    assert client.last("iter_messages") == ((INPUT,), {"limit": 10, "min_id": 50})
    # reads only: no callback, nothing marked read
    assert engine.incoming == [] and engine.manual == [] and engine.order == []
    assert "send_read_acknowledge" not in client.names()


async def test_recent_messages_keeps_the_newest_limit(
    gateway, client: StubClient
) -> None:
    client.history = [_message(text=f"№{n}", msg_id=n) for n in range(1001, 1131)]

    read = await gateway.recent_messages(PEER, 1000, 5)

    assert [m.tg_msg_id for m in read] == list(range(1126, 1131))


async def test_recent_messages_reads_a_chat_that_is_not_listened_to(
    gateway, client: StubClient
) -> None:
    # a person the agent has finished with: a goodbye whose send broke off is looked for
    await gateway.set_allowed(set())
    client.history = [_message(text="До свидания", msg_id=51, out=True)]

    assert [m.text for m in await gateway.recent_messages(PEER, 0, 10)] == [
        "До свидания"
    ]


async def test_recent_messages_with_no_room_asks_telegram_nothing(
    gateway, client: StubClient
) -> None:
    client.history = [_message(text="x", msg_id=51)]
    assert await gateway.recent_messages(PEER, 0, 0) == []
    assert "iter_messages" not in client.names()


async def test_recent_messages_errors_follow_the_read_contract(
    gateway, client: StubClient
) -> None:
    client.history = [_message(text="x", msg_id=51)]
    client.failures["iter_messages"] = errors.FloodWaitError(request=None, capture=120)
    with pytest.raises(TgActionError) as caught:
        await gateway.recent_messages(PEER, 50, 10)
    assert (caught.value.kind, caught.value.wait_s) == (TgErrorKind.FLOOD_WAIT, 120)

    client.failures["iter_messages"] = ConnectionError("Привет, как дела")
    with pytest.raises(TgActionError) as caught:
        await gateway.recent_messages(PEER, 50, 10)
    assert (caught.value.kind, caught.value.detail) == (
        TgErrorKind.OTHER,
        "ConnectionError",
    )

    del client.input_peers[PEER]  # a user nobody knows
    with pytest.raises(TgActionError):
        await gateway.recent_messages(PEER, 50, 10)


# -- the chats' sound and the folder (rulings R105, R106) --

FOLDER = "ИИ агент"


def _filter(id: int, title: str, *include: object, **extra: object) -> types.DialogFilter:
    return types.DialogFilter(
        id=id,
        title=types.TextWithEntities(text=title, entities=[]),
        pinned_peers=list(extra.pop("pinned", [])),
        include_peers=list(include),
        exclude_peers=list(extra.pop("exclude", [])),
        **extra,
    )


async def test_is_muted_reads_until_when_the_chat_is_muted(
    gateway, client: StubClient
) -> None:
    for until, muted in [
        (datetime(2038, 1, 19, tzinfo=UTC), True),
        (datetime(1970, 1, 1, tzinfo=UTC), False),  # Telegram's 0: not muted
        (datetime(2020, 5, 1, tzinfo=UTC), False),  # muted for a while, long ago
        (None, False),
    ]:
        client.answers["GetNotifySettingsRequest"] = types.PeerNotifySettings(
            mute_until=until
        )
        assert await gateway.is_muted(PEER) is muted

    [request] = client.last("GetNotifySettingsRequest")[0]
    assert request.peer == types.InputNotifyPeer(INPUT)
    bytes(request)  # a request Telegram can be sent


async def test_set_muted_mutes_for_good_and_gives_the_sound_back(
    gateway, client: StubClient
) -> None:
    client.answers["UpdateNotifySettingsRequest"] = True

    await gateway.set_muted(PEER, True)
    [request] = client.last("UpdateNotifySettingsRequest")[0]
    assert request.peer == types.InputNotifyPeer(INPUT)
    assert request.settings == types.InputPeerNotifySettings(mute_until=2**31 - 1)
    bytes(request)

    await gateway.set_muted(PEER, False)
    [request] = client.last("UpdateNotifySettingsRequest")[0]
    assert request.settings == types.InputPeerNotifySettings(mute_until=0)
    bytes(request)


async def test_a_refused_mute_is_a_tg_action_error(gateway, client: StubClient) -> None:
    client.failures["UpdateNotifySettingsRequest"] = errors.FloodWaitError(
        request=None, capture=40
    )
    with pytest.raises(TgActionError) as caught:
        await gateway.set_muted(PEER, True)
    assert (caught.value.kind, caught.value.wait_s) == (TgErrorKind.FLOOD_WAIT, 40)

    client.failures["GetNotifySettingsRequest"] = ConnectionError("Привет")
    with pytest.raises(TgActionError) as caught:
        await gateway.is_muted(PEER)
    assert caught.value.detail == "ConnectionError"


async def test_folder_peers_finds_the_folder_by_its_title(
    gateway, client: StubClient
) -> None:
    group = types.InputPeerChat(5)
    client.answers["GetDialogFiltersRequest"] = types.messages.DialogFilters(
        filters=[
            types.DialogFilterDefault(),
            _filter(2, "Работа", types.InputPeerUser(77, 1)),
            types.DialogFilterChatlist(
                id=3,
                title=types.TextWithEntities(text=FOLDER, entities=[]),
                pinned_peers=[],
                include_peers=[types.InputPeerUser(88, 1)],
            ),  # a shared folder of the same name is not the account's own
            _filter(4, FOLDER, INPUT, group, pinned=[types.InputPeerSelf()]),
        ]
    )

    assert await gateway.folder_peers(FOLDER) == [0, PEER, -5]
    assert await gateway.folder_peers("Нет такой") is None


async def test_add_to_folder_makes_the_folder_with_the_first_chat(
    gateway, client: StubClient
) -> None:
    client.answers["GetDialogFiltersRequest"] = types.messages.DialogFilters(
        filters=[types.DialogFilterDefault(), _filter(2, "Работа"), _filter(4, "Дом")]
    )
    client.answers["UpdateDialogFilterRequest"] = True

    await gateway.add_to_folder(FOLDER, PEER)

    [request] = client.last("UpdateDialogFilterRequest")[0]
    assert request.id == 3  # the smallest free id
    assert request.filter == _filter(3, FOLDER, INPUT)
    bytes(request)


async def test_add_to_folder_changes_nothing_else_of_the_folder(
    gateway, client: StubClient
) -> None:
    owners = types.InputPeerUser(55, 9)
    ours = types.InputPeerUser(66, 9)
    other = _filter(2, "Работа", types.InputPeerUser(77, 1))
    client.answers["GetDialogFiltersRequest"] = types.messages.DialogFilters(
        filters=[
            other,
            _filter(
                5, FOLDER, owners, ours, exclude=[types.InputPeerUser(99, 1)],
                emoticon="🤖", contacts=True,
            ),
        ]
    )  # fmt: skip
    client.answers["UpdateDialogFilterRequest"] = True

    await gateway.add_to_folder(FOLDER, PEER, remove={66})

    [request] = client.last("UpdateDialogFilterRequest")[0]
    assert request.id == 5
    assert request.filter == _filter(
        5, FOLDER, owners, INPUT, exclude=[types.InputPeerUser(99, 1)],
        emoticon="🤖", contacts=True,
    )  # fmt: skip
    assert client.names().count("UpdateDialogFilterRequest") == 1  # «Работа» untouched
    bytes(request)


async def test_a_chat_already_in_the_folder_is_not_put_there_twice(
    gateway, client: StubClient
) -> None:
    client.answers["GetDialogFiltersRequest"] = types.messages.DialogFilters(
        filters=[_filter(2, FOLDER, pinned=[INPUT])]
    )
    client.answers["UpdateDialogFilterRequest"] = True

    await gateway.add_to_folder(FOLDER, PEER)

    [request] = client.last("UpdateDialogFilterRequest")[0]
    assert request.filter == _filter(2, FOLDER, pinned=[INPUT])


async def test_a_refused_folder_is_a_tg_action_error(gateway, client: StubClient) -> None:
    client.answers["GetDialogFiltersRequest"] = types.messages.DialogFilters(filters=[])
    client.failures["UpdateDialogFilterRequest"] = errors.FloodWaitError(
        request=None, capture=90
    )
    with pytest.raises(TgActionError) as caught:
        await gateway.add_to_folder(FOLDER, PEER)
    assert (caught.value.kind, caught.value.wait_s) == (TgErrorKind.FLOOD_WAIT, 90)
