"""The remote screen's cipher and session, on a fake screen and a fake socket (no browser)."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import struct
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from tg_agent.clock import FakeClock
from tg_agent.search import remote
from tg_agent.search.browser import BrowserError
from tg_agent.search.remote import (
    CdpScreen,
    RemoteKey,
    RemoteSession,
    open_sealed,
    seal,
)

KEY = RemoteKey(bytes(range(64)))
NONCE = bytes(range(100, 116))
PLAIN = '{"t":"text","s":"привет, мир — 42"}'.encode()
# nonce ‖ ciphertext ‖ tag for KEY, NONCE and PLAIN; the page's JavaScript must give the same.
VECTOR = bytes.fromhex(
    "6465666768696a6b6c6d6e6f707172733e9fab77f2c96a54f6e0230dc15df670a1acfb4b59e1a225e45c"
    "f80d1042ecce25c05496089f1719fba06c2a6c76b44d1d5a2016a55665d47a8df166630450d3c5878f34"
    "e7bcd6fa27d668cf7870"
)


def _by_hand(key: bytes, nonce: bytes, plain: bytes) -> bytes:
    """The design's cipher written out with hmac, independently of `seal`."""
    blocks = (len(plain) + 31) // 32
    stream = b"".join(
        hmac.new(key[:32], nonce + struct.pack(">I", i), hashlib.sha256).digest()
        for i in range(blocks)
    )
    cipher = bytes(a ^ b for a, b in zip(plain, stream, strict=False))
    tag = hmac.new(key[32:], nonce + cipher, hashlib.sha256).digest()
    return nonce + cipher + tag


# --- the cipher ------------------------------------------------------------------------------


def test_the_fixed_vector_is_the_design_cipher() -> None:
    assert len(PLAIN) > 32  # two keystream blocks
    assert _by_hand(KEY.raw, NONCE, PLAIN) == VECTOR
    assert open_sealed(KEY, VECTOR) == PLAIN


def test_seal_gives_the_fixed_vector_for_a_fixed_nonce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(remote, "_new_nonce", lambda: NONCE)
    assert seal(KEY, PLAIN) == VECTOR


def test_a_round_trip_with_a_new_key() -> None:
    key = RemoteKey.new()
    for plain in (b"", b"a", b"x" * 32, b"y" * 33, "ввод".encode() * 200):
        assert open_sealed(key, seal(key, plain)) == plain


@pytest.mark.parametrize("index", [0, 15, 16, 30, 16 + len(PLAIN), len(VECTOR) - 1])
def test_a_flipped_bit_anywhere_is_refused(index: int) -> None:
    broken = bytearray(VECTOR)
    broken[index] ^= 0x01
    assert open_sealed(KEY, bytes(broken)) is None


def test_a_short_message_is_refused_and_an_empty_one_opens() -> None:
    assert open_sealed(KEY, VECTOR[:47]) is None
    assert open_sealed(KEY, b"") is None
    empty = _by_hand(KEY.raw, NONCE, b"")
    assert len(empty) == 48
    assert open_sealed(KEY, empty) == b""


def test_another_key_does_not_open_it() -> None:
    assert open_sealed(RemoteKey.new(), VECTOR) is None


def test_two_seals_of_one_plaintext_differ() -> None:
    key = RemoteKey.new()
    first, second = seal(key, PLAIN), seal(key, PLAIN)
    assert first != second
    assert first[:16] != second[:16]


def test_the_key_travels_as_url_safe_base64_without_padding() -> None:
    key = RemoteKey.new()
    fragment = key.fragment()
    assert len(fragment) == 86
    assert set(fragment) <= set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    )
    assert RemoteKey.from_fragment(fragment) == key
    assert RemoteKey.new() != key
    assert fragment not in repr(key) and key.raw.hex() not in repr(key)
    for bad in ("", "abc", fragment[:-2], fragment + "AA", fragment[:-1] + "*"):
        with pytest.raises(ValueError):
            RemoteKey.from_fragment(bad)
    with pytest.raises(ValueError):
        RemoteKey(b"short")


# --- the session -----------------------------------------------------------------------------

START = datetime(2026, 11, 11, 12, 0, tzinfo=ZoneInfo("Asia/Tashkent"))


class FakeScreen:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.on_frame: Callable[[bytes], None] | None = None
        self.on_refused: Callable[[], None] | None = None
        self.on_notice: Callable[[str], None] | None = None
        self.focused = "none"
        self.start_error: Exception | None = None
        self.text_error: Exception | None = None
        self.address = "https://example.com/login"

    async def start(
        self,
        url: str,
        on_frame: Callable[[bytes], None],
        on_refused: Callable[[], None],
        on_notice: Callable[[str], None] | None = None,
    ) -> None:
        self.calls.append(("start", url))
        self.on_frame, self.on_refused = on_frame, on_refused
        self.on_notice = on_notice
        if self.start_error is not None:
            raise self.start_error

    async def pointer(self, kind: str, x: float, y: float) -> None:
        self.calls.append(("pointer", kind, x, y))

    async def wheel(self, dy: float) -> None:
        self.calls.append(("wheel", dy))

    async def text(self, s: str) -> None:
        if self.text_error is not None:
            raise self.text_error
        self.calls.append(("text", s))

    async def key(self, k: str) -> None:
        self.calls.append(("key", k))

    async def back(self) -> None:
        self.calls.append(("back",))

    async def focus(self) -> str:
        return self.focused

    def url(self) -> str:
        return self.address

    async def stop(self) -> None:
        self.calls.append(("stop",))

    @property
    def inputs(self) -> list[tuple[Any, ...]]:
        return [c for c in self.calls if c[0] not in ("start", "stop")]


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[bytes | str] = []
        self.closed = False
        self._incoming: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.load = secrets.token_hex(16)  # the page load that writes on this socket
        self.counter = 0

    def feed(self, message: bytes) -> None:
        self._incoming.put_nowait(message)

    def hang_up(self) -> None:
        self._incoming.put_nowait(None)

    async def send_bytes(self, data: bytes) -> None:
        if self.closed:
            raise ConnectionResetError
        self.sent.append(data)

    async def send_str(self, data: str) -> None:
        if self.closed:
            raise ConnectionResetError
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True
        self._incoming.put_nowait(None)

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._read()

    async def _read(self) -> AsyncIterator[bytes]:
        while True:
            message = await self._incoming.get()
            if message is None:
                return
            yield message

    @property
    def texts(self) -> list[dict[str, Any]]:
        return [json.loads(m) for m in self.sent if isinstance(m, str)]

    @property
    def frames(self) -> list[bytes]:
        return [m for m in self.sent if isinstance(m, bytes)]


async def until(check: Callable[[], bool], seconds: float = 3.0) -> None:
    for _ in range(int(seconds / 0.01)):
        if check():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the condition never came true")


def _session(screen: FakeScreen, clock: FakeClock) -> RemoteSession:
    return RemoteSession(
        "olx.uz", "https://www.olx.uz/account/", screen, clock, 10, poll_seconds=0.01
    )


def _stamped(socket: FakeSocket, message: dict[str, Any]) -> bytes:
    """The message as the page writes it: its load and the next number of that load."""
    socket.counter += 1
    return json.dumps({"p": socket.load, "n": socket.counter, **message}).encode()


def _send(socket: FakeSocket, session: RemoteSession, message: dict[str, Any]) -> bytes:
    sealed = seal(session.key, _stamped(socket, message))
    socket.feed(sealed)
    return sealed


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(START)


async def test_a_new_session_has_a_token_a_key_and_ten_minutes(
    clock: FakeClock,
) -> None:
    screen = FakeScreen()
    session = _session(screen, clock)
    other = _session(screen, clock)
    assert len(session.token) >= 43 and session.token != other.token
    assert session.key != other.key
    assert session.site == "olx.uz"
    assert session.expires_at == START + timedelta(minutes=10)
    assert session.outcome is None
    assert session.alive is True


async def test_every_input_reaches_the_screen_in_order(clock: FakeClock) -> None:
    screen, socket = FakeScreen(), FakeSocket()
    session = _session(screen, clock)
    served = asyncio.create_task(session.serve(socket))
    messages: list[dict[str, Any]] = [
        {"t": "down", "x": 0.25, "y": 0.5},
        {"t": "up", "x": 0.25, "y": 0.5},
        {"t": "down", "x": 0.1, "y": 0.9},
        {"t": "move", "x": 0.4, "y": 0.9},
        {"t": "move", "x": 0.8, "y": 0.9},
        {"t": "up", "x": 0.8, "y": 0.9},
        {"t": "wheel", "dy": 300},
        {"t": "text", "s": "said@example.com"},
        {"t": "key", "k": "Tab"},
        {"t": "text", "s": "секрет 123"},
        {"t": "key", "k": "Backspace"},
        {"t": "key", "k": "Enter"},
        {"t": "back"},
        {"t": "done"},
    ]
    for message in messages:
        _send(socket, session, message)
    await asyncio.wait_for(served, 3)

    assert screen.calls[0] == ("start", "https://www.olx.uz/account/")
    assert screen.inputs == [
        ("pointer", "down", 0.25, 0.5),
        ("pointer", "up", 0.25, 0.5),
        ("pointer", "down", 0.1, 0.9),
        ("pointer", "move", 0.4, 0.9),
        ("pointer", "move", 0.8, 0.9),
        ("pointer", "up", 0.8, 0.9),
        ("wheel", 300.0),
        ("text", "said@example.com"),
        ("key", "Tab"),
        ("text", "секрет 123"),
        ("key", "Backspace"),
        ("key", "Enter"),
        ("back",),
    ]
    assert session.outcome == "done"
    assert screen.calls[-1] == ("stop",)
    assert {"end": "done"} in socket.texts
    assert socket.closed


async def test_unknown_and_malformed_messages_are_ignored(clock: FakeClock) -> None:
    screen, socket = FakeScreen(), FakeSocket()
    session = _session(screen, clock)
    served = asyncio.create_task(session.serve(socket))
    # Written by the page (its load, rising numbers), but nothing the screen takes.
    junk: list[dict[str, Any]] = [
        {"t": "paste", "s": "x"},
        {"t": "down", "x": 1.5, "y": 0.5},
        {"t": "down", "x": -0.1, "y": 0.5},
        {"t": "down", "x": True, "y": 0.5},
        {"t": "down", "x": "0.5", "y": 0.5},
        {"t": "move", "x": 0.5},
        {"t": "up", "x": float("nan"), "y": 0.5},  # json writes NaN
        {"t": "wheel", "dy": "down"},
        {"t": "wheel", "dy": float("inf")},
        {"t": "text", "s": 42},
        {"t": "text", "s": ""},
        {"t": "key", "k": "Escape"},
        {"t": "key", "k": "enter"},
        {"x": 0.5, "y": 0.5},
    ]
    for message in junk:
        _send(socket, session, message)
    # Not the page's at all: no load and number. Dropped and counted as bad (four of ten).
    for plain in (b"not json", b"[1, 2]", b'"text"', b"\xff\xfe"):
        socket.feed(seal(session.key, plain))
    _send(socket, session, {"t": "text", "s": "дальше"})
    await until(lambda: len(screen.inputs) == 1)
    assert screen.inputs == [("text", "дальше")]
    assert session.outcome is None  # authentic junk is not a bad message
    _send(socket, session, {"t": "done"})
    await asyncio.wait_for(served, 3)


async def test_text_over_500_characters_is_ignored(clock: FakeClock) -> None:
    screen, socket = FakeScreen(), FakeSocket()
    session = _session(screen, clock)
    served = asyncio.create_task(session.serve(socket))
    _send(socket, session, {"t": "text", "s": "я" * 501})
    _send(socket, session, {"t": "text", "s": "я" * 500})
    _send(socket, session, {"t": "done"})
    await asyncio.wait_for(served, 3)
    assert screen.inputs == [("text", "я" * 500)]


async def test_the_time_running_out_ends_it_as_expired(clock: FakeClock) -> None:
    screen, socket = FakeScreen(), FakeSocket()
    session = _session(screen, clock)
    served = asyncio.create_task(session.serve(socket))
    await until(lambda: screen.on_frame is not None)
    clock.advance(timedelta(minutes=9, seconds=59))
    await asyncio.sleep(0.05)
    assert session.outcome is None
    clock.advance(timedelta(seconds=1))
    await asyncio.wait_for(served, 3)
    assert session.outcome == "expired"
    assert session.alive is False
    assert screen.calls[-1] == ("stop",)
    assert {"end": "expired"} in socket.texts
    assert socket.closed
    assert await asyncio.wait_for(session.wait(), 1) == "expired"


async def test_it_expires_even_when_the_page_is_never_opened(
    clock: FakeClock,
) -> None:
    screen = FakeScreen()
    session = _session(screen, clock)
    await session.start()
    waiting = asyncio.create_task(session.wait())
    await asyncio.sleep(0.05)
    assert not waiting.done()
    clock.advance(timedelta(minutes=10))
    assert await asyncio.wait_for(waiting, 3) == "expired"
    assert screen.calls == [("start", "https://www.olx.uz/account/"), ("stop",)]


async def test_ten_messages_that_do_not_open_end_it(clock: FakeClock) -> None:
    screen, socket = FakeScreen(), FakeSocket()
    session = _session(screen, clock)
    served = asyncio.create_task(session.serve(socket))
    stranger = RemoteKey.new()
    for _ in range(8):
        socket.feed(seal(stranger, b'{"t":"back"}'))
    socket.feed(b"short")  # the ninth
    _send(socket, session, {"t": "key", "k": "Enter"})
    await until(lambda: len(screen.inputs) == 1)
    assert session.outcome is None  # nine bad ones so far, then a good one
    socket.feed(seal(stranger, b'{"t":"back"}'))  # the tenth bad one
    _send(socket, session, {"t": "back"})  # never reaches the screen
    await asyncio.wait_for(served, 3)
    assert session.outcome == "stopped"
    assert screen.inputs == [("key", "Enter")]
    assert screen.calls[-1] == ("stop",)


async def test_a_second_socket_replaces_the_first(clock: FakeClock) -> None:
    screen, first, second = FakeScreen(), FakeSocket(), FakeSocket()
    session = _session(screen, clock)
    first_served = asyncio.create_task(session.serve(first))
    await until(lambda: screen.on_frame is not None)
    second_served = asyncio.create_task(session.serve(second))
    await asyncio.wait_for(first_served, 3)  # the old phone page is let go
    assert first.closed
    assert session.outcome is None
    assert [c[0] for c in screen.calls].count("start") == 1

    _send(second, session, {"t": "text", "s": "снова"})
    await until(lambda: len(screen.inputs) == 1)
    assert screen.inputs == [("text", "снова")]
    _send(second, session, {"t": "done"})
    await asyncio.wait_for(second_served, 3)
    assert session.outcome == "done"


async def test_a_dropped_socket_does_not_end_the_session(clock: FakeClock) -> None:
    screen, first, second = FakeScreen(), FakeSocket(), FakeSocket()
    session = _session(screen, clock)
    served = asyncio.create_task(session.serve(first))
    await until(lambda: screen.on_frame is not None)
    assert screen.on_frame is not None
    screen.on_frame(b"\xff\xd8 frame one")
    await until(lambda: len(first.frames) == 1)
    first.hang_up()
    await asyncio.wait_for(served, 3)
    assert session.outcome is None and session.alive

    served = asyncio.create_task(session.serve(second))
    await until(lambda: len(second.frames) == 1)  # the last picture, at once
    assert second.frames == [b"\xff\xd8 frame one"]
    await session.stop()
    await asyncio.wait_for(served, 3)
    assert session.outcome == "stopped"


async def test_frames_go_out_and_the_time_left_every_15_seconds(
    clock: FakeClock,
) -> None:
    screen, socket = FakeScreen(), FakeSocket()
    session = _session(screen, clock)
    served = asyncio.create_task(session.serve(socket))
    await until(lambda: screen.on_frame is not None)

    def lefts() -> list[dict[str, Any]]:
        return [t for t in socket.texts if "left" in t]

    await until(lambda: lefts() == [{"left": 600}])
    assert screen.on_frame is not None
    screen.on_frame(b"\xff\xd8 one")
    screen.on_frame(b"\xff\xd8 two")
    await until(lambda: socket.frames[-1:] == [b"\xff\xd8 two"])
    clock.advance(timedelta(seconds=14))
    await asyncio.sleep(0.05)
    assert lefts() == [{"left": 600}]
    clock.advance(timedelta(seconds=1))
    await until(lambda: lefts() == [{"left": 600}, {"left": 585}])
    await session.stop()
    await asyncio.wait_for(served, 3)


async def test_a_refused_navigation_stops_it(clock: FakeClock) -> None:
    screen, socket = FakeScreen(), FakeSocket()
    session = _session(screen, clock)
    served = asyncio.create_task(session.serve(socket))
    await until(lambda: screen.on_refused is not None)
    assert screen.on_refused is not None
    screen.on_refused()
    await asyncio.wait_for(served, 3)
    assert session.outcome == "stopped"
    assert screen.calls[-1] == ("stop",)


async def test_a_socket_after_the_end_is_closed_at_once(clock: FakeClock) -> None:
    screen, socket = FakeScreen(), FakeSocket()
    session = _session(screen, clock)
    await session.start()
    await session.stop()
    await asyncio.wait_for(session.serve(socket), 3)
    assert socket.closed
    assert socket.frames == []
    assert screen.calls.count(("stop",)) == 1


async def test_nothing_typed_no_token_and_no_key_reach_the_log(
    clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    screen, socket = FakeScreen(), FakeSocket()
    session = _session(screen, clock)
    served = asyncio.create_task(session.serve(socket))
    typed = "МойПароль-9431"
    _send(socket, session, {"t": "text", "s": typed})
    _send(socket, session, {"t": "key", "k": "Enter"})
    socket.feed(seal(RemoteKey.new(), json.dumps({"t": "text", "s": typed}).encode()))
    socket.feed(b'{"t":"text","s":"' + typed.encode() + b'"}')  # not sealed at all
    await until(lambda: len(screen.inputs) == 2)
    clock.advance(timedelta(minutes=3))
    _send(socket, session, {"t": "done"})
    await asyncio.wait_for(served, 3)

    records = caplog.records
    assert records, "the end of a session is logged"
    secrets = (
        typed,
        typed.encode().hex(),
        session.token,
        session.key.fragment(),
        session.key.raw.hex(),
    )
    for record in records:
        shown = record.getMessage() + repr(record.args) + repr(record.__dict__)
        for secret in secrets:
            assert secret not in shown
    ended = [r.getMessage() for r in records if "done" in r.getMessage()]
    assert ended and "180" in ended[0] and "3 " in ended[0]


# --- fix round 1: once and in order (F1), the focus (F2), notices (F3), a failed start (F4) ---


async def _served(
    clock: FakeClock,
) -> tuple[FakeScreen, FakeSocket, RemoteSession, asyncio.Task[None]]:
    screen, socket = FakeScreen(), FakeSocket()
    session = _session(screen, clock)
    served = asyncio.create_task(session.serve(socket))
    await until(lambda: screen.on_frame is not None)
    return screen, socket, session, served


async def test_a_replayed_message_does_not_reach_the_screen_twice(
    clock: FakeClock,
) -> None:
    screen, socket, session, served = await _served(clock)
    password = _send(socket, session, {"t": "text", "s": "P@ss"})
    _send(socket, session, {"t": "key", "k": "Tab"})
    socket.feed(password)  # the operator sends it again
    _send(socket, session, {"t": "back"})
    await until(lambda: len(screen.inputs) == 3)
    assert screen.inputs == [("text", "P@ss"), ("key", "Tab"), ("back",)]
    await session.stop()
    await asyncio.wait_for(served, 3)


async def test_ten_replays_end_the_session(clock: FakeClock) -> None:
    screen, socket, session, served = await _served(clock)
    sealed = _send(socket, session, {"t": "key", "k": "Enter"})
    for _ in range(10):
        socket.feed(sealed)
    await asyncio.wait_for(served, 3)
    assert session.outcome == "stopped"
    assert screen.inputs == [("key", "Enter")]


async def test_a_reordered_pair_keeps_only_the_newer(clock: FakeClock) -> None:
    screen, socket, session, served = await _served(clock)
    first = seal(session.key, _stamped(socket, {"t": "text", "s": "первое"}))
    second = seal(session.key, _stamped(socket, {"t": "text", "s": "второе"}))
    socket.feed(second)
    socket.feed(first)  # older than what was taken: dropped
    _send(socket, session, {"t": "back"})
    await until(lambda: len(screen.inputs) == 2)
    assert screen.inputs == [("text", "второе"), ("back",)]
    await session.stop()
    await asyncio.wait_for(served, 3)


async def test_a_message_of_another_load_or_without_a_number_is_dropped(
    clock: FakeClock,
) -> None:
    screen, socket, session, served = await _served(clock)
    _send(socket, session, {"t": "text", "s": "моё"})
    other = {"p": secrets.token_hex(16), "n": 99, "t": "text", "s": "чужое"}
    socket.feed(seal(session.key, json.dumps(other).encode()))
    for bad_load in ("", "xyz", 7, None):
        message = {"p": bad_load, "n": 100, "t": "back"}
        socket.feed(seal(session.key, json.dumps(message).encode()))
    for bad_number in (0, "5", True, 1.5):
        message = {"p": socket.load, "n": bad_number, "t": "back"}
        socket.feed(seal(session.key, json.dumps(message).encode()))
    _send(socket, session, {"t": "key", "k": "Tab"})  # nine bad so far: it still runs
    await until(lambda: len(screen.inputs) == 2)
    assert screen.inputs == [("text", "моё"), ("key", "Tab")]
    assert session.outcome is None
    await session.stop()
    await asyncio.wait_for(served, 3)


async def test_a_repeated_nonce_is_dropped(
    clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    screen, socket, session, served = await _served(clock)
    monkeypatch.setattr(remote, "_new_nonce", lambda: NONCE)
    _send(socket, session, {"t": "text", "s": "раз"})
    _send(socket, session, {"t": "text", "s": "два"})  # newer, but the same nonce
    monkeypatch.undo()
    _send(socket, session, {"t": "back"})
    await until(lambda: len(screen.inputs) == 2)
    assert screen.inputs == [("text", "раз"), ("back",)]
    await session.stop()
    await asyncio.wait_for(served, 3)


async def test_a_fresh_load_after_a_reconnect_works_and_the_old_load_is_dead(
    clock: FakeClock,
) -> None:
    screen, first, session, served = await _served(clock)
    _send(first, session, {"t": "text", "s": "до"})
    await until(lambda: len(screen.inputs) == 1)
    second = FakeSocket()  # the page reloaded: a new load, counting from 1 again
    served_again = asyncio.create_task(session.serve(second))
    await asyncio.wait_for(served, 3)
    old = {"p": first.load, "n": 50, "t": "text", "s": "старое"}
    second.feed(seal(session.key, json.dumps(old).encode()))  # an old load: dropped
    _send(second, session, {"t": "text", "s": "после"})
    await until(lambda: len(screen.inputs) == 2)
    assert screen.inputs == [("text", "до"), ("text", "после")]
    await session.stop()
    await asyncio.wait_for(served_again, 3)


def test_text_messages_are_padded_to_a_fixed_length() -> None:
    short = json.dumps({"p": "a" * 32, "n": 1, "t": "text", "s": "1"}).encode()
    longer = json.dumps(
        {"p": "a" * 32, "n": 2, "t": "text", "s": "длинный пароль 12345"},
        ensure_ascii=False,
    )
    padded_short, padded_long = remote.pad(short), remote.pad(longer.encode())
    assert len(padded_short) == len(padded_long) == 640
    assert json.loads(padded_long) == json.loads(longer)
    key = RemoteKey.new()
    assert len(seal(key, padded_short)) == len(seal(key, padded_long)) == 640 + 48
    # The page's JSON.stringify keeps Cyrillic as UTF-8, two bytes a letter.
    huge = json.dumps(
        {"p": "a" * 32, "n": 3, "t": "text", "s": "я" * 500}, ensure_ascii=False
    ).encode()
    assert len(remote.pad(huge)) == 2560


async def test_the_focused_field_is_reported_after_a_tap_and_a_key(
    clock: FakeClock,
) -> None:
    screen, socket, session, served = await _served(clock)
    screen.focused = "password"
    _send(socket, session, {"t": "down", "x": 0.5, "y": 0.5})
    _send(socket, session, {"t": "up", "x": 0.5, "y": 0.5})
    await until(lambda: {"focus": "password"} in socket.texts)
    screen.focused = "text"
    _send(socket, session, {"t": "key", "k": "Tab"})
    await until(lambda: {"focus": "text"} in socket.texts)
    focus = [t for t in socket.texts if "focus" in t]
    assert focus == [{"focus": "password"}, {"focus": "text"}]  # not after a press
    await session.stop()
    await asyncio.wait_for(served, 3)


async def test_a_notice_of_the_screen_reaches_the_page(clock: FakeClock) -> None:
    screen, socket, session, served = await _served(clock)
    assert screen.on_notice is not None
    notice = "Сайт открыл новое окно: показываю его здесь"
    screen.on_notice(notice)
    await until(lambda: {"notice": notice} in socket.texts)
    await session.stop()
    await asyncio.wait_for(served, 3)


async def test_the_page_is_told_which_site_the_tab_is_on(clock: FakeClock) -> None:
    # Review 4б.2, I2: at the start and after every move of the tab, before its picture.
    screen, socket = FakeScreen(), FakeSocket()
    screen.address = "https://www.olx.uz/account/"
    session = _session(screen, clock)  # the card's site: olx.uz
    served = asyncio.create_task(session.serve(socket))
    await until(lambda: {"site": "olx.uz", "other": False} in socket.texts)
    assert screen.on_frame is not None

    screen.address = "https://olx-uz.example/account/login"  # a link led it away
    screen.on_frame(b"\xff\xd8 look-alike")
    await until(lambda: socket.frames[-1:] == [b"\xff\xd8 look-alike"])
    sites = [t for t in socket.texts if "site" in t]
    assert sites[-1] == {"site": "olx-uz.example", "other": True}
    told = socket.sent.index(json.dumps(sites[-1], ensure_ascii=False))
    assert told < socket.sent.index(
        b"\xff\xd8 look-alike"
    )  # the name before the picture

    screen.address = "https://m.olx.uz/d/item"  # a part of the card's site is that site
    screen.on_frame(b"\xff\xd8 back")
    await until(lambda: {"site": "m.olx.uz", "other": False} in socket.texts)
    screen.on_frame(b"\xff\xd8 same site")  # told once per change, not per picture
    await until(lambda: socket.frames[-1:] == [b"\xff\xd8 same site"])
    assert len([t for t in socket.texts if "site" in t]) == 3
    await session.stop()
    await asyncio.wait_for(served, 3)


async def test_a_screen_that_cannot_start_ends_the_session(clock: FakeClock) -> None:
    screen = FakeScreen()
    screen.start_error = RuntimeError("no browser")
    session = _session(screen, clock)
    with pytest.raises(RuntimeError):
        await session.start()
    assert session.outcome == "stopped"
    assert await asyncio.wait_for(session.wait(), 0.5) == "stopped"


async def test_serve_with_a_screen_that_cannot_start_closes_the_socket(
    clock: FakeClock,
) -> None:
    screen, socket = FakeScreen(), FakeSocket()
    screen.start_error = RuntimeError("no browser")
    session = _session(screen, clock)
    await asyncio.wait_for(session.serve(socket), 3)
    assert socket.closed
    assert session.outcome == "stopped"


async def test_a_screen_error_carrying_the_typed_text_is_not_logged(
    clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    typed = "СекретныйПароль-777"
    screen, socket, session, served = await _served(clock)
    screen.text_error = RuntimeError(f"cannot insert {typed}")
    _send(socket, session, {"t": "text", "s": typed})
    _send(socket, session, {"t": "done"})
    await asyncio.wait_for(served, 3)
    assert any("RuntimeError" in r.getMessage() for r in caplog.records)
    for record in caplog.records:
        shown = record.getMessage() + repr(record.args) + repr(record.__dict__)
        assert typed not in shown


# --- CdpScreen on a fake page: a start that fails undoes what it attached (F4) ---------------


class _FakeCdp:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.handlers: dict[str, Any] = {}
        self.detached = False

    def on(self, event: str, handler: Any) -> None:
        self.handlers[event] = handler

    async def send(self, method: str, params: Any = None) -> None:
        self.sent.append(method)

    async def detach(self) -> None:
        self.detached = True


class _FakeContext:
    def __init__(self, page: _FakePage) -> None:
        self.page = page
        self.cdp = _FakeCdp()
        self.pages: list[_FakePage] = [page]

    async def new_cdp_session(self, page: Any) -> _FakeCdp:
        return self.cdp


class _FakePage:
    def __init__(self, goto_error: Exception | None = None) -> None:
        self.listeners: dict[str, list[Any]] = {}
        self.routes: list[Any] = []
        self.goto_error = goto_error
        self.closed = False
        self.url = "about:blank"
        self.main_frame = object()
        self.context = _FakeContext(self)

    async def set_viewport_size(self, size: dict[str, int]) -> None:
        pass

    def on(self, event: str, handler: Any) -> None:
        self.listeners.setdefault(event, []).append(handler)

    def remove_listener(self, event: str, handler: Any) -> None:
        self.listeners[event].remove(handler)

    async def route(self, pattern: str, handler: Any) -> None:
        self.routes.append(handler)

    async def unroute(self, pattern: str, handler: Any = None) -> None:
        self.routes.remove(handler)

    async def goto(self, url: str, **_: Any) -> None:
        if self.goto_error is not None and url != "about:blank":
            raise self.goto_error
        self.url = url

    async def opener(self) -> None:
        return None

    def is_closed(self) -> bool:
        return self.closed

    async def close(self) -> None:
        self.closed = True


class _FakeBrowser:
    def __init__(self, page: _FakePage) -> None:
        self.page = page

    async def tab(self, name: str) -> _FakePage:
        return self.page


async def test_a_screen_whose_page_does_not_open_undoes_what_it_attached() -> None:
    page = _FakePage(goto_error=TimeoutError("net::ERR_TIMED_OUT"))
    screen = CdpScreen(_FakeBrowser(page))  # type: ignore[arg-type]
    with pytest.raises(BrowserError):
        await screen.start("https://example.com/", lambda _: None, lambda: None)
    assert all(not handlers for handlers in page.listeners.values())
    assert page.routes == []
    assert page.context.cdp.detached
    assert page.closed  # the next session gets a fresh tab


async def test_stop_closes_the_remote_tab() -> None:
    page = _FakePage()
    screen = CdpScreen(_FakeBrowser(page))  # type: ignore[arg-type]
    await screen.start("https://example.com/", lambda _: None, lambda: None)
    assert page.routes and page.listeners["popup"] and page.listeners["download"]
    await screen.stop()
    assert page.closed
    assert page.routes == []
    assert page.context.cdp.detached
    await screen.stop()  # twice is harmless
