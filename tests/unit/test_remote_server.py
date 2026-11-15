"""The page and the socket of a remote session, through aiohttp's test client (no browser)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp
import pytest
from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from tg_agent.clock import FakeClock
from tg_agent.search.remote import RemoteSession, open_sealed, seal
from tg_agent.search.remote_server import RemoteServer

START = datetime(2026, 11, 11, 12, 0, tzinfo=ZoneInfo("Asia/Tashkent"))
ORIGIN = "https://quiet-words.trycloudflare.com"  # the base the link was made with


class FakeScreen:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.on_frame: Callable[[bytes], None] | None = None

    async def start(
        self,
        url: str,
        on_frame: Callable[[bytes], None],
        on_refused: Callable[[], None],
        on_notice: Callable[[str], None] | None = None,
    ) -> None:
        self.on_frame = on_frame

    async def pointer(self, kind: str, x: float, y: float) -> None:
        self.calls.append(("pointer", kind, x, y))

    async def wheel(self, dy: float) -> None:
        self.calls.append(("wheel", dy))

    async def text(self, s: str) -> None:
        self.calls.append(("text", s))

    async def key(self, k: str) -> None:
        self.calls.append(("key", k))

    async def back(self) -> None:
        self.calls.append(("back",))

    async def focus(self) -> str:
        return "none"

    def url(self) -> str:
        return ""

    async def stop(self) -> None:
        self.calls.append(("stop",))


class PageLoad:
    """What one load of the page writes: its random id and rising numbers."""

    def __init__(self, session: RemoteSession) -> None:
        self.session = session
        self.load = secrets.token_hex(16)
        self.number = 0

    def seal(self, message: dict[str, Any]) -> bytes:
        self.number += 1
        plain = {"p": self.load, "n": self.number, **message}
        return seal(self.session.key, json.dumps(plain).encode())


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(START)


@pytest.fixture
def screen() -> FakeScreen:
    return FakeScreen()


@pytest.fixture
def session(screen: FakeScreen, clock: FakeClock) -> RemoteSession:
    return RemoteSession(
        "olx.uz", "https://www.olx.uz/", screen, clock, 10, poll_seconds=0.01
    )


@pytest.fixture
async def client(session: RemoteSession) -> AsyncIterator[TestClient[Any, Any]]:
    server = RemoteServer(0)
    server.attach(session, origin=ORIGIN + "/")
    test_client = TestClient(TestServer(server.app))
    await test_client.start_server()
    try:
        yield test_client
    finally:
        await session.stop()
        await test_client.close()


async def test_the_policy_names_the_socket_s_own_address(
    session: RemoteSession,
) -> None:
    # Task 4б.2: old Safari does not read 'self' as covering WebSockets.
    server = RemoteServer(0)
    for origin, socket in (
        (ORIGIN + "/", "wss://quiet-words.trycloudflare.com"),
        ("https://agent.example:8443", "wss://agent.example:8443"),
        ("http://127.0.0.1:5123", "ws://127.0.0.1:5123"),
    ):
        server.attach(session, origin=origin)
        test_client = TestClient(TestServer(server.app))
        await test_client.start_server()
        try:
            response = await test_client.get(f"/s/{session.token}")
            policy = response.headers["Content-Security-Policy"]
        finally:
            await test_client.close()
        assert f"connect-src 'self' {socket};" in policy, policy
    await session.stop()


async def _is_404(client: TestClient[Any, Any], path: str, method: str = "GET") -> None:
    response = await client.request(method, path)
    assert response.status == 404, path
    assert await response.read() == b"", path


async def test_the_page_is_served_for_the_live_token(
    client: TestClient[Any, Any], session: RemoteSession
) -> None:
    response = await client.get(f"/s/{session.token}")
    assert response.status == 200
    assert response.content_type == "text/html"
    headers = response.headers
    policy = headers["Content-Security-Policy"]
    assert "default-src 'none'" in policy
    assert "img-src blob: data:" in policy
    assert "connect-src 'self'" in policy
    assert re.search(r"script-src 'sha256-[A-Za-z0-9+/=]+'(;|$)", policy)
    assert re.search(r"style-src 'sha256-[A-Za-z0-9+/=]+'(;|$)", policy)
    assert "unsafe" not in policy
    assert headers["Cache-Control"] == "no-store"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers["X-Content-Type-Options"] == "nosniff"

    page = await response.text()
    assert "Готово" in page and "Ввести" in page and "Назад" in page
    assert session.key.fragment() not in page
    assert session.token not in page
    assert not re.search(r"https?://", page)  # no outside address
    # No attribute loads or sends anything anywhere.
    assert not re.search(r"\s(src|href|action)\s*=\s*[\"']", page)
    assert "crypto.subtle" in page


async def test_the_typed_text_is_hidden_by_default(
    client: TestClient[Any, Any], session: RemoteSession
) -> None:
    page = await (await client.get(f"/s/{session.token}")).text()
    field = re.search(r"<input id=\"field\"[^>]*>", page)
    assert field is not None
    for attribute in (
        'type="password"',
        'autocomplete="off"',
        'autocapitalize="off"',
        'autocorrect="off"',
        'spellcheck="false"',
    ):
        assert attribute in field.group(0)
    assert "Показать" in page


async def test_the_bar_shows_which_site_the_tab_is_on(
    client: TestClient[Any, Any], session: RemoteSession
) -> None:
    # Review 4б.2, I2: «Сайт: olx.uz» at the top of the bar, red for another site.
    page = await (await client.get(f"/s/{session.token}")).text()
    bar = page.split('<div id="bar">', 1)[1]
    assert bar.lstrip().startswith('<div id="site"></div>')  # first in the bar
    assert "Сайт: " in page and " — это другой сайт" in page
    assert "message.site" in page and "message.other" in page
    assert re.search(r"#site\.other\s*\{[^}]*color", page)  # drawn in red


async def test_every_other_address_is_an_empty_404(
    client: TestClient[Any, Any], session: RemoteSession
) -> None:
    token = session.token
    for path in (
        "/",
        "/s",
        "/s/",
        f"/s/{token[:-1]}",
        f"/s/{token}x",
        f"/s/{token}/",
        f"/s/{token}/other",
        f"/s/{token[:-1]}/ws",
        f"/x/{token}",
        "/favicon.ico",
    ):
        await _is_404(client, path)
    await _is_404(client, f"/s/{token}", method="POST")


async def test_no_page_after_detach(session: RemoteSession) -> None:
    server = RemoteServer(0)
    server.attach(session, origin=ORIGIN)
    async with TestClient(TestServer(server.app)) as test_client:
        assert (await test_client.get(f"/s/{session.token}")).status == 200
        server.detach()
        await _is_404(test_client, f"/s/{session.token}")
        await _is_404(test_client, f"/s/{session.token}/ws")


async def test_no_page_after_the_session_ends_or_expires(
    client: TestClient[Any, Any], session: RemoteSession, clock: FakeClock
) -> None:
    clock.advance(timedelta(minutes=10))
    await _is_404(client, f"/s/{session.token}")
    clock.advance(timedelta(minutes=-5))
    assert (await client.get(f"/s/{session.token}")).status == 200
    await session.stop()
    await _is_404(client, f"/s/{session.token}")
    await _is_404(client, f"/s/{session.token}/ws")


async def test_the_socket_of_another_origin_is_refused(
    client: TestClient[Any, Any], session: RemoteSession
) -> None:
    path = f"/s/{session.token}/ws"
    for headers in ({}, {"Origin": "https://evil.example"}, {"Origin": "null"}):
        with pytest.raises(aiohttp.WSServerHandshakeError) as refused:
            await client.ws_connect(path, headers=headers)
        assert refused.value.status == 403
    response = await client.get(path, headers={"Origin": "https://evil.example"})
    assert response.status == 403 and await response.read() == b""
    assert session.outcome is None


async def test_the_socket_pumps_frames_out_and_input_in(
    client: TestClient[Any, Any], session: RemoteSession, screen: FakeScreen
) -> None:
    page = PageLoad(session)
    async with client.ws_connect(
        f"/s/{session.token}/ws", headers={"Origin": ORIGIN}
    ) as ws:
        first = await asyncio.wait_for(ws.receive(), 3)
        assert first.type == WSMsgType.TEXT
        assert json.loads(first.data) == {"left": 600}

        assert screen.on_frame is not None
        screen.on_frame(b"\xff\xd8\xff a picture")
        frame = await asyncio.wait_for(ws.receive(), 3)
        assert frame.type == WSMsgType.BINARY
        assert frame.data == b"\xff\xd8\xff a picture"

        await ws.send_str('{"t":"text","s":"plain text is ignored"}')
        message = page.seal({"t": "text", "s": "said"})
        await ws.send_bytes(message)
        await ws.send_bytes(page.seal({"t": "down", "x": 0.5, "y": 0.25}))
        for _ in range(300):
            if len(screen.calls) == 2:
                break
            await asyncio.sleep(0.01)
        assert screen.calls == [("text", "said"), ("pointer", "down", 0.5, 0.25)]

        await ws.send_bytes(page.seal({"t": "done"}))
        end = await asyncio.wait_for(ws.receive(), 3)
        assert end.type == WSMsgType.TEXT and json.loads(end.data) == {"end": "done"}
        closed = await asyncio.wait_for(ws.receive(), 3)
        assert closed.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.CLOSING)
    assert session.outcome == "done"
    assert open_sealed(session.key, message) is not None


async def test_an_oversized_message_closes_the_socket_only(
    client: TestClient[Any, Any], session: RemoteSession
) -> None:
    async with client.ws_connect(
        f"/s/{session.token}/ws", headers={"Origin": ORIGIN}
    ) as ws:
        await ws.receive()  # the time left
        await ws.send_bytes(b"x" * 100_000)
        for _ in range(5):
            message = await asyncio.wait_for(ws.receive(), 3)
            if message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                break
        else:
            raise AssertionError("the socket stayed open")
    assert session.outcome is None  # the page may be opened again


async def test_the_server_listens_on_this_computer_only_and_logs_no_token(
    session: RemoteSession, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    server = RemoteServer(0)
    await server.start()
    try:
        assert server.port > 0
        assert server.host == "127.0.0.1"
        server.attach(session, origin=ORIGIN)
        base = f"http://127.0.0.1:{server.port}"
        async with aiohttp.ClientSession() as http:
            async with http.get(f"{base}/s/{session.token}") as response:
                assert response.status == 200
            async with http.get(f"{base}/s/{session.token}nope") as response:
                assert response.status == 404
    finally:
        await server.stop()
        await session.stop()
    await server.stop()  # twice is harmless
    for record in caplog.records:
        assert session.token not in record.getMessage() + repr(record.__dict__)
