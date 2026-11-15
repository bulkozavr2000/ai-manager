"""The remote screen in a real installed browser (Edge) with a temporary profile, against small
pages served from this computer: tapping, typing and Tab through the DevTools protocol, the
pictures, going back, and an address the tab may not show. Skipped where Edge is not installed."""

from __future__ import annotations

import asyncio
import functools
import json
import os
import threading
from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from tg_agent.clock import FakeClock
from tg_agent.search import remote
from tg_agent.search.browser import AgentBrowser, BrowserError
from tg_agent.search.remote import REMOTE_TAB, CdpScreen, RemoteKey, RemoteSession
from tg_agent.search.remote_server import RemoteServer

PAGES = {
    "index.html": """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<title>Главная</title></head><body style="margin:0">
<a id="enter" href="/login.html" style="display:block;padding:40px;font-size:30px">Войти</a>
<button id="away" style="display:block;margin:40px;font-size:30px" onclick="
  location.href = window.URL.createObjectURL(new Blob(['<p>чужое</p>'], {type: 'text/html'}))
">Уйти</button>
<button id="pop" style="display:block;margin:40px;font-size:30px"
  onclick="window.open('/login.html', '_blank')">Окно</button>
<a id="file" download="note.txt" href="/login.html"
  style="display:block;margin:40px;font-size:30px">Скачать</a>
</body></html>""",
    "login.html": """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<title>Вход</title></head><body style="margin:0;font-size:24px">
<input id="login" name="login" style="display:block;margin:30px;width:300px;height:50px">
<input id="password" type="password" name="password"
  style="display:block;margin:30px;width:300px;height:50px">
<button id="go" style="display:block;margin:30px;width:300px;height:60px" onclick="
  document.getElementById('who').textContent = 'Вошли как ' +
    document.getElementById('login').value + ' / ' +
    document.getElementById('password').value.length
">Войти</button>
<p id="who"></p>
<form action="/login.html"><input id="q" name="login" style="margin:30px;width:300px"></form>
</body></html>""",
}


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass  # the test output stays quiet


@pytest.fixture
def site(tmp_path: Path) -> Iterator[str]:
    root = tmp_path / "site"
    root.mkdir()
    for name, html in PAGES.items():
        (root / name).write_text(html, encoding="utf-8")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_Quiet, directory=str(root))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
async def browser(tmp_path: Path) -> AsyncIterator[AgentBrowser]:
    # The browser the settings would name: Edge here, Playwright's Chromium on a server
    # (TG_AGENT_TEST_BROWSER=chromium). Never the agent's real profile.
    channel = os.environ.get("TG_AGENT_TEST_BROWSER", "msedge")
    agent = AgentBrowser(tmp_path / "profile", channel=channel)
    try:
        await agent.tab(REMOTE_TAB)
    except BrowserError:
        pytest.skip(f"the browser {channel!r} is not installed here")
    try:
        yield agent
    finally:
        await agent.close()


async def _centre(browser: AgentBrowser, element_id: str) -> tuple[float, float]:
    """Where an element is, as fractions of the tab, the way the phone page reports a tap."""
    page = await browser.tab(REMOTE_TAB)
    box: dict[str, float] = await page.evaluate(
        """(id) => { const r = document.getElementById(id).getBoundingClientRect();
        return {x: (r.left + r.width / 2) / innerWidth, y: (r.top + r.height / 2) / innerHeight}; }""",
        element_id,
    )
    return box["x"], box["y"]


async def _tap(screen: CdpScreen, browser: AgentBrowser, element_id: str) -> None:
    x, y = await _centre(browser, element_id)
    await screen.pointer("down", x, y)
    await screen.pointer("up", x, y)


async def _wait_for(check: Any, seconds: float = 10.0) -> None:
    for _ in range(int(seconds / 0.05)):
        if await check():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("the page never got there")


async def test_signing_in_through_the_remote_screen(
    site: str, browser: AgentBrowser
) -> None:
    frames: list[bytes] = []
    refused: list[bool] = []
    screen = CdpScreen(browser, allow_private=True)
    await screen.start(
        f"{site}/index.html", frames.append, lambda: refused.append(True)
    )
    page = await browser.tab(REMOTE_TAB)
    assert page.viewport_size == {"width": 412, "height": 780}

    await _tap(screen, browser, "enter")
    await _wait_for(lambda: _is(screen.url(), f"{site}/login.html"))
    await page.wait_for_load_state("domcontentloaded")

    await _tap(screen, browser, "login")
    await screen.text("said@пример.uz")
    await screen.key("Tab")
    await screen.text("пароль 123")
    await screen.key("Backspace")
    await _tap(screen, browser, "go")

    async def signed_in() -> bool:
        return "Вошли как said@пример.uz / 9" in await page.inner_text("#who")

    await _wait_for(signed_in)
    await _wait_for(lambda: _is(bool(frames), True))
    assert all(frame[:3] == b"\xff\xd8\xff" for frame in frames)  # JPEG pictures

    await screen.back()
    assert screen.url() == f"{site}/index.html"
    assert refused == []
    await screen.stop()
    assert page.is_closed()  # every session gets a fresh tab, with no history


async def test_enter_sends_a_form(site: str, browser: AgentBrowser) -> None:
    screen = CdpScreen(browser, allow_private=True)
    await screen.start(f"{site}/login.html", lambda _: None, lambda: None)
    await _tap(screen, browser, "q")
    await screen.text("abc")
    await screen.key("Enter")  # a form with one button sends itself on Enter
    await _wait_for(lambda: _is("login=abc" in screen.url(), True))
    await screen.stop()


async def test_an_address_the_tab_may_not_show_is_left_and_reported(
    site: str, browser: AgentBrowser
) -> None:
    refused: list[bool] = []
    screen = CdpScreen(browser, allow_private=True)
    await screen.start(
        f"{site}/index.html", lambda _: None, lambda: refused.append(True)
    )
    await _tap(screen, browser, "away")  # to a blob: page, which is not http(s)
    await _wait_for(lambda: _is(refused, [True]))
    assert screen.url() == "about:blank"
    await screen.stop()


async def test_this_computer_is_refused_by_default(
    site: str, browser: AgentBrowser
) -> None:
    with pytest.raises(BrowserError):
        await CdpScreen(browser).start(
            f"{site}/index.html", lambda _: None, lambda: None
        )


async def test_the_focused_field_is_known(site: str, browser: AgentBrowser) -> None:
    screen = CdpScreen(browser, allow_private=True)
    await screen.start(f"{site}/login.html", lambda _: None, lambda: None)
    assert await screen.focus() == "none"
    await _tap(screen, browser, "login")
    assert await screen.focus() == "text"
    await screen.key("Tab")
    assert await screen.focus() == "password"
    await _tap(screen, browser, "who")
    assert await screen.focus() == "none"
    await screen.stop()


async def test_a_popup_is_shown_in_the_remote_tab_and_no_second_page_stays(
    site: str, browser: AgentBrowser
) -> None:
    notices: list[str] = []
    screen = CdpScreen(browser, allow_private=True)
    await screen.start(
        f"{site}/index.html", lambda _: None, lambda: None, notices.append
    )
    page = await browser.tab(REMOTE_TAB)
    context = page.context
    await _tap(screen, browser, "pop")
    await _wait_for(lambda: _is(screen.url(), f"{site}/login.html"))
    await _wait_for(lambda: _is(len(context.pages), 1))
    assert context.pages == [page]
    assert notices == ["Сайт открыл новое окно: показываю его здесь"]

    await screen.back()
    await _tap(screen, browser, "pop")  # told once per session
    await _wait_for(lambda: _is(screen.url(), f"{site}/login.html"))
    await _wait_for(lambda: _is(len(context.pages), 1))
    assert len(notices) == 1
    await screen.stop()
    assert context.pages == []  # nothing of the session is left


async def test_a_download_is_cancelled(site: str, browser: AgentBrowser) -> None:
    screen = CdpScreen(browser, allow_private=True)
    await screen.start(f"{site}/index.html", lambda _: None, lambda: None)
    page = await browser.tab(REMOTE_TAB)
    async with page.expect_download() as started:
        await _tap(screen, browser, "file")
    download = await started.value
    # Cancelled here, or refused by the browser itself (it is launched with downloads off):
    # either way nothing is saved.
    assert await download.failure() is not None
    await screen.stop()


async def test_this_computer_and_the_home_network_are_never_requested(
    site: str, browser: AgentBrowser, tmp_path: Path
) -> None:
    hits: list[str] = []

    class _Counting(_Quiet):
        def do_GET(self) -> None:
            hits.append(self.path)
            super().do_GET()

    other = ThreadingHTTPServer(("127.0.0.1", 0), _Counting)
    threading.Thread(target=other.serve_forever, daemon=True).start()
    elsewhere = f"http://127.0.0.1:{other.server_address[1]}"
    (tmp_path / "site" / "jump.html").write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"></head><body>
<img src="{elsewhere}/pixel.png">
<iframe src="{elsewhere}/frame.html"></iframe>
<a id="jump" href="{elsewhere}/" style="display:block;padding:40px;font-size:30px">Туда</a>
</body></html>""",
        encoding="utf-8",
    )
    refused: list[bool] = []
    try:
        screen = CdpScreen(
            browser, allow_origins={site}
        )  # only the test site is let in
        await screen.start(
            f"{site}/jump.html", lambda _: None, lambda: refused.append(True)
        )
        await _tap(screen, browser, "jump")
        await _wait_for(lambda: _is(refused, [True]))
        await screen.stop()
        assert hits == []  # aborted before they were sent, not undone afterwards
    finally:
        other.shutdown()
        other.server_close()


async def _is(value: object, expected: object) -> bool:
    return value == expected


# --- the phone page, in the same real browser ------------------------------------------------


class _RecordingScreen:
    """A screen that only records: the page is what is tested here."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.on_frame: Any = None
        self.on_notice: Any = None
        self.focused = "password"

    async def start(
        self, url: str, on_frame: Any, on_refused: Any, on_notice: Any = None
    ) -> None:
        self.on_frame, self.on_notice = on_frame, on_notice

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
        return self.focused

    def url(self) -> str:
        return ""

    async def stop(self) -> None:
        self.calls.append(("stop",))


async def test_the_phone_page_seals_input_the_way_python_opens_it(
    browser: AgentBrowser, monkeypatch: pytest.MonkeyPatch
) -> None:
    screen = _RecordingScreen()
    clock = FakeClock(datetime(2026, 11, 11, 12, 0, tzinfo=ZoneInfo("Asia/Tashkent")))
    session = RemoteSession(
        "example.com", "https://example.com/", screen, clock, 10, poll_seconds=0.05
    )
    server = RemoteServer(0)
    await server.start()
    server.attach(session, origin=f"http://127.0.0.1:{server.port}")
    phone = await browser.tab("phone")
    problems: list[str] = []
    phone.on(
        "console", lambda m: problems.append(m.text) if m.type == "error" else None
    )
    try:
        await phone.set_viewport_size({"width": 390, "height": 844})
        await phone.goto(
            f"http://127.0.0.1:{server.port}/s/{session.token}#{session.key.fragment()}"
        )

        async def status_has(text: str) -> bool:
            return text in await phone.inner_text("#status")

        await _wait_for(lambda: status_has("Осталось 10:00"))

        # The page's own seal gives the Python vector byte for byte.
        key = RemoteKey(bytes(range(64)))
        nonce = bytes(range(100, 116))
        plain = '{"t":"text","s":"привет, мир — 42"}'
        monkeypatch.setattr(remote, "_new_nonce", lambda: nonce)
        expected = remote.seal(key, plain.encode())
        sealed = await phone.evaluate(
            """async ([fragment, plain, nonce]) => {
                const k = await importKeys(fromBase64Url(fragment));
                const s = await seal(k, new TextEncoder().encode(plain), Uint8Array.from(nonce));
                return Array.from(s); }""",
            [key.fragment(), plain, list(nonce)],
        )
        assert bytes(sealed) == expected
        monkeypatch.undo()

        # Texts of different lengths are padded to one length, as remote.pad pads them; the
        # page stamps its load and rising numbers.
        encoded = await phone.evaluate(
            """() => [encode({t: "text", s: "1"}), encode({t: "text", s: "пароль-подлиннее 123"}),
                      encode({t: "key", k: "Tab"})].map((b) => new TextDecoder().decode(b))"""
        )
        short, longer, key_message = (json.loads(e) for e in encoded)
        assert len(encoded[0].encode()) == len(encoded[1].encode()) == 640
        assert encoded[1].encode() == remote.pad(encoded[1].rstrip(" ").encode())
        assert short["p"] == longer["p"] == key_message["p"]
        assert len(short["p"]) == 32
        assert [short["n"], longer["n"], key_message["n"]] == [1, 2, 3]
        assert len(encoded[2]) < 640  # only text is padded

        # The field hides what is typed; «Показать» shows it, «Ввести» hides it again.
        assert await phone.get_attribute("#field", "type") == "password"
        await phone.fill("#field", "Пароль 1 — да")
        await phone.click("#show")
        assert await phone.get_attribute("#field", "type") == "text"
        assert await phone.inner_text("#show") == "Скрыть"
        # What he types reaches the screen, opened with the session's key.
        await phone.click("#put")
        await phone.click("#enter")
        await phone.click("#back")
        await _wait_for(lambda: _is(len(screen.calls), 3))
        assert screen.calls == [("text", "Пароль 1 — да"), ("key", "Enter"), ("back",)]
        assert await phone.input_value("#field") == ""
        assert await phone.get_attribute("#field", "type") == "password"
        assert await phone.inner_text("#show") == "Показать"

        # The key moved the focus: the page says where a text would go.
        async def target_has(text: str) -> bool:
            return text in await phone.inner_text("#target")

        await _wait_for(lambda: target_has("поле пароля"))
        before_notice = await phone.evaluate(
            "document.getElementById('screen').getBoundingClientRect().top"
        )
        screen.on_notice("Сайт открыл новое окно: показываю его здесь")

        async def notice_shown() -> bool:
            return "новое окно" in await phone.inner_text("#notice")

        await _wait_for(notice_shown)
        # The notice has its place: it does not push the picture down (fix round 2).
        assert before_notice == await phone.evaluate(
            "document.getElementById('screen').getBoundingClientRect().top"
        )

        # A picture is shown, and a tap on it is reported in fractions of the picture.
        picture = await phone.screenshot(
            type="jpeg", clip={"x": 0, "y": 0, "width": 206, "height": 390}
        )
        screen.on_frame(picture)
        await _wait_for(
            lambda: phone.evaluate(
                "document.getElementById('screen').naturalWidth === 206"
            )
        )
        box = await phone.evaluate(
            """() => { const r = document.getElementById('screen').getBoundingClientRect();
            const s = Math.min(r.width / 206, r.height / 390);
            return {left: r.left + (r.width - 206 * s) / 2, top: r.top, w: 206 * s, h: 390 * s}; }"""
        )
        await phone.mouse.click(
            box["left"] + box["w"] * 0.5, box["top"] + box["h"] * 0.25
        )
        await _wait_for(lambda: _is(len(screen.calls), 5))
        (_, down, x, y), (_, up, _x, _y) = screen.calls[3:5]
        assert (down, up) == ("down", "up")
        assert abs(x - 0.5) < 0.01 and abs(y - 0.25) < 0.01

        # A drag released off the picture is released at its last point inside.
        await phone.mouse.move(
            box["left"] + box["w"] * 0.2, box["top"] + box["h"] * 0.5
        )
        await phone.mouse.down()
        await phone.mouse.move(
            box["left"] + box["w"] * 0.9, box["top"] + box["h"] * 0.5, steps=5
        )
        await asyncio.sleep(0.1)
        await phone.mouse.move(box["left"] + box["w"] * 0.9, box["top"] + box["h"] + 40)
        await phone.mouse.up()
        await _wait_for(lambda: _is(screen.calls[-1][:2], ("pointer", "up")))
        moves = [c for c in screen.calls[5:] if c[1] == "move"]
        assert moves and screen.calls[-1][2:] == moves[-1][2:]
        assert screen.calls[-1][3] > 0.4  # not the top-left corner

        # Fix round 2: the picture never moves when the line under the field changes (two
        # lines for "none", one for "password"), or a tap aimed at a field lands below it.
        async def picture_top() -> float:
            return float(
                await phone.evaluate(
                    "document.getElementById('screen').getBoundingClientRect().top"
                )
            )

        tops = []
        for focused, words in (
            ("none", "не выбрано поле"),
            ("password", "поле пароля"),
            ("text", "обычное поле"),
            ("none", "не выбрано поле"),
        ):
            screen.focused = focused
            await phone.click("#enter")  # a key: the page is told the focus again
            await _wait_for(functools.partial(target_has, words))
            tops.append(await picture_top())
        assert len(set(tops)) == 1, tops

        await phone.click("#done")
        assert await asyncio.wait_for(session.wait(), 5) == "done"
        await _wait_for(lambda: status_has("Можно закрыть страницу"))
        assert not [p for p in problems if "Content Security Policy" in p]
    finally:
        await session.stop()
        await server.stop()
