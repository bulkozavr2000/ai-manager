"""The remote screen: one tab of the agent's browser, shown on the owner's phone.

Some sites show contacts only after a sign-in. Саид works from his phone, so he signs in himself,
in the bot's own browser: a one-time link opens a page (`search.remote_server`) that shows the
picture of one tab and passes his taps and typing back to it. The sign-in then stays in the
agent's browser profile. Nobody else drives that tab: not a model, not the search (R47).

What he types is nobody's (R46). On the way from the phone every input message is sealed with a
key that travels only in the link's fragment (`#…`, which browsers never send to a server); here
it is opened, handed to the tab, and forgotten: never logged, stored or shown to a model. The
cipher uses the standard library only and the page does the same with `crypto.subtle`; both
sides must agree byte for byte:

    plaintext         = JSON {"p": <page load id>, "n": <1, 2, 3 …>, "t": …}
                        (a text message padded with spaces to 640 or 2560 bytes)
    keystream block i = HMAC-SHA256(enc_key, nonce ‖ uint32_be(i))
    ciphertext        = plaintext XOR keystream
    tag               = HMAC-SHA256(mac_key, nonce ‖ ciphertext)
    message           = nonce (16) ‖ ciphertext ‖ tag (32)

A message is taken once and in order: a nonce seen before, a number not above the last one, or
another page load is dropped.

What this gives, and what it does not. It hides what he types from a tunnel operator who only
reads the traffic (the operator ends the TLS and sees the pictures, but not the typed text, nor
its length beyond "short or long"). An operator who changes the traffic is not stopped: he
serves the page itself and can serve one that reads the key from the address. So the way in
must be one the owner trusts, or his own server with its own certificate.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hashlib
import hmac
import json
import logging
import math
import re
import secrets
import struct
from collections.abc import Callable, Collection
from datetime import datetime, timedelta
from typing import Any, Protocol
from urllib.parse import urlsplit

from tg_agent.clock import Clock
from tg_agent.search.browser import AgentBrowser, BrowserError
from tg_agent.search.sites import on_site, site_of, url_problem

_log = logging.getLogger(__name__)

KEY_SIZE = 64  # the first 32 bytes encrypt, the last 32 authenticate
NONCE_SIZE = 16
TAG_SIZE = 32
TEXT_MAX = 500  # characters of one text message
BAD_LIMIT = 10  # messages that do not open before the session ends
TICK_SECONDS = 15  # how often the page is told the time left

REMOTE_TAB = "remote"
VIEWPORT = (412, 780)  # a phone-size tab
JPEG_QUALITY = 60
FRAME_MAX_WIDTH = 720

DONE, EXPIRED, STOPPED = "done", "expired", "stopped"
POINTER_KINDS = frozenset({"down", "move", "up"})
KEYS = frozenset({"Enter", "Backspace", "Tab"})
FOCUS_KINDS = frozenset({"password", "text", "none"})
# A text message is padded with spaces to the first size it fits, so its length does not tell
# how long the typed text is (only whether it is longer than about 270 Cyrillic letters).
PAD_SIZES = (640, 2560)
_LOAD = re.compile(r"^[0-9a-f]{16,64}$")  # the page's random id of one page load
POPUP_NOTICE = "Сайт открыл новое окно: показываю его здесь"

# The kind of the focused element: "frame" when the focus is inside a frame of the page.
_FOCUS_SCRIPT = """() => {
  const e = document.activeElement;
  if (!e) return "none";
  const tag = e.tagName;
  if (tag === "IFRAME" || tag === "FRAME") return "frame";
  if (tag === "INPUT") {
    const type = (e.type || "text").toLowerCase();
    if (type === "password") return "password";
    return ["text", "email", "tel", "number", "search", "url"].includes(type) ? "text" : "none";
  }
  if (tag === "TEXTAREA" || e.isContentEditable) return "text";
  return "none";
}"""


def _origin_of(url: str) -> str:
    try:
        parts = urlsplit(url.strip())
        return f"{parts.scheme.lower()}://{parts.netloc.lower()}"
    except ValueError:
        return ""


# --- the cipher ------------------------------------------------------------------------------


class RemoteKey:
    """The 64 random bytes of one session; shown only in the link's fragment."""

    __slots__ = ("raw",)

    def __init__(self, raw: bytes) -> None:
        if len(raw) != KEY_SIZE:
            raise ValueError("a remote key is 64 bytes")
        self.raw = bytes(raw)

    @classmethod
    def new(cls) -> RemoteKey:
        return cls(secrets.token_bytes(KEY_SIZE))

    @classmethod
    def from_fragment(cls, fragment: str) -> RemoteKey:
        """The key back from `fragment()`; ValueError for anything else."""
        if len(fragment) != 86 or not fragment.isascii():
            raise ValueError("not a remote key")
        try:
            raw = base64.urlsafe_b64decode(fragment + "==")
        except (binascii.Error, ValueError):
            raise ValueError("not a remote key") from None
        if base64.urlsafe_b64encode(raw).rstrip(b"=").decode() != fragment:
            raise ValueError("not a remote key")  # stray characters base64 skipped
        return cls(raw)

    def fragment(self) -> str:
        """URL-safe base64 without padding: what goes after `#` in the link."""
        return base64.urlsafe_b64encode(self.raw).rstrip(b"=").decode()

    @property
    def enc_key(self) -> bytes:
        return self.raw[:32]

    @property
    def mac_key(self) -> bytes:
        return self.raw[32:]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RemoteKey) and hmac.compare_digest(self.raw, other.raw)

    def __hash__(self) -> int:
        return hash(self.raw)

    def __repr__(self) -> str:
        return "RemoteKey(…)"  # never the bytes: a repr can end up in a log


def _new_nonce() -> bytes:
    return secrets.token_bytes(NONCE_SIZE)


def _keystream_xor(key: RemoteKey, nonce: bytes, data: bytes) -> bytes:
    out = bytearray(len(data))
    for block in range((len(data) + 31) // 32):
        stream = hmac.new(
            key.enc_key, nonce + struct.pack(">I", block), hashlib.sha256
        ).digest()
        start = block * 32
        for offset, byte in enumerate(data[start : start + 32]):
            out[start + offset] = byte ^ stream[offset]
    return bytes(out)


def _tag(key: RemoteKey, nonce: bytes, cipher: bytes) -> bytes:
    return hmac.new(key.mac_key, nonce + cipher, hashlib.sha256).digest()


def seal(key: RemoteKey, plaintext: bytes) -> bytes:
    """nonce ‖ ciphertext ‖ tag, as the page sends its input messages."""
    nonce = _new_nonce()
    cipher = _keystream_xor(key, nonce, plaintext)
    return nonce + cipher + _tag(key, nonce, cipher)


def pad(plain: bytes) -> bytes:
    """A text message's JSON padded with spaces, as the page pads it before sealing."""
    for size in PAD_SIZES:
        if len(plain) <= size:
            return plain + b" " * (size - len(plain))
    return plain


def open_sealed(key: RemoteKey, message: bytes) -> bytes | None:
    """The plaintext of a sealed message, or None when it is too short or its tag is wrong."""
    if len(message) < NONCE_SIZE + TAG_SIZE:
        return None
    nonce = message[:NONCE_SIZE]
    cipher = message[NONCE_SIZE:-TAG_SIZE]
    if not hmac.compare_digest(_tag(key, nonce, cipher), message[-TAG_SIZE:]):
        return None
    return _keystream_xor(key, nonce, cipher)


# --- the screen ------------------------------------------------------------------------------


class Screen(Protocol):
    """One browser tab as the session drives it; a fake in tests."""

    async def start(
        self,
        url: str,
        on_frame: Callable[[bytes], None],
        on_refused: Callable[[], None],
        on_notice: Callable[[str], None] | None = None,
    ) -> None:
        """Open `url` and start sending JPEG pictures of the tab to `on_frame`; `on_refused` is
        called when the tab went to an address it may not show (it is already left);
        `on_notice` gets a sentence for the owner (a site opened a new window)."""
        ...

    async def pointer(self, kind: str, x: float, y: float) -> None:
        """A press ("down"), a move or a release ("up") at fractions 0..1 of the picture."""
        ...

    async def wheel(self, dy: float) -> None: ...

    async def text(self, s: str) -> None: ...

    async def key(self, k: str) -> None: ...

    async def back(self) -> None: ...

    async def focus(self) -> str:
        """What a typed text would go into: "password", "text" or "none"."""
        ...

    def url(self) -> str: ...

    async def stop(self) -> None: ...


class CdpScreen:
    """`Screen` on the agent's browser: the tab `remote`, phone-size, through Chrome's DevTools
    protocol (a screencast out, mouse and keyboard events in).

    The tab is guarded (R47): a request to an address `sites.url_problem` refuses (this
    computer, the home network) is aborted before it is sent, in the page and in its frames; a
    navigation the tab may not show ends the session; a new window a site opens is closed and
    its address shown in this tab instead; a download is cancelled. `stop()` closes the tab and
    every window opened from it, so the next session starts on a fresh tab with no history.
    """

    def __init__(
        self,
        browser: AgentBrowser,
        *,
        tab: str = REMOTE_TAB,
        viewport: tuple[int, int] = VIEWPORT,
        allow_private: bool = False,
        allow_origins: Collection[str] = (),
    ) -> None:
        self._browser = browser
        self._tab = tab
        self._width, self._height = viewport
        self._allow_private = allow_private  # tests serve pages from this computer
        # Tests: origins on this computer that may be shown though others there may not.
        self._allow_origins = frozenset(_origin_of(o) for o in allow_origins)
        self._page: Any = None
        self._cdp: Any = None
        self._routed = False
        self._on_frame: Callable[[bytes], None] | None = None
        self._on_refused: Callable[[], None] | None = None
        self._on_notice: Callable[[str], None] | None = None
        self._popup_told = False
        self._last = (viewport[0] / 2, viewport[1] / 2)
        self._tasks: set[asyncio.Task[Any]] = set()
        self._stopped = False

    def _problem(self, url: str) -> str | None:
        if _origin_of(url) in self._allow_origins:
            return None
        return url_problem(url, allow_private=self._allow_private)

    async def start(
        self,
        url: str,
        on_frame: Callable[[bytes], None],
        on_refused: Callable[[], None],
        on_notice: Callable[[str], None] | None = None,
    ) -> None:
        problem = self._problem(url)
        if problem:
            raise BrowserError(problem)
        self._on_frame, self._on_refused = on_frame, on_refused
        self._on_notice = on_notice
        try:
            page = await self._browser.tab(self._tab)
            self._page = page
            await page.set_viewport_size({"width": self._width, "height": self._height})
            self._cdp = await page.context.new_cdp_session(page)
            self._cdp.on("Page.screencastFrame", self._frame)
            await page.route("**/*", self._guard)
            self._routed = True
            page.on("framenavigated", self._navigated)
            page.on("popup", self._popup)
            page.on("download", self._download)
            await page.goto(url, wait_until="domcontentloaded")
            await self._cdp.send(
                "Page.startScreencast",
                {
                    "format": "jpeg",
                    "quality": JPEG_QUALITY,
                    "maxWidth": FRAME_MAX_WIDTH,
                    "maxHeight": FRAME_MAX_WIDTH * 2,
                    "everyNthFrame": 1,
                },
            )
        except Exception as exc:  # noqa: BLE001 - a page that does not open, a closed browser
            await self.stop()  # nothing it attached is left behind for a later session
            raise BrowserError(
                f"страница не открылась ({type(exc).__name__})"
            ) from None

    def _spawn(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _frame(self, params: dict[str, Any]) -> None:
        # Every frame is acknowledged, or Chrome stops sending them.
        if self._cdp is not None and not self._stopped:
            self._spawn(self._ack(params.get("sessionId")))
        try:
            data = base64.b64decode(params.get("data", ""))
        except (binascii.Error, ValueError):
            return
        if data and self._on_frame is not None and not self._stopped:
            self._on_frame(data)

    async def _ack(self, session_id: Any) -> None:
        with contextlib.suppress(Exception):  # the tab is closing
            await self._cdp.send("Page.screencastFrameAck", {"sessionId": session_id})

    async def _guard(self, route: Any, request: Any) -> None:
        """Every request of the tab and its frames: one to a refused address is never sent."""
        if not self._problem(request.url):
            with contextlib.suppress(Exception):
                await route.continue_()
            return
        with contextlib.suppress(Exception):
            await route.abort("blockedbyclient")
        page = self._page
        with contextlib.suppress(Exception):
            if request.is_navigation_request() and request.frame == page.main_frame:
                self._refuse()  # the tab itself was sent there: as a refused navigation

    def _navigated(self, frame: Any) -> None:
        page = self._page
        if page is None or self._stopped or frame != page.main_frame:
            return
        url = frame.url
        # A blank tab and Chrome's own error page (a link that did not open) are no address.
        if url == "about:blank" or url.startswith("chrome-error:"):
            return
        if self._problem(url):
            self._refuse()

    def _refuse(self) -> None:
        if not self._stopped:
            self._spawn(self._leave())

    async def _leave(self) -> None:
        with contextlib.suppress(Exception):
            await self._page.goto("about:blank")
        if self._on_refused is not None:
            self._on_refused()

    def _popup(self, popup: Any) -> None:
        self._spawn(self._take_popup(popup))

    async def _take_popup(self, popup: Any) -> None:
        """A window a site opened (a «Войти через Google» popup, a target=_blank link): it is
        not on the phone's picture, so it is closed and its address opened in this tab."""
        url = ""
        with contextlib.suppress(Exception):
            url = popup.url
            if not url or url == "about:blank":  # a window opened first and sent later
                await popup.wait_for_url(
                    lambda u: u != "about:blank", wait_until="commit", timeout=3000
                )
                url = popup.url
        with contextlib.suppress(Exception):
            await popup.close()
        if self._stopped or not url or url == "about:blank":
            return
        if self._problem(url):
            self._refuse()
            return
        if not self._popup_told and self._on_notice is not None:
            self._popup_told = True
            self._on_notice(POPUP_NOTICE)
        with contextlib.suppress(Exception):
            await self._page.goto(url, wait_until="domcontentloaded")

    def _download(self, download: Any) -> None:
        self._spawn(self._cancel(download))

    async def _cancel(self, download: Any) -> None:
        with contextlib.suppress(Exception):
            await download.cancel()

    def _point(self, x: float, y: float) -> tuple[float, float]:
        return x * self._width, y * self._height

    async def pointer(self, kind: str, x: float, y: float) -> None:
        px, py = self._point(x, y)
        self._last = (px, py)
        event = {"down": "mousePressed", "move": "mouseMoved", "up": "mouseReleased"}[
            kind
        ]
        params: dict[str, Any] = {"type": event, "x": px, "y": py, "button": "left"}
        if kind == "move":
            params["buttons"] = 1  # a drag: the button is held
        else:
            params["clickCount"] = 1
        await self._cdp.send("Input.dispatchMouseEvent", params)

    async def wheel(self, dy: float) -> None:
        px, py = self._last
        await self._cdp.send(
            "Input.dispatchMouseEvent",
            {"type": "mouseWheel", "x": px, "y": py, "deltaX": 0, "deltaY": dy},
        )

    async def text(self, s: str) -> None:
        await self._cdp.send("Input.insertText", {"text": s})

    async def key(self, k: str) -> None:
        code = {"Enter": 13, "Backspace": 8, "Tab": 9}[k]
        down: dict[str, Any] = {
            "type": "rawKeyDown",
            "key": k,
            "code": k,
            "windowsVirtualKeyCode": code,
        }
        if k == "Enter":  # with its text, so a form is sent the way a keyboard sends it
            down.update(type="keyDown", text="\r", unmodifiedText="\r")
        await self._cdp.send("Input.dispatchKeyEvent", down)
        await self._cdp.send(
            "Input.dispatchKeyEvent",
            {"type": "keyUp", "key": k, "code": k, "windowsVirtualKeyCode": code},
        )

    async def back(self) -> None:
        await self._page.go_back(wait_until="domcontentloaded")

    async def focus(self) -> str:
        """What a typed text would go into, looked up in the frame that holds the focus."""
        page = self._page
        if page is None:
            return "none"
        try:
            kind = await page.main_frame.evaluate(_FOCUS_SCRIPT)
        except Exception:  # noqa: BLE001 - the page is changing
            return "none"
        if kind != "frame":
            return kind if kind in FOCUS_KINDS else "none"
        for frame in page.frames[1:]:  # the focus is inside a frame: find the one
            with contextlib.suppress(Exception):
                inner = await frame.evaluate(_FOCUS_SCRIPT)
                if inner in ("password", "text"):
                    return str(inner)
        return "none"

    def url(self) -> str:
        return "" if self._page is None else str(self._page.url)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        page, cdp = self._page, self._cdp
        if page is not None:
            for event, handler in (
                ("framenavigated", self._navigated),
                ("popup", self._popup),
                ("download", self._download),
            ):
                with contextlib.suppress(Exception):
                    page.remove_listener(event, handler)
            if self._routed:
                with contextlib.suppress(Exception):
                    await page.unroute("**/*", self._guard)
        if cdp is not None:
            with contextlib.suppress(Exception):
                await cdp.send("Page.stopScreencast")
            with contextlib.suppress(Exception):
                await cdp.detach()
        if page is not None:
            # Every window opened from the tab, then the tab: nothing of the session stays.
            with contextlib.suppress(Exception):
                for other in list(page.context.pages):
                    if other is not page and await other.opener() == page:
                        await other.close()
            with contextlib.suppress(Exception):
                await page.close()
        for task in list(self._tasks):
            if task is not asyncio.current_task():
                task.cancel()


# --- the session -----------------------------------------------------------------------------


class Socket(Protocol):
    """The phone page's connection: binary messages in (sealed input), frames and text out."""

    async def send_bytes(self, data: bytes) -> None: ...

    async def send_str(self, data: str) -> None: ...

    async def close(self) -> None: ...

    def __aiter__(self) -> Any: ...


def _fraction(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if 0.0 <= number <= 1.0 else None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class RemoteSession:
    """One remote sign-in: a token for the link, a key for its fragment, `minutes` to live.

    `serve(socket)` pumps the tab's pictures to the phone page and its input to the tab until
    «Готово», the time is up, or the socket ends (the page may be reopened: a second socket
    replaces the first). `outcome` is None while it runs, then `done`, `expired` or `stopped`.
    """

    def __init__(
        self,
        site: str,
        start_url: str,
        screen: Screen,
        clock: Clock,
        minutes: int = 10,
        *,
        poll_seconds: float = 0.5,
    ) -> None:
        self.site = site
        self.start_url = start_url
        self.token = secrets.token_urlsafe(32)
        self.key = RemoteKey.new()
        self._screen = screen
        self._clock = clock
        self._poll = poll_seconds
        self.started_at: datetime = clock.now()
        self.expires_at: datetime = self.started_at + timedelta(minutes=minutes)
        self.outcome: str | None = None
        self._started = False
        self._socket: Socket | None = None
        self._frame: bytes | None = None
        self._frame_number = 0
        self._frame_ready = asyncio.Event()
        self._ended = asyncio.Event()
        self._inputs = 0
        self._bad = 0
        self._nonces: set[bytes] = set()  # every nonce taken, for the session's life
        self._load: str | None = None  # the page load the current socket speaks for
        self._last_number = 0
        self._dead_loads: set[str] = set()
        self._watcher: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

    def __repr__(self) -> str:
        return f"RemoteSession({self.site!r}, outcome={self.outcome!r})"  # no token, no key

    @property
    def alive(self) -> bool:
        return self.outcome is None and self._clock.now() < self.expires_at

    def seconds_left(self) -> int:
        left = (self.expires_at - self._clock.now()).total_seconds()
        return max(0, math.ceil(left))

    async def start(self) -> None:
        """Open the start page in the tab; BrowserError when it cannot be opened."""
        if self._started:
            return
        self._started = True
        try:
            await self._screen.start(
                self.start_url, self._on_frame, self._on_refused, self._on_notice
            )
        except BaseException:
            await self._finish(STOPPED)  # so wait() returns and the way in gets closed
            raise
        self._watcher = asyncio.create_task(self._watch())

    async def wait(self) -> str:
        """Until the session ends; its outcome."""
        await self._ended.wait()
        assert self.outcome is not None
        return self.outcome

    async def stop(self) -> None:
        """End it now (the program stops)."""
        await self._finish(STOPPED)

    async def serve(self, socket: Socket) -> None:
        if self.outcome is None and not self._started:
            with contextlib.suppress(
                Exception
            ):  # it ends the session, which is seen below
                await self.start()
        if self.outcome is not None:
            with contextlib.suppress(Exception):
                await socket.close()
            return
        previous, self._socket = self._socket, socket
        # A new socket speaks for a new page load; the old load is never taken again.
        if self._load is not None:
            self._dead_loads.add(self._load)
        self._load, self._last_number = None, 0
        if previous is not None:  # the phone reconnected: the old page is let go
            with contextlib.suppress(Exception):
                await previous.close()
        pump = asyncio.create_task(self._pump(socket))
        read = asyncio.create_task(self._read(socket))
        ended = asyncio.create_task(self._ended.wait())
        try:
            await asyncio.wait({read, ended}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (pump, read, ended):
                task.cancel()
            await asyncio.gather(pump, read, ended, return_exceptions=True)
            if self._socket is socket:
                self._socket = None
                with contextlib.suppress(Exception):
                    await socket.close()

    # -- inside --

    def _spawn(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_frame(self, data: bytes) -> None:
        self._frame = data
        self._frame_number += 1
        self._frame_ready.set()

    def _on_refused(self) -> None:
        self._spawn(self._finish(STOPPED))

    def _on_notice(self, text: str) -> None:
        self._spawn(self._tell({"notice": text}))

    async def _tell(self, message: dict[str, Any]) -> None:
        socket = self._socket
        if socket is not None and self.outcome is None:
            with contextlib.suppress(Exception):
                await socket.send_str(json.dumps(message, ensure_ascii=False))

    async def _watch(self) -> None:
        while self.outcome is None:
            if self._clock.now() >= self.expires_at:
                self._spawn(self._finish(EXPIRED))
                return
            await asyncio.sleep(self._poll)

    async def _pump(self, socket: Socket) -> None:
        """Pictures and the time left out to one socket while it is the current one."""
        sent_frame = 0
        last_tick: datetime | None = None
        told_site = ""
        try:
            told_site = await self._tell_site(socket, told_site)  # once at the start
            while self.outcome is None and self._socket is socket:
                now = self._clock.now()
                if last_tick is None or now - last_tick >= timedelta(
                    seconds=TICK_SECONDS
                ):
                    last_tick = now
                    await socket.send_str(json.dumps({"left": self.seconds_left()}))
                self._frame_ready.clear()
                if self._frame is not None and self._frame_number != sent_frame:
                    sent_frame = self._frame_number
                    # The tab may have moved: its site is told before its picture.
                    told_site = await self._tell_site(socket, told_site)
                    await socket.send_bytes(self._frame)
                    continue
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._frame_ready.wait(), self._poll)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the socket is gone; the reader sees its end
            return

    async def _tell_site(self, socket: Socket, told: str) -> str:
        """Which site the tab is on, when it changed (review 4б.2, I2): the bar shows it, in
        red when it is not the card's site. Nothing is refused: «Войти через Google» is a
        move to another site too, and the owner now sees where he is typing."""
        site = site_of(self._screen.url())
        if not site or site == told:
            return told
        other = not on_site(site, frozenset({self.site}))
        await socket.send_str(
            json.dumps({"site": site, "other": other}, ensure_ascii=False)
        )
        return site

    async def _read(self, socket: Socket) -> None:
        try:
            async for message in socket:
                if self.outcome is not None or self._socket is not socket:
                    return
                if not isinstance(message, bytes | bytearray):
                    continue
                accepted = self._accept(bytes(message))
                if accepted is None:
                    self._bad += 1
                    if self._bad >= BAD_LIMIT:
                        await self._finish(STOPPED)
                        return
                    continue
                self._inputs += 1
                await self._handle(accepted)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a broken connection ends this socket only
            return

    def _accept(self, sealed: bytes) -> dict[str, Any] | None:
        """The input message if it opens, is new, and comes from the current page load in
        order; None for anything else (it counts as bad). A replayed or reordered message is
        refused: its nonce was seen, or its number is not above the last one taken."""
        plain = open_sealed(self.key, sealed)
        if plain is None:
            return None
        nonce = sealed[:NONCE_SIZE]
        if nonce in self._nonces:
            return None
        self._nonces.add(nonce)
        try:
            message = json.loads(plain.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None
        if not isinstance(message, dict):
            return None
        load, number = message.get("p"), message.get("n")
        if (
            not isinstance(load, str)
            or not _LOAD.match(load)
            or load in self._dead_loads
        ):
            return None
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            return None
        if self._load is None:
            self._load = load  # the first message of a socket names its load
        if load != self._load or number <= self._last_number:
            return None
        self._last_number = number
        return message

    async def _handle(self, message: dict[str, Any]) -> None:
        kind = message.get("t")
        screen = self._screen
        try:
            if kind in POINTER_KINDS:
                x, y = _fraction(message.get("x")), _fraction(message.get("y"))
                if x is not None and y is not None:
                    await screen.pointer(kind, x, y)
                    if kind == "up":
                        await self._report_focus()
            elif kind == "wheel":
                dy = _finite(message.get("dy"))
                if dy is not None:
                    await screen.wheel(max(-5000.0, min(5000.0, dy)))
            elif kind == "text":
                s = message.get("s")
                if isinstance(s, str) and 0 < len(s) <= TEXT_MAX:
                    await screen.text(s)
            elif kind == "key":
                k = message.get("k")
                if k in KEYS:
                    await screen.key(k)
                    await self._report_focus()
            elif kind == "back":
                await screen.back()
            elif kind == "done":
                await self._finish(DONE)
        except Exception as error:  # noqa: BLE001 - one input the tab did not take
            # Only the kind of error: its text could carry what was typed.
            _log.warning("remote input not applied: %s", type(error).__name__)

    async def _report_focus(self) -> None:
        """Tell the page where a typed text would go, so a password is not typed into a plain
        field by mistake."""
        try:
            focus = await self._screen.focus()
        except Exception:  # noqa: BLE001 - the page is changing
            focus = "none"
        await self._tell({"focus": focus if focus in FOCUS_KINDS else "none"})

    async def _finish(self, outcome: str) -> None:
        if self.outcome is not None:
            return
        self.outcome = outcome
        seconds = int((self._clock.now() - self.started_at).total_seconds())
        _log.info(
            "remote session ended: %s after %d s, %d input messages",
            outcome,
            seconds,
            self._inputs,
        )
        watcher = self._watcher
        if watcher is not None and watcher is not asyncio.current_task():
            watcher.cancel()
        socket, self._socket = self._socket, None
        if socket is not None:
            with contextlib.suppress(Exception):
                await socket.send_str(json.dumps({"end": outcome}))
            with contextlib.suppress(Exception):
                await socket.close()
        if self._started:
            with contextlib.suppress(Exception):
                await self._screen.stop()
        self._ended.set()
