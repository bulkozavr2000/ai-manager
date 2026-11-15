"""The page and the socket of a remote session (see `search.remote`).

A small aiohttp server on this computer only (127.0.0.1); a tunnel (`search.tunnel`) brings the
owner's phone to it. Two addresses exist while a session lives: `/s/<token>` (the page) and
`/s/<token>/ws` (its WebSocket). Anything else, a wrong token, a session that ended or expired:
404 with an empty body, so a guessed or old link tells nothing. The token is never logged (the
access log is off), and the key never reaches the server at all: the page reads it from the
link's fragment. The socket opens only for a page of the link's own origin (403 otherwise).

The tunnel's operator serves this page to the phone. One who only reads the traffic cannot read
what is typed (it is sealed); one who changes it can serve another page and is not stopped by
anything here, the Content-Security-Policy included, since it travels through him too. The way
in must be one the owner trusts, or his own server.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import re
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

from aiohttp import WSMsgType, web

from tg_agent.search.remote import RemoteSession

DEFAULT_PORT = 8765
MESSAGE_MAX = (
    16 * 1024
)  # one sealed input message is far smaller: 500 characters at most

_PATH = re.compile(r"^/s/([A-Za-z0-9_-]{16,128})(/ws)?$")

# The page: one file, Russian, no outside resource. Its script seals every input message
# exactly as `remote.seal` does (HMAC-SHA256 through crypto.subtle):
#   nonce (16 random bytes) ‖ plaintext XOR keystream ‖ HMAC(mac_key, nonce ‖ ciphertext) (32)
#   keystream block i = HMAC(enc_key, nonce ‖ uint32 big-endian i)
_STYLE = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: #111; color: #eee;
  font: 16px/1.35 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
body { display: flex; flex-direction: column; height: 100vh; height: 100dvh; }
#bar { flex: none; padding: 8px; background: #1e1e20; border-bottom: 1px solid #333; }
#typing { display: flex; gap: 6px; margin: 0 0 6px; }
#field { flex: 1; min-width: 0; font-size: 16px; padding: 9px 10px; border-radius: 8px;
  border: 1px solid #444; background: #111; color: #eee; }
.keys { display: flex; gap: 6px; flex-wrap: wrap; }
button { font: inherit; padding: 8px 12px; border-radius: 8px; border: 1px solid #444;
  background: #2b2b2e; color: #eee; touch-action: manipulation; }
button:active { background: #3a3a3e; }
#done { margin-left: auto; background: #1f6f3f; border-color: #2a8a50; }
#status { margin-top: 6px; font-size: 14px; color: #aaa; }
#target { margin: 0 0 6px; font-size: 14px; color: #ddd; min-height: 2.7em; }
#site { margin: 0 0 6px; font-size: 18px; font-weight: 600; color: #eee; }
#site.other { color: #ff5c5c; }
/* #target and #notice: a place of two lines each, kept when empty, so the picture below
   never moves when their text changes and a tap aimed at a field lands on that field. */
#notice { font-size: 14px; color: #f0c674; min-height: 2.7em; }
#site:empty { display: none; }
#screen { flex: 1; min-height: 0; width: 100%; object-fit: contain; object-position: top;
  touch-action: none; user-select: none; -webkit-user-select: none; display: block; }
"""

_SCRIPT = """
"use strict";
const $ = (id) => document.getElementById(id);
const picture = $("screen"), field = $("field"), statusLine = $("status");
let keys = null, ws = null, chain = Promise.resolve(), left = null, over = false;
let shown = null, pressed = false, lastMove = 0, lastPoint = { x: 0.5, y: 0.5 };
// This page load's random id and its message numbers: the session takes a message once, in
// order, and only from the load it serves, so a message sent again is never typed twice.
const LOAD = Array.from(crypto.getRandomValues(new Uint8Array(16)),
  (b) => b.toString(16).padStart(2, "0")).join("");
let number = 0;
// A text message is padded with spaces to the first size it fits (remote.pad in Python), so
// its length does not tell how long the typed text is.
const PAD_SIZES = [640, 2560];
const TARGETS = {
  password: "Текст пойдёт в поле пароля",
  text: "Текст пойдёт в обычное поле (его видно на картинке)",
  none: "Сейчас не выбрано поле ввода: нажмите на него на картинке",
};

function say(text) { statusLine.textContent = text; }

function fromBase64Url(s) {
  s = s.replace(/-/g, "+").replace(/_/g, "/");
  while (s.length % 4) s += "=";
  const bin = atob(s), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function importKeys(raw) {
  const how = { name: "HMAC", hash: "SHA-256" };
  return {
    enc: await crypto.subtle.importKey("raw", raw.slice(0, 32), how, false, ["sign"]),
    mac: await crypto.subtle.importKey("raw", raw.slice(32, 64), how, false, ["sign"]),
  };
}

async function hmac(key, data) {
  return new Uint8Array(await crypto.subtle.sign("HMAC", key, data));
}

// nonce (16) | ciphertext | tag (32), as remote.seal in Python.
async function seal(k, plain, nonce) {
  nonce = nonce || crypto.getRandomValues(new Uint8Array(16));
  const cipher = new Uint8Array(plain.length);
  for (let i = 0; i * 32 < plain.length; i++) {
    const block = new Uint8Array(20);
    block.set(nonce);
    new DataView(block.buffer).setUint32(16, i, false);
    const stream = await hmac(k.enc, block);
    for (let j = 0; j < 32 && i * 32 + j < plain.length; j++) {
      cipher[i * 32 + j] = plain[i * 32 + j] ^ stream[j];
    }
  }
  const signed = new Uint8Array(16 + cipher.length);
  signed.set(nonce);
  signed.set(cipher, 16);
  const tag = await hmac(k.mac, signed);
  const out = new Uint8Array(16 + cipher.length + 32);
  out.set(signed);
  out.set(tag, 16 + cipher.length);
  return out;
}

function pad(plain) {
  for (const size of PAD_SIZES) {
    if (plain.length <= size) {
      const out = new Uint8Array(size).fill(32);
      out.set(plain);
      return out;
    }
  }
  return plain;
}

// The bytes the page seals for one input message: {"p": load, "n": number, ...}.
function encode(message) {
  number += 1;
  const plain = new TextEncoder().encode(
    JSON.stringify(Object.assign({ p: LOAD, n: number }, message)));
  return message.t === "text" ? pad(plain) : plain;
}

function send(message) {
  if (!keys || over) return;
  const plain = encode(message);
  // One after another: a press must reach the tab before its release.
  chain = chain.then(async () => {
    const sealed = await seal(keys, plain);
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(sealed);
  }).catch(() => {});
}

function where(event) {
  const r = picture.getBoundingClientRect();
  const w0 = picture.naturalWidth, h0 = picture.naturalHeight;
  if (!w0 || !h0) return null;
  const scale = Math.min(r.width / w0, r.height / h0);
  const w = w0 * scale, h = h0 * scale;
  const x = (event.clientX - r.left - (r.width - w) / 2) / w;
  const y = (event.clientY - r.top) / h;
  if (x < 0 || x > 1 || y < 0 || y > 1) return null;
  return { x: x, y: y };
}

picture.addEventListener("pointerdown", (event) => {
  const p = where(event);
  if (!p) return;
  event.preventDefault();
  pressed = true;
  lastPoint = p;
  picture.setPointerCapture(event.pointerId);
  send({ t: "down", x: p.x, y: p.y });
});
picture.addEventListener("pointermove", (event) => {
  if (!pressed || event.timeStamp - lastMove < 40) return;
  const p = where(event);
  if (!p) return;
  lastMove = event.timeStamp;
  lastPoint = p;
  send({ t: "move", x: p.x, y: p.y });
});
function release(event) {
  if (!pressed) return;
  pressed = false;
  const p = where(event) || lastPoint; // released off the picture: at its last point inside
  send({ t: "up", x: p.x, y: p.y });
}
picture.addEventListener("pointerup", release);
picture.addEventListener("pointercancel", release);
picture.addEventListener("wheel", (event) => {
  event.preventDefault();
  send({ t: "wheel", dy: event.deltaY });
}, { passive: false });

$("typing").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = field.value;
  if (!text) return;
  send({ t: "text", s: text.slice(0, 500) });
  field.value = "";
  hideField();
});
function hideField() {
  field.type = "password";
  $("show").textContent = "Показать";
}
$("show").addEventListener("click", () => {
  const hidden = field.type === "password";
  field.type = hidden ? "text" : "password";
  $("show").textContent = hidden ? "Скрыть" : "Показать";
});
$("erase").addEventListener("click", () => send({ t: "key", k: "Backspace" }));
$("enter").addEventListener("click", () => send({ t: "key", k: "Enter" }));
$("up").addEventListener("click", () => send({ t: "wheel", dy: -400 }));
$("down").addEventListener("click", () => send({ t: "wheel", dy: 400 }));
$("back").addEventListener("click", () => send({ t: "back" }));
$("done").addEventListener("click", () => send({ t: "done" }));

function showLeft() {
  if (over || left === null) return;
  const m = Math.floor(left / 60), s = left % 60;
  say("Осталось " + m + ":" + (s < 10 ? "0" : "") + s);
}
setInterval(() => {
  if (left !== null && left > 0) left -= 1;
  showLeft();
}, 1000);

const ENDS = {
  done: "Готово. Можно закрыть страницу.",
  expired: "Время вышло. Можно закрыть страницу.",
  stopped: "Сеанс закончен. Можно закрыть страницу.",
};

// Which site the tab is on; red when it is not the site of the bot's card.
function showSite(site, other) {
  const line = $("site");
  line.textContent = "Сайт: " + site + (other ? " — это другой сайт" : "");
  line.classList.toggle("other", other);
}

function finish(outcome) {
  over = true;
  say(ENDS[outcome] || ENDS.stopped);
  document.body.classList.add("over");
}

function connect() {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(scheme + "//" + location.host + location.pathname + "/ws");
  ws.binaryType = "arraybuffer";
  ws.onopen = () => say("Соединено");
  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      let message = null;
      try { message = JSON.parse(event.data); } catch (e) { return; }
      if (typeof message.left === "number") { left = Math.max(0, Math.floor(message.left)); showLeft(); }
      if (typeof message.end === "string") finish(message.end);
      if (typeof message.focus === "string") $("target").textContent = TARGETS[message.focus] || "";
      if (typeof message.notice === "string") $("notice").textContent = message.notice;
      if (typeof message.site === "string") showSite(message.site, message.other === true);
      return;
    }
    const next = URL.createObjectURL(new Blob([event.data], { type: "image/jpeg" }));
    const old = shown;
    shown = next;
    picture.src = next;
    if (old) setTimeout(() => URL.revokeObjectURL(old), 1000);
  };
  ws.onclose = () => { if (!over) say("Связь прервалась. Обновите страницу."); };
}

async function begin() {
  if (!window.crypto || !crypto.subtle) {
    say("Этот браузер не подходит: откройте ссылку в Chrome или Safari.");
    return;
  }
  let raw = null;
  try { raw = fromBase64Url(location.hash.slice(1)); } catch (e) { raw = null; }
  if (!raw || raw.length !== 64) {
    say("Ссылка неполная: откройте её целиком из сообщения бота.");
    return;
  }
  keys = await importKeys(raw);
  say("Соединяюсь…");
  connect();
}
begin();
"""

_PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<meta name="referrer" content="no-referrer">
<title>Вход на сайт</title>
<style>{style}</style>
</head>
<body>
<div id="bar">
  <div id="site"></div>
  <form id="typing" autocomplete="off">
    <input id="field" type="password" maxlength="500" placeholder="Текст для поля на сайте"
      autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false">
    <button id="show" type="button">Показать</button>
    <button id="put" type="submit">Ввести</button>
  </form>
  <div id="target"></div>
  <div class="keys">
    <button id="erase" type="button" aria-label="Стереть">⌫</button>
    <button id="enter" type="button">Enter</button>
    <button id="up" type="button" aria-label="Прокрутить вверх">▲</button>
    <button id="down" type="button" aria-label="Прокрутить вниз">▼</button>
    <button id="back" type="button">Назад</button>
    <button id="done" type="button">Готово</button>
  </div>
  <div id="status">Загружаю…</div>
  <div id="notice"></div>
</div>
<img id="screen" alt="Страница сайта">
<script>{script}</script>
</body>
</html>
"""


def _hash(source: str) -> str:
    return (
        "'sha256-"
        + base64.b64encode(hashlib.sha256(source.encode()).digest()).decode()
        + "'"
    )


PAGE = _PAGE.format(style=_STYLE, script=_SCRIPT).encode("utf-8")


def _policy(socket: str = "") -> str:
    """The page's Content-Security-Policy. `socket` names the WebSocket's own address beside
    'self' (`wss://<host of the link's base>`): old Safari does not read 'self' as covering
    WebSockets (Task 4б.2)."""
    return "; ".join(
        (
            "default-src 'none'",
            f"script-src {_hash(_SCRIPT)}",
            f"style-src {_hash(_STYLE)}",
            "img-src blob: data:",
            f"connect-src 'self' {socket}".rstrip(),
            "base-uri 'none'",
            "form-action 'none'",
            "frame-ancestors 'none'",
        )
    )


def _socket_source(origin: str) -> str:
    """`wss://host[:port]` of an https base (`ws://` of a plain-http one, in tests); "" for
    anything else."""
    try:
        parts = urlsplit(origin)
    except ValueError:
        return ""
    scheme = {"https": "wss", "http": "ws"}.get(parts.scheme)
    if (
        scheme is None
        or not parts.netloc
        or not re.fullmatch(r"[\w.:\-\[\]]+", parts.netloc)
    ):
        return ""
    return f"{scheme}://{parts.netloc}"


PAGE_HEADERS = {
    "Content-Security-Policy": _policy(),
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}


class _PageSocket:
    """aiohttp's WebSocket as the session sees a socket: binary messages in."""

    def __init__(self, ws: web.WebSocketResponse) -> None:
        self._ws = ws

    async def send_bytes(self, data: bytes) -> None:
        await self._ws.send_bytes(data)

    async def send_str(self, data: str) -> None:
        await self._ws.send_str(data)

    async def close(self) -> None:
        await self._ws.close()

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._messages()

    async def _messages(self) -> AsyncIterator[bytes]:
        async for message in self._ws:  # ends with the connection
            if message.type == WSMsgType.BINARY:
                yield message.data
            elif message.type == WSMsgType.ERROR:
                return


def _not_found() -> web.Response:
    return web.Response(status=404, body=b"")


def _origin(value: str) -> str:
    return value.strip().rstrip("/").lower()


class RemoteServer:
    """The page server of the one remote session that may live at a time (R45)."""

    def __init__(self, port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> None:
        self.host = host
        self._port = port
        self._session: RemoteSession | None = None
        self._origin = ""
        self._headers = PAGE_HEADERS
        self._runner: web.AppRunner | None = None
        self.app = web.Application()
        self.app.router.add_route("*", "/{tail:.*}", self._route)

    @property
    def port(self) -> int:
        """The port it listens on (the one chosen by the system when 0 was asked)."""
        if self._runner is not None:
            for address in self._runner.addresses:
                if isinstance(address, tuple) and len(address) >= 2:
                    return int(address[1])
        return self._port

    async def start(self) -> None:
        if self._runner is not None:
            return
        # No access log: every line of it would carry the token.
        runner = web.AppRunner(self.app, access_log=None, handle_signals=False)
        await runner.setup()
        try:
            await web.TCPSite(runner, self.host, self._port).start()
        except BaseException:
            await runner.cleanup()
            raise
        self._runner = runner

    async def stop(self) -> None:
        runner, self._runner = self._runner, None
        if runner is not None:
            with contextlib.suppress(Exception):
                await runner.cleanup()

    def attach(self, session: RemoteSession, *, origin: str) -> None:
        """Serve `session`; its socket only for a page of `origin`, the public base the link
        was made with (`https://….trycloudflare.com`)."""
        self._session = session
        self._origin = _origin(origin)
        self._headers = {
            **PAGE_HEADERS,
            "Content-Security-Policy": _policy(_socket_source(self._origin)),
        }

    def detach(self) -> None:
        self._session = None
        self._origin = ""
        self._headers = PAGE_HEADERS

    async def _route(self, request: web.Request) -> web.StreamResponse:
        session = self._session
        match = _PATH.match(request.path)
        if (
            session is None
            or match is None
            or request.method != "GET"
            or not hmac.compare_digest(match.group(1).encode(), session.token.encode())
            or not session.alive
        ):
            return _not_found()
        if match.group(2) is None:
            return web.Response(
                body=PAGE,
                content_type="text/html",
                charset="utf-8",
                headers=self._headers,
            )
        # Only the page itself may open the socket: not another site, not a stray client.
        origin = request.headers.get("Origin", "")
        if not origin or not hmac.compare_digest(_origin(origin), self._origin):
            return web.Response(status=403, body=b"")
        ws = web.WebSocketResponse(max_msg_size=MESSAGE_MAX, heartbeat=30)
        if not ws.can_prepare(request).ok:
            return _not_found()
        await ws.prepare(request)
        await session.serve(_PageSocket(ws))
        return ws
