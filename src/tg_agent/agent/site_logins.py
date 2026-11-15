"""The owner signs in to a site from his phone, in the bot's own browser (stage 4б, R45–R51).

A task's search meets a site that shows contacts only after a sign-in, or behind a captcha
(`search/agent.Wall`). The `research` tool hands the walls of its search to `met`: the site is
remembered with the tasks that met it, and — when a way in from the internet is configured,
no sign-in is live and the site was not offered today — the owner gets one card (R48).
«Войти» opens the site's page in the remote tab, the page server on 127.0.0.1 and the way in
(`search/tunnel`), and sends him one message whose URL button carries the link; he signs in
on the phone and presses «Готово». The server and the tunnel are closed on every way out.
After «Готово» the site is one of the signed-in sites the search may roam, and every task in
work that met the wall gets a reminder for its agent (R49).

Nothing typed, no link, no token and no key reaches a log, the database, the chat log or a
model: the link exists only in the URL button (Telegram does not fetch a button's address),
and only counts and outcomes are logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import sys
from collections.abc import Callable, Sequence
from pathlib import PurePath, PureWindowsPath
from typing import Any, Protocol

from tg_agent.agent.approvals import ALREADY_DECIDED, UNKNOWN_BUTTON
from tg_agent.clock import Clock
from tg_agent.config import Settings
from tg_agent.control.base import Card, CardGone
from tg_agent.control.cards import WAITING_MARK
from tg_agent.control.layout import ChatLayout
from tg_agent.db import NotFound, Repo
from tg_agent.models import EventKind, TaskStatus
from tg_agent.search.agent import Wall
from tg_agent.search.remote import RemoteSession, Screen
from tg_agent.search.remote_server import RemoteServer
from tg_agent.search.tunnel import Tunnel, TunnelError, tunnel_from

_log = logging.getLogger(__name__)

# kv: the walls met, {site: {"url", "kind", "tasks": [task ids]}}; the signed-in sites (a
# JSON list); the day a site was last offered; the cards' one-time codes; the running
# tunnel's process, so that one a crash left behind is ended at the next start.
WALLS_KEY = "site_walls"
SIGNED_IN_KEY = "signed_in_sites"
OFFER_SEQ_KEY = "site_offer_seq"
TUNNEL_KEY = "remote_tunnel"


def offered_key(site: str) -> str:
    return f"site_offered:{site}"


def offer_key(number: int) -> str:
    return f"site_offer:{number}"


LOGIN_CARD = (
    f"{WAITING_MARK} Сайт {{site}} показывает контакты только после входа. Войти с"
    " телефона? Вход останется в браузере бота: дальше он сам увидит там номера."
)
CAPTCHA_CARD = f"{WAITING_MARK} Сайт {{site}} просит пройти проверку «я не робот». Пройти с телефона?"
SIGN_IN = "Войти"
NOT_NOW = "Не сейчас"
FINE = "Хорошо"
LINK_TEXT = (
    "Открой страницу и войди. Ссылка работает {minutes}; когда закончишь — нажми"
    " «Готово» на странице."
)
OPEN_PAGE = "Открыть страницу входа"
LINK_CLOSED = "Страница входа на {site} закрыта."
LINK_BELOW = "Вход на {site}: ссылка ниже"
NOT_NOW_LINE = "Вход на {site}: не сейчас"
OPENING = "Открываю страницу входа: ссылка придёт через несколько секунд."
ALREADY_OPEN = "Вход уже открыт: ссылка в сообщении выше"
OTHER_OPEN = "Сейчас открыт вход на {site}: закончи его и нажми «Войти» ещё раз."
TURNED_OFF = "Вход с телефона выключен в настройках."
SAVED = "Вход на {site} сохранён."
EXPIRED = "Время вышло: вход на {site} не завершён. Нажми «Войти» ещё раз, если нужно."
FAILED = "Не получилось открыть страницу входа: {why}"
NO_BROWSER = "браузер бота не открыл эту страницу."
PORT_BUSY = "порт {port} на компьютере занят другой программой."
BROKE_OFF = "сеанс прервался раньше времени."
PROGRAM_STOPPED = "программа остановилась."
UNEXPECTED = "что-то сломалось ({error})."
# What the agent of a task that met the wall hears once the owner signed in (R49).
SIGNED_IN_EVENT = (
    "Владелец вошёл на сайт {site}: поиск теперь видит там контакты. Если людей не"
    " хватает — поищи ещё раз."
)


def _minutes(count: int) -> str:
    """«1 минуту», «3 минуты», «10 минут»."""
    if count % 10 == 1 and count % 100 != 11:
        word = "минуту"
    elif count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        word = "минуты"
    else:
        word = "минут"
    return f"{count} {word}"


def _sentence(text: str) -> str:
    """A tunnel's error as the end of «Не получилось открыть страницу входа: …»."""
    text = " ".join(text.split()).strip()
    if not text:
        return BROKE_OFF
    text = text[0].lower() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


class Processes(Protocol):
    """The computer's processes, for the clean-up after a crash."""

    def name(self, pid: int) -> str | None:
        """The program file of a running process, or None when there is none."""
        ...

    def terminate(self, pid: int) -> None: ...


class SystemProcesses:
    """`Processes` on this computer, with the standard library only."""

    def name(self, pid: int) -> str | None:
        if sys.platform == "win32":
            return _windows_image(pid)
        try:
            return os.readlink(f"/proc/{pid}/exe")
        except OSError:
            return None

    def terminate(self, pid: int) -> None:
        import signal

        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)  # TerminateProcess on Windows


def _windows_image(pid: int) -> str | None:
    """The full path of a process's program file, or None (no such process, no access)."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    query_limited_information = 0x1000
    handle = kernel.OpenProcess(query_limited_information, False, pid)
    if not handle:
        return None
    try:
        size = wintypes.DWORD(1024)
        buffer = ctypes.create_unicode_buffer(size.value)
        kernel.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        if not kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return None
        return buffer.value
    finally:
        kernel.CloseHandle(handle)


def _program_name(path: str) -> str:
    """ "C:\\x\\cloudflared.exe" -> "cloudflared"."""
    name = PureWindowsPath(path).name if "\\" in path else PurePath(path).name
    return name.casefold().removesuffix(".exe")


class _Failed(Exception):
    """The sign-in could not be opened; the text ends «Не получилось открыть …: »."""


class SiteLogins:
    """The sign-in walls the searches met, the cards they bring, and the one live sign-in.

    `screen` makes the remote tab (`search/remote.CdpScreen` on the agent's browser; None
    when there is no browser), `tunnel` the way in (`tunnel_from` the settings by default;
    None from it: off), `server` the page server (on `remote.port` by default)."""

    def __init__(
        self,
        repo: Repo,
        layout: ChatLayout,
        clock: Clock,
        settings: Settings,
        *,
        screen: Callable[[], Screen] | None = None,
        tunnel: Callable[[], Tunnel | None] | None = None,
        server: Callable[[], RemoteServer] | None = None,
        poll_seconds: float = 0.5,
        processes: Processes | None = None,
    ) -> None:
        self._repo = repo
        self._layout = layout
        self._clock = clock
        self._settings = settings
        self._make_screen = screen
        self._make_tunnel = tunnel or (lambda: tunnel_from(settings))
        self._make_server = server or (lambda: RemoteServer(settings.remote.port))
        self._poll = poll_seconds
        self._processes = processes or SystemProcesses()
        self._lock = asyncio.Lock()
        # The one sign-in at a time (R45): its site while it opens or lives.
        self._site: str | None = None
        self._job: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._closing = False
        self.session: RemoteSession | None = None

    def configured(self) -> bool:
        """Is a way in chosen (R50), and is there a browser to sign in with?"""
        remote = self._settings.remote
        chosen = bool(remote.public_url.strip()) or remote.tunnel != "none"
        return chosen and self._make_screen is not None

    # -- what the searches met --

    async def met(self, task_id: int, walls: Sequence[Wall]) -> None:
        """The walls one search of `task_id` met: remembered, and a card for each site that
        may be offered now (R48)."""
        if not walls:
            return
        async with self._lock:
            known = await self._walls()
            for wall in walls:
                entry = known.setdefault(wall.site, {"tasks": []})
                entry["url"], entry["kind"] = wall.url, wall.kind
                if task_id not in entry["tasks"]:
                    entry["tasks"].append(task_id)
            await self._repo.set_kv(WALLS_KEY, json.dumps(known, ensure_ascii=False))
            if not self.configured():
                return
            today = self._today()
            for site in dict.fromkeys(wall.site for wall in walls):
                if self._site is not None:
                    return  # one sign-in at a time: no card while one is open
                if await self._repo.get_kv(offered_key(site)) == today:
                    continue
                await self._repo.set_kv(offered_key(site), today)
                await self._offer(site, known[site]["kind"])

    async def tasks_that_met(self, site: str) -> list[int]:
        entry = (await self._walls()).get(site)
        return list(entry["tasks"]) if entry else []

    async def signed_in(self) -> list[str]:
        """The sites the owner signed in to (R49): the search may roam them."""
        return list(json.loads(await self._repo.get_kv(SIGNED_IN_KEY) or "[]"))

    async def _walls(self) -> dict[str, Any]:
        return dict(json.loads(await self._repo.get_kv(WALLS_KEY) or "{}"))

    def _today(self) -> str:
        return (
            self._clock.now()
            .astimezone(self._settings.schedule.zone)
            .date()
            .isoformat()
        )

    async def _new_offer(self, site: str) -> tuple[int, str]:
        number = int(await self._repo.get_kv(OFFER_SEQ_KEY) or 0) + 1
        nonce = secrets.token_hex(3)
        await self._repo.set_kv(OFFER_SEQ_KEY, str(number))
        await self._repo.set_kv(
            offer_key(number), json.dumps({"site": site, "nonce": nonce})
        )
        return number, nonce

    async def _offer(self, site: str, kind: str) -> None:
        number, nonce = await self._new_offer(site)
        text = (CAPTCHA_CARD if kind == "captcha" else LOGIN_CARD).format(site=site)
        buttons = [
            [(SIGN_IN, f"sl:{number}:in:{nonce}"), (NOT_NOW, f"sl:{number}:no:{nonce}")]
        ]
        await self._layout.send(Card(text, buttons, kind="site_login", key=site))
        _log.info("sign-in offered for %s", site)

    async def _sign_in_button(self, site: str) -> list[list[tuple[str, str]]]:
        number, nonce = await self._new_offer(site)
        return [[(SIGN_IN, f"sl:{number}:in:{nonce}")]]

    # -- the buttons --

    async def button(self, number: int, action: str, nonce: str) -> str:
        """«Войти» or «Не сейчас» (callback `sl:<number>:<in|no>:<nonce>`); the short text
        Telegram shows."""
        raw = await self._repo.get_kv(offer_key(number))
        offer = json.loads(raw) if raw else None
        if not offer or not secrets.compare_digest(
            str(offer.get("nonce")).encode(), nonce.encode()
        ):
            return ALREADY_DECIDED
        site = str(offer["site"])
        if action == "no":
            await self._repo.set_kv(offered_key(site), self._today())
            await self._layout.release(
                "site_login", site, Card(NOT_NOW_LINE.format(site=site))
            )
            return FINE
        if action == "in":
            return await self.open(site)
        return UNKNOWN_BUTTON

    async def open(self, site: str) -> str:
        """Open a sign-in on `site` (in the background: a tunnel may take seconds); the
        short text for the button."""
        if self._site is not None:
            if self._site == site:
                return ALREADY_OPEN
            return OTHER_OPEN.format(site=self._site)
        if self._closing:
            return ""
        if not self.configured():
            return TURNED_OFF
        # Taken before the first await: presses run at once, and a double tap on «Войти»
        # must find this sign-in in progress, not open a second one (review 4б.2, M1).
        self._site = site
        self._ready = asyncio.Event()
        try:
            entry = (await self._walls()).get(site) or {}
            url = str(entry.get("url") or f"https://{site}/")
            await self._layout.release(
                "site_login", site, Card(LINK_BELOW.format(site=site))
            )
        except BaseException:
            self._site = None  # nothing was started: a later press may try again
            self._ready.set()
            raise
        if self._closing:  # the program began to stop meanwhile: nothing is opened
            self._site = None
            self._ready.set()
            return ""
        self._job = asyncio.create_task(self._run(site, url), name=f"site_login {site}")
        return OPENING

    # -- one sign-in --

    async def _run(self, site: str, url: str) -> None:
        parts = _Parts()
        outcome: str | None = None
        why = ""
        try:
            outcome = await self._open(site, url, parts)
        except _Failed as failed:
            why = str(failed)
        except Exception as error:  # noqa: BLE001 - the owner hears of it, the rest closes
            _log.warning("the sign-in failed: %s", type(error).__name__)
            why = UNEXPECTED.format(error=type(error).__name__)
        finally:
            self._ready.set()
            await self._shut(parts.session, parts.server, parts.tunnel)
            self.session = None
        link_msg = parts.link_msg
        try:
            await self._tell(site, outcome, why, link_msg)
        except Exception:  # noqa: BLE001 - the program may be stopping
            _log.warning(
                "could not tell the owner how the sign-in ended", exc_info=True
            )
        finally:
            self._site = None

    async def _open(self, site: str, url: str, parts: _Parts) -> str:
        """Open everything, send the link, wait for the end; the outcome. What was opened is
        kept in `parts`, so that it is closed however this ends."""
        if self._make_screen is None:
            raise _Failed(NO_BROWSER)
        minutes = self._settings.remote.session_minutes
        session = RemoteSession(
            site,
            url,
            self._make_screen(),
            self._clock,
            minutes,
            poll_seconds=self._poll,
        )
        parts.session = self.session = session
        try:
            await session.start()
        except Exception as error:  # noqa: BLE001 - BrowserError, or the browser broke
            _log.info("the remote tab did not open: %s", type(error).__name__)
            raise _Failed(NO_BROWSER) from None
        self._still_wanted()
        parts.server = server = self._make_server()
        try:
            await server.start()
        except OSError:
            raise _Failed(PORT_BUSY.format(port=self._settings.remote.port)) from None
        self._still_wanted()
        parts.tunnel = tunnel = self._make_tunnel()
        if tunnel is None:
            raise _Failed(_sentence(TURNED_OFF))
        try:
            base = (await tunnel.open(server.port)).rstrip("/")
        except TunnelError as error:
            raise _Failed(_sentence(str(error))) from None
        await self._keep_process(tunnel)
        self._still_wanted()
        server.attach(session, origin=base)
        parts.link_msg = await self._layout.send(
            Card(
                LINK_TEXT.format(minutes=_minutes(minutes)),
                # The link lives only here: never in a text, a log or a record.
                [
                    [
                        (
                            OPEN_PAGE,
                            f"url:{base}/s/{session.token}#{session.key.fragment()}",
                        )
                    ]
                ],
                kind="site_link",
            )
        )
        self._ready.set()
        _log.info("sign-in opened for %s", site)
        return await session.wait()

    def _still_wanted(self) -> None:
        """The program began to stop while the sign-in opened: it opens no further."""
        if self._closing or (self.session is not None and not self.session.alive):
            raise _Failed(PROGRAM_STOPPED if self._closing else BROKE_OFF)

    async def _shut(
        self,
        session: RemoteSession | None,
        server: RemoteServer | None,
        tunnel: Tunnel | None,
    ) -> None:
        """Close what was opened, each part whatever the others do."""
        if session is not None:
            with contextlib.suppress(Exception):
                await session.stop()
        if server is not None:
            server.detach()
            with contextlib.suppress(Exception):
                await server.stop()
        if tunnel is not None:
            with contextlib.suppress(Exception):
                await tunnel.close()
            with contextlib.suppress(Exception):
                await self._repo.set_kv(TUNNEL_KEY, "")

    async def _tell(
        self, site: str, outcome: str | None, why: str, link_msg: int | None
    ) -> None:
        """What the owner hears at the end, and R49 after «Готово»."""
        if link_msg is not None:
            with contextlib.suppress(CardGone):
                await self._layout.edit(link_msg, Card(LINK_CLOSED.format(site=site)))
        if outcome == "done":
            await self._signed_in(site)
            await self._layout.notify(SAVED.format(site=site))
            return
        if outcome == "expired":
            await self._layout.send(
                Card(EXPIRED.format(site=site), await self._sign_in_button(site))
            )
            return
        if outcome == "stopped":
            why = PROGRAM_STOPPED if self._closing else BROKE_OFF
            await self._layout.notify(FAILED.format(why=why))
            return
        await self._layout.send(
            Card(FAILED.format(why=why or BROKE_OFF), await self._sign_in_button(site))
        )

    async def _signed_in(self, site: str) -> None:
        """R49: the site is the owner's now; each task in work that met it hears so once."""
        async with self._lock:
            signed = await self.signed_in()
            if site not in signed:
                signed.append(site)
                await self._repo.set_kv(SIGNED_IN_KEY, json.dumps(signed))
            known = await self._walls()
            entry = known.pop(site, None)
            await self._repo.set_kv(WALLS_KEY, json.dumps(known, ensure_ascii=False))
        told = 0
        for task_id in entry["tasks"] if entry else []:
            try:
                task = await self._repo.get_task(task_id)
            except NotFound:
                continue
            if task.status is not TaskStatus.ACTIVE:
                continue
            await self._repo.add_event(
                task_id,
                EventKind.REMINDER,
                payload={"about": SIGNED_IN_EVENT.format(site=site)},
            )
            told += 1
        _log.info("signed in to %s; %d tasks told", site, told)

    # -- the tunnel's process --

    async def _keep_process(self, tunnel: Tunnel) -> None:
        pid = getattr(tunnel, "pid", None)
        name = str(getattr(tunnel, "name", "") or "")
        if isinstance(pid, int) and name:
            await self._repo.set_kv(TUNNEL_KEY, json.dumps({"pid": pid, "name": name}))

    async def recover(self) -> None:
        """At start-up: a tunnel process a crash left running is ended — only when the
        process of that number is still that program (the number may be reused)."""
        raw = await self._repo.get_kv(TUNNEL_KEY)
        if not raw:
            return
        try:
            kept = json.loads(raw)
            pid, name = int(kept["pid"]), str(kept["name"])
        except (ValueError, KeyError, TypeError):
            await self._repo.set_kv(TUNNEL_KEY, "")
            return
        running = self._processes.name(pid)
        if running is not None and _program_name(running) == name.casefold():
            self._processes.terminate(pid)
            _log.info("a tunnel a crash left running was ended")
        await self._repo.set_kv(TUNNEL_KEY, "")

    # -- the program's life --

    async def ready(self) -> None:
        """Until the sign-in that is opening has sent its link or failed (tests)."""
        if self._job is not None:
            ready = asyncio.ensure_future(self._ready.wait())
            await asyncio.wait({ready, self._job}, return_when=asyncio.FIRST_COMPLETED)
            ready.cancel()

    async def idle(self) -> None:
        """Until the sign-in, if any, has ended and the owner was told (tests)."""
        job = self._job
        if job is not None:
            with contextlib.suppress(Exception):
                await job

    async def close(self) -> None:
        """The program stops: a live sign-in ends, its server and tunnel close."""
        self._closing = True
        session = self.session
        if session is not None:
            await session.stop()
        await self.idle()


class _Parts:
    """What one sign-in opened so far, for the clean-up."""

    def __init__(self) -> None:
        self.session: RemoteSession | None = None
        self.server: RemoteServer | None = None
        self.tunnel: Tunnel | None = None
        self.link_msg: int | None = None  # the message with the link's button
