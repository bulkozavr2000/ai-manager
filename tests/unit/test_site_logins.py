"""The owner's sign-in to a site from his phone (stage 4б, Task 4б.2): the card a sign-in wall
brings, «Войти» and its link, the three ends of a session and what follows a sign-in (R49).

The session, the page server and the socket are the real ones (`RemoteSession`,
`RemoteServer` on a free port of 127.0.0.1); the tab is a fake screen, the way in a fake
tunnel. The phone is played by a WebSocket client that speaks the page's cipher.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import aiohttp
import pytest

from tg_agent.agent.site_logins import SiteLogins
from tg_agent.clock import FakeClock
from tg_agent.config import Remote, Settings
from tg_agent.control.base import Card, FakeUI
from tg_agent.control.chatlog import ChatLog, LoggedUI
from tg_agent.control.layout import ChatLayout
from tg_agent.db import Repo
from tg_agent.models import EventKind, TaskStatus
from tg_agent.search.agent import Wall
from tg_agent.search.remote import RemoteKey, seal
from tg_agent.search.remote_server import RemoteServer
from tg_agent.search.tunnel import TunnelError

BASE = "https://quiet-words.trycloudflare.com"
OLX = Wall("olx.uz", "https://www.olx.uz/account/", "login")
CARD = (
    "❓ Сайт olx.uz показывает контакты только после входа. Войти с телефона? Вход"
    " останется в браузере бота: дальше он сам увидит там номера."
)
LINK_TEXT = (
    "Открой страницу и войди. Ссылка работает 10 минут; когда закончишь — нажми"
    " «Готово» на странице."
)
TYPED = "секретный-пароль-Ж7"


class FakeScreen:
    """The remote tab: what it was asked to open and type."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.started: list[str] = []
        self.typed: list[str] = []
        self.stopped = 0
        self.fail = fail

    async def start(
        self,
        url: str,
        on_frame: Callable[[bytes], None],
        on_refused: Callable[[], None],
        on_notice: Callable[[str], None] | None = None,
    ) -> None:
        if self.fail is not None:
            raise self.fail
        self.started.append(url)
        on_frame(b"\xff\xd8 a picture")

    async def pointer(self, kind: str, x: float, y: float) -> None: ...

    async def wheel(self, dy: float) -> None: ...

    async def text(self, s: str) -> None:
        self.typed.append(s)

    async def key(self, k: str) -> None: ...

    async def back(self) -> None: ...

    async def focus(self) -> str:
        return "password"

    def url(self) -> str:
        return self.started[-1] if self.started else ""

    async def stop(self) -> None:
        self.stopped += 1


class FakeTunnel:
    name = "cloudflared"

    def __init__(self, fail: TunnelError | None = None) -> None:
        self.fail = fail
        self.opened: list[int] = []
        self.closed = 0
        self.pid: int | None = None

    async def open(self, port: int) -> str:
        self.opened.append(port)
        if self.fail is not None:
            raise self.fail
        self.pid = 4242
        return BASE

    async def close(self) -> None:
        self.closed += 1
        self.pid = None


class Processes:
    """The computer's processes, as the clean-up after a crash sees them."""

    def __init__(self, names: dict[int, str]) -> None:
        self.names = names
        self.terminated: list[int] = []

    def name(self, pid: int) -> str | None:
        return self.names.get(pid)

    def terminate(self, pid: int) -> None:
        self.terminated.append(pid)


class World:
    def __init__(
        self, repo: Repo, clock: FakeClock, settings: Settings, *, way_in: bool = True
    ) -> None:
        self.repo, self.clock = repo, clock
        if way_in:
            settings = Settings(
                owner=settings.owner,
                data_dir=settings.data_dir,
                remote=Remote(tunnel="cloudflared"),
            )
        self.settings = settings
        self.bot = FakeUI()
        self.layout = ChatLayout(repo, LoggedUI(self.bot, ChatLog(repo)), clock=clock)
        self.screens: list[FakeScreen] = []
        self.tunnels: list[FakeTunnel] = []
        self.servers: list[RemoteServer] = []
        self.screen_fail: Exception | None = None
        self.tunnel_fail: TunnelError | None = None
        self.processes = Processes({})
        self.logins = SiteLogins(
            repo,
            self.layout,
            clock,
            settings,
            screen=self.new_screen,
            tunnel=self.new_tunnel if way_in else (lambda: None),
            server=self.new_server,
            poll_seconds=0.01,
            processes=self.processes,
        )

    def new_screen(self) -> FakeScreen:
        self.screens.append(FakeScreen(self.screen_fail))
        return self.screens[-1]

    def new_tunnel(self) -> FakeTunnel:
        self.tunnels.append(FakeTunnel(self.tunnel_fail))
        return self.tunnels[-1]

    def new_server(self) -> RemoteServer:
        self.servers.append(RemoteServer(0))
        return self.servers[-1]

    def cards(self, kind: str) -> list[Card]:
        return [card for card in self.bot.sent if card.kind == kind]

    def button(self, card: Card, label: str) -> str:
        return next(data for row in card.buttons for text, data in row if text == label)

    async def press(self, data: str) -> str:
        """What `ControlLogic.on_callback` does with a «sl:…» button."""
        kind, number, action, nonce = data.split(":")
        assert kind == "sl"
        return await self.logins.button(int(number), action, nonce)

    async def open(self) -> str:
        """The card for olx.uz, «Войти», and the link of the message that follows."""
        await self.logins.met(1, [OLX])
        (card,) = self.cards("site_login")
        await self.press(self.button(card, "Войти"))
        await self.logins.ready()
        link_card = self.bot.sent[-1]
        data = self.button(link_card, "Открыть страницу входа")
        assert data.startswith("url:")
        return data[4:]


class Phone:
    """The page on the phone: its socket, speaking the page's cipher."""

    def __init__(self, link: str, server: RemoteServer) -> None:
        address, fragment = link.split("#", 1)
        self.token = address.rsplit("/s/", 1)[1]
        self.key = RemoteKey.from_fragment(fragment)
        self.url = f"http://127.0.0.1:{server.port}/s/{self.token}/ws"
        self.load = secrets.token_hex(16)
        self.number = 0

    def sealed(self, message: dict[str, Any]) -> bytes:
        self.number += 1
        plain = {"p": self.load, "n": self.number, **message}
        return seal(self.key, json.dumps(plain, ensure_ascii=False).encode())

    async def send(self, *messages: dict[str, Any]) -> None:
        """Send the messages; return once the session took them all: it ended, or it
        reported the focus after each release and key (the last of them comes last)."""
        expected = sum(1 for m in messages if m["t"] in ("up", "key"))
        async with (
            aiohttp.ClientSession() as http,
            http.ws_connect(self.url, origin=BASE) as ws,
        ):
            for message in messages:
                await ws.send_bytes(self.sealed(message))
            focused = 0
            async for reply in ws:
                if reply.type is not aiohttp.WSMsgType.TEXT:
                    continue
                if "end" in reply.data:
                    break
                if "focus" in reply.data:
                    focused += 1
                    if focused == expected:
                        break

    async def junk(self, count: int) -> None:
        """Messages that do not open, then the socket until the session closes it."""
        async with (
            aiohttp.ClientSession() as http,
            http.ws_connect(self.url, origin=BASE) as ws,
        ):
            for _ in range(count):
                await ws.send_bytes(b"\x00" * 64)
            async for _ in ws:
                pass


@pytest.fixture
def world(repo: Repo, fake_clock: FakeClock, settings: Settings) -> World:
    return World(repo, fake_clock, settings)


async def a_task(repo: Repo, status: TaskStatus = TaskStatus.ACTIVE) -> int:
    task_id = await repo.create_task(
        "3D-печать", "Найти, кто напечатает 5 корпусов", {}, None, "UZS", ["name"], []
    )
    await repo.set_task_status(task_id, status)
    return task_id


# --- the card ---------------------------------------------------------------------------------


async def test_a_wall_brings_one_card_a_day_and_again_the_next_day(
    world: World,
) -> None:
    await world.logins.met(1, [OLX])
    await world.logins.met(2, [OLX])

    (card,) = world.cards("site_login")
    assert card.text == CARD
    assert card.key == "olx.uz"
    assert [label for row in card.buttons for label, _ in row] == ["Войти", "Не сейчас"]
    assert all(data.startswith("sl:") for row in card.buttons for _, data in row)
    assert world.bot.last() is card  # it waits: the last message of the chat

    world.clock.advance(timedelta(days=1))
    await world.logins.met(1, [OLX])
    assert len(world.cards("site_login")) == 2


async def test_a_captcha_card_asks_to_pass_the_check(world: World) -> None:
    await world.logins.met(1, [Wall("prom.uz", "https://prom.uz/", "captcha")])

    (card,) = world.cards("site_login")
    assert card.text == (
        "❓ Сайт prom.uz просит пройти проверку «я не робот». Пройти с телефона?"
    )


async def test_without_a_way_in_there_is_no_card_and_the_site_is_still_remembered(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    world = World(repo, fake_clock, settings, way_in=False)

    await world.logins.met(1, [OLX])

    assert world.bot.sent == [] and world.bot.notes == []
    assert await world.logins.tasks_that_met("olx.uz") == [1]


async def test_not_now_says_fine_and_the_site_waits_for_tomorrow(world: World) -> None:
    await world.logins.met(1, [OLX])
    (card,) = world.cards("site_login")

    assert await world.press(world.button(card, "Не сейчас")) == "Хорошо"

    assert world.bot.screen[world.bot.sent_ids[0]].text == "Вход на olx.uz: не сейчас"
    await world.logins.met(1, [OLX])
    assert len(world.cards("site_login")) == 1
    assert world.tunnels == []


# --- «Войти» and the link ---------------------------------------------------------------------


async def test_sign_in_sends_the_link_only_in_a_button(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)

    link = await world.open()

    assert link.startswith(f"{BASE}/s/") and "#" in link
    assert world.screens[0].started == ["https://www.olx.uz/account/"]
    assert world.tunnels[0].opened == [world.servers[0].port]
    # The card became its settled line; the link came in a message of its own.
    first = world.bot.sent_ids[0]
    assert world.bot.screen[first].text == "Вход на olx.uz: ссылка ниже"
    assert world.bot.sent[-1].text == LINK_TEXT
    await world.logins.close()
    await nothing_kept(world, caplog, link)


async def test_the_session_s_three_ends_and_what_the_owner_hears(world: World) -> None:
    # «Готово» on the page.
    link = await world.open()
    await Phone(link, world.servers[0]).send({"t": "done"})
    await world.logins.idle()
    assert world.bot.notes[-1] == "Вход на olx.uz сохранён."
    assert_closed(world, 0)

    # The ten minutes pass.
    world.clock.advance(timedelta(days=1))
    await world.logins.met(1, [OLX])
    await world.press(world.button(world.cards("site_login")[-1], "Войти"))
    await world.logins.ready()
    world.clock.advance(timedelta(minutes=11))
    await world.logins.idle()
    expired = world.bot.sent[-1]
    assert expired.text == (
        "Время вышло: вход на olx.uz не завершён. Нажми «Войти» ещё раз, если нужно."
    )
    assert_closed(world, 1)

    # «Войти» under that message opens a new session; this one breaks off.
    await world.press(world.button(expired, "Войти"))
    await world.logins.ready()
    link = world.button(world.bot.sent[-1], "Открыть страницу входа")[4:]
    await Phone(link, world.servers[2]).junk(10)  # ten messages that do not open
    await world.logins.idle()
    assert world.bot.notes[-1] == (
        "Не получилось открыть страницу входа: сеанс прервался раньше времени."
    )
    assert_closed(world, 2)


def assert_closed(world: World, number: int) -> None:
    """The way in and the page server of session `number` are closed, the tab stopped."""
    assert world.tunnels[number].closed == 1
    assert world.servers[number]._runner is None
    assert world.screens[number].stopped >= 1
    assert world.logins.session is None


async def test_after_done_the_site_is_signed_in_and_active_tasks_hear_of_it(
    world: World,
) -> None:
    active, done = await a_task(world.repo), await a_task(world.repo, TaskStatus.DONE)
    await world.logins.met(active, [OLX])
    await world.logins.met(done, [OLX])
    (card,) = world.cards("site_login")
    await world.press(world.button(card, "Войти"))
    await world.logins.ready()
    link = world.button(world.bot.sent[-1], "Открыть страницу входа")[4:]

    await Phone(link, world.servers[0]).send({"t": "done"})
    await world.logins.idle()

    assert await world.logins.signed_in() == ["olx.uz"]
    (event,) = await world.repo.open_events(active)
    assert event.kind is EventKind.REMINDER
    assert event.payload == {
        "about": "Владелец вошёл на сайт olx.uz: поиск теперь видит там контакты."
        " Если людей не хватает — поищи ещё раз."
    }
    assert await world.repo.open_events(done) == []
    # The link's message no longer offers the dead link.
    assert all(
        not data.startswith("url:")
        for _, card in world.bot.chat()
        for row in card.buttons
        for _, data in row
    )


async def test_a_tunnel_that_does_not_open_is_told_in_plain_words(world: World) -> None:
    world.tunnel_fail = TunnelError(
        "Не найдена программа cloudflared: tools/cloudflared.exe. Сам я её не скачиваю"
        " и не устанавливаю."
    )
    await world.logins.met(1, [OLX])
    await world.press(world.button(world.cards("site_login")[0], "Войти"))
    await world.logins.idle()

    failed = world.bot.sent[-1]
    assert failed.text == (
        "Не получилось открыть страницу входа: не найдена программа cloudflared:"
        " tools/cloudflared.exe. Сам я её не скачиваю и не устанавливаю."
    )
    assert [label for row in failed.buttons for label, _ in row] == ["Войти"]
    assert world.logins.session is None
    assert world.servers[0]._runner is None
    assert world.screens[0].stopped >= 1
    assert world.tunnels[0].closed == 1


async def test_a_browser_that_does_not_start_is_told_and_nothing_is_left_open(
    world: World,
) -> None:
    from tg_agent.search.browser import BrowserError

    world.screen_fail = BrowserError("браузер не запустился")
    await world.logins.met(1, [OLX])
    await world.press(world.button(world.cards("site_login")[0], "Войти"))
    await world.logins.idle()

    assert world.bot.sent[-1].text == (
        "Не получилось открыть страницу входа: браузер бота не открыл эту страницу."
    )
    assert world.tunnels == [] and world.servers == []
    assert world.logins.session is None


async def test_a_second_press_while_the_session_lives(world: World) -> None:
    await world.open()
    (card,) = world.cards("site_login")

    assert await world.press(world.button(card, "Войти")) == (
        "Вход уже открыт: ссылка в сообщении выше"
    )
    assert len(world.tunnels) == 1
    # And a wall met meanwhile brings no card.
    await world.logins.met(2, [Wall("prom.uz", "https://prom.uz/", "login")])
    assert len(world.cards("site_login")) == 1
    await world.logins.close()


async def test_a_button_with_a_wrong_code_does_nothing(world: World) -> None:
    await world.logins.met(1, [OLX])
    (card,) = world.cards("site_login")
    data = world.button(card, "Войти")

    assert await world.press(data[: data.rindex(":")] + ":000000") != ""
    assert world.screens == [] and world.tunnels == []


async def test_a_clean_stop_ends_the_session_and_closes_everything(
    world: World,
) -> None:
    link = await world.open()

    await world.logins.close()

    assert_closed(world, 0)
    async with aiohttp.ClientSession() as http:
        with pytest.raises(aiohttp.ClientError):
            await http.get(
                link.split("#")[0].replace(
                    BASE, f"http://127.0.0.1:{world.servers[0].port}"
                )
            )
    assert await world.repo.get_kv("remote_tunnel") in (None, "")


async def test_a_double_tap_on_voiti_opens_one_sign_in(world: World) -> None:
    # Review 4б.2, M1: button presses run at once; the second must see the first.
    await world.logins.met(1, [OLX])
    (card,) = world.cards("site_login")
    data = world.button(card, "Войти")
    answers = await asyncio.gather(world.press(data), world.press(data))
    await world.logins.ready()
    await asyncio.sleep(0.1)
    opening = "Открываю страницу входа: ссылка придёт через несколько секунд."
    assert answers.count(opening) == 1
    links = [c for c in world.bot.sent if c.kind == "site_link"]
    assert len(world.tunnels) == 1 and len(links) == 1
    await world.logins.close()
    assert world.tunnels[0].closed == 1  # nothing is left open
    assert await world.repo.get_kv("remote_tunnel") in ("", None)


async def test_the_tunnel_s_process_is_kept_and_a_crash_s_is_ended_at_start(
    world: World,
) -> None:
    await world.open()
    kept = json.loads(await world.repo.get_kv("remote_tunnel") or "{}")
    assert kept == {"pid": 4242, "name": "cloudflared"}

    # A crash: the program never closed it. At the next start it is still that program.
    world.processes.names[4242] = "C:\\agent\\tools\\cloudflared.exe"
    await world.logins.recover()
    assert world.processes.terminated == [4242]
    assert await world.repo.get_kv("remote_tunnel") == ""

    # Another program took the number since: it is left alone.
    await world.repo.set_kv(
        "remote_tunnel", json.dumps({"pid": 77, "name": "cloudflared"})
    )
    world.processes.names[77] = "C:\\Windows\\notepad.exe"
    await world.logins.recover()
    assert world.processes.terminated == [4242]
    assert await world.repo.get_kv("remote_tunnel") == ""
    await world.logins.close()


# --- nothing typed is kept --------------------------------------------------------------------


async def test_what_he_types_and_the_link_are_kept_nowhere(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    link = await world.open()

    await Phone(link, world.servers[0]).send(
        {"t": "down", "x": 0.5, "y": 0.3},
        {"t": "up", "x": 0.5, "y": 0.3},
        {"t": "text", "s": TYPED},
        {"t": "key", "k": "Enter"},
    )
    assert world.screens[0].typed == [TYPED]
    await Phone(link, world.servers[0]).send({"t": "done"})
    await world.logins.idle()

    await nothing_kept(world, caplog, link, TYPED)


async def nothing_kept(
    world: World, caplog: pytest.LogCaptureFixture, link: str, *more: str
) -> None:
    """No log record, message, chat-log row or other row holds the link, its token, its
    key, or what was typed."""
    address, fragment = link.split("#", 1)
    token = address.rsplit("/s/", 1)[1]
    secrets_ = [link, token, fragment, *more]
    logged = "\n".join(
        f"{record.getMessage()} {record.args!r}" for record in caplog.records
    )
    shown = "\n".join(
        [card.text for card in world.bot.sent]
        + [card.text for _, card in world.bot.edits]
        + world.bot.notes
    )
    tables = [
        row["name"]
        for row in await world.repo._all(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    ]
    rows = []
    for table in tables:
        rows += [
            repr(tuple(row)) for row in await world.repo._all(f"SELECT * FROM {table}")
        ]
    stored = "\n".join(rows)
    for secret in secrets_:
        assert secret not in logged
        assert secret not in shown
        assert secret not in stored
    assert "owner_chat" in tables


async def test_a_session_that_stops_with_the_program_tells_the_owner(
    world: World,
) -> None:
    await world.open()
    await world.logins.close()
    assert world.bot.notes[-1] == (
        "Не получилось открыть страницу входа: программа остановилась."
    )


async def test_nothing_runs_in_the_background_after_the_end(world: World) -> None:
    link = await world.open()
    await Phone(link, world.servers[0]).send({"t": "done"})
    await world.logins.idle()
    await asyncio.sleep(0.05)
    assert world.logins.session is None
    assert [t for t in asyncio.all_tasks() if "site_login" in t.get_name()] == []
