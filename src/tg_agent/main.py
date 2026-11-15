"""The program: one process that runs the userbot, the owner's bot and the scheduler together.

The program runs on the agent core (`tg_agent.agent`): the userbot reports to its inbox, the
Scheduler wakes its runner, the Sender reports to the runner, and the owner's bot works through
`TaskOps` and the runner's approvals.

At start-up, in this order: the settings and the secrets are checked before anything is
touched; one copy only (`acquire_single_instance`); the prompts are read, once for the whole
run (`brain/context.load_prompts`); sends a crash cut short become UNKNOWN (the
core reads the chat to learn whether they went); the userbot learns which chats to listen to,
and where each chat's catch-up starts is read, before it connects; what reached those chats
while the program was off is read; the wakes a crash cut short are released for a new wake.
Then the bot's polling, the scheduler and a watch on the userbot run together until Ctrl+C, or
until the link to the Telegram account is gone for good (the owner is told).
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import logging
import random
import secrets
import socket
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

from aiogram import Bot, Dispatcher

from tg_agent import logging_setup
from tg_agent.agent import chats, people
from tg_agent.agent.inbox import Inbox
from tg_agent.agent.ops import TaskOps
from tg_agent.agent.runner import Runner
from tg_agent.agent.site_logins import SiteLogins
from tg_agent.agent.tools import TaskResearch
from tg_agent.brain.context import load_prompts
from tg_agent.brain.llm import LLM, AnthropicLLM
from tg_agent.clock import Clock, SystemClock
from tg_agent.config import ConfigError, Settings, load_settings
from tg_agent.control.base import UI, Card
from tg_agent.control.bot import BotUI, build_dispatcher
from tg_agent.control.chatlog import ChatLog, LoggedUI
from tg_agent.control.layout import ChatLayout
from tg_agent.control.logic import ControlLogic
from tg_agent.db import Repo
from tg_agent.gateway.base import Gateway
from tg_agent.gateway.userbot import TelethonGateway
from tg_agent.search.agent import Searcher
from tg_agent.search.browser import AgentBrowser, agent_profile
from tg_agent.search.olx import OlxBrowser, OlxSource, RevealBudget
from tg_agent.search.remote import CdpScreen, Screen
from tg_agent.search.remote_server import RemoteServer
from tg_agent.search.sites import SiteBrowser, SitePages
from tg_agent.search.tunnel import Tunnel
from tg_agent.secrets import Key, load_owner_secrets, require_secret, use_data_dir
from tg_agent.tasks.recovery import (
    replay_missed,
    replay_starts,
)
from tg_agent.tasks.scheduler import HEARTBEAT_KEY, Scheduler
from tg_agent.tasks.sender import Sender

_log = logging.getLogger(__name__)

LOCK_PORT = 47321
DB_FILE = "agent.db"
LINK_LOST = (
    "Связь с твоим Telegram-аккаунтом оборвалась ({reason}), агент остановлен. Проверь"
    " вход: uv run python -m tg_agent.setup check — и запусти агента снова"
)
# Ruling R88: a run that ended cleanly leaves its last heartbeat marked; a start after a
# crash, or after more than two minutes away, says so in one silent line.
STOPPED = " stopped"
AWAY = timedelta(minutes=2)
BACK = "Бот снова работает. Не работал с {since} до {until}"
_IN_USE = {errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", errno.EADDRINUSE)}


class AlreadyRunning(RuntimeError):
    """Another copy of the agent runs on this computer."""


class UserbotLost(RuntimeError):
    """The link to the Telegram account ended and will not come back by itself."""


class Userbot(Gateway, Protocol):
    """The gateway as the program drives it: started, stopped and watched."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def wait_disconnected(self) -> None: ...


def acquire_single_instance(port: int = LOCK_PORT) -> socket.socket:
    """Hold a local port while the program runs, so that a second copy cannot start.

    Two copies would answer the same people twice. The port is free again as soon as the
    socket is closed or the process ends, however it ends.
    """
    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock.bind(("127.0.0.1", port))
    except OSError as exc:
        lock.close()
        if exc.errno in _IN_USE:
            raise AlreadyRunning(
                f"Агент уже запущен: порт {port} занят другой его копией"
            ) from None
        raise
    return lock


def load_config(path: Path) -> Settings:
    """The settings, with what the program cannot run without checked first."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(
            f"Нет файла настроек {path}: скопируй config.example.toml в config.toml"
            " и впиши свой user_id"
        )
    settings = load_settings(path)
    if not settings.owner.control_user_ids:
        raise ConfigError(
            f"В {path.name} пуст owner.control_user_ids: боту некого слушаться"
        )
    return settings


@dataclass(frozen=True)
class Secrets:
    """The keys the program runs on, read from the keyring."""

    anthropic_api_key: str
    bot_token: str
    api_id: int
    api_hash: str
    session: str

    @classmethod
    def from_keyring(cls) -> Secrets:
        anthropic_api_key = require_secret(Key.ANTHROPIC_API_KEY)
        bot_token = require_secret(Key.BOT_TOKEN)
        api_id = require_secret(Key.TG_API_ID).strip()
        if not api_id.isdigit():
            raise ConfigError(
                "api_id в хранилище — не число: uv run python -m tg_agent.setup secrets --force"
            )
        api_hash = require_secret(Key.TG_API_HASH)
        session = require_secret(Key.TG_SESSION)
        return cls(anthropic_api_key, bot_token, int(api_id), api_hash, session)

    def values(self) -> list[str]:
        """What the log must never show."""
        return [
            self.anthropic_api_key,
            self.bot_token,
            str(self.api_id),
            self.api_hash,
            self.session,
        ]


@dataclass(kw_only=True)
class App:
    """The wired parts; `start_up` and `run` set them going."""

    settings: Settings
    clock: Clock
    repo: Repo
    gateway: Userbot
    ui: UI
    events: Inbox  # what the userbot reports goes here
    runner: Runner
    ops: TaskOps
    logic: ControlLogic
    sender: Sender
    scheduler: Scheduler
    dispatcher: Dispatcher
    # The search behind the task agent's `research` tool (the Runner hands it to each wake).
    searcher: TaskResearch
    # The owner's sign-ins to sites from his phone (stage 4б).
    site_logins: SiteLogins


def assemble(
    settings: Settings,
    clock: Clock,
    repo: Repo,
    llm: LLM,
    gateway: Userbot,
    ui: UI,
    *,
    owner_secrets: Iterable[str] = (),
    canary: str | None = None,
    rng: random.Random | None = None,
    olx: OlxSource | None = None,
    pages: SitePages | None = None,
    chatlog: ChatLog | None = None,
    inline_wakes: bool = False,
    searcher: TaskResearch | None = None,
    remote_screen: Callable[[], Screen] | None = None,
    remote_tunnel: Callable[[], Tunnel | None] | None = None,
    remote_server: Callable[[], RemoteServer] | None = None,
) -> App:
    """Wire the parts together; nothing is started. The Repo must take its time from `clock`.

    The program draws its own canary and random pauses. The evals pass theirs: they must know
    the canary to look for it in what the agent writes, and repeat the pauses from run to run.
    `olx` and `pages` are how the vendor search reads OLX and other sites in the agent's
    browser; without them, those are not searched. With a `chatlog`, the bot writes down what
    the owner sends it (what it shows him is written down by the `LoggedUI` passed as `ui`).
    `inline_wakes` runs each wake inside the Scheduler's round instead of beside it (tests).
    `searcher` replaces the program's own search behind the agent's `research` tool (the
    evals pass a scripted one). `remote_screen` makes the tab the owner signs in to sites in
    from his phone (none: no browser, no sign-ins); `remote_tunnel` and `remote_server`
    replace the way in the settings choose and the page server (tests).
    """
    rng = rng if rng is not None else random.Random()
    # Everything the program shows the owner goes through the chat's layout, which keeps
    # the right message last (stage 5, R52).
    ui = ChatLayout(repo, ui, clock=clock)
    if searcher is None:
        searcher = Searcher(llm, repo, settings, gateway, olx, pages=pages)
    site_logins = SiteLogins(
        repo,
        ui,
        clock,
        settings,
        screen=remote_screen,
        tunnel=remote_tunnel,
        server=remote_server,
    )
    runner = Runner(
        repo,
        llm,
        gateway,
        ui,
        clock,
        settings,
        rng,
        canary=canary if canary is not None else f"ZX-{secrets.token_hex(4)}",
        owner_secrets=list(owner_secrets),
        inline=inline_wakes,
        searcher=searcher,
        site_logins=site_logins,
    )
    # One set of per-conversation locks for the inbox, the send pipeline and the cards.
    inbox = Inbox(repo, clock, settings, outcomes=runner, locks=runner.locks)
    ops = TaskOps(repo, gateway, clock, settings, runner, rng)
    logic = ControlLogic(
        repo,
        ops,
        runner.approvals,
        llm,
        settings,
        clock,
        ui,
        gateway=gateway,
        site_logins=site_logins,
    )
    sender = Sender(repo, gateway, ui, clock, settings, outcomes=runner)
    scheduler = Scheduler(
        runner,
        sender,
        repo,
        clock,
        settings,
        ui,
        reminder=ui.resend_newest,
        board=logic.layout.settle,
    )
    dispatcher = build_dispatcher(logic, settings.data_dir / "files", chatlog)
    return App(
        settings=settings,
        clock=clock,
        repo=repo,
        gateway=gateway,
        ui=ui,
        events=inbox,
        runner=runner,
        ops=ops,
        logic=logic,
        sender=sender,
        scheduler=scheduler,
        dispatcher=dispatcher,
        searcher=searcher,
        site_logins=site_logins,
    )


async def start_up(app: App) -> None:
    """What must happen before the loops run, in this order."""
    # Once per process: an edit of the working tree reaches the bot only with its next start.
    load_prompts()
    await _say_back(app)
    # A tunnel process a crash left running would keep a public address open.
    await app.site_logins.recover()
    unclear = await app.sender.recover()
    await people.listen(app.repo, app.gateway)
    # Before the userbot is live: a message that arrives while it connects is stored at once
    # and would move a chat's starting point past the messages that were missed.
    starts = await replay_starts(app.repo)
    _log.info("connecting to Telegram (waits for the network if it is down)")
    await app.gateway.start()
    replayed = await replay_missed(app.repo, app.gateway, starts)
    # A chat whose sound Telegram refused to give back at a task's end (R105).
    await chats.give_back_left_over(app.repo, app.gateway)
    interrupted = await app.runner.recover()
    # The owner's messages a stop left waiting to be read together (ruling R71).
    await app.logic.read_waiting_burst()
    _log.info(
        "started: %s unclear sends, %s missed messages read, %s interrupted wakes",
        unclear,
        replayed,
        interrupted,
    )


async def _say_back(app: App) -> None:
    """The line «Бот снова работает» when the last run crashed or ended long ago (R88)."""
    beat = await app.repo.get_kv(HEARTBEAT_KEY)
    if not beat:
        return  # the program never ran on this database
    try:
        since, now = datetime.fromisoformat(beat.removesuffix(STOPPED)), app.clock.now()
        if beat.endswith(STOPPED) and now - since <= AWAY:
            return
        tz = app.settings.schedule.zone
        since, now = since.astimezone(tz), now.astimezone(tz)
        shown = "%H:%M" if since.date() == now.date() else "%d.%m %H:%M"
        line = BACK.format(since=since.strftime(shown), until=now.strftime(shown))
        await app.ui.send(Card(line), silent=True)
    except Exception:  # noqa: BLE001 - the program starts without the line: a server
        # restarts by itself, and a start that failed here would loop
        _log.exception("the line about the time away did not go")


async def run(app: App, bot: Bot) -> None:
    """The bot's polling, the scheduler and the watch on the userbot, until one of them ends.

    Whichever ends first ends the others; its error, if it had one, is raised.
    """
    # The owner's messages of a burst are read together once it ends (ruling R71).
    app.logic.timer = _event_loop_timer()
    jobs = [
        asyncio.create_task(
            app.dispatcher.start_polling(
                bot, handle_signals=False, close_bot_session=False
            ),
            name="bot",
        ),
        asyncio.create_task(app.scheduler.run_forever(), name="scheduler"),
        asyncio.create_task(_watch(app), name="userbot"),
    ]
    try:
        done, _ = await asyncio.wait(jobs, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        # A live sign-in ends, and its page server and tunnel close (R45), while the bot
        # can still tell the owner.
        with contextlib.suppress(Exception):
            await app.site_logins.close()
        # Before the caller closes the userbot and the database: a wake left running would
        # write to them after they are gone.
        await app.runner.stop()
        with contextlib.suppress(Exception):  # the next start reads it (R88)
            await app.repo.set_kv(HEARTBEAT_KEY, app.clock.now().isoformat() + STOPPED)
    for job in done:
        job.result()


def _event_loop_timer() -> Callable[[float, Callable[[], Awaitable[None]]], None]:
    """Runs a job after a delay on this event loop; a failure is logged, never lost."""
    loop = asyncio.get_running_loop()
    running: set[asyncio.Task[None]] = set()

    async def guarded(job: Callable[[], Awaitable[None]]) -> None:
        try:
            await job()
        except Exception:  # noqa: BLE001 - the next message tries again
            _log.exception("a delayed job of the bot failed")

    def start(job: Callable[[], Awaitable[None]]) -> None:
        task = loop.create_task(guarded(job))
        running.add(task)  # kept until done: the loop holds tasks only weakly
        task.add_done_callback(running.discard)

    def timer(delay: float, job: Callable[[], Awaitable[None]]) -> None:
        loop.call_later(delay, start, job)

    return timer


async def _watch(app: App) -> None:
    """Wait for the link to the Telegram account to end for good; tell the owner; end the run.

    A link that drops comes back by itself; it ends for good on an error Telethon cannot get
    past, such as a revoked session.
    """
    try:
        await app.gateway.wait_disconnected()
        reason = "соединение закрыто"
    except Exception as error:  # noqa: BLE001 - whatever ended the link ends the run
        reason = type(error).__name__
    _log.error("the link to the Telegram account is gone: %s", reason)
    with contextlib.suppress(Exception):
        await app.ui.notify(LINK_LOST.format(reason=reason))
    raise UserbotLost(f"Связь с Telegram-аккаунтом оборвалась: {reason}")


async def amain(config_path: Path) -> None:
    """Run the agent until Ctrl+C, or until the link to the Telegram account is gone."""
    settings = load_config(config_path)
    use_data_dir(settings.data_dir)  # keys.toml, where the system has no key store (R84)
    keys = Secrets.from_keyring()
    owner_secrets = load_owner_secrets()
    logging_setup.configure(settings.data_dir, [*keys.values(), *owner_secrets])
    async with contextlib.AsyncExitStack() as stack:
        stack.callback(acquire_single_instance().close)
        clock = SystemClock(settings.schedule.zone)
        repo = await Repo.open(settings.data_dir / DB_FILE, clock=clock)
        stack.push_async_callback(repo.close)
        bot = Bot(keys.bot_token)
        stack.push_async_callback(bot.session.close)
        # The core needs the userbot first; the userbot's callbacks find `app.events` when
        # they are called, which is only once the userbot is started.
        gateway = TelethonGateway(
            keys.api_id,
            keys.api_hash,
            keys.session,
            lambda msg: app.events.on_incoming(msg),
            lambda peer_id, msg_id, text: app.events.on_edit(peer_id, msg_id, text),
            lambda peer_id, text, msg_id: app.events.on_owner_manual(
                peer_id, text, msg_id
            ),
            on_delete=lambda msg_ids: app.events.on_delete(msg_ids),
        )
        stack.push_async_callback(gateway.stop)
        chatlog = ChatLog(repo)
        ui = LoggedUI(
            BotUI(bot, settings.owner.control_user_ids[0], clock, settings.schedule),
            chatlog,
        )
        llm = AnthropicLLM(keys.anthropic_api_key, settings.models)
        olx = pages = remote_screen = None
        if settings.search.olx or settings.search.browse:
            # One browser for OLX and the other sites; it starts when a search first needs it.
            browser = AgentBrowser(
                agent_profile(settings),
                channel=settings.search.browser,
                headless=settings.search.browser_headless,
            )
            stack.push_async_callback(browser.close)
            olx = (
                OlxBrowser(browser, reveals=RevealBudget(repo, clock, settings))
                if settings.search.olx
                else None
            )
            pages = SiteBrowser(browser) if settings.search.browse else None

            def remote_screen() -> Screen:
                # The owner's own tab of the same browser: his sign-in stays in its profile.
                return CdpScreen(browser)

        app = assemble(
            settings,
            clock,
            repo,
            llm,
            gateway,
            ui,
            owner_secrets=owner_secrets,
            olx=olx,
            pages=pages,
            chatlog=chatlog,
            remote_screen=remote_screen,
        )
        await start_up(app)
        _log.info("running; Ctrl+C stops")
        await run(app, bot)
