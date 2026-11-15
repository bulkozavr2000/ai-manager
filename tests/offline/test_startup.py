"""Start-up: what is checked before anything is touched, the wiring, and the order things start in."""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tg_agent import main as main_module
from tg_agent.__main__ import main
from tg_agent.agent import chats, people
from tg_agent.agent.inbox import Inbox
from tg_agent.brain.llm import FakeLLM
from tg_agent.clock import FakeClock
from tg_agent.config import ConfigError, Settings
from tg_agent.control.base import FakeUI
from tg_agent.control.chatlog import ChatLog
from tg_agent.db import Repo
from tg_agent.gateway.base import Incoming, PeerInfo
from tg_agent.gateway.fake import FakeGateway
from tg_agent.main import STOPPED, App, _event_loop_timer, amain, assemble, start_up
from tg_agent.models import (
    ConvState,
    Direction,
    EventKind,
    MsgSource,
    OutboxStatus,
    TaskStatus,
)
from tg_agent.search.agent import Searcher
from tg_agent.search.fake import FakeSearch
from tg_agent.secrets import Key, MissingSecret, set_secret
from tg_agent.tasks.scheduler import HEARTBEAT_KEY

PEER = 501
END = [{"type": "text", "text": "Жду ответа."}]


class StartableGateway(FakeGateway):
    """A fake userbot that can be started; `calls` shows the order of what it was asked.
    `on_start` runs inside `start`: what happens while the userbot connects."""

    on_start = None

    async def start(self) -> None:
        self.calls.append("start")
        if self.on_start is not None:
            await self.on_start()


def write_config(tmp_path: Path, owner: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(f'data_dir = "data"\n\n[owner]\n{owner}\n', encoding="utf-8")
    return path


async def test_amain_needs_the_settings_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config.example.toml"):
        await amain(tmp_path / "config.toml")


async def test_amain_needs_someone_to_obey(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="control_user_ids"):
        await amain(write_config(tmp_path, 'city = "Ташкент"'))


async def test_amain_names_a_missing_secret_before_it_touches_anything(
    tmp_path: Path,
) -> None:
    with pytest.raises(MissingSecret, match="anthropic_api_key"):
        await amain(write_config(tmp_path, "control_user_ids = [111]"))

    assert not (tmp_path / "data").exists()


def test_the_command_says_what_is_missing_and_exits_with_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--config", str(tmp_path / "config.toml")])

    assert code == 2
    assert "config.example.toml" in capsys.readouterr().err


# --- the wiring -----------------------------------------------------------------------------


def build(
    repo: Repo, clock: FakeClock, settings: Settings, **kw: object
) -> tuple[App, StartableGateway, FakeUI, FakeLLM]:
    gateway, ui, llm = StartableGateway(clock), FakeUI(), FakeLLM()
    app = assemble(settings, clock, repo, llm, gateway, ui, inline_wakes=True, **kw)
    return app, gateway, ui, llm


async def test_assemble_wires_the_agent_core(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    app, gateway, ui, _ = build(repo, fake_clock, settings)

    assert not hasattr(app, "engine")
    assert isinstance(app.events, Inbox)
    # One set of locks per conversation for the inbox, the send pipeline and the cards.
    assert app.events._locks is app.runner.locks
    assert app.events._outcomes is app.runner
    assert app.scheduler._wakeups is app.runner
    assert app.sender._outcomes is app.runner
    assert app.ops._runner is app.runner
    assert app.logic.ops is app.ops and app.logic.approvals is app.runner.approvals
    # What the program shows goes through the chat's layout (stage 5, R52).
    assert (app.gateway, app.ui.inner) == (gateway, ui)
    assert app.logic.layout is app.ui and app.runner._ui is app.ui


async def test_assemble_hands_the_program_s_search_to_the_task_agent(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    app, gateway, _, llm = build(repo, fake_clock, settings)

    assert isinstance(app.searcher, Searcher)
    assert app.runner._searcher is app.searcher
    assert (app.searcher._llm, app.searcher._gateway) == (llm, gateway)


async def test_assemble_uses_a_search_it_is_given(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    # The evals pass a scripted search: no model and no network behind it.
    search = FakeSearch()
    app, _, _, _ = build(repo, fake_clock, settings, searcher=search)

    assert app.searcher is search and app.runner._searcher is search


async def launched_task(repo: Repo, gateway: FakeGateway) -> tuple[int, int]:
    """An ACTIVE task of the core with one person attached (PLANNED)."""
    task_id = await repo.create_task(
        "3D-печать", "Цена печати", {"what": "корпус"}, None, "UZS", ["name"], []
    )
    gateway.peers["shop"] = PeerInfo(PEER, "shop", "Магазин", False)
    attached = await people.attach(repo, gateway, task_id, "shop")
    assert isinstance(attached, people.Attached)
    await repo.set_task_status(task_id, TaskStatus.ACTIVE)
    return task_id, attached.conv_id


async def first_wake_prompt(app: App, repo: Repo, llm: FakeLLM) -> str:
    """Run the first wake of a launched task and return the rules it was given."""
    await launched_task(repo, app.gateway)
    (task,) = await repo.all_tasks()
    await repo.add_event(task.id, EventKind.CREATED)
    llm.steps["task"].append(END)
    await app.runner.tick()
    ((_, request),) = [r for r in llm.requests if r[0] == "task"]
    return request["system"][0]


async def test_assemble_plants_the_canary_and_uses_the_rng_it_is_given(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    # The evals have to know the canary to look for it, and to repeat the pauses.
    rng = random.Random(5)
    app, _, _, llm = build(repo, fake_clock, settings, canary="ZX-eval-0001", rng=rng)

    prompt = await first_wake_prompt(app, repo, llm)

    assert "ZX-eval-0001" in prompt
    assert app.ops._rng is rng and app.runner._outgoing._rng is rng


async def test_assemble_without_them_draws_a_canary_and_an_rng_of_its_own(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    app, _, _, llm = build(repo, fake_clock, settings)

    prompt = await first_wake_prompt(app, repo, llm)

    assert re.search(r"ZX-[0-9a-f]{8}", prompt)
    assert isinstance(app.ops._rng, random.Random)


async def test_assemble_hands_the_chat_log_to_the_bot(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    from aiogram import Bot
    from aiogram.types import Update

    ui = FakeUI()
    app = assemble(
        settings, fake_clock, repo, FakeLLM(), FakeGateway(), ui, chatlog=ChatLog(repo)
    )
    owner = settings.owner.control_user_ids[0]
    update = Update.model_validate(
        {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "date": 1762750000,
                "chat": {"id": owner, "type": "private"},
                "from": {"id": owner, "is_bot": False, "first_name": "S"},
                "text": "/help",
            },
        }
    )
    bot = Bot("42:TEST")
    try:
        await app.dispatcher.feed_update(bot, update)
    finally:
        await bot.session.close()

    assert ui.sent  # the help card
    (entry,) = await repo.owner_chat_tail(5)
    assert (entry.direction, entry.kind, entry.text) == (
        Direction.IN,
        "command",
        "/help",
    )


# --- start-up -------------------------------------------------------------------------------


async def contacted(repo: Repo, gateway: FakeGateway, clock: FakeClock) -> int:
    """A person of an ACTIVE task whose first message (Telegram id 900) went an hour ago."""
    _, conv_id = await launched_task(repo, gateway)
    await repo.set_state(conv_id, ConvState.CONTACTED)
    sent_at = clock.now() - timedelta(hours=1)
    await repo.add_message(
        conv_id, Direction.OUT, MsgSource.AGENT, "text", "Добрый день!", 900, sent_at
    )
    await repo.update_conversation(conv_id, last_out_at=sent_at)
    return conv_id


async def test_start_up_gives_back_a_sound_a_task_s_end_could_not(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    # Fix round 9.2 (D5): the unmute at a task's end was refused; the next start tries again.
    app, gateway, _, _ = build(repo, fake_clock, settings)
    contact_id = await repo.upsert_contact(777, "old_friend", "Old", False)
    gateway.muted[777] = True
    await repo.set_kv(chats.muted_key(contact_id), "1")

    await start_up(app)

    assert gateway.muted[777] is False
    assert not await repo.get_kv(chats.muted_key(contact_id))


async def test_start_up_listens_to_the_open_chats_before_it_connects(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    app, gateway, ui, _ = build(repo, fake_clock, settings)
    conv_id = await contacted(repo, gateway, fake_clock)
    # A send the crash cut short: it may or may not have left.
    item_id = await repo.enqueue(conv_id, "Сколько стоит доставка?", fake_clock.now())
    assert await repo.claim_outbox(item_id)
    gateway.calls.clear()

    await start_up(app)

    assert gateway.calls == ["set_allowed", "start", "replay"]
    assert gateway.allowed == {PEER}
    assert gateway.replays == [(PEER, 900, {900})]
    (item,) = await repo.outbox_of(conv_id)
    # The core reads the chat a minute later to learn whether it went; the owner is not asked.
    assert item.status is OutboxStatus.UNKNOWN and item.tried_at is not None
    assert ui.sent == [] and ui.notes == []


async def test_start_up_loads_every_prompt_before_the_loops_run(
    repo: Repo, fake_clock: FakeClock, settings: Settings, monkeypatch
) -> None:
    # Fix pass D1.9: one read per prompt per process, at start-up, so that an edit of the
    # working tree never reaches the running bot in the middle of a task.
    from tg_agent.brain import context

    app, _, _, _ = build(repo, fake_clock, settings)
    read: list[str] = []

    class Counted:
        def __init__(self, name: str) -> None:
            self.name = name

        def __truediv__(self, part: str) -> Counted:
            return Counted(part)

        def read_text(self, encoding: str) -> str:
            read.append(self.name)
            return "правила"

    context.load_prompt.cache_clear()
    monkeypatch.setattr(context, "files", lambda package: Counted(package))
    try:
        await start_up(app)
        context.load_prompt("task")  # asked for again later: no second read
    finally:
        context.load_prompt.cache_clear()

    assert sorted(read) == [
        "judge.md",
        "persona.md",
        "search.md",
        "secretary.md",
        "task.md",
    ]


async def test_start_up_runs_its_five_steps_in_order(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    app, gateway, _, _ = build(repo, fake_clock, settings)
    await contacted(repo, gateway, fake_clock)
    gateway.calls.clear()

    def recorded(name: str, step):  # a test double of a step
        async def run(*args: object) -> object:
            gateway.calls.append(name)
            return await step(*args)

        return run

    app.sender.recover = recorded("sender.recover", app.sender.recover)
    app.runner.recover = recorded("runner.recover", app.runner.recover)

    await start_up(app)

    assert gateway.calls == [
        "sender.recover",
        "set_allowed",
        "start",
        "replay",
        "runner.recover",
    ]


async def test_start_up_goes_on_when_the_waiting_burst_fails(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    # Review M4: a refused send while the waiting burst is read must not stop the program.
    app, _, ui, _ = build(repo, fake_clock, settings)
    burst = {
        "first": fake_clock.now().isoformat(),
        "messages": [{"text": "Найди мастера"}],
    }
    await repo.set_kv("owner_burst", json.dumps(burst, ensure_ascii=False))

    async def refused(*args: object, **kwargs: object) -> object:
        raise ConnectionError("network down")

    app.logic.secretary.handle_messages = refused

    await start_up(app)

    assert "Что-то сломалось (ConnectionError)" in ui.sent[-1].text


async def test_the_bot_sets_its_menu_to_stop_and_help_at_its_start(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    app, _, _, _ = build(repo, fake_clock, settings)
    menus: list[list[str]] = []

    class MenuBot:
        async def set_my_commands(self, commands: list[Any], **kwargs: object) -> bool:
            menus.append([command.command for command in commands])
            return True

    class RefusingMenuBot:
        async def set_my_commands(self, commands: list[Any], **kwargs: object) -> bool:
            raise ConnectionError("network down")

    await app.dispatcher.emit_startup(bot=MenuBot())
    await app.dispatcher.emit_startup(bot=RefusingMenuBot())  # only logged

    assert menus == [["stop", "help"]]


async def test_a_message_stored_while_connecting_does_not_hide_an_older_missed_one(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    # The person wrote 901 while the program was off and 905 just as the userbot connected:
    # 905 is stored first, and the replay must still start from 900, or 901 is lost.
    app, gateway, _, _ = build(repo, fake_clock, settings)
    await contacted(repo, gateway, fake_clock)

    async def live_message() -> None:
        await app.events.on_incoming(
            Incoming(PEER, 905, "Ещё актуально?", "text", fake_clock.now())
        )

    gateway.on_start = live_message

    await start_up(app)

    assert gateway.replays == [(PEER, 900, {900})]


async def test_a_wake_interrupted_before_the_start_is_recovered(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    app, gateway, _, _ = build(repo, fake_clock, settings)
    task_id, _ = await launched_task(repo, gateway)
    await repo.add_event(task_id, EventKind.CREATED)
    wake_id, _ = await repo.claim_events(task_id, fake_clock.now())  # then the crash
    assert wake_id

    await start_up(app)

    assert await repo.unfinished_wakes() == []
    events = await repo.open_events(task_id)
    assert all(event.wake_id is None for event in events)
    assert {EventKind.CREATED, EventKind.RECOVERED} <= {e.kind for e in events}


# --- the userbot's callbacks ----------------------------------------------------------------


class _Started(Exception):
    """Ends `amain` once the fake userbot has called its callbacks."""


async def test_the_userbot_reports_to_the_inbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_clock: FakeClock
) -> None:
    reached: list[tuple[str, tuple[object, ...]]] = []

    class RecordingInbox(Inbox):
        async def on_incoming(self, msg: Incoming) -> None:
            reached.append(("on_incoming", (msg,)))

        async def on_edit(self, peer_id: int, tg_msg_id: int, text: str) -> None:
            reached.append(("on_edit", (peer_id, tg_msg_id, text)))

        async def on_owner_manual(
            self, peer_id: int, text: str, tg_msg_id: int | None = None
        ) -> None:
            reached.append(("on_owner_manual", (peer_id, text, tg_msg_id)))

        async def on_delete(self, tg_msg_ids: Sequence[int]) -> None:
            reached.append(("on_delete", (tuple(tg_msg_ids),)))

    incoming = Incoming(PEER, 7, "Привет", "text", fake_clock.now())

    class FakeUserbot(FakeGateway):
        def __init__(  # TelethonGateway's signature
            self, api_id, api_hash, session, on_incoming, on_edit, on_owner_manual,
            *, on_delete,
        ) -> None:  # fmt: skip
            super().__init__()
            self.callbacks = (on_incoming, on_edit, on_owner_manual, on_delete)

        async def start(self) -> None:
            on_incoming, on_edit, on_owner_manual, on_delete = self.callbacks
            await on_incoming(incoming)
            await on_edit(PEER, 7, "Привет!")
            await on_owner_manual(PEER, "Сам отвечу", 8)
            await on_delete([7])
            raise _Started

        async def stop(self) -> None:
            pass

    class Lock:
        def close(self) -> None:
            pass

    for key, value in [
        (Key.ANTHROPIC_API_KEY, "sk-test"),
        (Key.BOT_TOKEN, "42:TEST"),
        (Key.TG_API_ID, "12345"),
        (Key.TG_API_HASH, "hash"),
        (Key.TG_SESSION, "session"),
    ]:
        set_secret(key, value)
    monkeypatch.setattr(main_module, "Inbox", RecordingInbox)
    monkeypatch.setattr(main_module, "TelethonGateway", FakeUserbot)
    monkeypatch.setattr(main_module, "acquire_single_instance", Lock)
    # The log stays the test's: the folder it would have made is made here.
    monkeypatch.setattr(main_module.logging_setup, "configure", lambda *a: None)
    (tmp_path / "data").mkdir()

    with pytest.raises(_Started):
        await amain(write_config(tmp_path, "control_user_ids = [111]"))

    assert reached == [
        ("on_incoming", (incoming,)),
        ("on_edit", (PEER, 7, "Привет!")),
        ("on_owner_manual", (PEER, "Сам отвечу", 8)),
        ("on_delete", ((7,),)),
    ]


# --- a clean stop -----------------------------------------------------------------------------


class WaitingLLM(FakeLLM):
    """A model whose step waits until it is let go: a wake in flight."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.go = asyncio.Event()

    async def agent_step(self, **kw: Any) -> Any:
        self.entered.set()
        await self.go.wait()
        return await super().agent_step(**kw)


class LosableGateway(StartableGateway):
    """A userbot whose link can be cut by the test."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__(clock)
        self.lost = asyncio.Event()

    async def wait_disconnected(self) -> None:
        await self.lost.wait()


async def test_a_stop_cancels_the_wake_in_flight_and_the_next_start_recovers_it(
    db_path: Path,
    fake_clock: FakeClock,
    settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    repo = await Repo.open(db_path, clock=fake_clock)
    llm, gateway = WaitingLLM(), LosableGateway(fake_clock)
    app = assemble(settings, fake_clock, repo, llm, gateway, FakeUI())

    async def no_polling(*args: object, **kwargs: object) -> None:
        await asyncio.Event().wait()

    app.dispatcher.start_polling = no_polling  # type: ignore[method-assign]
    task_id, _ = await launched_task(repo, gateway)
    await repo.add_event(task_id, EventKind.CREATED)
    llm.steps["task"].append(END)

    running = asyncio.create_task(main_module.run(app, object()))  # type: ignore[arg-type]
    await asyncio.wait_for(llm.entered.wait(), 5)  # the wake is in flight
    gateway.lost.set()  # the link is gone: the run ends
    with pytest.raises(main_module.UserbotLost):
        await running
    await repo.close()  # what amain's exit stack does next
    llm.go.set()  # a wake still running would now write to a closed database
    for _ in range(5):
        await asyncio.sleep(0)

    broken = [
        r
        for r in caplog.records
        if r.levelname == "ERROR" and r.name != "tg_agent.main"
    ]
    assert broken == []

    repo = await Repo.open(db_path, clock=fake_clock)
    try:
        again = assemble(
            settings,
            fake_clock,
            repo,
            FakeLLM(),
            StartableGateway(fake_clock),
            FakeUI(),
        )
        await start_up(again)
        assert await repo.unfinished_wakes() == []
        events = await repo.open_events(task_id)
        assert EventKind.CREATED in [e.kind for e in events]
        assert all(event.wake_id is None for event in events)
    finally:
        await repo.close()


async def test_the_bots_timer_runs_a_job_later_and_survives_one_that_breaks() -> None:
    # Ruling R71: the owner's burst is read when its timer fires (`main.run` sets it).
    timer = _event_loop_timer()
    done: list[str] = []

    async def breaks() -> None:
        raise RuntimeError("boom")

    async def works() -> None:
        done.append("read")

    timer(0, breaks)
    timer(0, works)
    for _ in range(20):
        if done:
            break
        await asyncio.sleep(0.01)

    assert done == ["read"]


# --- «Бот снова работает» (ruling R88) ------------------------------------------------------


async def test_after_ten_minutes_away_the_start_up_names_both_times_silently(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    # A clean stop at 09:50 and a start at 10:00: ten minutes away.
    await repo.set_kv(
        HEARTBEAT_KEY, (fake_clock.now() - timedelta(minutes=10)).isoformat() + STOPPED
    )
    app, _, ui, _ = build(repo, fake_clock, settings)

    await start_up(app)

    (card,) = ui.sent
    assert card.text == "Бот снова работает. Не работал с 09:50 до 10:00"
    assert ui.silent == [ui.sent_ids[0]] and ui.notes == []


async def test_a_clean_restart_of_thirty_seconds_says_nothing(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    await repo.set_kv(
        HEARTBEAT_KEY, (fake_clock.now() - timedelta(seconds=30)).isoformat() + STOPPED
    )
    app, _, ui, _ = build(repo, fake_clock, settings)

    await start_up(app)

    assert ui.sent == [] and ui.notes == []


async def test_a_crash_is_told_however_short_and_a_first_start_is_not(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    app, _, ui, _ = build(repo, fake_clock, settings)
    await start_up(app)  # no heartbeat yet: the program never ran here
    assert ui.sent == []

    # The last round at 09:59:30, no clean stop after it.
    await repo.set_kv(
        HEARTBEAT_KEY, (fake_clock.now() - timedelta(seconds=30)).isoformat()
    )
    await start_up(app)

    assert [card.text for card in ui.sent] == [
        "Бот снова работает. Не работал с 09:59 до 10:00"
    ]


@pytest.mark.parametrize("stored", ["garbage", STOPPED, "2026-13-45T99:00"])
async def test_an_unreadable_heartbeat_never_stops_the_start(
    repo: Repo, fake_clock: FakeClock, settings: Settings, stored: str
) -> None:
    await repo.set_kv(HEARTBEAT_KEY, stored)
    app, _, ui, _ = build(repo, fake_clock, settings)

    await start_up(app)  # a server restarts by itself: a start that fails here would loop

    assert ui.sent == []


async def test_days_away_name_the_dates_too(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    await repo.set_kv(
        HEARTBEAT_KEY, (fake_clock.now() - timedelta(days=2, hours=1)).isoformat()
    )
    app, _, ui, _ = build(repo, fake_clock, settings)

    await start_up(app)

    assert ui.sent[0].text == (
        "Бот снова работает. Не работал с 08.11 09:00 до 10.11 10:00"
    )


async def test_the_end_of_a_run_marks_a_clean_stop(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    gateway = LosableGateway(fake_clock)
    app = assemble(settings, fake_clock, repo, FakeLLM(), gateway, FakeUI())

    async def no_polling(*args: object, **kwargs: object) -> None:
        await asyncio.Event().wait()

    app.dispatcher.start_polling = no_polling  # type: ignore[method-assign]
    running = asyncio.create_task(main_module.run(app, object()))  # type: ignore[arg-type]
    for _ in range(20):
        await asyncio.sleep(0)
    gateway.lost.set()
    with pytest.raises(main_module.UserbotLost):
        await running

    assert await repo.get_kv(HEARTBEAT_KEY) == fake_clock.now().isoformat() + STOPPED
