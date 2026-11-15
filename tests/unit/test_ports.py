"""The three protocols between the core and the Sender, the Scheduler and the gateway.

The Sender and the Scheduler are driven here with recording stubs in place of the core: they
must work with whatever implements `SendOutcomes` and `Wakeups`.
"""

from __future__ import annotations

import ast
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tg_agent.clock import FakeClock
from tg_agent.config import Settings
from tg_agent.control.base import FakeUI
from tg_agent.db import Repo
from tg_agent.gateway.base import TgActionError, TgSendUncertain
from tg_agent.gateway.fake import FakeGateway
from tg_agent.models import (
    Conversation,
    ConvState,
    Direction,
    OutboxItem,
    OutboxStatus,
)
from tg_agent.policy.tg_safety import TgErrorKind
from tg_agent.tasks.outbox import unclear_text
from tg_agent.tasks.scheduler import Scheduler
from tg_agent.tasks.sender import Sender
from tg_agent.tasks.texts import ADVICE, NOT_SENT, who

PEER = 555
PRIVACY_REASON = "его настройки приватности не дают написать"


def _imported_by(module: str) -> set[str]:
    """The modules `module`'s own source imports (not what those import in turn)."""
    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin is not None
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_the_sender_does_not_use_the_state_machine() -> None:
    # The state machine is loaded all the same, by the database (`db.py` checks every move
    # of a conversation's state): the Sender itself must not use it.
    assert "tg_agent.tasks.state" not in _imported_by("tg_agent.tasks.sender")


class Outcomes:
    """A `SendOutcomes` that records what it is told; `fail` makes the named call raise."""

    def __init__(
        self,
        *,
        reanswer: bool = True,
        fail: str | None = None,
        took_care: bool = False,
    ) -> None:
        self.reanswer = reanswer
        self.fail = fail
        self.took_care = took_care  # what `on_failed` answers
        self.failed_breaks = False  # `on_failed` raises instead
        self.calls: list[tuple] = []

    def _maybe_fail(self, name: str) -> None:
        if self.fail == name:
            raise RuntimeError(f"{name} broke")

    def can_reanswer(self, conv: Conversation) -> bool:
        self._maybe_fail("can_reanswer")
        return self.reanswer

    async def excluded(self, conv: Conversation) -> bool:
        return False

    async def on_failed(
        self, item: OutboxItem, conv: Conversation, code: str, reason: str
    ) -> bool:
        self.calls.append(("on_failed", item.id, conv.id, code, reason))
        if self.failed_breaks:
            raise RuntimeError("on_failed broke")
        return self.took_care

    async def on_sent(
        self, item: OutboxItem, conv: Conversation, at: datetime, *, first: bool
    ) -> None:
        self.calls.append(("on_sent", item.id, conv.id, at, first))
        self._maybe_fail("on_sent")

    async def on_stale(self, conv_id: int) -> None:
        self.calls.append(("on_stale", conv_id))
        self._maybe_fail("on_stale")

    async def on_unclear(self, item: OutboxItem) -> None:
        self.calls.append(("on_unclear", item.id))
        self._maybe_fail("on_unclear")

    async def on_refused(
        self,
        item: OutboxItem,
        conv: Conversation,
        kind: TgErrorKind,
        reason: str,
        cancelled: int,
    ) -> None:
        self.calls.append(("on_refused", item.id, conv.id, kind, reason, cancelled))
        self._maybe_fail("on_refused")

    def named(self, name: str) -> list[tuple]:
        return [call for call in self.calls if call[0] == name]


class World:
    """A conversation in CONTACTED, a fake Telegram and bot, and a Sender on `outcomes`."""

    def __init__(
        self,
        repo: Repo,
        clock: FakeClock,
        settings: Settings,
        outcomes: Outcomes,
        conv_id: int,
    ) -> None:
        self.repo = repo
        self.clock = clock
        self.settings = settings
        self.outcomes = outcomes
        self.conv_id = conv_id
        self.gateway = FakeGateway()
        self.ui = FakeUI()
        self.sender = Sender(repo, self.gateway, self.ui, clock, settings, outcomes)

    async def queue(self, text: str, after: float = 0) -> int:
        return await self.repo.enqueue(
            self.conv_id, text, self.clock.now() + timedelta(seconds=after)
        )

    async def item(self, item_id: int) -> OutboxItem:
        [item] = [i for i in await self.repo.outbox_of(self.conv_id) if i.id == item_id]
        return item


async def make_world(
    repo: Repo, clock: FakeClock, settings: Settings, outcomes: Outcomes
) -> World:
    task_id = await repo.create_task(
        "3D-печать", "Узнать цену", {"what": "корпус"}, None, "UZS", [], []
    )
    contact_id = await repo.upsert_contact(PEER, "print_lab", "Акмаль", False)
    conv_id = await repo.create_conversation(task_id, contact_id)
    await repo.set_state(conv_id, ConvState.CONTACTED)
    return World(repo, clock, settings, outcomes, conv_id)


@pytest.fixture
async def world(repo: Repo, fake_clock: FakeClock, settings: Settings) -> World:
    return await make_world(repo, fake_clock, settings, Outcomes())


async def test_a_due_message_goes_out_and_the_core_hears_whether_it_was_the_first(
    world: World,
) -> None:
    first = await world.queue("Здравствуйте")

    assert await world.sender.tick() == 1

    item = await world.item(first)
    assert item.status is OutboxStatus.SENT
    [row] = await world.repo.last_messages(world.conv_id, 10)
    assert (row.direction, row.text, row.tg_msg_id) == (
        Direction.OUT,
        "Здравствуйте",
        item.tg_msg_id,
    )
    assert world.outcomes.named("on_sent") == [
        ("on_sent", first, world.conv_id, world.clock.now(), True)
    ]

    second = await world.queue("Сколько стоит?")
    assert await world.sender.tick() == 1
    assert world.outcomes.named("on_sent")[1] == (
        "on_sent",
        second,
        world.conv_id,
        world.clock.now(),
        False,
    )


async def test_a_reply_made_stale_by_a_newer_message_is_dropped_when_the_core_can_reanswer(
    world: World,
) -> None:
    item_id = await world.queue("Ответ")
    world.clock.advance(timedelta(seconds=10))
    await world.repo.update_conversation(world.conv_id, last_in_at=world.clock.now())

    assert await world.sender.tick() == 0

    assert world.gateway.sent == []
    assert (await world.item(item_id)).status is OutboxStatus.CANCELLED
    assert world.outcomes.named("on_stale") == [("on_stale", world.conv_id)]


async def test_a_reply_made_stale_goes_out_as_it_is_when_the_core_cannot_reanswer(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    world = await make_world(repo, fake_clock, settings, Outcomes(reanswer=False))
    item_id = await world.queue("До свидания")
    world.clock.advance(timedelta(seconds=10))
    await world.repo.update_conversation(world.conv_id, last_in_at=world.clock.now())

    assert await world.sender.tick() == 1

    assert world.gateway.sent == [(PEER, "До свидания")]
    assert (await world.item(item_id)).status is OutboxStatus.SENT
    assert world.outcomes.named("on_stale") == []


@pytest.mark.parametrize("card_breaks", [False, True])
async def test_a_send_that_broke_off_is_unknown_and_the_core_asks_the_owner(
    repo: Repo, fake_clock: FakeClock, settings: Settings, card_breaks: bool
) -> None:
    outcomes = Outcomes(fail="on_unclear" if card_breaks else None)
    world = await make_world(repo, fake_clock, settings, outcomes)
    item_id = await world.queue("Сообщение")
    world.gateway.fail_next = TgSendUncertain("ConnectionError")

    assert await world.sender.tick() == 0

    item = await world.item(item_id)
    assert item.status is OutboxStatus.UNKNOWN
    assert outcomes.named("on_unclear") == [("on_unclear", item_id)]
    if card_breaks:
        contact = await repo.get_contact(
            (await repo.get_conversation(world.conv_id)).contact_id
        )
        assert world.ui.notes == [unclear_text(item, contact)]
    else:
        assert world.ui.notes == []


@pytest.mark.parametrize("core_breaks", [False, True])
async def test_a_final_refusal_fails_the_message_drops_the_queue_and_tells_the_core(
    repo: Repo, fake_clock: FakeClock, settings: Settings, core_breaks: bool
) -> None:
    outcomes = Outcomes(fail="on_refused" if core_breaks else None)
    world = await make_world(repo, fake_clock, settings, outcomes)
    item_id = await world.queue("Первое")
    others = [await world.queue("Второе", 60), await world.queue("Третье", 120)]
    world.gateway.fail_next = TgActionError(
        TgErrorKind.PRIVACY, "USER_PRIVACY_RESTRICTED"
    )

    assert await world.sender.tick() == 0

    assert (await world.item(item_id)).status is OutboxStatus.FAILED
    assert [(await world.item(i)).status for i in others] == [
        OutboxStatus.CANCELLED,
        OutboxStatus.CANCELLED,
    ]
    assert outcomes.named("on_refused") == [
        (
            "on_refused",
            item_id,
            world.conv_id,
            TgErrorKind.PRIVACY,
            PRIVACY_REASON,
            2,
        )
    ]
    if core_breaks:
        conv = await repo.get_conversation(world.conv_id)
        contact = await repo.get_contact(conv.contact_id)
        notice = NOT_SENT.format(who=who(contact), reason=PRIVACY_REASON)
        assert world.ui.notes == [notice + ADVICE]
    else:
        assert world.ui.notes == []


async def _fail_by_itself(world: World, case: str) -> int:
    """Make the Sender fail a message without Telegram's refusal; return the item's id."""
    if case == "file":
        return await world.repo.enqueue(
            world.conv_id, "", world.clock.now(), attach_path="gone.pdf"
        )
    if case == "typing":
        item_id = await world.queue("Сообщение")
        world.gateway.fail_typing = ConnectionResetError(
            "the connection keeps dropping"
        )
        for _ in range(3):  # the first try and two retries put it back
            await world.sender.tick()
            world.clock.advance(timedelta(minutes=1))
        return item_id
    # "prepare": a decision of the core raises before the message is claimed
    item_id = await world.queue("Ответ")
    world.clock.advance(timedelta(seconds=10))
    await world.repo.update_conversation(world.conv_id, last_in_at=world.clock.now())
    world.outcomes.fail = "can_reanswer"
    return item_id


_FAILED_BY_ITSELF = {
    "file": ("file_missing", "файла «gone.pdf» нет на диске", ""),
    "typing": (
        "typing: ConnectionResetError",
        "не получилось начать отправку (ConnectionResetError)",
        ADVICE,
    ),
    "prepare": ("RuntimeError", "не получилось его подготовить (RuntimeError)", ADVICE),
}


@pytest.mark.parametrize("case", list(_FAILED_BY_ITSELF))
@pytest.mark.parametrize("answer", ["took_care", "did_not", "breaks"])
async def test_a_message_the_sender_fails_by_itself_goes_to_the_core_first(
    repo: Repo, fake_clock: FakeClock, settings: Settings, case: str, answer: str
) -> None:
    outcomes = Outcomes(took_care=answer == "took_care")
    outcomes.failed_breaks = answer == "breaks"
    world = await make_world(repo, fake_clock, settings, outcomes)
    item_id = await _fail_by_itself(world, case)

    await world.sender.tick()

    code, reason, advice = _FAILED_BY_ITSELF[case]
    item = await world.item(item_id)
    assert (item.status, item.error) == (OutboxStatus.FAILED, code)
    assert outcomes.named("on_failed") == [
        ("on_failed", item_id, world.conv_id, code, reason)
    ]
    assert world.gateway.sent == []
    if answer == "took_care":
        assert world.ui.notes == []
    else:
        contact = await repo.get_contact(
            (await repo.get_conversation(world.conv_id)).contact_id
        )
        assert world.ui.notes == [
            NOT_SENT.format(who=who(contact), reason=reason) + advice
        ]


class Wakeups:
    """A `Wakeups` that counts its ticks; `fail` makes each tick raise."""

    def __init__(self, *, fail: bool = False) -> None:
        self.ticks = 0
        self.fail = fail

    async def tick(self) -> None:
        self.ticks += 1
        if self.fail:
            raise RuntimeError("the core broke")


def scheduler_on(world: World, wakeups: Wakeups, rounds: list[str]) -> Scheduler:
    async def digest() -> str:
        return ""

    async def board() -> None:
        rounds.append("board")

    return Scheduler(
        wakeups,
        world.sender,
        world.repo,
        world.clock,
        world.settings,
        world.ui,
        digest,
        board=board,
    )


async def test_one_round_of_the_scheduler_wakes_the_core_once(world: World) -> None:
    wakeups = Wakeups()
    rounds: list[str] = []

    await scheduler_on(world, wakeups, rounds).tick()

    assert wakeups.ticks == 1
    assert await world.repo.events_of("scheduler_error") == []


async def test_a_core_that_fails_to_wake_is_recorded_and_the_round_goes_on(
    world: World,
) -> None:
    wakeups = Wakeups(fail=True)
    rounds: list[str] = []
    await world.queue("Сообщение")

    await scheduler_on(world, wakeups, rounds).tick()  # must not raise

    assert wakeups.ticks == 1
    assert await world.repo.events_of("scheduler_error") == [
        {"step": "timers", "error": "RuntimeError"}
    ]
    assert world.gateway.sent == [(PEER, "Сообщение")]  # the Sender still ran
    assert rounds == ["board"]  # and so did the rest
