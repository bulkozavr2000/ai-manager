"""One lock per conversation, shared by the inbox, the send pipeline and the cards
(`agent/locks.py`). Each test makes the interleaving happen on purpose: a Repo call inside the
send pipeline's last section starts another coroutine and yields before it goes on."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pytest

from tg_agent.agent.approvals import Approvals
from tg_agent.agent.inbox import Inbox
from tg_agent.agent.locks import ConvLocks
from tg_agent.agent.outgoing import Outgoing, Queued, Refused
from tg_agent.brain.llm import FakeLLM
from tg_agent.clock import FakeClock
from tg_agent.config import Settings
from tg_agent.control.base import FakeUI
from tg_agent.db import Repo
from tg_agent.gateway.base import Incoming
from tg_agent.gateway.fake import FakeGateway
from tg_agent.models import (
    Conversation,
    ConvState,
    Direction,
    MsgSource,
    OutboxItem,
    OutboxStatus,
    Task,
    TaskStatus,
)
from tg_agent.policy.tg_safety import TgErrorKind

S = ConvState
PEER = 501


class Outcomes:
    """A `SendOutcomes` the inbox does not call in these tests."""

    def can_reanswer(self, conv: Conversation) -> bool:
        return True

    async def on_sent(
        self, item: OutboxItem, conv: Conversation, at: datetime, *, first: bool
    ) -> None:
        raise AssertionError("not expected")

    async def on_stale(self, conv_id: int) -> None:
        raise AssertionError("not expected")

    async def on_unclear(self, item: OutboxItem) -> None:
        raise AssertionError("not expected")

    async def on_refused(
        self,
        item: OutboxItem,
        conv: Conversation,
        kind: TgErrorKind,
        reason: str,
        cancelled: int,
    ) -> None:
        raise AssertionError("not expected")


@dataclass
class World:
    repo: Repo
    clock: FakeClock
    outgoing: Outgoing
    approvals: Approvals
    inbox: Inbox
    task: Task
    conv_id: int


async def make_world(
    repo: Repo, clock: FakeClock, settings: Settings, *, shared: bool
) -> World:
    locks = {"locks": ConvLocks()} if shared else {}
    gateway = FakeGateway()
    gateway.name = "ИИ помощник Саида"  # no introduction
    outgoing = Outgoing(
        repo,
        gateway,
        FakeLLM(),
        clock,
        settings,
        random.Random(5),
        canary="ZX-CANARY-7731",
        owner_secrets=[],
        **locks,
    )
    approvals = Approvals(repo, FakeUI(), clock, settings, outgoing, **locks)
    inbox = Inbox(repo, clock, settings, Outcomes(), **locks)
    task_id = await repo.create_task("Дрель", "купить дрель", {}, None, "UZS", [], [])
    await repo.set_task_status(task_id, TaskStatus.ACTIVE)
    contact_id = await repo.upsert_contact(PEER, "ivan_tools", "Иван", False)
    conv_id = await repo.create_conversation(task_id, contact_id, state=S.PLANNED)
    for state in (S.CONTACTED, S.TALKING):
        await repo.set_state(conv_id, state)
    now = clock.now()
    await repo.add_message(
        conv_id, Direction.OUT, MsgSource.AGENT, "text", "Есть дрель?", None, now
    )
    await repo.add_message(
        conv_id, Direction.IN, MsgSource.COUNTERPART, "text", "Есть", None, now
    )
    await repo.update_conversation(
        conv_id, first_sent=True, last_out_at=now, last_in_at=now
    )
    clock.advance(timedelta(minutes=1))
    return World(
        repo, clock, outgoing, approvals, inbox, await repo.get_task(task_id), conv_id
    )


@pytest.fixture
async def world(repo: Repo, fake_clock: FakeClock, settings: Settings) -> World:
    return await make_world(repo, fake_clock, settings, shared=True)


async def send(world: World, text: str = "Сколько стоит?") -> Queued | Refused:
    return await world.outgoing.send(
        world.task,
        world.conv_id,
        [text],
        language="ru",
        commitment=False,
        attach_files=False,
        seen_message_id=await world.repo.max_incoming_id(world.conv_id),
    )


def meanwhile(
    monkeypatch: pytest.MonkeyPatch,
    repo: Repo,
    name: str,
    other: Callable[[], Awaitable[Any]],
    *,
    before: bool = True,
    after: Callable[[], None] | None = None,
) -> list[asyncio.Task]:
    """At the first call of `repo.<name>`, start `other` as a task and let it run (before the
    real call, or after it); `after` runs once it has had its turn."""
    real = getattr(repo, name)
    started: list[asyncio.Task] = []

    async def interleave() -> None:
        task = asyncio.create_task(other())
        started.append(task)
        # The database runs in a thread: give the other coroutine real time to go as far as
        # it can (to its end, or to the lock it waits for).
        await asyncio.wait([task], timeout=0.3)
        if after is not None:
            after()

    async def patched(*args: Any, **kwargs: Any) -> Any:
        first = not started
        if first and before:
            await interleave()
        result = await real(*args, **kwargs)
        if first and not before:
            await interleave()
        return result

    monkeypatch.setattr(repo, name, patched)
    return started


async def test_one_lock_per_conversation() -> None:
    locks = ConvLocks()
    assert locks.of(1) is locks.of(1)
    assert locks.of(1) is not locks.of(2)


async def test_write_myself_pressed_while_a_reply_is_being_queued(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def card_then_write_myself() -> None:
        await world.approvals.ask(
            world.task,
            kind="phone",
            question="Дать?",
            conv_id=world.conv_id,
            suggested=[],
            amount=None,
            terms="",
        )
        [approval] = await world.repo.pending_approvals(world.conv_id)
        await world.approvals.owner_decision(approval.id, approval.nonce, "self")

    started = meanwhile(
        monkeypatch, world.repo, "cancel_pending", card_then_write_myself
    )

    assert isinstance(await send(world), Queued)
    await asyncio.gather(*started)

    conv = await world.repo.get_conversation(world.conv_id)
    assert conv.state is S.HANDED_TO_OWNER
    assert all(
        item.status is not OutboxStatus.PENDING
        for item in await world.repo.outbox_of(world.conv_id)
    )


async def test_a_card_asked_while_a_reply_is_being_queued(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def card() -> None:
        await world.approvals.ask(
            world.task,
            kind="question",
            question="Брать?",
            conv_id=world.conv_id,
            suggested=[],
            amount=None,
            terms="",
        )

    started = meanwhile(monkeypatch, world.repo, "cancel_pending", card)

    assert isinstance(await send(world), Queued)
    await asyncio.gather(*started)

    [item] = await world.repo.outbox_of(world.conv_id)
    assert item.status is OutboxStatus.PENDING  # it came before the card
    assert (await world.repo.get_conversation(world.conv_id)).state is S.WAITING_OWNER
    assert isinstance(await send(world, "И ещё"), Refused)


async def test_a_message_reported_while_a_reply_is_being_queued(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    message = Incoming(PEER, 77, "Алло?", "text", world.clock.now())
    started = meanwhile(
        monkeypatch,
        world.repo,
        "enqueue",
        lambda: world.inbox.on_incoming(message),
        before=False,
        after=lambda: world.clock.advance(timedelta(seconds=5)),
    )

    assert isinstance(await send(world), Queued)
    await asyncio.gather(*started)

    [item] = await world.repo.outbox_of(world.conv_id)
    conv = await world.repo.get_conversation(world.conv_id)
    assert conv.last_in_at is not None
    # The inbox waited for the queue: the Sender's stale check drops this reply.
    assert conv.last_in_at > item.created_at


async def test_without_a_shared_lock_everything_works_as_before(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    world = await make_world(repo, fake_clock, settings, shared=False)

    assert isinstance(await send(world), Queued)
    result = await world.approvals.ask(
        world.task,
        kind="phone",
        question="Дать?",
        conv_id=world.conv_id,
        suggested=[],
        amount=None,
        terms="",
    )
    assert result[1] is False
    [approval] = await repo.pending_approvals(world.conv_id)
    assert await world.approvals.owner_decision(approval.id, approval.nonce, "yes") == (
        "Ок, телефон дадим"
    )
    await world.inbox.on_incoming(
        Incoming(PEER, 78, "Спасибо", "text", fake_clock.now())
    )
    conv = await repo.get_conversation(world.conv_id)
    assert conv.state is S.TALKING
    assert conv.last_in_at == fake_clock.now()
