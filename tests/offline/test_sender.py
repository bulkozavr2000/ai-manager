"""The Sender and the Scheduler, offline, against the core's contract only.

A fake Telegram, a fake bot, fake time and a temporary database; in place of the core a
recording stub of `ports.SendOutcomes` (`Core`) and of `ports.Wakeups` (`Wake`). The
conversations stand in the states the agent core uses, and messages are queued with
`repo.enqueue`, as the core queues them. What the core does with what the Sender tells it is
proved by the core's own flows (`test_agent_flow.py`).
"""

from __future__ import annotations

import asyncio
import inspect
import sqlite3
from collections.abc import Awaitable, Callable
from contextlib import closing
from dataclasses import dataclass, field, replace
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import keyring
import pytest
from keyring.errors import KeyringLocked

from tg_agent.clock import FakeClock
from tg_agent.config import Schedule, Settings
from tg_agent.control.base import FakeUI
from tg_agent.db import FIRST_SENT, Repo
from tg_agent.gateway.base import TgActionError, TgSendUncertain
from tg_agent.gateway.fake import FakeGateway
from tg_agent.models import (
    ApprovalKind,
    ApprovalStatus,
    Conversation,
    ConvState,
    Direction,
    MsgSource,
    OutboxItem,
    OutboxStatus,
    TaskStatus,
)
from tg_agent.policy.tg_safety import TgErrorKind, typing_seconds
from tg_agent.secrets import Key, set_secret
from tg_agent.tasks.outbox import unclear_text
from tg_agent.tasks.scheduler import Scheduler
from tg_agent.tasks.sender import Sender
from tg_agent.tasks.texts import ADVICE, NOT_SENT, who

S = ConvState
Out = OutboxStatus  # the short name reads better in the assertions
TASHKENT = ZoneInfo("Asia/Tashkent")
PEER, OTHER_PEER = 555, 777
WHO = "@print_lab"
OWNER_PHONE = "+998 90 123 45 67"
OWNER_ADDRESS = "Ташкент, ул. Навои, 15"
FIRST = "Ассаляму алейкум! Сколько будет стоить печать 5 корпусов?\n— ИИ-помощник Саида"
PRIVACY = "его настройки приватности не дают написать"
ROUND_THE_CLOCK = Schedule(work_start=time(0), work_end=time(0))


def at(day: int, hour: int, minute: int = 0, second: int = 0) -> datetime:
    """A moment of November 2026 in Tashkent."""
    return datetime(2026, 11, day, hour, minute, second, tzinfo=TASHKENT)


# --- the stand-ins for the core --------------------------------------------------------------


@dataclass
class Core:
    """A stand-in for the core: records what the Sender reports, answers what it is told to."""

    reanswer: bool = True  # what can_reanswer returns
    handles_failures: bool = False  # what on_failed returns
    raises: set[str] = field(default_factory=set)  # these methods raise RuntimeError
    asked: list[ConvState] = field(default_factory=list)  # the states can_reanswer saw
    sent: list[tuple] = field(default_factory=list)
    stale: list[int] = field(default_factory=list)
    unclear: list[int] = field(default_factory=list)
    refused: list[tuple] = field(default_factory=list)
    failed: list[tuple] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)  # the item texts the core was handed
    banned: set[int] = field(default_factory=set)  # conversations whose contact is excluded

    def _maybe_raise(self, name: str) -> None:
        if name in self.raises:
            raise RuntimeError(f"{name} broke")

    def can_reanswer(self, conv: Conversation) -> bool:
        self.asked.append(conv.state)
        self._maybe_raise("can_reanswer")
        return self.reanswer

    async def excluded(self, conv: Conversation) -> bool:
        return conv.id in self.banned

    async def on_sent(
        self, item: OutboxItem, conv: Conversation, at: datetime, *, first: bool
    ) -> None:
        self.sent.append((item.id, conv.id, at, first))
        self.texts.append(item.text)
        self._maybe_raise("on_sent")

    async def on_stale(self, conv_id: int) -> None:
        self.stale.append(conv_id)
        self._maybe_raise("on_stale")

    async def on_unclear(self, item: OutboxItem) -> None:
        self.unclear.append(item.id)
        self._maybe_raise("on_unclear")

    async def on_refused(
        self,
        item: OutboxItem,
        conv: Conversation,
        kind: TgErrorKind,
        reason: str,
        cancelled: int,
    ) -> None:
        self.refused.append((item.id, conv.id, kind, reason, cancelled))
        self._maybe_raise("on_refused")

    async def on_failed(
        self, item: OutboxItem, conv: Conversation, code: str, reason: str
    ) -> bool:
        self.failed.append((item.id, conv.id, code, reason))
        self._maybe_raise("on_failed")
        return self.handles_failures


class Wake:
    """A stand-in for the core's `Wakeups`: counts its ticks, does `work`, can raise."""

    def __init__(self) -> None:
        self.ticks = 0
        self.raises: Exception | None = None
        self.work: Callable[[], Awaitable[object]] | None = None

    async def tick(self) -> None:
        self.ticks += 1
        if self.work is not None:
            await self.work()
        if self.raises is not None:
            raise self.raises


# --- the world -------------------------------------------------------------------------------


@dataclass
class World:
    repo: Repo
    gateway: FakeGateway
    ui: FakeUI
    clock: FakeClock
    settings: Settings
    core: Core
    wake: Wake
    sender: Sender
    scheduler: Scheduler
    task_id: int
    conv_id: int
    reminders: list[str] = field(default_factory=list)

    def now(self) -> datetime:
        return self.clock.now()

    def later(self, seconds: float) -> None:
        self.clock.advance(timedelta(seconds=seconds))

    async def remind(self) -> bool:
        self.reminders.append("asked")
        return True

    def new_sender(self, settings: Settings | None = None) -> Sender:
        """Another Sender on the same world: a restart, or other settings."""
        return Sender(
            self.repo,
            self.gateway,
            self.ui,
            self.clock,
            settings or self.settings,
            self.core,
        )

    def new_scheduler(
        self,
        reminder: Callable[[], Awaitable[object]] | None = None,
        board: Callable[[], Awaitable[object]] | None = None,
    ) -> Scheduler:
        return Scheduler(
            self.wake,
            self.sender,
            self.repo,
            self.clock,
            self.settings,
            self.ui,
            reminder or self.remind,
            board=board,
        )

    async def conv(self, conv_id: int | None = None) -> Conversation:
        return await self.repo.get_conversation(conv_id or self.conv_id)

    async def walk(self, *states: ConvState, conv_id: int | None = None) -> None:
        for state in states:
            await self.repo.set_state(conv_id or self.conv_id, state)

    async def talking(self) -> None:
        """The first message went out half an hour ago and the vendor is answering."""
        sent_at = self.now() - timedelta(minutes=30)
        await self.repo.add_message(
            self.conv_id, Direction.OUT, MsgSource.AGENT, "text", FIRST, 900, sent_at
        )
        await self.repo.update_conversation(
            self.conv_id, first_sent=True, last_out_at=sent_at
        )
        await self.walk(S.TALKING)

    async def writes(self, conv_id: int | None = None) -> None:
        """The counterpart writes now (what the core's inbox records of it)."""
        await self.repo.update_conversation(
            conv_id or self.conv_id, last_in_at=self.now()
        )

    async def add_chat(
        self,
        peer_id: int = OTHER_PEER,
        username: str = "second_lab",
        *,
        stranger: bool = False,
    ) -> int:
        """Another person of the same task, written to (CONTACTED); his conversation id.

        His first message went out already, so the pacing of first messages does not hold
        his chat back; a `stranger` was never written to.
        """
        contact_id = await self.repo.upsert_contact(
            peer_id, username, "Второй цех", False
        )
        conv_id = await self.repo.create_conversation(
            self.task_id, contact_id, state=S.CONTACTED
        )
        if not stranger:
            await self.repo.update_conversation(conv_id, first_sent=True)
        return conv_id

    async def queue(
        self,
        text: str,
        *,
        after: float = 0,
        conv_id: int | None = None,
        attach: str | None = None,
    ) -> int:
        """Put a message in the outbox, to be sent `after` seconds from now."""
        return await self.repo.enqueue(
            conv_id or self.conv_id,
            text,
            self.now() + timedelta(seconds=after),
            attach_path=attach,
        )

    async def item(self, item_id: int) -> OutboxItem:
        for status in OutboxStatus:
            for item in await self.repo.outbox_in(status):
                if item.id == item_id:
                    return item
        raise AssertionError(f"no outbox item {item_id}")

    async def statuses(self, *item_ids: int) -> list[OutboxStatus]:
        return [(await self.item(item_id)).status for item_id in item_ids]

    async def pending(self, conv_id: int | None = None) -> list[OutboxItem]:
        items = await self.repo.outbox_of(conv_id or self.conv_id)
        return [item for item in items if item.status is Out.PENDING]


@pytest.fixture
async def world(repo: Repo, fake_clock: FakeClock, settings: Settings) -> World:
    gateway, ui, core, wake = FakeGateway(), FakeUI(), Core(), Wake()
    sender = Sender(repo, gateway, ui, fake_clock, settings, core)
    task_id = await repo.create_task(
        "3D-печать",
        "Узнать цену и срок печати 5 корпусов",
        {"what": "корпус из PETG", "quantity": 5},
        300_000.0,
        "UZS",
        ["name"],
        ["files/1/korpus.stl"],
    )
    await repo.set_task_status(task_id, TaskStatus.ACTIVE)
    contact_id = await repo.upsert_contact(
        PEER, "print_lab", "Акмаль | 3D Print", False
    )
    # The core moves a person from PLANNED to CONTACTED when it queues his first message.
    conv_id = await repo.create_conversation(task_id, contact_id, state=S.PLANNED)
    await repo.set_state(conv_id, S.CONTACTED)
    built = World(
        repo, gateway, ui, fake_clock, settings, core, wake, sender, None, task_id, conv_id
    )  # fmt: skip
    built.scheduler = built.new_scheduler()
    return built


async def approve(world: World, kind: ApprovalKind) -> None:
    """The owner has pressed the button for this kind of approval."""
    approval = await world.repo.create_approval(world.task_id, world.conv_id, kind, {})
    await world.repo.resolve_approval(
        approval.id, approval.nonce, ApprovalStatus.APPROVED
    )


# --- sending ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Да",  # shorter than the shortest "typing…"
        "Здравствуйте! Подскажите, пожалуйста, цену на печать пяти корпусов?",
        "Очень длинное сообщение. " * 40,  # longer than the longest
    ],
)
async def test_typing_shows_before_the_send(world: World, text: str) -> None:
    await world.queue(text)

    assert await world.sender.tick() == 1

    assert world.gateway.calls == ["typing", "send_text"]
    assert world.gateway.typed == [(PEER, typing_seconds(text))]
    assert world.gateway.sent == [(PEER, text)]


async def test_a_message_that_is_not_due_yet_waits(world: World) -> None:
    item = await world.queue("Через минуту", after=60)

    assert await world.sender.tick() == 0
    assert world.gateway.calls == []
    world.later(59)
    assert await world.sender.tick() == 0
    world.later(1)
    assert await world.sender.tick() == 1
    assert (await world.item(item)).status is Out.SENT


async def test_a_sent_message_is_recorded_and_the_core_hears_it_was_the_first(
    world: World,
) -> None:
    sent_at = world.now()
    item_id = await world.queue(FIRST)

    assert await world.sender.tick() == 1

    item = await world.item(item_id)
    assert (item.status, item.tg_msg_id, item.sent_at, item.error) == (
        Out.SENT,
        1,
        sent_at,
        None,
    )
    conv = await world.conv()
    assert (conv.first_sent, conv.last_out_at) == (True, sent_at)
    [message] = await world.repo.last_messages(world.conv_id, 5)
    assert (message.direction, message.source, message.kind) == (
        Direction.OUT,
        MsgSource.AGENT,
        "text",
    )
    assert (message.text, message.tg_msg_id, message.created_at) == (FIRST, 1, sent_at)
    assert await world.repo.events_of("first_sent", world.conv_id) == [{}]
    assert await world.repo.sent_texts(world.conv_id) == [FIRST]
    assert world.core.sent == [(item_id, world.conv_id, sent_at, True)]


async def test_a_reply_to_a_person_who_wrote_first_is_not_a_new_contact(
    world: World,
) -> None:
    # He wrote to us before our first message went: no place in the day's quota of new
    # people, and no 10-minute gap for the next stranger.
    await world.writes()
    world.later(1)
    replied_at = world.now()
    item_id = await world.queue("Здравствуйте! Да, дрель ещё нужна")
    stranger = await world.add_chat(stranger=True)
    await world.queue(FIRST, conv_id=stranger, after=1)

    assert await world.sender.tick() == 1
    world.later(1)
    assert await world.sender.tick() == 1

    conv = await world.conv()
    assert (conv.first_sent, conv.last_out_at) == (True, replied_at)
    assert world.core.sent[0] == (item_id, world.conv_id, replied_at, True)
    assert await world.repo.events_of("first_sent", world.conv_id) == []
    assert await world.repo.events_of("first_sent", stranger) == [{}]
    assert await world.repo.new_contacts_on(world.now().date()) == 1
    assert world.gateway.sent[-1] == (OTHER_PEER, FIRST)


async def test_only_the_first_message_counts_as_first(world: World) -> None:
    first = await world.queue(FIRST)
    second = await world.queue("Файл скину позже")

    assert await world.sender.tick() == 2
    assert await world.repo.events_of("first_sent", world.conv_id) == [{}]
    world.later(3600)
    third = await world.queue("Ещё вопрос")
    assert await world.sender.tick() == 1

    assert await world.repo.events_of("first_sent", world.conv_id) == [{}]
    assert (await world.conv()).last_out_at == world.now()
    assert [(item, flag) for item, _, _, flag in world.core.sent] == [
        (first, True),
        (second, False),
        (third, False),
    ]
    assert world.core.sent[-1][2] == world.now()
    assert await world.repo.new_contacts_on(world.now().date()) == 1


async def test_the_address_comes_from_the_conversation_not_from_the_item(
    world: World,
) -> None:
    other = await world.add_chat()
    await world.queue("Первому")
    await world.queue("Второму", conv_id=other)

    assert await world.sender.tick() == 2

    assert world.gateway.sent == [(PEER, "Первому"), (OTHER_PEER, "Второму")]


@pytest.mark.parametrize(
    "states",
    [
        (),
        (S.TALKING,),
        (S.TALKING, S.WAITING_OWNER),
        (S.HANDED_TO_OWNER,),
        (S.CLOSED,),
    ],
)
async def test_the_core_hears_of_every_sent_message_whatever_the_state(
    world: World, states: tuple[ConvState, ...]
) -> None:
    # What follows a sent message (a wake if the person stays silent) is the core's business:
    # the Sender arms nothing itself and tells the core in every state.
    if S.CLOSED in states:
        # He wrote to us: a first message to a closed person who never wrote is dropped
        # (fix pass D1.1), a message that answers him goes.
        await world.repo.update_conversation(
            world.conv_id, last_in_at=world.now() - timedelta(minutes=1)
        )
    item_id = await world.queue("Сообщение")
    await world.walk(*states)

    assert await world.sender.tick() == 1

    assert world.core.sent == [(item_id, world.conv_id, world.now(), True)]
    assert (await world.conv()).state is (states[-1] if states else S.CONTACTED)


# --- a crash in the middle of a send ---------------------------------------------------------


class Crash(BaseException):
    """The process dies here (power cut, kill): nothing after the raising line runs."""


async def test_crash_between_sending_and_sent_is_not_resent(world: World) -> None:
    text = "Сколько будет стоить печать?"
    item_id = await world.queue(text)
    send = world.gateway.send_text

    async def send_then_die(peer_id: int, body: str) -> int:
        await send(peer_id, body)  # the message reaches Telegram ...
        raise Crash  # ... and the program dies before it writes that down

    world.gateway.send_text = send_then_die
    with pytest.raises(Crash):
        await world.sender.tick()
    assert (await world.item(item_id)).status is Out.SENDING

    restarted = world.new_sender()  # the program starts again on the same database
    assert await restarted.recover() == 1

    assert (await world.item(item_id)).status is Out.UNKNOWN
    assert world.core.unclear == [item_id]  # the core asks the owner whether it went
    assert world.ui.notes == []
    assert await world.repo.events_of("outbox_unknown", world.conv_id) == [
        {"reason": "interrupted"}
    ]
    world.later(3600)
    assert await restarted.tick() == 0
    assert await world.sender.tick() == 0
    assert world.gateway.sent == [(PEER, text)]  # once, and not again
    assert world.core.sent == []


async def test_recover_touches_only_what_was_being_sent(world: World) -> None:
    other = await world.add_chat()
    interrupted = await world.queue("Это отправлялось")
    interrupted_file = await world.queue("", attach="files/1/korpus.stl")
    waiting = await world.queue("Это ждало", conv_id=other)
    done = await world.queue("Это ушло")
    await world.repo.mark_outbox(done, Out.SENT, tg_msg_id=7, at=world.now())
    assert await world.repo.claim_outbox(interrupted)
    assert await world.repo.claim_outbox(interrupted_file)

    assert await world.sender.recover() == 2

    assert await world.statuses(interrupted, interrupted_file, waiting, done) == [
        Out.UNKNOWN,
        Out.UNKNOWN,
        Out.PENDING,
        Out.SENT,
    ]
    assert world.core.unclear == [interrupted, interrupted_file]  # one question each
    assert await world.sender.recover() == 0
    assert len(world.core.unclear) == 2
    assert await world.sender.tick() == 1  # what was waiting goes out as usual
    assert world.gateway.sent == [(OTHER_PEER, "Это ждало")]


async def test_recover_tells_the_owner_itself_when_the_core_cannot_ask(
    world: World,
) -> None:
    world.core.raises = {"on_unclear"}
    item_id = await world.queue("Это отправлялось")
    assert await world.repo.claim_outbox(item_id)

    assert await world.sender.recover() == 1

    item = await world.item(item_id)
    assert item.status is Out.UNKNOWN
    contact = await world.repo.get_contact((await world.conv()).contact_id)
    assert world.ui.notes == [unclear_text(item, contact)]
    assert "Это отправлялось" in world.ui.notes[0] and WHO in world.ui.notes[0]


async def test_after_a_restart_an_interrupted_send_holds_its_chat_too(
    world: World,
) -> None:
    first = await world.queue("Первое")
    second = await world.queue("Второе")
    assert await world.repo.claim_outbox(
        first
    )  # it was being sent when the program died

    assert await world.sender.recover() == 1
    assert await world.sender.tick() == 0

    assert await world.statuses(first, second) == [Out.UNKNOWN, Out.PENDING]
    assert world.core.unclear == [first]
    assert world.gateway.calls == []


async def test_a_message_cancelled_after_the_list_was_read_is_not_sent(
    world: World,
) -> None:
    item_id = await world.queue("Ответ, который отменят")
    due_outbox = world.repo.due_outbox

    async def due_then_cancelled(now: datetime) -> list[OutboxItem]:
        items = await due_outbox(now)
        await world.repo.cancel_pending(world.conv_id)  # the core drops it right after
        return items

    world.repo.due_outbox = due_then_cancelled

    assert await world.sender.tick() == 0

    assert world.gateway.calls == []
    assert (await world.item(item_id)).status is Out.CANCELLED  # and it stays cancelled
    assert world.core.sent == world.core.stale == []


# --- a newer message from the counterpart ----------------------------------------------------


async def test_a_reply_made_stale_by_a_newer_message_is_dropped_and_the_core_told(
    world: World,
) -> None:
    await world.talking()
    other = await world.add_chat()
    stale = await world.queue("Цена 240 000 сум", after=90)
    stale_too = await world.queue("Срок три дня", after=90)
    elsewhere = await world.queue("Здравствуйте", conv_id=other)
    world.later(30)
    await world.writes()  # after the reply was written
    world.later(60)

    assert await world.sender.tick() == 1  # the other chat's message only

    assert await world.statuses(stale, stale_too, elsewhere) == [
        Out.CANCELLED,
        Out.CANCELLED,
        Out.SENT,
    ]
    assert world.gateway.sent == [(OTHER_PEER, "Здравствуйте")]
    assert world.core.stale == [world.conv_id]  # once for the chat, not once a message
    assert S.TALKING in world.core.asked
    assert await world.repo.events_of("outbox_cancelled", world.conv_id) == [
        {"reason": "stale", "items": 2}
    ]


async def test_nothing_queued_goes_to_a_person_the_owner_excluded(world: World) -> None:
    # R97 (b): a goodbye or the agent's own line queued before «не
    # работай с @x» is dropped by the Sender; the drop is logged, nobody is told.
    await world.talking()
    other = await world.add_chat()
    bye = await world.queue("Спасибо, пока не нужно")
    elsewhere = await world.queue("Здравствуйте", conv_id=other)
    world.core.banned.add(world.conv_id)

    assert await world.sender.tick() == 1

    assert await world.statuses(bye, elsewhere) == [Out.CANCELLED, Out.SENT]
    assert world.gateway.sent == [(OTHER_PEER, "Здравствуйте")]
    assert await world.repo.events_of("outbox_cancelled", world.conv_id) == [
        {"reason": "excluded", "items": 1}
    ]
    assert world.core.stale == [] and world.ui.sent == [] and world.ui.notes == []


async def test_a_person_excluded_while_the_message_is_typed_gets_nothing(
    world: World,
) -> None:
    await world.talking()
    bye = await world.queue("Спасибо, пока не нужно")
    typing = world.gateway.typing

    async def excluded_meanwhile(peer: int, seconds: float) -> None:
        world.core.banned.add(world.conv_id)
        await typing(peer, seconds)

    world.gateway.typing = excluded_meanwhile  # type: ignore[method-assign]

    assert await world.sender.tick() == 0

    assert await world.statuses(bye) == [Out.CANCELLED]
    assert world.gateway.sent == []


async def test_a_reply_written_after_the_message_is_not_stale(world: World) -> None:
    await world.talking()
    await world.writes()
    world.later(45)
    item_id = await world.queue("Ответ на то сообщение")  # written after it

    assert await world.sender.tick() == 1

    assert (await world.item(item_id)).status is Out.SENT
    assert world.core.stale == []


@pytest.mark.parametrize(
    "path",
    [
        (S.TALKING, S.WAITING_OWNER),  # the stalling words while the owner is asked
        (S.HANDED_TO_OWNER,),  # the handoff words
        (S.TALKING, S.CLOSED),  # the goodbye
    ],
)
async def test_where_the_core_cannot_answer_again_the_queued_words_go_out_as_they_are(
    world: World, path: tuple[ConvState, ...]
) -> None:
    # R22: nobody decides anything again in such a chat, so a newer message does not drop
    # what was queued for it: it is all the counterpart will hear.
    world.core.reanswer = False
    farewell = await world.queue("Понял, спасибо! Если что, напишу)")
    await world.walk(*path)
    world.later(60)
    await world.writes()

    assert await world.sender.tick() == 1

    assert (await world.item(farewell)).status is Out.SENT
    assert world.core.stale == []
    assert world.core.asked[0] is path[-1]  # the core was asked about the chat as it is


# --- the agent is stopped, or the owner has the chat -----------------------------------------


async def test_a_stopped_agent_sends_nothing_and_keeps_the_queue(world: World) -> None:
    item = await world.queue("Ответ")
    await world.repo.set_stopped(True)

    assert await world.sender.tick() == 0
    world.later(3600)
    assert await world.sender.tick() == 0

    assert world.gateway.calls == []
    assert (await world.item(item)).status is Out.PENDING  # it may still go out
    await world.repo.set_stopped(False)  # /resume_all
    assert await world.sender.tick() == 1
    assert world.gateway.sent == [(PEER, "Ответ")]


async def test_a_chat_the_owner_has_loses_what_was_queued_for_it(world: World) -> None:
    first = await world.queue("Первое")
    second = await world.queue("Второе", after=600)  # not due yet: dropped all the same
    other = await world.add_chat()
    elsewhere = await world.queue("Третье", conv_id=other)
    # The core's own pause: the owner wrote in the chat himself.
    until = world.now() + timedelta(hours=12)
    await world.repo.update_conversation(world.conv_id, paused_until=until)

    assert await world.sender.tick() == 1

    assert await world.statuses(first, second, elsewhere) == [
        Out.CANCELLED,
        Out.CANCELLED,
        Out.SENT,
    ]
    assert world.gateway.sent == [(OTHER_PEER, "Третье")]
    assert await world.repo.events_of("outbox_cancelled", world.conv_id) == [
        {"reason": "paused", "items": 2}
    ]
    assert world.core.stale == []


async def test_an_owners_pause_that_is_over_holds_nothing_back(world: World) -> None:
    until = world.now() - timedelta(seconds=1)
    await world.repo.update_conversation(world.conv_id, paused_until=until)
    await world.queue("Ответ")

    assert await world.sender.tick() == 1

    assert world.gateway.sent == [(PEER, "Ответ")]


async def test_the_owner_writing_while_the_agent_is_typing_stops_the_message(
    world: World,
) -> None:
    item = await world.queue("Ответ агента")
    typing = world.gateway.typing

    async def typing_while_the_owner_writes(peer_id: int, seconds: float) -> None:
        await typing(peer_id, seconds)
        until = world.now() + timedelta(hours=12)
        await world.repo.update_conversation(world.conv_id, paused_until=until)

    world.gateway.typing = typing_while_the_owner_writes

    assert await world.sender.tick() == 0

    assert world.gateway.sent == []
    assert (await world.item(item)).status is Out.CANCELLED
    assert world.core.sent == world.core.stale == []


async def test_a_stop_while_the_agent_is_typing_keeps_the_message_waiting(
    world: World,
) -> None:
    item = await world.queue("Ответ агента")
    typing = world.gateway.typing

    async def typing_while_the_owner_stops(peer_id: int, seconds: float) -> None:
        await typing(peer_id, seconds)
        await world.repo.set_stopped(True)

    world.gateway.typing = typing_while_the_owner_stops

    assert await world.sender.tick() == 0

    assert world.gateway.sent == []
    assert (await world.item(item)).status is Out.PENDING


async def test_a_message_that_arrives_while_the_agent_types_makes_the_reply_stale(
    world: World,
) -> None:
    # "Typing…" takes seconds, in which the counterpart may have written: the checks are made
    # once more before the send.
    await world.talking()
    stale = await world.queue("Цена 240 000 сум")
    later = await world.queue("И срок три дня", after=600)
    typing = world.gateway.typing

    async def typing_while_the_vendor_writes(peer_id: int, seconds: float) -> None:
        await typing(peer_id, seconds)
        world.later(5)
        await world.writes()

    world.gateway.typing = typing_while_the_vendor_writes

    assert await world.sender.tick() == 0

    assert world.gateway.sent == []
    assert await world.statuses(stale, later) == [Out.CANCELLED, Out.CANCELLED]
    assert world.core.stale == [world.conv_id]
    assert await world.repo.events_of("outbox_cancelled", world.conv_id) == [
        {"reason": "stale", "items": 2}
    ]


async def test_a_message_that_arrives_while_the_agent_types_in_a_chat_nobody_answers_in(
    world: World,
) -> None:
    world.core.reanswer = False
    await world.walk(S.TALKING)
    stalling = await world.queue("Уточню у Саида и вернусь")
    await world.walk(S.WAITING_OWNER)
    typing = world.gateway.typing

    async def typing_while_the_vendor_writes(peer_id: int, seconds: float) -> None:
        await typing(peer_id, seconds)
        world.later(5)
        await world.writes()

    world.gateway.typing = typing_while_the_vendor_writes

    assert await world.sender.tick() == 1  # the same rule as before "typing…"

    assert (await world.item(stalling)).status is Out.SENT
    assert world.core.stale == []


# --- an error while "typing…": nothing has been sent -----------------------------------------


async def test_an_error_while_typing_leaves_the_message_waiting(world: World) -> None:
    item_id = await world.queue("Сколько стоит печать?")
    world.gateway.fail_typing = ConnectionResetError("network flap while typing")

    assert await world.sender.tick() == 0

    item = await world.item(item_id)
    assert (item.status, item.not_before) == (
        Out.PENDING,
        world.now() + timedelta(minutes=1),
    )
    assert (item.tg_msg_id, item.sent_at, item.error) == (None, None, None)
    assert world.gateway.calls == ["typing"]  # no send was even tried
    assert world.ui.notes == []  # nobody is alarmed: it never left
    assert world.core.failed == world.core.unclear == []
    assert not await world.repo.is_stopped()
    assert await world.repo.events_of("typing_failed", world.conv_id) == [
        {"attempt": 1, "error": "ConnectionResetError"}
    ]

    world.gateway.fail_typing = None  # the connection is back
    world.later(59)
    assert await world.sender.tick() == 0
    world.later(1)
    assert await world.sender.tick() == 1
    assert world.gateway.sent == [(PEER, "Сколько стоит печать?")]
    assert world.ui.notes == []


async def test_typing_that_keeps_failing_is_tried_three_more_times_then_the_message_fails(
    world: World,
) -> None:
    item_id = await world.queue("Сообщение")
    world.gateway.fail_typing = ConnectionResetError("the connection keeps dropping")

    for attempt in (1, 2, 3):  # the first try, and two of the three retries
        assert await world.sender.tick() == 0
        item = await world.item(item_id)
        assert (item.status, item.not_before) == (
            Out.PENDING,
            world.now() + timedelta(minutes=1),
        ), attempt
        assert world.ui.notes == []
        world.later(60)
    assert await world.sender.tick() == 0  # the last retry fails as well

    item = await world.item(item_id)
    code = "typing: ConnectionResetError"
    assert (item.status, item.error) == (Out.FAILED, code)
    reason = "не получилось начать отправку (ConnectionResetError)"
    assert world.core.failed == [(item_id, world.conv_id, code, reason)]
    [note] = world.ui.notes  # the core did not take care of it: the owner is told
    assert WHO in note and "ConnectionResetError" in note
    assert note.endswith(ADVICE)
    assert world.gateway.calls == ["typing"] * 4  # and not a message was sent
    assert world.gateway.sent == []
    attempts = await world.repo.events_of("typing_failed", world.conv_id)
    assert [event["attempt"] for event in attempts] == [1, 2, 3, 4]
    world.later(3600)
    assert await world.sender.tick() == 0  # nothing is tried again
    assert len(world.ui.notes) == 1


async def test_a_typing_failure_keeps_the_order_of_a_chats_messages(
    world: World,
) -> None:
    await world.queue("Первое")
    await world.queue("Второе")
    world.gateway.fail_typing = ConnectionResetError("a flap")

    assert (
        await world.sender.tick() == 0
    )  # the second is not tried in front of the first

    assert world.gateway.calls == ["typing"]
    world.gateway.fail_typing = None
    world.later(60)
    assert await world.sender.tick() == 2
    assert [text for _, text in world.gateway.sent] == ["Первое", "Второе"]


async def test_cancelling_while_typing_puts_the_message_back(world: World) -> None:
    item_id = await world.queue("Ответ")
    typing_started = asyncio.Event()

    async def typing_for_ever(peer_id: int, seconds: float) -> None:
        typing_started.set()
        await asyncio.Event().wait()

    world.gateway.typing = typing_for_ever
    tick = asyncio.create_task(world.sender.tick())
    await typing_started.wait()
    assert (await world.item(item_id)).status is Out.SENDING  # claimed, and "typing…"

    tick.cancel()  # the program is being stopped
    with pytest.raises(asyncio.CancelledError):
        await tick

    assert (
        await world.item(item_id)
    ).status is Out.PENDING  # nothing left: no question
    restarted = world.new_sender()
    assert await restarted.recover() == 0
    assert world.ui.notes == [] and world.core.unclear == []
    del world.gateway.typing  # the fake's own method again
    assert await restarted.tick() == 1
    assert world.gateway.sent == [(PEER, "Ответ")]


async def test_an_unclassified_telegram_error_while_typing_is_retried_like_any_other(
    world: World,
) -> None:
    item_id = await world.queue("Сообщение")
    world.gateway.fail_typing = TgActionError(TgErrorKind.OTHER, "SET_TYPING_FAILED")

    assert await world.sender.tick() == 0

    item = await world.item(item_id)
    assert (item.status, item.not_before) == (
        Out.PENDING,
        world.now() + timedelta(minutes=1),
    )
    assert world.core.refused == []  # not a refusal of the chat
    assert (await world.conv()).state is S.CONTACTED
    assert world.ui.notes == []


async def test_a_flood_wait_while_typing_waits_as_telegram_says(world: World) -> None:
    item_id = await world.queue("Сообщение")
    world.gateway.fail_typing = TgActionError(
        TgErrorKind.FLOOD_WAIT, "FLOOD_WAIT_300", 300
    )

    assert await world.sender.tick() == 0

    item = await world.item(item_id)
    assert (item.status, item.not_before) == (
        Out.PENDING,
        world.now() + timedelta(seconds=300),
    )
    assert world.ui.notes == []
    assert await world.repo.events_of("typing_failed", world.conv_id) == []
    world.gateway.fail_typing = None
    world.later(299)
    assert await world.sender.tick() == 0
    world.later(1)
    assert await world.sender.tick() == 1


async def test_a_final_refusal_while_typing_is_handled_as_for_a_send(
    world: World,
) -> None:
    item_id = await world.queue("Сообщение")
    world.gateway.fail_typing = TgActionError(
        TgErrorKind.PRIVACY, "USER_PRIVACY_RESTRICTED"
    )

    assert await world.sender.tick() == 0

    item = await world.item(item_id)
    assert (item.status, item.error) == (Out.FAILED, "privacy: USER_PRIVACY_RESTRICTED")
    assert world.core.refused == [
        (item_id, world.conv_id, TgErrorKind.PRIVACY, PRIVACY, 0)
    ]
    assert world.ui.notes == []  # the core tells the owner
    assert world.gateway.sent == []


async def test_peer_flood_while_typing_stops_the_agent(world: World) -> None:
    item_id = await world.queue("Сообщение")
    world.gateway.fail_typing = TgActionError(TgErrorKind.PEER_FLOOD, "PEER_FLOOD")

    assert await world.sender.tick() == 0

    assert await world.repo.is_stopped()
    assert (await world.item(item_id)).status is Out.PENDING
    [alarm] = world.ui.notes
    assert "peer_flood" in alarm
    assert await world.repo.stopped_why() == "peer_flood"  # the board says why
    assert world.core.refused == []


# --- out of hours ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("now", "morning"),
    [
        (at(10, 21, 30), at(11, 9)),  # after the evening
        (at(10, 21, 0), at(11, 9)),  # the very end of the day
        (at(11, 5, 0), at(11, 9)),  # before dawn
    ],
)
async def test_night_items_wait_until_morning(
    world: World, now: datetime, morning: datetime
) -> None:
    await world.talking()  # not a first message: only the working hours hold it
    item_id = await world.queue("Добрый вечер")  # queued while it was still day
    world.clock.set(now)

    assert await world.sender.tick() == 0

    item = await world.item(item_id)
    assert (item.status, item.not_before) == (Out.PENDING, morning)
    assert world.gateway.calls == []
    assert world.ui.notes == []  # nobody is told about a night's wait
    world.clock.set(morning - timedelta(seconds=1))
    assert await world.sender.tick() == 0
    world.clock.set(morning)
    assert await world.sender.tick() == 1
    assert world.gateway.sent == [(PEER, "Добрый вечер")]


# --- first messages to new people (Task 3.1) -------------------------------------------------


async def written_first(world: World, moment: datetime, peer_id: int) -> int:
    """Another new person of the task, whose first message went out at `moment`."""
    conv_id = await world.add_chat(peer_id=peer_id, username=f"lab_{peer_id}")
    now = world.now()
    world.clock.set(moment)
    await world.repo.log_event(FIRST_SENT, conv_id)  # as `mark_sent` logs it
    world.clock.set(now)
    await world.repo.update_conversation(conv_id, first_sent=True)
    return conv_id


async def quota_used(world: World) -> None:
    """Today's quota of first messages (3 by default) went out this morning."""
    for minute, peer_id in ((0, 801), (20, 802), (40, 803)):
        await written_first(world, at(10, 9, minute), peer_id)


async def test_every_sender_paces_first_messages(world: World) -> None:
    # No switch any more: the Sender the program builds holds a first message too.
    await quota_used(world)
    await world.queue(FIRST)

    assert await world.sender.tick() == 0

    [item] = await world.pending()
    assert item.not_before == at(11, 9)
    assert world.gateway.sent == []
    assert await world.repo.events_of("first_contact_deferred", world.conv_id) == [
        {"reason": "quota"}
    ]


async def test_a_first_message_due_at_night_waits_for_the_morning_with_the_chat(
    world: World,
) -> None:
    first = await world.queue(FIRST)
    second = await world.queue("Можно прислать файл?", after=1)
    world.clock.set(at(10, 22))

    assert await world.new_sender().tick() == 0

    assert [item.not_before for item in await world.pending()] == [at(11, 9)] * 2
    assert await world.statuses(first, second) == [Out.PENDING, Out.PENDING]
    assert await world.repo.events_of("first_contact_deferred", world.conv_id) == [
        {"reason": "hours"}
    ]
    assert world.gateway.calls == []


async def test_round_the_clock_a_first_message_still_waits_for_new_contacts_start(
    world: World,
) -> None:
    night = replace(world.settings, schedule=ROUND_THE_CLOCK)
    await world.queue(FIRST)
    world.clock.set(at(11, 3))

    assert await world.new_sender(night).tick() == 0

    [item] = await world.pending()
    assert item.not_before == at(11, 9)
    world.clock.set(at(11, 9))
    assert await world.new_sender(night).tick() == 1


async def test_over_the_quota_a_first_message_waits_for_tomorrow(world: World) -> None:
    await quota_used(world)
    await world.queue(FIRST)

    assert await world.new_sender().tick() == 0

    [item] = await world.pending()
    assert item.not_before == at(11, 9)
    assert await world.repo.events_of("first_contact_deferred", world.conv_id) == [
        {"reason": "quota"}
    ]


async def test_a_first_message_goes_and_the_next_new_person_waits_for_the_gap(
    world: World,
) -> None:
    other = await world.add_chat(stranger=True)
    await world.queue(FIRST)
    later = await world.queue(
        "Здравствуйте! Печатаете корпуса?", after=1, conv_id=other
    )
    sender = world.new_sender()
    sent_at = world.now()

    assert await sender.tick() == 1
    world.later(1)
    assert await sender.tick() == 0

    assert world.gateway.sent == [(PEER, FIRST)]
    assert (await world.item(later)).not_before == sent_at + timedelta(minutes=1)
    assert await world.repo.events_of("first_contact_deferred", other) == [
        {"reason": "gap"}
    ]
    world.clock.set(sent_at + timedelta(minutes=1))
    assert await sender.tick() == 1
    assert world.gateway.sent[-1] == (OTHER_PEER, "Здравствуйте! Печатаете корпуса?")


async def situation_of(world: World, situation: str) -> Settings:
    """The world arranged so that a first message to a new person would be held now."""
    if situation == "hours":  # round the clock: the night is closed to new people only
        world.clock.set(at(11, 3))
        return replace(world.settings, schedule=ROUND_THE_CLOCK)
    if situation == "quota":
        await quota_used(world)
    else:
        await written_first(world, world.now() - timedelta(minutes=2), 804)
    return world.settings


@pytest.mark.parametrize("situation", ["hours", "quota", "gap"])
async def test_a_message_to_a_person_written_to_goes_at_once(
    world: World, situation: str
) -> None:
    settings = await situation_of(world, situation)
    await world.talking()
    await world.queue("Понял, спасибо")

    assert await world.new_sender(settings).tick() == 1

    assert world.gateway.sent == [(PEER, "Понял, спасибо")]
    assert await world.repo.events_of("first_contact_deferred") == []


@pytest.mark.parametrize("situation", ["hours", "quota", "gap"])
async def test_a_reply_to_a_person_who_wrote_first_goes_at_once(
    world: World, situation: str
) -> None:
    settings = await situation_of(world, situation)
    await world.writes()
    world.later(1)
    await world.queue("Здравствуйте! Да, дрель ещё нужна")

    assert await world.new_sender(settings).tick() == 1

    assert world.gateway.sent == [(PEER, "Здравствуйте! Да, дрель ещё нужна")]
    assert await world.repo.events_of("first_contact_deferred") == []


# --- what Telegram answers -------------------------------------------------------------------


async def test_flood_wait_reschedules(world: World) -> None:
    now = world.now()
    other = await world.add_chat()
    first = await world.queue("Первое")
    second = await world.queue("Второе", conv_id=other)
    world.gateway.fail_next = TgActionError(
        TgErrorKind.FLOOD_WAIT, "FLOOD_WAIT_120", wait_s=120
    )

    assert await world.sender.tick() == 0

    item = await world.item(first)
    assert item.status is Out.PENDING
    assert item.not_before == now + timedelta(seconds=120)
    assert (item.tg_msg_id, item.sent_at) == (None, None)
    assert world.gateway.sent == []
    assert world.gateway.calls.count("send_text") == 1  # Telegram is not asked again
    assert (await world.item(second)).status is Out.PENDING
    assert world.ui.notes == []  # waiting is the normal answer to a flood wait
    assert world.core.refused == world.core.unclear == []
    assert not await world.repo.is_stopped()
    assert await world.repo.events_of("flood_wait", world.conv_id) == [{"seconds": 120}]

    assert (
        await world.sender.tick() == 0
    )  # nothing goes out while Telegram asks to wait
    world.later(119)
    assert await world.sender.tick() == 0
    world.later(1)
    assert await world.sender.tick() == 2
    # the chat that was not refused goes first: it was waiting since before the refusal
    assert world.gateway.sent == [(OTHER_PEER, "Второе"), (PEER, "Первое")]


async def test_a_flood_wait_keeps_the_order_of_a_chats_messages(world: World) -> None:
    first = await world.queue("Первое")
    second = await world.queue("Второе")
    third = await world.queue("Третье", after=30)  # not due yet, but behind the refused
    world.gateway.fail_next = TgActionError(
        TgErrorKind.FLOOD_WAIT, "FLOOD_WAIT_120", 120
    )

    assert await world.sender.tick() == 0

    until = world.now() + timedelta(seconds=120)
    items = [await world.item(item_id) for item_id in (first, second, third)]
    assert [item.not_before for item in items] == [until] * 3
    world.later(120)
    assert await world.sender.tick() == 3
    assert [text for _, text in world.gateway.sent] == ["Первое", "Второе", "Третье"]


async def test_a_flood_wait_counts_from_now_not_from_when_the_message_was_due(
    world: World,
) -> None:
    item_id = await world.queue("Давно пора было")
    world.later(3 * 3600)  # the program was asleep
    world.gateway.fail_next = TgActionError(
        TgErrorKind.FLOOD_WAIT, "FLOOD_WAIT_300", 300
    )

    assert await world.sender.tick() == 0

    assert (await world.item(item_id)).not_before == world.now() + timedelta(
        seconds=300
    )


async def test_a_flood_wait_without_a_time_still_waits(world: World) -> None:
    item_id = await world.queue("Сообщение")
    world.gateway.fail_next = TgActionError(TgErrorKind.FLOOD_WAIT, "FLOOD_WAIT")

    assert await world.sender.tick() == 0

    assert (await world.item(item_id)).not_before == world.now() + timedelta(seconds=60)


@pytest.mark.parametrize(
    "kind", [TgErrorKind.PEER_FLOOD, TgErrorKind.FROZEN, TgErrorKind.DEAD_SESSION]
)
async def test_peer_flood_stops_all_sending(world: World, kind: TgErrorKind) -> None:
    other = await world.add_chat()
    first = await world.queue("Первое")
    second = await world.queue("Второе", conv_id=other)
    world.gateway.fail_next = TgActionError(kind, kind.value.upper())

    assert await world.sender.tick() == 0

    assert await world.repo.is_stopped()
    assert world.gateway.sent == []
    assert world.gateway.calls.count("send_text") == 1  # nothing else is tried
    assert await world.statuses(first, second) == [Out.PENDING, Out.PENDING]
    assert (await world.item(first)).error == kind.value
    [alarm] = world.ui.notes
    # no command to remember: the button is under the pinned status
    assert kind.value in alarm and "«Продолжить работу»" in alarm
    assert "/resume_all" not in alarm
    assert await world.repo.stopped_why() == kind.value
    assert await world.repo.events_of("stopped") == [{"reason": kind.value}]
    assert world.core.refused == []  # the chats themselves are not refused
    assert (await world.conv()).state is S.CONTACTED

    world.later(3600)
    assert await world.sender.tick() == 0  # stopped stays stopped
    await world.repo.set_stopped(False)  # the owner has looked into it
    assert await world.sender.tick() == 2
    assert world.gateway.sent == [(PEER, "Первое"), (OTHER_PEER, "Второе")]


async def test_a_privacy_refusal_fails_the_message_drops_the_chats_queue_and_tells_the_core(
    world: World,
) -> None:
    await world.walk(S.TALKING)
    other = await world.add_chat()
    first = await world.queue("Первое")
    second = await world.queue("Второе")
    elsewhere = await world.queue("Третье", conv_id=other)
    world.gateway.fail_next = TgActionError(
        TgErrorKind.PRIVACY, "USER_PRIVACY_RESTRICTED"
    )

    assert await world.sender.tick() == 1  # the other chat is not affected

    failed = await world.item(first)
    assert (failed.status, failed.error) == (
        Out.FAILED,
        "privacy: USER_PRIVACY_RESTRICTED",
    )
    assert await world.statuses(second, elsewhere) == [Out.CANCELLED, Out.SENT]
    # The core pauses the chat and tells the owner; the Sender only reports.
    assert world.core.refused == [
        (first, world.conv_id, TgErrorKind.PRIVACY, PRIVACY, 1)
    ]
    assert world.ui.notes == [] and world.ui.sent == []
    assert (await world.conv()).state is S.TALKING
    assert not await world.repo.is_stopped()
    assert world.gateway.sent == [(OTHER_PEER, "Третье")]


@pytest.mark.parametrize(
    ("kind", "words"),
    [
        (TgErrorKind.BLOCKED, "заблокировал"),
        (TgErrorKind.PREMIUM_REQUIRED, "Premium"),
        (TgErrorKind.PAYMENT_REQUIRED, "заплатить"),
        (TgErrorKind.OTHER, "Telegram отказал: RPC_ERROR"),
    ],
)
async def test_the_other_refusals_fail_the_message_and_tell_the_core_why(
    world: World, kind: TgErrorKind, words: str
) -> None:
    item_id = await world.queue("Сообщение")
    world.gateway.fail_next = TgActionError(kind, "RPC_ERROR")

    assert await world.sender.tick() == 0

    assert (await world.item(item_id)).status is Out.FAILED
    [(refused_id, conv_id, refused_kind, reason, cancelled)] = world.core.refused
    assert (refused_id, conv_id, refused_kind, cancelled) == (
        item_id,
        world.conv_id,
        kind,
        0,
    )
    assert words in reason
    assert world.ui.notes == []


async def test_a_refusal_in_a_closed_chat_fails_the_message_and_the_state_stays(
    world: World,
) -> None:
    # He wrote to us (TALKING): the goodbye answers him, so it is not a first message to a
    # closed person, which would be dropped (fix pass D1.1).
    await world.repo.update_conversation(
        world.conv_id, last_in_at=world.now() - timedelta(minutes=1)
    )
    item_id = await world.queue("Понял, спасибо!")
    await world.walk(S.TALKING, S.CLOSED)
    world.core.reanswer = False
    world.gateway.fail_next = TgActionError(
        TgErrorKind.PRIVACY, "USER_PRIVACY_RESTRICTED"
    )

    assert await world.sender.tick() == 0

    assert (await world.item(item_id)).status is Out.FAILED
    assert (await world.conv()).state is S.CLOSED  # the Sender moves no state
    [(_, _, _, _, cancelled)] = world.core.refused
    assert cancelled == 0


async def test_a_refusal_the_core_cannot_take_is_told_to_the_owner_by_the_sender(
    world: World,
) -> None:
    world.core.raises = {"on_refused"}
    item_id = await world.queue("Первое")
    await world.queue("Второе")
    world.gateway.fail_next = TgActionError(
        TgErrorKind.PRIVACY, "USER_PRIVACY_RESTRICTED"
    )

    assert await world.sender.tick() == 0

    assert (await world.item(item_id)).status is Out.FAILED
    assert len(world.core.refused) == 1
    contact = await world.repo.get_contact((await world.conv()).contact_id)
    assert world.ui.notes == [
        NOT_SENT.format(who=who(contact), reason=PRIVACY) + ADVICE
    ]
    assert WHO in world.ui.notes[0]


async def test_an_unexpected_error_leaves_the_outcome_unknown_and_is_not_retried(
    world: World,
) -> None:
    other = await world.add_chat()
    first = await world.queue("Первое")
    second = await world.queue("Второе", conv_id=other)
    send = world.gateway.send_text

    async def broken(peer_id: int, text: str) -> int:
        raise ConnectionResetError(f"reset while sending {text}")

    world.gateway.send_text = broken

    assert await world.sender.tick() == 0

    assert await world.statuses(first, second) == [Out.UNKNOWN, Out.PENDING]
    assert world.core.unclear == [first]  # the core asks the owner
    assert world.ui.notes == []
    assert (await world.item(first)).error == "ConnectionResetError"  # no text in it

    world.gateway.send_text = send  # the connection is back
    assert await world.sender.tick() == 1
    assert world.gateway.sent == [(OTHER_PEER, "Второе")]  # the first never again
    assert (await world.item(first)).status is Out.UNKNOWN


async def test_a_send_the_gateway_cannot_vouch_for_is_asked_about_and_never_resent(
    world: World,
) -> None:
    # The message may be in the chat, so it is not a refusal: no failed message, nothing
    # refused; the core asks the owner and the message is never sent again.
    other = await world.add_chat()
    first = await world.queue("Первое")
    second = await world.queue("Второе", conv_id=other)
    world.gateway.fail_next = TgSendUncertain("ConnectionError")

    assert await world.sender.tick() == 0

    assert await world.statuses(first, second) == [Out.UNKNOWN, Out.PENDING]
    assert world.gateway.calls.count("send_text") == 1  # one at a time
    assert world.core.unclear == [first]
    assert world.core.refused == world.core.sent == []
    assert (await world.conv()).state is S.CONTACTED
    assert not await world.repo.is_stopped()
    assert (await world.item(first)).error == "TgSendUncertain"
    assert await world.repo.events_of("outbox_unknown", world.conv_id) == [
        {"reason": "TgSendUncertain"}
    ]

    assert await world.sender.tick() == 1
    assert world.gateway.sent == [(OTHER_PEER, "Второе")]
    assert (await world.item(first)).status is Out.UNKNOWN


async def test_a_send_that_broke_off_is_told_to_the_owner_when_the_core_cannot_ask(
    world: World,
) -> None:
    world.core.raises = {"on_unclear"}
    item_id = await world.queue("Первое")
    world.gateway.fail_next = TgSendUncertain("ConnectionError")

    assert await world.sender.tick() == 0

    item = await world.item(item_id)
    assert item.status is Out.UNKNOWN
    assert world.core.unclear == [item_id]
    contact = await world.repo.get_contact((await world.conv()).contact_id)
    assert world.ui.notes == [unclear_text(item, contact)]


async def unclear(world: World) -> tuple[int, int]:
    """«Первое» and «Второе» are queued for the vendor; the send of «Первое» breaks off."""
    await world.walk(S.TALKING)
    first = await world.queue("Первое")
    second = await world.queue("Второе")
    world.gateway.fail_next = TgSendUncertain("ConnectionError")
    await world.sender.tick()
    assert world.core.unclear == [first]
    return first, second


async def test_after_an_unclear_send_the_rest_of_that_chat_waits_for_the_owner(
    world: World,
) -> None:
    # The next messages of the chat must not go out without the first: the vendor could get
    # the second half without the first.
    first, second = await unclear(world)
    other = await world.add_chat()
    elsewhere = await world.queue("Третье", conv_id=other)

    for _ in range(12):
        world.later(300)
        await world.sender.tick()

    assert await world.statuses(first, second) == [Out.UNKNOWN, Out.PENDING]
    assert (await world.item(elsewhere)).status is Out.SENT  # other chats go on
    assert world.gateway.sent == [(OTHER_PEER, "Третье")]
    assert world.core.unclear == [first]  # asked once


@pytest.mark.parametrize("settled", [Out.SENT, Out.CANCELLED])
async def test_once_the_unclear_message_is_settled_the_rest_of_the_chat_goes(
    world: World, settled: OutboxStatus
) -> None:
    # The owner's answer (or the message found in the chat) is the core's: it settles the
    # unclear row, and then the Sender lets the chat go on.
    first, second = await unclear(world)
    await world.repo.mark_outbox(first, settled)

    assert await world.sender.tick() == 1

    assert world.gateway.sent == [(PEER, "Второе")]
    assert (await world.item(second)).status is Out.SENT


# --- the owner's phone and address -----------------------------------------------------------


def everything_stored(db_path: Path) -> str:
    """Every row of the database, as text: where the real phone must never be found."""
    with closing(sqlite3.connect(db_path)) as db:
        return "\n".join(db.iterdump())


async def test_a_placeholder_is_filled_in_once_the_owner_approved_it(
    world: World, db_path: Path
) -> None:
    set_secret(Key.OWNER_PHONE, OWNER_PHONE)
    set_secret(Key.OWNER_ADDRESS, OWNER_ADDRESS)
    await approve(world, ApprovalKind.SHARE_PHONE)
    await approve(world, ApprovalKind.SHARE_ADDRESS)
    text = "Мой номер {PHONE}, адрес {address}. Звоните!"
    item_id = await world.queue(text)

    assert await world.sender.tick() == 1

    assert world.gateway.sent == [
        (PEER, f"Мой номер {OWNER_PHONE}, адрес {OWNER_ADDRESS}. Звоните!")
    ]
    # What is stored, and what the core is handed, is the placeholder text, never the number.
    assert (await world.item(item_id)).text == text
    [message] = await world.repo.last_messages(world.conv_id, 5)
    assert message.text == text
    assert await world.repo.sent_texts(world.conv_id) == [text]
    assert world.core.texts == [text]
    stored = everything_stored(db_path)
    assert "123 45 67" not in stored
    assert "Навои" not in stored
    assert not any("123 45 67" in note or "Навои" in note for note in world.ui.notes)


async def test_only_the_placeholders_that_are_there_need_a_value(world: World) -> None:
    set_secret(Key.OWNER_PHONE, OWNER_PHONE)  # no address stored, and none asked for
    await approve(world, ApprovalKind.SHARE_PHONE)
    await world.queue("Телефон: {PHONE}")

    assert await world.sender.tick() == 1

    assert world.gateway.sent == [(PEER, f"Телефон: {OWNER_PHONE}")]


@pytest.mark.parametrize(
    ("text", "kind", "what"),
    [
        ("Звоните {PHONE}", ApprovalKind.SHARE_ADDRESS, "телефон"),  # the other one
        ("Адрес: {ADDRESS}", ApprovalKind.SHARE_PHONE, "адрес"),
        ("Звоните {PHONE}", None, "телефон"),
        ("Звоните {PHONE}", ApprovalKind.QUESTION, "телефон"),
        ("Звоните ｛PHONE｝", None, "телефон"),  # fullwidth braces: read as one
    ],
)
async def test_a_placeholder_without_the_owners_permission_is_not_sent(
    world: World, text: str, kind: ApprovalKind | None, what: str
) -> None:
    set_secret(Key.OWNER_PHONE, OWNER_PHONE)
    set_secret(Key.OWNER_ADDRESS, OWNER_ADDRESS)
    if kind is not None:
        await approve(world, kind)
    item_id = await world.queue(text)
    pending = await world.repo.create_approval(
        world.task_id, world.conv_id, ApprovalKind.SHARE_PHONE, {}
    )  # asked, not answered: no permission either
    assert pending.status is ApprovalStatus.PENDING

    assert await world.sender.tick() == 0

    item = await world.item(item_id)
    assert (item.status, item.error) == (Out.FAILED, "not_approved")
    assert world.gateway.calls == []  # not even "typing…"
    [(failed_id, _, code, reason)] = world.core.failed
    assert (failed_id, code) == (item_id, "not_approved")
    assert what in reason
    [note] = world.ui.notes
    assert what in note and WHO in note
    assert OWNER_PHONE not in note


async def test_a_message_that_cannot_be_prepared_does_not_block_the_others(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = await world.add_chat()
    await approve(world, ApprovalKind.SHARE_PHONE)
    broken = await world.queue("Звоните {PHONE}")
    plain = await world.queue("Обычное сообщение", conv_id=other)

    def locked(service: str, username: str) -> str:
        raise KeyringLocked("the keyring is locked")

    monkeypatch.setattr(keyring, "get_password", locked)

    assert await world.sender.tick() == 1  # the other chat's message goes out this tick

    assert await world.statuses(broken, plain) == [Out.FAILED, Out.SENT]
    assert (await world.item(broken)).error == "KeyringLocked"
    assert world.gateway.sent == [(OTHER_PEER, "Обычное сообщение")]
    assert world.gateway.calls == [
        "typing",
        "send_text",
    ]  # nothing tried for the broken
    reason = "не получилось его подготовить (KeyringLocked)"
    assert world.core.failed == [(broken, world.conv_id, "KeyringLocked", reason)]
    [note] = world.ui.notes
    assert WHO in note and "KeyringLocked" in note
    assert await world.repo.events_of("outbox_error", world.conv_id) == [
        {"reason": "KeyringLocked", "failed": True}
    ]
    world.later(5)
    assert await world.sender.tick() == 0  # it does not come back to block the line
    assert len(world.ui.notes) == 1


async def test_a_core_that_breaks_on_a_stale_reply_does_not_stop_the_other_chats(
    world: World,
) -> None:
    world.core.raises = {"on_stale"}
    await world.talking()
    other = await world.add_chat()
    stale = await world.queue("Устаревший ответ", after=60)
    elsewhere = await world.queue("Другому", conv_id=other, after=60)
    world.later(10)
    await world.writes()
    world.later(60)

    assert await world.sender.tick() == 1

    assert await world.statuses(stale, elsewhere) == [Out.CANCELLED, Out.SENT]
    assert world.gateway.sent == [(OTHER_PEER, "Другому")]
    assert world.core.stale == [world.conv_id]
    assert await world.repo.events_of("outbox_error", world.conv_id) == [
        {
            "reason": "RuntimeError",
            "failed": False,
        }  # the reply was dropped: nothing to fail
    ]
    assert world.core.failed == [] and world.ui.notes == []


async def test_a_placeholder_whose_value_is_not_stored_is_not_sent(
    world: World,
) -> None:
    await approve(world, ApprovalKind.SHARE_PHONE)  # the keyring has no phone
    item_id = await world.queue("Звоните {PHONE}")

    assert await world.sender.tick() == 0

    item = await world.item(item_id)
    assert (item.status, item.error) == (Out.FAILED, "no_value")
    assert world.gateway.calls == []
    assert [code for _, _, code, _ in world.core.failed] == ["no_value"]
    [note] = world.ui.notes
    assert "телефон" in note


async def test_a_blank_value_is_no_value(world: World) -> None:
    set_secret(Key.OWNER_PHONE, "   ")
    await approve(world, ApprovalKind.SHARE_PHONE)
    item_id = await world.queue("Звоните {PHONE}")

    assert await world.sender.tick() == 0

    assert (await world.item(item_id)).error == "no_value"


async def test_a_value_that_holds_a_placeholder_is_not_sent(world: World) -> None:
    set_secret(Key.OWNER_ADDRESS, "{PHONE}")  # nonsense, but no placeholder may go out
    await approve(world, ApprovalKind.SHARE_ADDRESS)
    item_id = await world.queue("Адрес: {ADDRESS}")

    assert await world.sender.tick() == 0

    assert (await world.item(item_id)).error == "placeholder_left"
    assert world.gateway.calls == []
    assert [code for _, _, code, _ in world.core.failed] == ["placeholder_left"]


# --- files -----------------------------------------------------------------------------------


async def test_a_task_file_goes_through_send_file(world: World) -> None:
    path = world.settings.data_dir / "files" / "1" / "korpus.stl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"solid korpus")
    absolute = world.settings.data_dir / "elsewhere.pdf"
    absolute.write_bytes(b"%PDF")
    text_id = await world.queue("Файл сейчас пришлю")
    file_id = await world.queue("", attach="files/1/korpus.stl")
    absolute_id = await world.queue("", attach=str(absolute))

    assert await world.sender.tick() == 3

    assert world.gateway.calls == ["typing", "send_text"] + ["typing", "send_file"] * 2
    assert world.gateway.sent == [(PEER, "Файл сейчас пришлю")]
    assert world.gateway.files == [(PEER, str(path)), (PEER, str(absolute))]
    assert await world.statuses(text_id, file_id, absolute_id) == [Out.SENT] * 3
    messages = await world.repo.last_messages(world.conv_id, 5)
    assert [(m.kind, m.text, m.source, m.direction) for m in messages[1:]] == [
        ("document", "", MsgSource.AGENT, Direction.OUT)
    ] * 2
    assert [m.tg_msg_id for m in messages] == [1, 2, 3]
    assert [first for _, _, _, first in world.core.sent] == [True, False, False]


async def test_a_file_that_is_not_there_fails_the_message_and_tells_the_owner(
    world: World,
) -> None:
    item_id = await world.queue("", attach="files/1/korpus.stl")  # nothing saved there

    assert await world.sender.tick() == 0

    item = await world.item(item_id)
    assert (item.status, item.error) == (Out.FAILED, "file_missing")
    assert world.gateway.calls == []
    assert world.core.failed == [
        (item_id, world.conv_id, "file_missing", "файла «korpus.stl» нет на диске")
    ]
    [note] = world.ui.notes
    assert "korpus.stl" in note and WHO in note
    assert (await world.conv()).state is S.CONTACTED  # the chat itself is fine


# --- what the Sender fails by itself goes to the core first (Task 2.6) -----------------------


async def fails_by_itself(
    world: World, how: str, monkeypatch: pytest.MonkeyPatch
) -> int:
    """A message the Sender will fail without Telegram's word, in one of four ways."""
    if how == "file_missing":
        return await world.queue("", attach="files/1/korpus.stl")
    if how == "not_approved":
        return await world.queue("Звоните {PHONE}")
    if how == "typing":
        item_id = await world.queue("Сообщение")
        world.gateway.fail_typing = ConnectionResetError(
            "the connection keeps dropping"
        )
        for _ in range(3):  # three failures put it back; the fourth fails it
            await world.sender.tick()
            world.later(60)
        return item_id
    # "broke": the message cannot be prepared (the keyring is locked)
    await approve(world, ApprovalKind.SHARE_PHONE)
    item_id = await world.queue("Звоните {PHONE}")

    def locked(service: str, username: str) -> str:
        raise KeyringLocked("the keyring is locked")

    monkeypatch.setattr(keyring, "get_password", locked)
    return item_id


HOWS = {
    "file_missing": "file_missing",
    "not_approved": "not_approved",
    "typing": "typing: ConnectionResetError",
    "broke": "KeyringLocked",
}


@pytest.mark.parametrize("how", list(HOWS))
async def test_a_failure_the_core_takes_care_of_is_not_told_to_the_owner(
    world: World, how: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.core.handles_failures = True
    item_id = await fails_by_itself(world, how, monkeypatch)

    assert await world.sender.tick() == 0

    item = await world.item(item_id)
    assert (item.status, item.error) == (Out.FAILED, HOWS[how])
    [(failed_id, conv_id, code, reason)] = world.core.failed
    assert (failed_id, conv_id, code) == (item_id, world.conv_id, HOWS[how])
    assert reason  # the phrase the owner would have read
    assert world.ui.notes == []
    assert world.gateway.sent == []


@pytest.mark.parametrize("how", list(HOWS))
@pytest.mark.parametrize("core", ["declines", "breaks"])
async def test_a_failure_the_core_does_not_take_is_told_to_the_owner(
    world: World, how: str, core: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if core == "breaks":
        world.core.raises = {"on_failed"}
    item_id = await fails_by_itself(world, how, monkeypatch)

    assert await world.sender.tick() == 0

    assert (await world.item(item_id)).status is Out.FAILED
    [(_, _, code, reason)] = world.core.failed
    assert code == HOWS[how]
    [note] = world.ui.notes
    assert note.startswith("Не отправил сообщение") and WHO in note
    assert reason in note


# --- reading the chat ------------------------------------------------------------------------


async def test_the_chat_is_marked_read_before_the_agent_types(world: World) -> None:
    await world.talking()
    await world.repo.add_message(
        world.conv_id,
        Direction.IN,
        MsgSource.COUNTERPART,
        "text",
        "45 000",
        901,
        world.now() - timedelta(seconds=1),
    )
    await world.queue("Спасибо!")

    assert await world.sender.tick() == 1

    # As a person reads before answering: the vendor sees his message read.
    assert world.gateway.calls == ["mark_read", "typing", "send_text"]
    assert world.gateway.read == [(PEER, 901)]

    await world.queue("И ещё одно")  # nothing new came in: not marked again
    assert await world.sender.tick() == 1
    assert world.gateway.read == [(PEER, 901)]


async def test_a_read_mark_that_fails_does_not_hold_the_message_back(
    world: World,
) -> None:
    await world.talking()
    await world.repo.add_message(
        world.conv_id,
        Direction.IN,
        MsgSource.COUNTERPART,
        "text",
        "45 000",
        901,
        world.now() - timedelta(seconds=1),
    )

    async def unreadable(peer_id: int, max_id: int) -> None:
        world.gateway.calls.append("mark_read")
        raise TgActionError(TgErrorKind.OTHER, "TimeoutError")

    world.gateway.mark_read = unreadable  # type: ignore[method-assign]
    await world.queue("Спасибо!")

    assert await world.sender.tick() == 1

    assert world.gateway.calls == ["mark_read", "typing", "send_text"]
    assert world.gateway.sent == [(PEER, "Спасибо!")]


# --- the Scheduler ---------------------------------------------------------------------------


async def test_one_round_runs_its_steps_in_order(world: World) -> None:
    order: list[str] = []

    async def flush() -> int:
        order.append("flush")
        return 0

    async def wake_queues_a_reply() -> None:
        order.append("timers")
        await world.queue("Ответ, поставленный ядром")

    real_sender_tick = world.sender.tick

    async def sender_tick() -> int:
        order.append("sender")
        return await real_sender_tick()

    async def reminder() -> bool:
        order.append("reminder")
        return False

    real_purge = world.repo.purge_messages_before

    async def purge(before: datetime) -> int:
        order.append("purge")
        return await real_purge(before)

    async def board() -> None:
        order.append("board")

    world.ui.flush = flush
    world.wake.work = wake_queues_a_reply
    world.sender.tick = sender_tick
    world.repo.purge_messages_before = purge
    world.clock.set(at(10, 20, 30))
    await world.talking()

    await world.new_scheduler(reminder, board).tick()

    assert order == ["flush", "timers", "sender", "reminder", "purge", "board"]
    assert world.wake.ticks == 1
    # what the core queued in this round goes out in the same round
    assert world.gateway.sent == [(PEER, "Ответ, поставленный ядром")]


async def test_the_scheduler_wakes_the_core_every_round(world: World) -> None:
    for _ in range(3):
        await world.scheduler.tick()
        world.later(5)

    assert world.wake.ticks == 3


async def test_the_scheduler_refreshes_the_status_board_every_round(
    world: World,
) -> None:
    rounds: list[str] = []

    async def board() -> None:
        rounds.append("board")

    scheduler = world.new_scheduler(board=board)
    await scheduler.tick()
    await scheduler.tick()

    assert rounds == ["board", "board"]


async def test_the_ui_is_flushed_on_every_tick(world: World) -> None:
    flushes: list[datetime] = []

    async def flush() -> int:
        flushes.append(world.now())
        return 0

    world.ui.flush = flush

    await world.scheduler.tick()
    world.later(5)
    await world.scheduler.tick()

    assert flushes == [at(10, 10, 0), at(10, 10, 0, 5)]


async def test_a_failing_step_does_not_stop_the_others(world: World) -> None:
    async def flush_that_fails() -> int:
        raise RuntimeError("the bot is down")

    async def board_that_fails() -> None:
        raise LookupError("no board message")

    world.ui.flush = flush_that_fails
    world.wake.raises = ValueError("a bug in the core")
    await world.queue("Сообщение")

    await world.new_scheduler(board=board_that_fails).tick()  # must not raise

    assert world.gateway.sent == [(PEER, "Сообщение")]  # the sender step still ran
    assert await world.repo.get_kv("purge_on") == "2026-11-10"  # and the daily job
    assert await world.repo.events_of("scheduler_error") == [
        {"step": "flush", "error": "RuntimeError"},
        {"step": "timers", "error": "ValueError"},
        {"step": "board", "error": "LookupError"},
    ]


async def test_a_step_that_keeps_failing_is_reported_once_in_ten_minutes(
    world: World,
) -> None:
    world.wake.raises = RuntimeError("the core is broken")

    async def rounds(count: int, seconds: float = 5) -> None:
        for _ in range(count):
            world.later(seconds)
            await world.scheduler.tick()

    async def reports() -> int:
        return len(await world.repo.events_of("scheduler_error"))

    await rounds(5)  # twenty-five seconds of the same failure
    assert await reports() == 1
    assert world.wake.ticks == 5  # the core is woken every round all the same
    await rounds(1, 10 * 60)  # still failing, ten minutes on
    assert await reports() == 2
    world.wake.raises = None
    await rounds(1)
    world.wake.raises = RuntimeError("again")  # it worked in between: news at once
    await rounds(1)
    assert await reports() == 3


async def test_run_forever_ticks_until_it_is_cancelled(world: World) -> None:
    assert (
        inspect.signature(Scheduler.run_forever).parameters["interval_s"].default == 5
    )
    ticks: list[int] = []

    async def counting_tick() -> None:
        ticks.append(len(ticks))

    world.scheduler.tick = counting_tick
    loop = asyncio.create_task(world.scheduler.run_forever(interval_s=0.01))
    for _ in range(300):
        if len(ticks) >= 3:
            break
        await asyncio.sleep(0.01)

    loop.cancel()
    with pytest.raises(asyncio.CancelledError):
        await loop
    assert len(ticks) >= 3


async def test_the_reminder_runs_once_a_day_from_2030(world: World) -> None:
    world.clock.set(at(10, 20, 29, 59))
    await world.scheduler.tick()
    assert world.reminders == []

    world.clock.set(at(10, 20, 30))
    await world.scheduler.tick()
    assert world.reminders == ["asked"]
    assert await world.repo.get_kv("digest_sent_on") == "2026-11-10"

    for later in (at(10, 20, 30, 5), at(10, 21, 0), at(10, 23, 59, 59)):
        world.clock.set(later)
        await world.scheduler.tick()
    assert world.reminders == ["asked"]  # once a day

    world.clock.set(at(11, 20, 30))  # and again the next day
    await world.scheduler.tick()
    assert world.reminders == ["asked", "asked"]
    assert await world.repo.get_kv("digest_sent_on") == "2026-11-11"


async def test_a_reminder_missed_at_2030_runs_when_the_program_starts_that_evening(
    world: World,
) -> None:
    world.clock.set(at(10, 22, 0))  # the computer was off at 20:30
    await world.scheduler.tick()
    assert world.reminders == ["asked"]

    world.clock.set(at(11, 7, 0))  # yesterday's is not repeated, today's is not due
    await world.scheduler.tick()
    assert world.reminders == ["asked"]


async def test_a_reminder_with_nothing_to_send_still_counts_the_day(
    world: World,
) -> None:
    async def nothing_waited() -> bool:
        world.reminders.append("asked")
        return False

    scheduler = world.new_scheduler(nothing_waited)
    world.clock.set(at(10, 20, 31))

    await scheduler.tick()
    await scheduler.tick()

    assert world.reminders == ["asked"]
    assert await world.repo.get_kv("digest_sent_on") == "2026-11-10"


async def test_a_reminder_that_fails_is_tried_again_and_does_not_stop_the_rest(
    world: World,
) -> None:
    async def failing() -> bool:
        world.reminders.append("asked")
        raise RuntimeError("the bot is down")

    scheduler = world.new_scheduler(failing)
    world.clock.set(at(10, 20, 31))
    await world.talking()  # not a first message: nothing but the reminder is in the way
    await world.queue("Сообщение")

    await scheduler.tick()
    await scheduler.tick()

    assert world.gateway.sent == [(PEER, "Сообщение")]
    assert world.reminders == ["asked", "asked"]  # not done, so it is tried again
    assert await world.repo.get_kv("digest_sent_on") is None
    assert await world.repo.get_kv("purge_on") == "2026-11-10"  # the other daily job


async def test_old_messages_are_purged_once_a_day(world: World) -> None:
    now = world.now()

    async def say(text: str, days: int) -> None:
        await world.repo.add_message(
            world.conv_id,
            Direction.IN,
            MsgSource.COUNTERPART,
            "text",
            text,
            None,
            now - timedelta(days=days),
        )

    await say("старое", 91)
    await say("свежее", 80)

    await world.scheduler.tick()

    assert [m.text for m in await world.repo.last_messages(world.conv_id, 10)] == [
        "свежее"
    ]
    assert await world.repo.get_kv("purge_on") == "2026-11-10"
    assert await world.repo.events_of("purge") == [{"deleted": 1}]

    await say("ещё старое", 95)
    world.later(3600)
    await world.scheduler.tick()  # the same day: not again
    assert len(await world.repo.last_messages(world.conv_id, 10)) == 2

    world.later(24 * 3600)  # the next day
    await world.scheduler.tick()
    assert [m.text for m in await world.repo.last_messages(world.conv_id, 10)] == [
        "свежее"
    ]
    assert await world.repo.get_kv("purge_on") == "2026-11-11"
