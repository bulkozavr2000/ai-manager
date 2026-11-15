"""The inbox of the agent core: what Telegram reports becomes stored messages and events."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from tg_agent.agent.inbox import Inbox, takes_events
from tg_agent.clock import FakeClock
from tg_agent.config import Settings
from tg_agent.db import Repo
from tg_agent.gateway.base import Incoming
from tg_agent.models import (
    AgentEvent,
    Conversation,
    ConvState,
    Direction,
    EventKind,
    MsgSource,
    OutboxItem,
    OutboxStatus,
    TaskStatus,
)
from tg_agent.policy.tg_safety import TgErrorKind

S = ConvState
PEER = 501
WINDOW = timedelta(seconds=45)
MAX_WAIT = timedelta(seconds=180)


class Outcomes:
    """A recording `SendOutcomes`."""

    def __init__(self) -> None:
        self.sent: list[tuple[int, int, datetime, bool]] = []

    def can_reanswer(self, conv: Conversation) -> bool:
        return True

    async def on_sent(
        self, item: OutboxItem, conv: Conversation, at: datetime, *, first: bool
    ) -> None:
        self.sent.append((item.id, conv.id, at, first))

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


@pytest.fixture
def outcomes() -> Outcomes:
    return Outcomes()


@pytest.fixture
def inbox(
    repo: Repo, fake_clock: FakeClock, settings: Settings, outcomes: Outcomes
) -> Inbox:
    return Inbox(repo, fake_clock, settings, outcomes)


async def new_conv(
    repo: Repo,
    *path: ConvState,
    status: TaskStatus = TaskStatus.ACTIVE,
    peer_id: int = PEER,
) -> Conversation:
    task_id = await repo.create_task("Мастер", "найти мастера", {}, None, "UZS", [], [])
    await repo.set_task_status(task_id, status)
    contact_id = await repo.upsert_contact(peer_id, "ivan", "Ivan", False)
    conv_id = await repo.create_conversation(task_id, contact_id, state=S.PLANNED)
    for state in path:
        await repo.set_state(conv_id, state)
    return await repo.get_conversation(conv_id)


def incoming(
    clock: FakeClock, tg_msg_id: int, text: str = "Здравствуйте", **kw: object
) -> Incoming:
    return Incoming(
        peer_id=kw.pop("peer_id", PEER),  # type: ignore[arg-type]
        tg_msg_id=tg_msg_id,
        text=text,
        kind=kw.pop("kind", "text"),  # type: ignore[arg-type]
        date=clock.now(),
        **kw,  # type: ignore[arg-type]
    )


async def events(repo: Repo, task_id: int, kind: EventKind) -> list[AgentEvent]:
    return [e for e in await repo.open_events(task_id) if e.kind is kind]


async def messages(repo: Repo, conv_id: int):
    return await repo.last_messages(conv_id, 100)


# -- on_incoming --


async def test_incoming_in_contacted(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    conv = await new_conv(repo, S.CONTACTED)
    await repo.update_conversation(conv.id, followups_sent=2, first_sent=True)
    await repo.arm_silence(
        conv.task_id, conv.id, fake_clock.now() + timedelta(hours=24)
    )
    now = fake_clock.now()

    await inbox.on_incoming(incoming(fake_clock, 10, "Да, могу"))

    [message] = await messages(repo, conv.id)
    assert (message.direction, message.source, message.kind) == (
        Direction.IN,
        MsgSource.COUNTERPART,
        "text",
    )
    assert (message.text, message.tg_msg_id, message.created_at) == (
        "Да, могу",
        10,
        now,
    )
    fresh = await repo.get_conversation(conv.id)
    assert fresh.state is S.TALKING
    assert fresh.last_in_at == now
    assert fresh.followups_sent == 0
    assert await events(repo, conv.task_id, EventKind.SILENCE) == []
    [event] = await events(repo, conv.task_id, EventKind.INCOMING)
    assert event.conversation_id == conv.id
    assert event.due_at == now + WINDOW


async def test_a_burst_makes_one_event_pushed_back_but_not_past_the_cap(
    repo: Repo, fake_clock: FakeClock, settings: Settings, outcomes: Outcomes
) -> None:
    limits = replace(settings.limits, batch_max_seconds=50)
    inbox = Inbox(repo, fake_clock, replace(settings, limits=limits), outcomes)
    conv = await new_conv(repo, S.CONTACTED)
    first = fake_clock.now()

    await inbox.on_incoming(incoming(fake_clock, 10, "раз"))
    [event] = await events(repo, conv.task_id, EventKind.INCOMING)
    assert event.due_at == first + WINDOW

    fake_clock.advance(timedelta(seconds=10))
    await inbox.on_incoming(incoming(fake_clock, 11, "два"))
    [event] = await events(repo, conv.task_id, EventKind.INCOMING)
    assert event.due_at == first + timedelta(seconds=50)  # 10 + 45 = 55 > the cap of 50

    fake_clock.advance(timedelta(seconds=10))
    await inbox.on_incoming(incoming(fake_clock, 12, "три"))
    [event] = await events(repo, conv.task_id, EventKind.INCOMING)
    assert event.due_at == first + timedelta(seconds=50)
    assert len(await messages(repo, conv.id)) == 3


async def test_a_burst_is_pushed_back_with_each_message(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    conv = await new_conv(repo, S.CONTACTED)
    first = fake_clock.now()
    for n in range(3):
        await inbox.on_incoming(incoming(fake_clock, 10 + n, f"часть {n}"))
        [event] = await events(repo, conv.task_id, EventKind.INCOMING)
        assert event.due_at == fake_clock.now() + WINDOW
        assert event.due_at <= first + MAX_WAIT
        fake_clock.advance(timedelta(seconds=10))


async def test_a_duplicate_is_stored_once(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    conv = await new_conv(repo, S.CONTACTED)

    await inbox.on_incoming(incoming(fake_clock, 10))
    await inbox.on_incoming(incoming(fake_clock, 10))

    assert len(await messages(repo, conv.id)) == 1
    assert len(await repo.events_of("duplicate_incoming", conv.id)) == 1


async def test_an_auto_reply_is_stored_without_an_event(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    conv = await new_conv(repo, S.CONTACTED)
    await repo.arm_silence(
        conv.task_id, conv.id, fake_clock.now() + timedelta(hours=24)
    )

    await inbox.on_incoming(
        incoming(fake_clock, 10, "Я сейчас не на связи", offline=True)
    )

    [message] = await messages(repo, conv.id)
    assert message.text == "Я сейчас не на связи"
    assert len(await repo.events_of("auto_reply", conv.id)) == 1
    assert await events(repo, conv.task_id, EventKind.INCOMING) == []
    assert len(await events(repo, conv.task_id, EventKind.SILENCE)) == 1
    fresh = await repo.get_conversation(conv.id)
    assert fresh.state is S.CONTACTED
    assert fresh.last_in_at is None


async def test_a_repeated_canned_text_is_an_auto_reply(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    conv = await new_conv(repo, S.CONTACTED)
    greeting = "Здравствуйте! Спасибо за обращение, мы ответим в течение рабочего дня."
    await repo.add_message(
        conv.id,
        Direction.IN,
        MsgSource.COUNTERPART,
        "text",
        greeting,
        5,
        fake_clock.now(),
    )
    await repo.update_conversation(conv.id, last_out_at=fake_clock.now())

    await inbox.on_incoming(incoming(fake_clock, 10, greeting))

    assert len(await repo.events_of("auto_reply", conv.id)) == 1
    assert await events(repo, conv.task_id, EventKind.INCOMING) == []


@pytest.mark.parametrize(
    ("path", "status"),
    [
        ((S.CONTACTED, S.WAITING_OWNER), TaskStatus.ACTIVE),
        ((S.CONTACTED, S.HANDED_TO_OWNER), TaskStatus.ACTIVE),
        ((S.CONTACTED,), TaskStatus.PAUSED),
    ],
    ids=["waiting_owner", "handed_to_owner", "task_paused"],
)
async def test_a_conversation_that_takes_no_events_stores_the_message(
    repo: Repo,
    inbox: Inbox,
    fake_clock: FakeClock,
    path: tuple[ConvState, ...],
    status: TaskStatus,
) -> None:
    conv = await new_conv(repo, *path, status=status)

    await inbox.on_incoming(incoming(fake_clock, 10))

    assert len(await messages(repo, conv.id)) == 1
    assert (await repo.get_conversation(conv.id)).last_in_at == fake_clock.now()
    assert await repo.open_events(conv.task_id) == []


async def test_the_owners_pause_stores_the_message_without_an_event(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    conv = await new_conv(repo, S.CONTACTED, S.TALKING)
    await repo.update_conversation(
        conv.id, paused_until=fake_clock.now() + timedelta(hours=1)
    )

    await inbox.on_incoming(incoming(fake_clock, 10))

    assert len(await messages(repo, conv.id)) == 1
    assert await repo.open_events(conv.task_id) == []


async def test_a_past_pause_does_not_hold_the_event(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    conv = await new_conv(repo, S.CONTACTED, S.TALKING)
    await repo.update_conversation(
        conv.id, paused_until=fake_clock.now() - timedelta(minutes=1)
    )

    await inbox.on_incoming(incoming(fake_clock, 10))

    assert len(await events(repo, conv.task_id, EventKind.INCOMING)) == 1


async def test_a_closed_person_of_an_active_task_is_heard_again_a_stranger_is_not(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    # Ruling R104 (b): his talk opens again and the agent has his message.
    conv = await new_conv(repo, S.CLOSED)

    await inbox.on_incoming(incoming(fake_clock, 10))
    await inbox.on_incoming(incoming(fake_clock, 11, peer_id=999))

    assert [m.tg_msg_id for m in await messages(repo, conv.id)] == [10]
    assert (await repo.get_conversation(conv.id)).state is S.TALKING
    assert len(await events(repo, conv.task_id, EventKind.INCOMING)) == 1
    assert len(await repo.events_of("closed_heard_again", conv.id)) == 1


# -- on_edit --


async def test_an_edit_replaces_the_text_and_bumps_the_event(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    conv = await new_conv(repo, S.CONTACTED)
    await inbox.on_incoming(incoming(fake_clock, 10, "Цена 100"))
    fake_clock.advance(timedelta(seconds=20))

    await inbox.on_edit(PEER, 10, "Цена 120")

    [message] = await messages(repo, conv.id)
    assert message.text == "Цена 120"
    assert (await repo.get_conversation(conv.id)).last_in_at == fake_clock.now()
    [event] = await events(repo, conv.task_id, EventKind.INCOMING)
    assert event.due_at == fake_clock.now() + WINDOW


async def test_an_edit_after_the_event_was_handled_adds_a_new_one(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    conv = await new_conv(repo, S.CONTACTED, S.TALKING)
    await repo.add_message(
        conv.id,
        Direction.IN,
        MsgSource.COUNTERPART,
        "text",
        "Цена 100",
        10,
        fake_clock.now(),
    )

    await inbox.on_edit(PEER, 10, "Цена 120")

    [event] = await events(repo, conv.task_id, EventKind.INCOMING)
    assert event.due_at == fake_clock.now() + WINDOW


async def test_an_edit_of_an_unknown_message_does_nothing(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    conv = await new_conv(repo, S.CONTACTED, S.TALKING)

    await inbox.on_edit(PEER, 77, "что-то")

    assert (await repo.get_conversation(conv.id)).last_in_at is None
    assert await repo.open_events(conv.task_id) == []


async def test_an_edit_in_waiting_owner_adds_no_event(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    conv = await new_conv(repo, S.CONTACTED, S.WAITING_OWNER)
    await repo.add_message(
        conv.id,
        Direction.IN,
        MsgSource.COUNTERPART,
        "text",
        "Цена 100",
        10,
        fake_clock.now(),
    )

    await inbox.on_edit(PEER, 10, "Цена 120")

    assert (await messages(repo, conv.id))[0].text == "Цена 120"
    assert await repo.open_events(conv.task_id) == []


# -- on_delete --


async def test_a_delete_marks_the_text_once_and_bumps_the_event(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    conv = await new_conv(repo, S.CONTACTED, S.TALKING)
    await repo.add_message(
        conv.id,
        Direction.IN,
        MsgSource.COUNTERPART,
        "text",
        "Цена 100",
        10,
        fake_clock.now(),
    )

    await inbox.on_delete([10, 99])
    await inbox.on_delete([10])

    [message] = await messages(repo, conv.id)
    assert message.text == "(удалено собеседником) Цена 100"
    assert len(await repo.events_of("incoming_deleted", conv.id)) == 1
    [event] = await events(repo, conv.task_id, EventKind.INCOMING)
    assert event.due_at == fake_clock.now() + WINDOW


async def test_a_delete_in_a_closed_conversation_is_left_alone(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    conv = await new_conv(repo, S.CONTACTED)
    await repo.add_message(
        conv.id,
        Direction.IN,
        MsgSource.COUNTERPART,
        "text",
        "Цена 100",
        10,
        fake_clock.now(),
    )
    await repo.set_state(conv.id, S.CLOSED)

    await inbox.on_delete([10])

    assert (await messages(repo, conv.id))[0].text == "Цена 100"
    assert await repo.open_events(conv.task_id) == []


# -- on_owner_manual --


async def test_the_owner_writes_himself(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    conv = await new_conv(repo, S.CONTACTED, S.TALKING)
    await inbox.on_incoming(incoming(fake_clock, 10, "Сколько?"))
    await repo.arm_silence(
        conv.task_id, conv.id, fake_clock.now() + timedelta(hours=24)
    )
    await repo.add_event(
        conv.task_id, EventKind.EDITED, conv_id=conv.id, due_at=fake_clock.now()
    )
    pending = await repo.enqueue(conv.id, "Ответ агента", fake_clock.now())
    fake_clock.advance(timedelta(seconds=5))
    now = fake_clock.now()

    await inbox.on_owner_manual(PEER, "Сам отвечу", 11)

    message = (await messages(repo, conv.id))[-1]
    assert (message.direction, message.source, message.text, message.tg_msg_id) == (
        Direction.OUT,
        MsgSource.OWNER_MANUAL,
        "Сам отвечу",
        11,
    )
    fresh = await repo.get_conversation(conv.id)
    assert fresh.paused_until == now + timedelta(hours=12)
    assert fresh.last_out_at == now
    assert (await repo.get_outbox(pending)).status is OutboxStatus.CANCELLED
    left = await repo.open_events(conv.task_id)
    assert [e.kind for e in left] == [EventKind.OWNER_WROTE]
    assert left[0].due_at == fresh.paused_until
    assert left[0].conversation_id == conv.id
    assert len(await repo.events_of("owner_manual", conv.id)) == 1

    fake_clock.advance(timedelta(hours=1))
    await inbox.on_owner_manual(PEER, "И ещё", 12)

    again = await repo.get_conversation(conv.id)
    assert again.paused_until == fake_clock.now() + timedelta(hours=12)
    [event] = await events(repo, conv.task_id, EventKind.OWNER_WROTE)
    assert event.due_at == again.paused_until


async def test_the_owners_message_already_stored_is_not_taken_again(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    conv = await new_conv(repo, S.CONTACTED, S.TALKING)
    await inbox.on_owner_manual(PEER, "Сам отвечу", 11)
    fake_clock.advance(timedelta(hours=1))

    await inbox.on_owner_manual(PEER, "Сам отвечу", 11)

    assert len(await messages(repo, conv.id)) == 1
    assert len(await repo.events_of("duplicate_outgoing", conv.id)) == 1
    assert (
        await repo.get_conversation(conv.id)
    ).paused_until == fake_clock.now() + timedelta(hours=11)


async def test_the_owner_in_a_closed_chat_of_an_active_task_starts_his_quiet(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    # His message is stored and the person's messages do not reopen the talk until the
    # quiet is over (R104 b); nothing wakes the agent.
    conv = await new_conv(repo, S.CLOSED)

    await inbox.on_owner_manual(PEER, "Привет", 11)
    await inbox.on_incoming(incoming(fake_clock, 12, "Да?"))

    fresh = await repo.get_conversation(conv.id)
    assert fresh.state is S.CLOSED
    assert fresh.paused_until == fake_clock.now() + timedelta(hours=12)
    assert [m.text for m in await messages(repo, conv.id)] == ["Привет", "Да?"]
    assert await repo.open_events(conv.task_id) == []


async def test_the_owner_in_a_closed_chat_cancels_what_is_queued_to_him(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    # Fix round 9.2 (mutation M16): the goodbye a wake queued does not follow his words.
    conv = await new_conv(repo, S.CLOSED)
    item_id = await repo.enqueue(conv.id, "Спасибо, всего доброго", fake_clock.now())

    await inbox.on_owner_manual(PEER, "Это Саид, я сам позвоню", 11)

    assert (await repo.get_outbox(item_id)).status is OutboxStatus.CANCELLED


async def test_the_owners_message_that_is_an_unclear_item_marks_it_sent(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock, outcomes: Outcomes
) -> None:
    conv = await new_conv(repo, S.CONTACTED)
    item_id = await repo.enqueue(
        conv.id, "Мой номер {PHONE}, звоните", fake_clock.now()
    )
    await repo.mark_outbox(item_id, OutboxStatus.UNKNOWN)
    other = await repo.enqueue(conv.id, "Следующее", fake_clock.now())
    now = fake_clock.now()

    await inbox.on_owner_manual(PEER, "Мой номер +998 90 123 45 67,  звоните", 21)

    item = await repo.get_outbox(item_id)
    assert item.status is OutboxStatus.SENT
    assert item.tg_msg_id == 21
    assert outcomes.sent == [(item_id, conv.id, now, True)]
    fresh = await repo.get_conversation(conv.id)
    assert fresh.paused_until is None
    assert fresh.first_sent is True
    assert (await repo.get_outbox(other)).status is OutboxStatus.PENDING
    [message] = await messages(repo, conv.id)
    assert (message.source, message.tg_msg_id) == (MsgSource.AGENT, 21)
    assert len(await repo.events_of("outbox_found_sent", conv.id)) == 1
    assert await events(repo, conv.task_id, EventKind.OWNER_WROTE) == []


@pytest.mark.parametrize("status", [TaskStatus.DONE, TaskStatus.CANCELLED])
async def test_a_finished_task_s_chat_is_not_listened_to(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock, status: TaskStatus
) -> None:
    # A person of a DONE or CANCELLED task is free for a new task: his conversation is not
    # live, so neither his messages nor the owner's in his chat are stored.
    conv = await new_conv(repo, S.CONTACTED, S.TALKING, status=status)
    pending = await repo.enqueue(conv.id, "Спасибо!", fake_clock.now())

    await inbox.on_incoming(incoming(fake_clock, 10))
    await inbox.on_owner_manual(PEER, "Это я, спасибо", 11)

    assert await messages(repo, conv.id) == []
    assert await repo.open_events(conv.task_id) == []
    assert (await repo.get_conversation(conv.id)).paused_until is None
    assert (await repo.get_outbox(pending)).status is OutboxStatus.PENDING


async def test_an_unclear_file_reported_by_the_replay_is_ours_not_the_owners(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock, outcomes: Outcomes
) -> None:
    conv = await new_conv(repo, S.CONTACTED)
    item_id = await repo.enqueue(conv.id, "", fake_clock.now(), "files/spec.pdf")
    await repo.mark_outbox(item_id, OutboxStatus.UNKNOWN)
    fake_clock.advance(timedelta(seconds=30))

    # the program died while the file was sent; at start the replay reports it (no caption)
    await inbox.on_owner_manual(PEER, "", 21)

    item = await repo.get_outbox(item_id)
    assert (item.status, item.tg_msg_id) == (OutboxStatus.SENT, 21)
    fresh = await repo.get_conversation(conv.id)
    assert fresh.paused_until is None  # not the owner's own message: no pause
    [message] = await messages(repo, conv.id)
    assert (message.source, message.kind) == (MsgSource.AGENT, "document")
    assert await events(repo, conv.task_id, EventKind.OWNER_WROTE) == []
    assert [sent[0] for sent in outcomes.sent] == [item_id]


async def test_an_unclear_text_is_not_matched_by_a_file_or_a_file_by_a_text(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    conv = await new_conv(repo, S.CONTACTED)
    text_id = await repo.enqueue(conv.id, "Здравствуйте", fake_clock.now())
    await repo.mark_outbox(text_id, OutboxStatus.UNKNOWN)

    await inbox.on_owner_manual(PEER, "", 21)  # an empty message is not the text

    assert (await repo.get_outbox(text_id)).status is OutboxStatus.UNKNOWN
    assert (await repo.get_conversation(conv.id)).paused_until is not None


# -- fix round 1: races, the order of writes, a handed-over chat --


@pytest.mark.parametrize("meanwhile", [S.WAITING_OWNER, S.CLOSED])
async def test_a_state_moved_meanwhile_is_left_where_the_wake_put_it(
    repo: Repo,
    inbox: Inbox,
    fake_clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
    meanwhile: ConvState,
) -> None:
    conv = await new_conv(repo, S.CONTACTED)
    last_messages = repo.last_messages

    async def moved_first(conv_id: int, n: int):
        # A wake moves the person after the Inbox read his state.
        await repo.set_state(conv.id, meanwhile)
        return await last_messages(conv_id, n)

    monkeypatch.setattr(repo, "last_messages", moved_first)

    await inbox.on_incoming(incoming(fake_clock, 10, "Да"))

    monkeypatch.undo()
    assert (await repo.get_conversation(conv.id)).state is meanwhile
    assert [m.text for m in await messages(repo, conv.id)] == ["Да"]
    assert await events(repo, conv.task_id, EventKind.INCOMING) == []


class Boom(Exception):
    pass


def fail_once(monkeypatch: pytest.MonkeyPatch, repo: Repo, name: str) -> None:
    real = getattr(repo, name)
    state = {"failed": False}

    async def flaky(*args: object, **kwargs: object):
        if not state["failed"]:
            state["failed"] = True
            raise Boom(name)
        return await real(*args, **kwargs)

    monkeypatch.setattr(repo, name, flaky)


async def test_the_event_is_added_before_the_message_is_stored(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    conv = await new_conv(repo, S.CONTACTED)
    fail_once(monkeypatch, repo, "add_message")

    with pytest.raises(Boom):
        await inbox.on_incoming(incoming(fake_clock, 10, "Да"))

    assert await messages(repo, conv.id) == []
    assert len(await events(repo, conv.task_id, EventKind.INCOMING)) == 1

    # The replay at the next start delivers it again.
    await inbox.on_incoming(incoming(fake_clock, 10, "Да"))

    assert [m.text for m in await messages(repo, conv.id)] == ["Да"]
    assert len(await events(repo, conv.task_id, EventKind.INCOMING)) == 1


async def test_an_edit_adds_its_event_before_the_text_changes(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    conv = await new_conv(repo, S.CONTACTED, S.TALKING)
    await repo.add_message(
        conv.id,
        Direction.IN,
        MsgSource.COUNTERPART,
        "text",
        "Цена 100",
        10,
        fake_clock.now(),
    )
    fail_once(monkeypatch, repo, "edit_message")

    with pytest.raises(Boom):
        await inbox.on_edit(PEER, 10, "Цена 120")

    assert (await messages(repo, conv.id))[0].text == "Цена 100"
    assert len(await events(repo, conv.task_id, EventKind.INCOMING)) == 1


async def test_a_delete_adds_its_event_before_the_text_is_marked(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    conv = await new_conv(repo, S.CONTACTED, S.TALKING)
    await repo.add_message(
        conv.id,
        Direction.IN,
        MsgSource.COUNTERPART,
        "text",
        "Цена 100",
        10,
        fake_clock.now(),
    )
    fail_once(monkeypatch, repo, "edit_message")

    with pytest.raises(Boom):
        await inbox.on_delete([10])

    assert (await messages(repo, conv.id))[0].text == "Цена 100"
    assert len(await events(repo, conv.task_id, EventKind.INCOMING)) == 1


async def test_the_owner_writes_in_a_chat_closed_meanwhile(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    conv = await new_conv(repo, S.CONTACTED, S.TALKING)
    by_peer = repo.conversation_by_peer

    async def closed_after_lookup(peer_id: int):
        found = await by_peer(peer_id)
        await repo.set_state(conv.id, S.CLOSED)  # a wake closes him meanwhile
        return found

    monkeypatch.setattr(repo, "conversation_by_peer", closed_after_lookup)

    await inbox.on_owner_manual(PEER, "Сам отвечу", 11)

    monkeypatch.undo()
    # Read again under the lock: closed now, so his message only starts his quiet.
    fresh = await repo.get_conversation(conv.id)
    assert fresh.paused_until == fake_clock.now() + timedelta(hours=12)
    assert [m.text for m in await messages(repo, conv.id)] == ["Сам отвечу"]
    assert await repo.open_events(conv.task_id) == []


async def test_the_owners_message_is_stored_last(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    conv = await new_conv(repo, S.CONTACTED, S.TALKING)
    fail_once(monkeypatch, repo, "add_message")

    with pytest.raises(Boom):
        await inbox.on_owner_manual(PEER, "Сам отвечу", 11)

    assert await messages(repo, conv.id) == []
    assert (await repo.get_conversation(conv.id)).paused_until is not None
    assert len(await events(repo, conv.task_id, EventKind.OWNER_WROTE)) == 1

    # The replay at the next start delivers it again.
    await inbox.on_owner_manual(PEER, "Сам отвечу", 11)

    assert [m.text for m in await messages(repo, conv.id)] == ["Сам отвечу"]
    assert len(await events(repo, conv.task_id, EventKind.OWNER_WROTE)) == 1


async def test_the_owner_in_a_handed_over_chat_is_only_stored(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock
) -> None:
    conv = await new_conv(repo, S.CONTACTED, S.HANDED_TO_OWNER)
    pending = await repo.enqueue(conv.id, "Ответ агента", fake_clock.now())

    await inbox.on_owner_manual(PEER, "Сам договорюсь", 11)

    [message] = await messages(repo, conv.id)
    assert (message.source, message.text) == (MsgSource.OWNER_MANUAL, "Сам договорюсь")
    fresh = await repo.get_conversation(conv.id)
    assert fresh.paused_until is None
    assert fresh.state is S.HANDED_TO_OWNER
    assert await repo.open_events(conv.task_id) == []
    assert (await repo.get_outbox(pending)).status is OutboxStatus.PENDING


@pytest.mark.parametrize(
    ("state", "status", "paused", "expected"),
    [
        (S.PLANNED, TaskStatus.ACTIVE, None, True),
        (S.CONTACTED, TaskStatus.ACTIVE, None, True),
        (S.TALKING, TaskStatus.ACTIVE, -1, True),
        (S.TALKING, TaskStatus.ACTIVE, 1, False),
        (S.WAITING_OWNER, TaskStatus.ACTIVE, None, False),
        (S.HANDED_TO_OWNER, TaskStatus.ACTIVE, None, False),
        (S.CLOSED, TaskStatus.ACTIVE, None, False),
        (S.TALKING, TaskStatus.PAUSED, None, False),
        (S.TALKING, TaskStatus.DONE, None, False),
    ],
)
async def test_takes_events(
    repo: Repo,
    fake_clock: FakeClock,
    state: ConvState,
    status: TaskStatus,
    paused: int | None,
    expected: bool,
) -> None:
    conv = await new_conv(repo, status=status)
    task = await repo.get_task(conv.task_id)
    now = fake_clock.now()
    until = None if paused is None else now + timedelta(hours=paused)
    conv = replace(conv, state=state, paused_until=until)

    assert takes_events(task, conv, now) is expected


# -- a wake that claims the event between the two writes --


def claim_inside(
    monkeypatch: pytest.MonkeyPatch,
    repo: Repo,
    name: str,
    task_id: int,
    clock: FakeClock,
) -> None:
    """A wake claims the task's waiting events in the middle of the Repo call `name`: an
    older event of the burst may be due already (a long wake, a retry after a failure)."""
    real = getattr(repo, name)

    async def claimed_first(*args: object, **kwargs: object):
        await repo.claim_events(task_id, clock.now() + timedelta(hours=1))
        return await real(*args, **kwargs)

    monkeypatch.setattr(repo, name, claimed_first)


async def waiting(repo: Repo, task_id: int) -> list[AgentEvent]:
    """The `incoming` events no wake has taken yet."""
    return [
        event
        for event in await events(repo, task_id, EventKind.INCOMING)
        if event.wake_id is None
    ]


async def test_a_message_stored_after_a_wake_took_its_event_gets_a_new_one(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    conv = await new_conv(repo, S.CONTACTED)
    claim_inside(monkeypatch, repo, "add_message", conv.task_id, fake_clock)

    await inbox.on_incoming(incoming(fake_clock, 10, "Да"))

    # The wake built its context before the message was there: without a new event the
    # message would never be answered.
    assert [m.text for m in await messages(repo, conv.id)] == ["Да"]
    assert len(await waiting(repo, conv.task_id)) == 1


async def test_an_edit_made_after_a_wake_took_its_event_gets_a_new_one(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    conv = await new_conv(repo, S.CONTACTED, S.TALKING)
    await repo.add_message(
        conv.id,
        Direction.IN,
        MsgSource.COUNTERPART,
        "text",
        "Цена 100",
        10,
        fake_clock.now(),
    )
    claim_inside(monkeypatch, repo, "edit_message", conv.task_id, fake_clock)

    await inbox.on_edit(PEER, 10, "Цена 120")

    assert (await messages(repo, conv.id))[0].text == "Цена 120"
    assert len(await waiting(repo, conv.task_id)) == 1


async def test_a_delete_marked_after_a_wake_took_its_event_gets_a_new_one(
    repo: Repo, inbox: Inbox, fake_clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    conv = await new_conv(repo, S.CONTACTED, S.TALKING)
    await repo.add_message(
        conv.id,
        Direction.IN,
        MsgSource.COUNTERPART,
        "text",
        "Цена 100",
        10,
        fake_clock.now(),
    )
    claim_inside(monkeypatch, repo, "edit_message", conv.task_id, fake_clock)

    await inbox.on_delete([10])

    assert len(await waiting(repo, conv.task_id)) == 1
