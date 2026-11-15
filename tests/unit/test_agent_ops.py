"""What the owner does to a task in the agent core (`agent/ops.py`), on fakes."""

from __future__ import annotations

import asyncio
import random
from datetime import timedelta

import pytest

from tg_agent.agent import people
from tg_agent.agent.context import NEW_MARKER
from tg_agent.agent.inbox import Inbox
from tg_agent.agent.ops import NOBODY_TO_WRITE, TaskOps
from tg_agent.agent.people import save_targets
from tg_agent.agent.runner import Runner
from tg_agent.brain.llm import FakeLLM
from tg_agent.clock import FakeClock
from tg_agent.config import Settings
from tg_agent.control.base import FakeUI
from tg_agent.db import Repo
from tg_agent.gateway.base import Incoming, PeerInfo
from tg_agent.gateway.fake import FakeGateway
from tg_agent.models import (
    ApprovalKind,
    ApprovalStatus,
    ConvState,
    EventKind,
    OutboxItem,
    OutboxStatus,
    TaskStatus,
)
from tg_agent.tasks.templates import DECLINE

S = ConvState
K = EventKind
END = [{"type": "text", "text": "Жду."}]


class Pieces:
    """The core's objects on fakes, with TaskOps over them."""

    def __init__(self, repo: Repo, clock: FakeClock, settings: Settings) -> None:
        self.repo, self.clock, self.settings = repo, clock, settings
        self.llm = FakeLLM()
        self.gateway = FakeGateway(clock)
        self.ui = FakeUI()
        self.runner = Runner(
            repo, self.llm, self.gateway, self.ui, clock, settings, random.Random(3),
            canary="ZX-OPS-1", owner_secrets=[], inline=True,
        )  # fmt: skip
        self.inbox = Inbox(
            repo, clock, settings, outcomes=self.runner, locks=self.runner.locks
        )
        self.ops = TaskOps(
            repo, self.gateway, clock, settings, self.runner, random.Random(5)
        )
        self._peer = 500

    def peer(self, username: str) -> int:
        self._peer += 1
        self.gateway.peers[username] = PeerInfo(
            self._peer, username, username.title(), False
        )
        return self._peer

    async def task(self, *targets: str, ceiling: float | None = 600_000) -> int:
        task_id = await self.repo.create_task(
            "Дрель", "Купить дрель Bosch не дороже 600 000", {"what": "дрель"},
            ceiling, "UZS", ["name"], [], source_text="Найди дрель",
        )  # fmt: skip
        await save_targets(self.repo, task_id, list(targets))
        return task_id

    async def attach(self, task_id: int, username: str) -> int:
        self.peer(username)
        attached = await people.attach(self.repo, self.gateway, task_id, username)
        assert isinstance(attached, people.Attached), attached
        return attached.conv_id

    async def spoke(self, conv_id: int) -> None:
        """Something of ours really went to him (the goodbye goes only then: L25)."""
        item = await self.repo.enqueue(conv_id, "Здравствуйте!", self.clock.now())
        await self.repo.mark_outbox(item, OutboxStatus.SENT, at=self.clock.now())

    async def pending(self, conv_id: int) -> list[OutboxItem]:
        return [
            item
            for item in await self.repo.outbox_of(conv_id)
            if item.status is OutboxStatus.PENDING
        ]

    async def events(self, type: str) -> list[dict]:
        return await self.repo.events_of(type)

    def context(self) -> str:
        [*_, request] = [req for purpose, req in self.llm.requests if purpose == "task"]
        return request["messages"][0]["content"][-1]["text"]


@pytest.fixture
async def w(repo, fake_clock, settings) -> Pieces:
    return Pieces(repo, fake_clock, settings)


# -- launch --


async def test_launch_attaches_who_it_can_and_starts_the_task(w: Pieces) -> None:
    w.peer("drill_shop")
    task_id = await w.task("drill_shop", "ghost_shop")

    launched, notes = await w.ops.launch(task_id)

    assert launched is True
    assert notes == ["@ghost_shop: не найден — проверь имя"]
    [conv] = await w.repo.task_conversations(task_id)
    assert conv.state is S.PLANNED
    assert (await w.repo.get_task(task_id)).status is TaskStatus.ACTIVE
    assert [e.kind for e in await w.repo.open_events(task_id)] == [K.CREATED]
    assert await w.events("task_launched") == [{"task_id": task_id}]
    assert w.gateway.allowed == {w.gateway.peers["drill_shop"].peer_id}

    assert await w.ops.launch(task_id) == (
        False,
        ["Задача уже запущена или закрыта"],
    )
    assert [e.kind for e in await w.repo.open_events(task_id)] == [K.CREATED]


async def test_a_launch_that_attaches_nobody_keeps_the_draft(w: Pieces) -> None:
    task_id = await w.task("ghost_shop")

    launched, notes = await w.ops.launch(task_id)

    assert launched is False
    assert notes == ["@ghost_shop: не найден — проверь имя", NOBODY_TO_WRITE]
    assert (await w.repo.get_task(task_id)).status is TaskStatus.DRAFT
    assert await w.repo.open_events(task_id) == []
    assert await w.events("task_not_launched") == [{"task_id": task_id}]


async def test_a_launch_with_people_already_in_the_draft_starts_it(w: Pieces) -> None:
    task_id = await w.task("drill_shop")
    await w.attach(task_id, "drill_shop")

    launched, notes = await w.ops.launch(task_id)

    assert launched is True and notes == []  # being in the draft is no failure
    assert (await w.repo.get_task(task_id)).status is TaskStatus.ACTIVE


async def test_a_draft_that_named_nobody_launches_at_once(w: Pieces) -> None:
    # Ruling R26: the owner named nobody, so the agent's first wake finds people.
    task_id = await w.task()

    launched, notes = await w.ops.launch(task_id)

    assert launched is True and notes == []
    assert (await w.repo.get_task(task_id)).status is TaskStatus.ACTIVE
    assert await w.repo.task_conversations(task_id) == []
    assert [e.kind for e in await w.repo.open_events(task_id)] == [K.CREATED]
    assert await w.events("task_launched") == [{"task_id": task_id}]
    assert await w.events("task_not_launched") == []


async def test_an_unknown_task(w: Pieces) -> None:
    assert await w.ops.launch(99) == (False, ["Задачи #99 нет"])
    assert await w.ops.instruct(99, "привет") == "Задачи #99 нет"
    assert await w.ops.pause(99) == "Задачи #99 нет"
    assert await w.ops.resume(99) == "Задачи #99 нет"
    assert await w.ops.close(99) == "Задачи #99 нет"


# -- instruct --


@pytest.mark.parametrize(
    ("status", "line"),
    [
        (TaskStatus.ACTIVE, "Передал агенту"),
        (
            TaskStatus.PAUSED,
            (
                "Передал. Задача на паузе: агент увидит это, когда ты её продолжишь"
                " — напиши «продолжи задачу {id}»"
            ),
        ),
    ],
)
async def test_an_instruction_to_a_task_in_work(w: Pieces, status, line) -> None:
    task_id = await w.task()
    conv_id = await w.attach(task_id, "drill_shop")
    await w.repo.set_task_status(task_id, status)

    answer = await w.ops.instruct(task_id, "  Торгуйся до 600 000  ", conv_id=conv_id)

    assert answer == line.format(id=task_id)
    [event] = await w.repo.open_events(task_id)
    assert event.kind is K.OWNER_INSTRUCTION and event.conversation_id == conv_id
    assert "600 000" not in event.payload["text"]  # the ceiling never reaches the model
    assert event.payload["text"].startswith("Торгуйся до")
    assert (await w.repo.get_task(task_id)).status is status
    assert await w.events("task_instructed") == [{"task_id": task_id}]


async def test_an_instruction_about_the_task_as_a_whole(w: Pieces) -> None:
    task_id = await w.task()
    await w.repo.set_task_status(task_id, TaskStatus.ACTIVE)

    await w.ops.instruct(task_id, "Ищи ещё")

    [event] = await w.repo.open_events(task_id)
    assert event.conversation_id is None and event.payload == {"text": "Ищи ещё"}


async def test_an_instruction_takes_a_finished_task_up_again(w: Pieces) -> None:
    task_id = await w.task()
    free = await w.attach(task_id, "drill_shop")
    busy = await w.attach(task_id, "maker_uz")
    await w.repo.set_task_status(task_id, TaskStatus.DONE)
    # maker_uz is in work in another task by now
    other = await w.task()
    maker = await w.repo.contact_by_peer(w.gateway.peers["maker_uz"].peer_id)
    assert maker is not None
    await w.repo.create_conversation(other, maker.id, state=S.PLANNED)
    await w.repo.set_task_status(other, TaskStatus.ACTIVE)

    answer = await w.ops.instruct(task_id, "Спроси про кейс")

    assert answer == f"Задача #{task_id} снова в работе: передал агенту"
    assert (await w.repo.get_task(task_id)).status is TaskStatus.ACTIVE
    assert (await w.repo.get_conversation(free)).state is S.PLANNED
    closed = await w.repo.get_conversation(busy)
    assert closed.state is S.CLOSED
    assert closed.close_reason == f"занят в задаче #{other}"
    [event] = await w.repo.open_events(task_id)
    assert event.kind is K.OWNER_INSTRUCTION
    assert w.gateway.allowed == await w.repo.active_peer_ids()


@pytest.mark.parametrize("status", [TaskStatus.DRAFT, TaskStatus.CANCELLED])
async def test_an_instruction_to_a_task_not_in_work(w: Pieces, status) -> None:
    task_id = await w.task()
    await w.repo.set_task_status(task_id, status)

    assert await w.ops.instruct(task_id, "Ищи ещё") == "Эта задача не в работе"
    assert await w.repo.open_events(task_id) == []
    assert (await w.repo.get_task(task_id)).status is status
    assert await w.events("task_instructed") == []


async def test_a_blank_instruction(w: Pieces) -> None:
    task_id = await w.task()
    await w.repo.set_task_status(task_id, TaskStatus.ACTIVE)

    assert await w.ops.instruct(task_id, "  \n ") == "Пустое указание"
    assert await w.repo.open_events(task_id) == []


# -- pause and resume --


async def test_pause_and_resume(w: Pieces) -> None:
    task_id = await w.task()
    conv_id = await w.attach(task_id, "drill_shop")
    await w.runner.start(task_id)
    w.llm.steps["task"].append(END)
    await w.runner.tick()  # a first wake: what it saw is not new any more
    queued = await w.repo.enqueue(conv_id, "Здравствуйте", w.clock.now())
    open_before = [e.id for e in await w.repo.open_events(task_id)]

    assert await w.ops.pause(task_id) == (
        f"Пауза в задаче #{task_id}. Продолжить — напиши «продолжи задачу {task_id}»"
    )

    assert (await w.repo.get_task(task_id)).status is TaskStatus.PAUSED
    assert (await w.repo.get_outbox(queued)).status is OutboxStatus.CANCELLED
    # the agent hears of the cancelled message when the task resumes
    [refused] = [
        e for e in await w.repo.open_events(task_id) if e.id not in open_before
    ]
    assert (refused.kind, refused.conversation_id) == (K.SEND_REFUSED, conv_id)
    open_before.append(refused.id)
    assert await w.events("task_paused") == [{"task_id": task_id}]
    assert await w.ops.pause(task_id) == f"Задача #{task_id} уже на паузе"
    # he writes while the task is paused: stored, no event
    peer = w.gateway.peers["drill_shop"].peer_id
    message = w.gateway.add_incoming(peer, "Есть GSB 13")
    await w.inbox.on_incoming(
        Incoming(peer, message.tg_msg_id, message.text, "text", message.date)
    )
    assert [m.text for m in await w.repo.last_messages(conv_id, 5)] == ["Есть GSB 13"]
    assert [e.id for e in await w.repo.open_events(task_id)] == open_before

    assert await w.ops.resume(task_id) == f"Задача #{task_id} продолжена"

    assert (await w.repo.get_task(task_id)).status is TaskStatus.ACTIVE
    resumed = [e for e in await w.repo.open_events(task_id) if e.kind is K.RESUMED]
    assert len(resumed) == 1
    assert await w.events("task_resumed") == [{"task_id": task_id}]
    assert await w.ops.resume(task_id) == f"Задача #{task_id} не на паузе"
    w.llm.steps["task"].append(END)
    await w.runner.tick()
    context = w.context()
    assert NEW_MARKER in context
    assert context.index(NEW_MARKER) < context.index("Есть GSB 13")


@pytest.mark.parametrize("status", [TaskStatus.DRAFT, TaskStatus.DONE])
async def test_pause_of_a_task_not_in_work(w: Pieces, status) -> None:
    task_id = await w.task()
    await w.repo.set_task_status(task_id, status)

    assert await w.ops.pause(task_id) == f"Задача #{task_id} не в работе"
    assert await w.ops.resume(task_id) == f"Задача #{task_id} не на паузе"
    assert (await w.repo.get_task(task_id)).status is status


async def test_resume_forgets_the_failed_wakes(w: Pieces) -> None:
    task_id = await w.task()
    await w.repo.set_task_status(task_id, TaskStatus.PAUSED)
    await w.repo.set_kv(f"task_failures:{task_id}", "3")

    await w.ops.resume(task_id)

    assert await w.repo.get_kv(f"task_failures:{task_id}") == "0"


# -- close --


async def test_close_of_a_draft(w: Pieces) -> None:
    task_id = await w.task()

    assert await w.ops.close(task_id) == "Задача закрыта"

    assert (await w.repo.get_task(task_id)).status is TaskStatus.CANCELLED
    assert await w.events("task_closed") == [{"task_id": task_id}]
    assert await w.ops.close(task_id) == f"Задача #{task_id} уже закрыта"


@pytest.mark.parametrize(
    ("language", "goodbye"), [("ru", DECLINE["ru"]), ("uz_latn", DECLINE["uz_latn"])]
)
async def test_close_of_a_task_in_work(w: Pieces, language, goodbye) -> None:
    task_id = await w.task()
    planned = await w.attach(task_id, "planned_one")
    contacted = await w.attach(task_id, "contacted_one")
    talking = await w.attach(task_id, "talking_one")
    waiting = await w.attach(task_id, "waiting_one")
    await w.runner.start(task_id)
    for conv_id in (contacted, talking, waiting):
        await w.repo.set_state(conv_id, S.CONTACTED)
    await w.repo.set_state(talking, S.TALKING)
    await w.spoke(talking)
    await w.repo.update_conversation(talking, language=language)
    stale = await w.repo.enqueue(talking, "Подумаю", w.clock.now())
    card = await w.repo.create_approval(
        task_id, waiting, ApprovalKind.QUESTION, {"question": "?"}
    )
    await w.repo.set_state(waiting, S.WAITING_OWNER)
    await w.repo.add_event(task_id, K.SILENCE, conv_id=contacted)

    answer = await w.ops.close(task_id)

    assert answer == f"Задача #{task_id} закрыта. Попрощался: @talking_one"
    for conv_id in (planned, contacted, talking, waiting):
        conv = await w.repo.get_conversation(conv_id)
        assert (conv.state, conv.close_reason) == (S.CLOSED, "задача закрыта")
    assert (await w.repo.get_outbox(stale)).status is OutboxStatus.CANCELLED
    [bye] = [
        item
        for item in await w.repo.outbox_of(talking)
        if item.status is OutboxStatus.PENDING
    ]
    assert bye.text == goodbye
    assert bye.not_before > w.clock.now()  # after a human pause
    for conv_id in (planned, contacted, waiting):
        assert not [
            item
            for item in await w.repo.outbox_of(conv_id)
            if item.status is OutboxStatus.PENDING
        ]
    assert (await w.repo.get_approval(card.id)).status is ApprovalStatus.EXPIRED
    assert await w.repo.open_events(task_id) == []
    assert (await w.repo.get_task(task_id)).status is TaskStatus.CANCELLED
    assert await w.events("task_closed") == [{"task_id": task_id}]
    assert w.gateway.allowed == set()
    assert await w.ops.close(task_id) == f"Задача #{task_id} уже закрыта"


async def test_close_takes_the_contact_s_language_when_the_chat_has_none(
    w: Pieces,
) -> None:
    task_id = await w.task()
    talking = await w.attach(task_id, "talking_one")
    await w.runner.start(task_id)
    await w.repo.set_state(talking, S.CONTACTED)
    await w.repo.set_state(talking, S.TALKING)
    await w.spoke(talking)
    conv = await w.repo.get_conversation(talking)
    contact = await w.repo.get_contact(conv.contact_id)
    await w.repo.update_contact_card(contact.id, contact.card, "uz_cyrl")

    await w.ops.close(task_id)

    [bye] = await w.pending(talking)
    assert bye.text == DECLINE["uz_latn"]  # Uzbek goes out in Latin


async def test_no_goodbye_to_one_nothing_was_sent_to(w: Pieces) -> None:
    """L25: he wrote first and our first reply still waits its pause; the close cancels it,
    and «Спасибо, пока не нужно» does not become the first message he gets."""
    task_id = await w.task()
    talking = await w.attach(task_id, "talking_one")
    await w.runner.start(task_id)
    await w.repo.set_state(talking, S.CONTACTED)
    await w.repo.set_state(talking, S.TALKING)
    first = await w.repo.enqueue(talking, "Здравствуйте! Да, хотел узнать цену", w.clock.now())

    assert await w.ops.close(task_id) == f"Задача #{task_id} закрыта"

    assert [item.id for item in await w.repo.outbox_of(talking)] == [first]
    assert (await w.repo.get_outbox(first)).status is OutboxStatus.CANCELLED
    assert (await w.repo.get_conversation(talking)).state is S.CLOSED


async def test_no_goodbye_in_a_chat_the_owner_writes_in_himself(w: Pieces) -> None:
    task_id = await w.task()
    talking = await w.attach(task_id, "talking_one")
    await w.runner.start(task_id)
    await w.repo.set_state(talking, S.CONTACTED)
    await w.repo.set_state(talking, S.TALKING)
    await w.spoke(talking)
    await w.repo.update_conversation(
        talking, paused_until=w.clock.now() + timedelta(minutes=30)
    )

    assert await w.ops.close(task_id) == f"Задача #{task_id} закрыта"

    assert await w.pending(talking) == []
    assert (await w.repo.get_conversation(talking)).state is S.CLOSED


async def test_close_of_a_finished_task_expires_a_card_about_the_task(
    w: Pieces,
) -> None:
    task_id = await w.task()
    await w.attach(task_id, "planned_one")
    await w.repo.set_task_status(task_id, TaskStatus.DONE)
    card = await w.repo.create_approval(
        task_id, None, ApprovalKind.QUESTION, {"question": "?"}
    )

    assert await w.ops.close(task_id) == f"Задача #{task_id} закрыта"

    assert (await w.repo.get_approval(card.id)).status is ApprovalStatus.EXPIRED
    assert (await w.repo.get_task(task_id)).status is TaskStatus.CANCELLED


# -- stop and resume all --


async def test_stop_all_and_resume_all(w: Pieces) -> None:
    await w.ops.stop_all()

    assert await w.repo.is_stopped() is True
    assert await w.events("stopped") == [{}]

    await w.ops.resume_all()

    assert await w.repo.is_stopped() is False
    assert await w.events("resumed") == [{}]


# -- fix round 1 --


async def test_two_closes_at_once_close_once(w: Pieces) -> None:
    task_id = await w.task()
    talking = await w.attach(task_id, "talking_one")
    await w.runner.start(task_id)
    await w.repo.set_state(talking, S.CONTACTED)
    await w.repo.set_state(talking, S.TALKING)
    await w.spoke(talking)

    lines = await asyncio.gather(w.ops.close(task_id), w.ops.close(task_id))

    assert sorted(lines) == sorted(
        [
            f"Задача #{task_id} закрыта. Попрощался: @talking_one",
            f"Задача #{task_id} уже закрыта",
        ]
    )
    assert await w.events("task_closed") == [{"task_id": task_id}]
    assert len(await w.pending(talking)) == 1  # one goodbye


async def test_two_launches_at_once_launch_once(w: Pieces) -> None:
    w.peer("drill_shop")
    task_id = await w.task("drill_shop")

    results = await asyncio.gather(w.ops.launch(task_id), w.ops.launch(task_id))

    assert sorted(launched for launched, _ in results) == [False, True]
    assert (False, ["Задача уже запущена или закрыта"]) in results
    assert [e.kind for e in await w.repo.open_events(task_id)] == [K.CREATED]
    assert await w.events("task_launched") == [{"task_id": task_id}]
    assert len(await w.repo.task_conversations(task_id)) == 1


async def test_a_lost_race_to_pause_answers_already(w: Pieces, monkeypatch) -> None:
    task_id = await w.task()
    await w.repo.set_task_status(task_id, TaskStatus.ACTIVE)
    real = w.repo.get_task

    async def read_then_paused(task_id_: int):
        task = await real(task_id_)
        # another press wins between the read and the move
        await w.repo.set_task_status(task_id_, TaskStatus.PAUSED)
        return task

    monkeypatch.setattr(w.repo, "get_task", read_then_paused)

    assert await w.ops.pause(task_id) == f"Задача #{task_id} уже на паузе"
    assert await w.events("task_paused") == []


async def test_closing_a_finished_task_says_no_goodbye(w: Pieces) -> None:
    task_id = await w.task()
    talking = await w.attach(task_id, "talking_one")
    await w.repo.set_state(talking, S.CONTACTED)
    await w.repo.set_state(talking, S.TALKING)
    await w.spoke(talking)
    await w.repo.set_task_status(task_id, TaskStatus.DONE)

    assert await w.ops.close(task_id) == f"Задача #{task_id} закрыта"

    assert await w.pending(talking) == []
    conv = await w.repo.get_conversation(talking)
    assert (conv.state, conv.close_reason) == (S.CLOSED, "задача закрыта")


async def test_no_goodbye_for_one_who_is_in_work_in_another_task(w: Pieces) -> None:
    task_id = await w.task()
    talking = await w.attach(task_id, "talking_one")
    await w.runner.start(task_id)
    await w.repo.set_state(talking, S.CONTACTED)
    await w.repo.set_state(talking, S.TALKING)
    await w.spoke(talking)
    other = await w.task()
    contact = await w.repo.contact_by_peer(w.gateway.peers["talking_one"].peer_id)
    assert contact is not None
    await w.repo.create_conversation(other, contact.id, state=S.PLANNED)
    await w.repo.set_task_status(other, TaskStatus.ACTIVE)

    assert await w.ops.close(task_id) == f"Задача #{task_id} закрыта"

    assert await w.pending(talking) == []
    assert (await w.repo.get_conversation(talking)).state is S.CLOSED


async def test_closing_a_draft_closes_the_people_attached_to_it(w: Pieces) -> None:
    task_id = await w.task()
    planned = await w.attach(task_id, "planned_one")

    assert await w.ops.close(task_id) == "Задача закрыта"

    conv = await w.repo.get_conversation(planned)
    assert (conv.state, conv.close_reason) == (S.CLOSED, "задача закрыта")
    assert await w.repo.outbox_of(planned) == []
    assert w.gateway.allowed == set()


async def test_a_pause_that_cancelled_a_message_tells_the_agent(w: Pieces) -> None:
    task_id = await w.task()
    queued_for = await w.attach(task_id, "drill_shop")
    await w.attach(task_id, "tool_shop")  # nothing queued for him: no event
    await w.repo.set_task_status(task_id, TaskStatus.ACTIVE)
    await w.repo.enqueue(queued_for, "Сколько стоит?", w.clock.now())

    await w.ops.pause(task_id)

    [event] = await w.repo.open_events(task_id)
    assert (event.kind, event.conversation_id) == (K.SEND_REFUSED, queued_for)
    assert event.payload == {
        "reason": "задача была на паузе: сообщение не ушло",
        "kind": "paused",
        "closed": False,
    }


async def test_take_up_closes_the_busy_before_the_task_is_in_work(
    w: Pieces, monkeypatch
) -> None:
    task_id = await w.task()
    busy = await w.attach(task_id, "maker_uz")
    await w.repo.set_task_status(task_id, TaskStatus.DONE)
    other = await w.task()
    await w.repo.set_task_status(other, TaskStatus.ACTIVE)  # he is in work there
    maker = await w.repo.contact_by_peer(w.gateway.peers["maker_uz"].peer_id)
    assert maker is not None
    await w.repo.create_conversation(other, maker.id, state=S.PLANNED)
    statuses = []
    real_close = people.close

    async def watching(repo, gateway, conv_id, reason):
        statuses.append((await repo.get_task(task_id)).status)
        return await real_close(repo, gateway, conv_id, reason)

    monkeypatch.setattr(people, "close", watching)

    await w.ops.instruct(task_id, "Спроси про кейс")

    assert statuses == [TaskStatus.DONE]
    assert (await w.repo.get_conversation(busy)).state is S.CLOSED
    assert (await w.repo.get_task(task_id)).status is TaskStatus.ACTIVE


async def test_an_instruction_about_a_person_of_another_task_drops_the_person(
    w: Pieces,
) -> None:
    task_id = await w.task()
    await w.repo.set_task_status(task_id, TaskStatus.ACTIVE)
    other = await w.task()
    stranger = await w.attach(other, "tool_shop")

    answer = await w.ops.instruct(task_id, "Ищи ещё", conv_id=stranger)

    assert answer == "Передал агенту"
    [event] = await w.repo.open_events(task_id)
    assert event.conversation_id is None


async def test_a_launch_says_nothing_about_who_is_in_the_draft_already(
    w: Pieces,
) -> None:
    task_id = await w.task("drill_shop")
    await w.attach(task_id, "drill_shop")

    assert await w.ops.launch(task_id) == (True, [])


async def test_a_gateway_error_while_launching_becomes_a_note(
    w: Pieces, monkeypatch
) -> None:
    w.peer("drill_shop")
    task_id = await w.task("broken_one", "drill_shop")
    real = w.gateway.resolve

    async def resolve(target: str):
        if target == "broken_one":
            raise ConnectionError("the link dropped")
        return await real(target)

    monkeypatch.setattr(w.gateway, "resolve", resolve)

    launched, notes = await w.ops.launch(task_id)

    assert launched is True
    assert notes == ["@broken_one: не удалось открыть (ConnectionError)"]
    assert len(await w.repo.task_conversations(task_id)) == 1


async def test_a_card_is_not_made_for_a_task_not_in_work(w: Pieces) -> None:
    from tg_agent.agent.outgoing import TASK_STOPPED

    task_id = await w.task()
    await w.repo.set_task_status(task_id, TaskStatus.ACTIVE)
    task = await w.repo.get_task(task_id)  # read while it was in work, as a wake would
    await w.repo.set_task_status(task_id, TaskStatus.CANCELLED)

    text, is_error = await w.runner.approvals.ask(
        task, kind="question", question="Искать ещё?", conv_id=None,
        suggested=[], amount=None, terms="",
    )  # fmt: skip

    assert (text, is_error) == (TASK_STOPPED, True)
    assert await w.repo.pending_approvals() == []
    assert w.ui.sent == []


# -- «Завершить»: the owner accepts the result (stage 5, R56) --


async def test_acknowledge_closes_the_people_quietly_and_the_task_stays_done(
    w: Pieces,
) -> None:
    task_id = await w.task()
    talking = await w.attach(task_id, "talking_one")
    await w.repo.set_state(talking, S.CONTACTED)
    await w.repo.set_state(talking, S.TALKING)
    # The last wake's reply to him still goes: «Завершить» cancels nothing (R25).
    await w.repo.enqueue(talking, "Спасибо, оформляем", w.clock.now())
    await w.repo.set_task_status(task_id, TaskStatus.DONE)
    await w.repo.set_kv(f"task_result_waits:{task_id}", "1")
    card = await w.repo.create_approval(
        task_id, None, ApprovalKind.QUESTION, {"question": "?"}
    )

    assert await w.ops.acknowledge(task_id) == f"Задача #{task_id} завершена"

    conv = await w.repo.get_conversation(talking)
    assert (conv.state, conv.close_reason) == (S.CLOSED, "задача завершена")
    [item] = await w.repo.outbox_of(talking)
    assert (item.text, item.status) == ("Спасибо, оформляем", OutboxStatus.PENDING)
    assert (await w.repo.get_approval(card.id)).status is ApprovalStatus.EXPIRED
    assert (await w.repo.get_task(task_id)).status is TaskStatus.DONE
    assert await w.repo.get_kv(f"task_result_waits:{task_id}") == ""
    assert w.gateway.allowed == await w.repo.active_peer_ids()
    assert await w.events("task_acknowledged") == [{"task_id": task_id}]


@pytest.mark.parametrize(
    "status",
    [TaskStatus.DRAFT, TaskStatus.ACTIVE, TaskStatus.PAUSED, TaskStatus.CANCELLED],
)
async def test_acknowledge_of_a_task_that_is_not_done_changes_nothing(
    w: Pieces, status: TaskStatus
) -> None:
    task_id = await w.task()
    person = await w.attach(task_id, "planned_one")
    await w.repo.set_task_status(task_id, status)

    assert await w.ops.acknowledge(task_id) == (
        f"Задача #{task_id} не завершена агентом — завершать нечего"
    )
    assert (await w.repo.get_task(task_id)).status is status
    assert (await w.repo.get_conversation(person)).state is S.PLANNED
    assert await w.ops.acknowledge(999) == "Задачи #999 нет"


async def test_a_take_up_stops_the_result_from_waiting(w: Pieces) -> None:
    task_id = await w.task()
    await w.attach(task_id, "drill_shop")
    await w.repo.set_task_status(task_id, TaskStatus.DONE)
    await w.repo.set_kv(f"task_result_waits:{task_id}", "1")

    await w.ops.instruct(task_id, "Спроси про кейс")

    assert await w.repo.get_kv(f"task_result_waits:{task_id}") == ""
