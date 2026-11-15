"""The runner of the agent core (`agent/runner.py`): what the end-to-end flows do not pin."""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from tg_agent.agent.context import seen_key
from tg_agent.agent.runner import DAY_CAP, MONTH_CAP, Runner
from tg_agent.brain.llm import AgentStep, FakeLLM, LLMError
from tg_agent.brain.schemas import Usage
from tg_agent.clock import FakeClock
from tg_agent.config import Settings
from tg_agent.control.base import FakeUI
from tg_agent.db import Repo
from tg_agent.gateway.base import TgActionError
from tg_agent.gateway.fake import FakeGateway
from tg_agent.models import (
    ApprovalKind,
    ApprovalStatus,
    ConvState,
    Direction,
    EventKind,
    MsgSource,
    OutboxStatus,
    TaskStatus,
)
from tg_agent.policy.tg_safety import TgErrorKind
from tg_agent.tasks.texts import HANDOFF_SELF

S = ConvState
END_TURN = [{"type": "text", "text": "Готово."}]


class BlockingLLM(FakeLLM):
    """A model whose steps wait until `go` is set: a wake stays running meanwhile."""

    def __init__(self) -> None:
        super().__init__()
        self.go = asyncio.Event()
        self.started = 0

    async def agent_step(self, **kw: Any) -> AgentStep:
        self.started += 1
        await self.go.wait()
        return await super().agent_step(**kw)


async def until(condition: Any, seconds: float = 5.0) -> None:
    """Wait until `condition()` holds (the database works in its own thread)."""
    for _ in range(int(seconds / 0.01)):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the condition never held")


def make_runner(
    repo: Repo,
    clock: FakeClock,
    settings: Settings,
    llm: FakeLLM | None = None,
    *,
    inline: bool = True,
) -> tuple[Runner, FakeUI]:
    ui = FakeUI()
    runner = Runner(
        repo,
        llm or FakeLLM(),
        FakeGateway(clock),
        ui,
        clock,
        settings,
        random.Random(1),
        canary="ZX-CANARY-0001",
        owner_secrets=[],
        inline=inline,
    )
    return runner, ui


async def make_task(repo: Repo, status: TaskStatus = TaskStatus.ACTIVE) -> int:
    task_id = await repo.create_task("Дрель", "Узнать цену", {}, None, "UZS", [], [])
    if status is not TaskStatus.DRAFT:
        await repo.set_task_status(task_id, status)
    return task_id


async def make_conv(repo: Repo, task_id: int, *path: ConvState) -> int:
    contact_id = await repo.upsert_contact(700, "drill_shop", "Дрели", False)
    conv_id = await repo.create_conversation(task_id, contact_id, state=S.PLANNED)
    for state in path:
        await repo.set_state(conv_id, state)
    return conv_id


@pytest.mark.parametrize("status", [TaskStatus.ACTIVE, TaskStatus.PAUSED])
async def test_start_launches_only_a_draft(repo, fake_clock, settings, status):
    runner, _ = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo, status)

    await runner.start(task_id)

    assert (await repo.get_task(task_id)).status is status
    assert await repo.open_events(task_id) == []


async def test_start_makes_a_draft_active_with_one_created_event(
    repo, fake_clock, settings
):
    runner, _ = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo, TaskStatus.DRAFT)

    await runner.start(task_id)

    assert (await repo.get_task(task_id)).status is TaskStatus.ACTIVE
    assert [e.kind for e in await repo.open_events(task_id)] == [EventKind.CREATED]


async def test_tick_does_not_start_a_second_wake_of_a_task_whose_wake_runs(
    repo, fake_clock, settings
):
    llm = BlockingLLM()
    llm.steps["task"].append(END_TURN)
    runner, _ = make_runner(repo, fake_clock, settings, llm, inline=False)
    task_id = await make_task(repo)
    await repo.add_event(task_id, EventKind.CREATED)

    await runner.tick()
    await until(lambda: llm.started == 1)
    # a new event while the wake runs: the task is due again, but its wake is running
    await repo.add_event(task_id, EventKind.REMINDER, payload={"about": "цена"})
    await runner.tick()
    await asyncio.sleep(0.2)  # time enough for a second wake to reach the model
    assert llm.started == 1

    llm.go.set()
    await runner.idle()
    assert [w.outcome for w in await repo.unfinished_wakes()] == []
    assert llm.started == 1  # no second wake, not even one queued behind the first
    [left] = await repo.open_events(task_id)  # the reminder waits for the next round
    assert (left.kind, left.wake_id, left.attempts) == (EventKind.REMINDER, None, 0)


async def test_a_wake_of_a_paused_task_leaves_its_events(repo, fake_clock, settings):
    runner, _ = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo, TaskStatus.PAUSED)
    event_id = await repo.add_event(task_id, EventKind.RESUMED)

    assert await runner.wake(task_id) is None

    [event] = await repo.open_events(task_id)
    assert (event.id, event.wake_id, event.attempts) == (event_id, None, 0)


@pytest.mark.parametrize("case", ["waiting_owner", "task_done"])
async def test_on_sent_arms_no_silence_where_nobody_waits_for_an_answer(
    repo, fake_clock, settings, case
):
    runner, _ = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id, S.CONTACTED)
    if case == "waiting_owner":
        await repo.set_state(conv_id, S.WAITING_OWNER)
    else:
        await repo.set_task_status(task_id, TaskStatus.DONE)
    item_id = await repo.enqueue(conv_id, "Здравствуйте", fake_clock.now())
    item = await repo.get_outbox(item_id)
    conv = await repo.get_conversation(conv_id)

    await runner.on_sent(item, conv, fake_clock.now(), first=True)

    assert await repo.open_events(task_id) == []


async def test_on_sent_arms_the_silence_of_a_contacted_person(
    repo, fake_clock, settings
):
    runner, _ = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id, S.CONTACTED)
    item = await repo.get_outbox(
        await repo.enqueue(conv_id, "Привет", fake_clock.now())
    )
    conv = await repo.get_conversation(conv_id)

    await runner.on_sent(item, conv, fake_clock.now(), first=True)

    [event] = await repo.open_events(task_id)
    assert (event.kind, event.conversation_id, event.due_at) == (
        EventKind.SILENCE,
        conv_id,
        fake_clock.now() + timedelta(hours=24),
    )


@pytest.mark.parametrize(
    "status", [TaskStatus.PAUSED, TaskStatus.DONE, TaskStatus.CANCELLED]
)
async def test_on_failed_leaves_the_notice_to_the_sender_when_no_wake_would_read_it(
    repo, fake_clock, settings, status
):
    runner, _ = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id, S.CONTACTED)
    await repo.set_task_status(task_id, status)
    item = await repo.get_outbox(await repo.enqueue(conv_id, "", fake_clock.now()))
    conv = await repo.get_conversation(conv_id)

    took = await runner.on_failed(item, conv, "file_missing", "файла нет")

    assert took is False
    assert await repo.open_events(task_id) == []


async def test_on_failed_of_an_active_task_wakes_the_agent_and_keeps_the_person(
    repo, fake_clock, settings
):
    runner, ui = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id, S.CONTACTED)
    item = await repo.get_outbox(await repo.enqueue(conv_id, "", fake_clock.now()))
    conv = await repo.get_conversation(conv_id)

    assert await runner.on_failed(item, conv, "file_missing", "файла нет") is True

    [event] = await repo.open_events(task_id)
    assert (event.kind, event.conversation_id, event.payload) == (
        EventKind.SEND_REFUSED,
        conv_id,
        {"reason": "файла нет", "kind": "file_missing", "closed": False},
    )
    assert (await repo.get_conversation(conv_id)).state is S.CONTACTED
    assert ui.notes == [] and ui.sent == []


async def test_on_unclear_marks_the_attempt_and_tells_the_owner_nothing(
    repo, fake_clock, settings
):
    runner, ui = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id, S.CONTACTED)
    item_id = await repo.enqueue(conv_id, "Привет", fake_clock.now())
    await repo.mark_outbox(item_id, OutboxStatus.UNKNOWN, error="ConnectionError")

    await runner.on_unclear(await repo.get_outbox(item_id))

    item = await repo.get_outbox(item_id)
    assert (item.status, item.tried_at) == (OutboxStatus.UNKNOWN, fake_clock.now())
    assert await repo.events_of("outbox_unknown_pending", conv_id) == [
        {"outbox_id": item_id}
    ]
    assert ui.notes == [] and ui.sent == []


async def unclear_item(
    repo: Repo, clock: FakeClock, conv_id: int, text: str, attach: str | None = None
) -> int:
    """A queued message whose send broke off just now (UNKNOWN, tried now)."""
    item_id = await repo.enqueue(conv_id, text, clock.now(), attach)
    await repo.mark_outbox(item_id, OutboxStatus.UNKNOWN, error="ConnectionError")
    await repo.mark_outbox_tried(item_id, clock.now())
    return item_id


async def test_an_unclear_file_is_found_after_the_person_replied(
    repo, fake_clock, settings
):
    runner, ui = make_runner(repo, fake_clock, settings)
    gateway = runner._gateway
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id, S.CONTACTED)
    item_id = await unclear_item(repo, fake_clock, conv_id, "", "files/spec.pdf")
    fake_clock.advance(timedelta(seconds=1))
    delivered = gateway._log(700, True, "")  # the file reached the chat
    reply = gateway.add_incoming(700, "Получил файл")
    await repo.add_message(
        conv_id,
        Direction.IN,
        MsgSource.COUNTERPART,
        "text",
        reply.text,
        reply.tg_msg_id,
        fake_clock.now(),
    )
    fake_clock.advance(timedelta(seconds=61))

    await runner.tick()

    item = await repo.get_outbox(item_id)
    assert (item.status, item.tg_msg_id) == (OutboxStatus.SENT, delivered.tg_msg_id)
    assert ui.notes == []


async def test_an_unclear_text_ignores_outgoing_messages_that_are_not_it(
    repo, fake_clock, settings
):
    runner, _ = make_runner(repo, fake_clock, settings)
    gateway = runner._gateway
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id, S.CONTACTED)
    # an identical greeting from an earlier task, long before this item was queued
    gateway._log(700, True, "Здравствуйте! Есть дрель?")
    fake_clock.advance(timedelta(days=3))
    item_id = await unclear_item(repo, fake_clock, conv_id, "Здравствуйте! Есть дрель?")
    gateway._log(700, True, "Другой текст")  # not this item: another text
    gateway._log(700, True, "")  # not this item: a file
    fake_clock.advance(timedelta(seconds=61))

    await runner.tick()

    item = await repo.get_outbox(item_id)
    assert (item.status, item.not_before) == (OutboxStatus.PENDING, fake_clock.now())


async def test_the_watermark_is_written_only_when_a_wake_finishes(
    repo, fake_clock, settings
):
    llm = FakeLLM()
    llm.steps["task"] += [LLMError("overloaded"), END_TURN]
    runner, _ = make_runner(repo, fake_clock, settings, llm)
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id, S.CONTACTED, S.TALKING)
    await repo.add_message(
        conv_id,
        Direction.IN,
        MsgSource.COUNTERPART,
        "text",
        "Есть?",
        5,
        fake_clock.now(),
    )
    await repo.add_event(task_id, EventKind.CREATED)

    assert await runner.wake(task_id) == "error"
    assert await repo.get_kv(seen_key(task_id)) is None

    fake_clock.advance(timedelta(minutes=5))
    assert await runner.wake(task_id) == "done"
    assert await repo.get_kv(seen_key(task_id)) == str(
        await repo.max_message_id(task_id)
    )


async def test_recover_finishes_a_wake_that_had_claimed_nothing(
    repo, fake_clock, settings
):
    runner, _ = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo)
    # a crash between the wake's row and the claim of its events
    wake_id = await repo._insert(
        "INSERT INTO wakes (task_id, started_at) VALUES (?, ?)",
        (task_id, fake_clock.now().isoformat()),
    )
    waiting = await repo.add_event(task_id, EventKind.REMINDER, payload={"about": "x"})

    assert await runner.recover() == 1

    assert await repo.unfinished_wakes() == []
    [event] = await repo.open_events(task_id)  # no `recovered` event was added
    assert (event.id, event.attempts) == (waiting, 0)
    rows = await repo._all("SELECT outcome FROM wakes WHERE id = ?", (wake_id,))
    assert rows[0]["outcome"] == "interrupted"


async def test_recover_re_arms_a_lost_silence_of_an_active_task(
    repo, fake_clock, settings
):
    runner, _ = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo)
    paused_id = await make_task(repo, TaskStatus.PAUSED)
    conv_id = await make_conv(repo, task_id, S.CONTACTED)
    sent_at = fake_clock.now()
    await repo.add_message(
        conv_id, Direction.OUT, MsgSource.AGENT, "text", "Привет", 3, sent_at
    )
    await repo.update_conversation(conv_id, last_out_at=sent_at, first_sent=True)
    fake_clock.advance(timedelta(hours=2))  # the silence was lost in a crash

    await runner.recover()

    [silence] = await repo.open_events(task_id)
    assert (silence.kind, silence.conversation_id, silence.due_at) == (
        EventKind.SILENCE,
        conv_id,
        sent_at + timedelta(hours=24),
    )
    assert await repo.open_events(paused_id) == []  # only ACTIVE tasks


async def test_an_owner_decision_re_arms_the_silence(repo, fake_clock, settings):
    runner, _ = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id, S.CONTACTED)
    await repo.add_message(
        conv_id, Direction.OUT, MsgSource.AGENT, "text", "Привет", 3, fake_clock.now()
    )
    await repo.update_conversation(
        conv_id, last_out_at=fake_clock.now(), first_sent=True
    )
    answer = await runner.approvals.ask(
        await repo.get_task(task_id),
        kind="question",
        question="Ждать?",
        conv_id=conv_id,
        suggested=[],
        amount=None,
        terms="",
    )
    assert answer[1] is False
    [approval] = await repo.pending_approvals()

    await runner.approvals.owner_decision(approval.id, approval.nonce, "answer", "Жди")

    kinds = [e.kind for e in await repo.open_events(task_id)]
    assert kinds == [EventKind.OWNER_ANSWER, EventKind.SILENCE]


async def test_recover_releases_a_wake_that_had_claimed_events(
    repo, fake_clock, settings
):
    runner, _ = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo)
    created = await repo.add_event(task_id, EventKind.CREATED)
    await repo.claim_events(task_id, fake_clock.now())

    assert await runner.recover() == 1

    events = await repo.open_events(task_id)
    assert [(e.id, e.kind, e.wake_id) for e in events] == [
        (created, EventKind.CREATED, None),
        (events[1].id, EventKind.RECOVERED, None),
    ]
    assert events[0].attempts == 0  # a crash is not the model's failure (A5)
    assert await repo.due_task_ids(fake_clock.now()) == [task_id]


# -- A3: the unclear send that is given up --


def unreadable(runner: Runner) -> None:
    """The next read of a chat fails."""
    runner._gateway.fail_next = TgActionError(TgErrorKind.FLOOD_WAIT, "FLOOD_WAIT")


async def test_an_unclear_send_is_given_up_after_five_reads_one_a_minute(
    repo, fake_clock, settings
):
    runner, ui = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id, S.CONTACTED)
    item_id = await unclear_item(repo, fake_clock, conv_id, "Здравствуйте")
    fake_clock.advance(timedelta(seconds=61))
    reads = runner._gateway.calls

    for n in range(1, 5):
        unreadable(runner)
        await runner.tick()
        assert reads.count("recent_messages") == n
        fake_clock.advance(timedelta(seconds=5))
        await runner.tick()  # a round later: not read again within the minute
        assert reads.count("recent_messages") == n
        fake_clock.advance(timedelta(seconds=55))
    assert (await repo.get_outbox(item_id)).status is OutboxStatus.UNKNOWN
    assert ui.notes == []
    unreadable(runner)
    await runner.tick()

    assert (await repo.get_outbox(item_id)).status is OutboxStatus.FAILED
    [notice] = ui.notes
    assert notice.startswith("Не могу проверить, ушло ли сообщение")
    [event] = await repo.open_events(task_id)
    assert (event.kind, event.conversation_id, event.payload) == (
        EventKind.SEND_REFUSED,
        conv_id,
        {
            "reason": "не удалось проверить, ушло ли сообщение",
            "kind": "unclear",
            "closed": False,
        },
    )


async def test_an_unclear_send_found_meanwhile_is_not_given_up(
    repo, fake_clock, settings
):
    runner, ui = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id, S.CONTACTED)
    item_id = await unclear_item(repo, fake_clock, conv_id, "Здравствуйте")
    fake_clock.advance(timedelta(seconds=61))
    for _ in range(4):
        unreadable(runner)
        await runner.tick()
        fake_clock.advance(timedelta(minutes=1))
    # the inbox found it (the owner's app reported our message) while the fifth read ran
    original = runner._gateway.recent_messages

    async def found_meanwhile(*args: Any, **kw: Any) -> Any:
        await repo.mark_outbox(item_id, OutboxStatus.SENT, tg_msg_id=40)
        return await original(*args, **kw)

    runner._gateway.recent_messages = found_meanwhile  # type: ignore[method-assign]
    unreadable(runner)
    await runner.tick()

    assert (await repo.get_outbox(item_id)).status is OutboxStatus.SENT
    assert ui.notes == []
    assert await repo.open_events(task_id) == []


# -- A4, A6, A7, A8: the wake's bookkeeping --


async def test_tick_releases_a_wake_left_unfinished_ten_minutes_ago(
    repo, fake_clock, settings
):
    llm = FakeLLM()
    llm.steps["task"].append(END_TURN)
    runner, _ = make_runner(repo, fake_clock, settings, llm)
    task_id = await make_task(repo)
    young_task = await make_task(repo)
    await repo.add_event(task_id, EventKind.REMINDER, payload={"about": "цена"})
    dead, _ = await repo.claim_events(task_id, fake_clock.now())
    fake_clock.advance(timedelta(minutes=6))
    await repo.add_event(young_task, EventKind.CREATED)
    young, _ = await repo.claim_events(young_task, fake_clock.now())
    fake_clock.advance(timedelta(minutes=5))  # dead: 11 minutes old; young: 5

    await runner.tick()

    rows = await repo._all("SELECT id, outcome FROM wakes ORDER BY id")
    outcomes = {row["id"]: row["outcome"] for row in rows}
    assert outcomes[dead] == "interrupted"
    assert outcomes[young] is None  # left alone: it may still be running
    assert len(llm.requests) == 1  # the released event woke its task in the same round
    assert await repo.claimed_events(young) != []


async def failing_wakes(runner: Runner, llm: FakeLLM, clock: FakeClock, task_id: int):
    """One failed wake of the task, on an event of its own."""
    await repo_of(runner).add_event(task_id, EventKind.REMINDER, payload={"about": "x"})
    llm.steps["task"].append(LLMError("overloaded"))
    assert await runner.wake(task_id) == "error"
    clock.advance(timedelta(minutes=5))


def repo_of(runner: Runner) -> Repo:
    return runner._repo


async def test_three_failed_wakes_in_a_row_pause_the_task_whatever_their_events(
    repo, fake_clock, settings
):
    llm = FakeLLM()
    runner, ui = make_runner(repo, fake_clock, settings, llm)
    task_id = await make_task(repo)

    for n in (1, 2):
        await failing_wakes(runner, llm, fake_clock, task_id)
        # each failure's event is gone by the next one (a newer message superseded it)
        await repo.drop_task_events(task_id)
        assert await repo.get_kv(f"task_failures:{task_id}") == str(n)
    assert (await repo.get_task(task_id)).status is TaskStatus.ACTIVE
    await failing_wakes(runner, llm, fake_clock, task_id)

    assert (await repo.get_task(task_id)).status is TaskStatus.PAUSED
    assert len(ui.notes) == 1 and "три сбоя подряд" in ui.notes[0]


async def test_a_finished_wake_and_a_resume_forget_the_failures(
    repo, fake_clock, settings
):
    llm = FakeLLM()
    runner, _ = make_runner(repo, fake_clock, settings, llm)
    task_id = await make_task(repo)
    key = f"task_failures:{task_id}"
    await failing_wakes(runner, llm, fake_clock, task_id)
    await failing_wakes(runner, llm, fake_clock, task_id)
    llm.steps["task"].append(END_TURN)

    assert await runner.wake(task_id) == "done"

    assert await repo.get_kv(key) == "0"
    await failing_wakes(runner, llm, fake_clock, task_id)
    await failing_wakes(runner, llm, fake_clock, task_id)
    await runner.forget_failures(task_id)
    assert await repo.get_kv(key) == "0"
    await failing_wakes(runner, llm, fake_clock, task_id)
    assert (await repo.get_task(task_id)).status is TaskStatus.ACTIVE


class ClosedMeanwhile(FakeLLM):
    """The owner closes the task while the model works; then the call fails."""

    def __init__(self, repo: Repo, task_id: int, status: TaskStatus) -> None:
        super().__init__()
        self.repo, self.task_id, self.status = repo, task_id, status

    async def agent_step(self, **kw: Any) -> AgentStep:
        await self.repo.set_task_status(self.task_id, self.status)
        raise LLMError("overloaded")


@pytest.mark.parametrize(
    ("status", "left"),
    [(TaskStatus.CANCELLED, 0), (TaskStatus.DONE, 0), (TaskStatus.PAUSED, 1)],
)
async def test_a_failed_wake_of_a_task_that_is_over_finishes_its_events(
    repo, fake_clock, settings, status, left
):
    task_id = await make_task(repo)
    llm = ClosedMeanwhile(repo, task_id, status)
    runner, _ = make_runner(repo, fake_clock, settings, llm)
    await repo.add_event(task_id, EventKind.CREATED)

    await runner.wake(task_id)

    assert len(await repo.open_events(task_id)) == left


class FailingNotices(FakeUI):
    def __init__(self) -> None:
        super().__init__()
        self.broken = True

    async def notify(self, text: str) -> int:
        if self.broken:
            raise ConnectionError("bot is down")
        return await super().notify(text)


async def test_the_day_s_cap_notice_is_marked_only_once_it_was_sent(
    repo, fake_clock, settings
):
    runner, _ = make_runner(repo, fake_clock, settings)
    ui = FailingNotices()
    runner._ui = ui
    await spend(repo, settings.limits.daily_usd)
    task_id = await make_task(repo)
    await repo.add_event(task_id, EventKind.CREATED)

    await runner.tick()
    assert ui.notes == []
    ui.broken = False
    await runner.tick()

    assert ui.notes == [DAY_CAP]


# -- A12: tick does not hide a broken core --


async def test_tick_raises_when_its_own_work_fails(
    repo, fake_clock, settings, monkeypatch
):
    runner, _ = make_runner(repo, fake_clock, settings)

    async def broken(now: Any) -> list[int]:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(repo, "due_task_ids", broken)

    with pytest.raises(RuntimeError, match="locked"):
        await runner.tick()


async def test_an_unclear_send_that_cannot_be_checked_does_not_stop_the_others(
    repo, fake_clock, settings, monkeypatch
):
    runner, _ = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id, S.CONTACTED)
    broken = await unclear_item(repo, fake_clock, conv_id, "Первое")
    fine = await unclear_item(repo, fake_clock, conv_id, "Второе")
    fake_clock.advance(timedelta(seconds=61))
    check_one = runner._check_one

    async def check(item: Any) -> None:
        if item.id == broken:
            raise RuntimeError("bug")
        await check_one(item)

    monkeypatch.setattr(runner, "_check_one", check)

    with pytest.raises(RuntimeError, match="bug"):
        await runner.tick()

    assert (await repo.get_outbox(fine)).status is OutboxStatus.PENDING  # checked


async def test_an_error_inside_a_wake_stays_inside_the_wake(
    repo, fake_clock, settings, monkeypatch
):
    llm = FakeLLM()
    llm.steps["task"].append(END_TURN)
    runner, _ = make_runner(repo, fake_clock, settings, llm)
    task_id = await make_task(repo)
    await repo.add_event(task_id, EventKind.CREATED)

    async def broken(*args: Any, **kw: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(repo, "finish_wake", broken)

    await runner.tick()  # does not raise

    assert len(llm.requests) == 1


# -- A13, A14: recovery around cards; one `recovered` event per task --


async def held_person(repo: Repo, task_id: int, state: ConvState) -> int:
    """A person who answered, then was held for the owner (`state`)."""
    conv_id = await make_conv(repo, task_id, S.CONTACTED, S.TALKING)
    await repo.update_conversation(conv_id, last_in_at=repo._clock.now())
    await repo.set_state(conv_id, state)
    return conv_id


def kinds_of(events: list[Any]) -> list[EventKind]:
    return [event.kind for event in events]


@pytest.mark.parametrize("status", [TaskStatus.ACTIVE, TaskStatus.PAUSED])
async def test_recover_expires_an_approval_that_never_got_its_card(
    repo, fake_clock, settings, status
):
    runner, ui = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo, status)
    conv_id = await held_person(repo, task_id, S.WAITING_OWNER)
    # the program died between the approval's row and its card
    approval_id = (
        await repo.create_approval(
            task_id, conv_id, ApprovalKind.QUESTION, {"question": "Брать?"}
        )
    ).id

    await runner.recover()

    assert (await repo.get_approval(approval_id)).status is ApprovalStatus.EXPIRED
    assert (await repo.get_conversation(conv_id)).state is S.TALKING
    assert kinds_of(await repo.open_events(task_id)) == [EventKind.RECOVERED]
    assert ui.sent == []


@pytest.mark.parametrize("status", [TaskStatus.ACTIVE, TaskStatus.PAUSED])
async def test_recover_reopens_a_person_held_by_no_card(
    repo, fake_clock, settings, status
):
    runner, _ = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo, status)
    # the program died between the owner's decision and its event
    conv_id = await held_person(repo, task_id, S.WAITING_OWNER)

    await runner.recover()

    assert (await repo.get_conversation(conv_id)).state is S.TALKING
    assert kinds_of(await repo.open_events(task_id)) == [EventKind.RECOVERED]


async def test_recover_keeps_a_person_held_by_a_card_the_owner_has(
    repo, fake_clock, settings
):
    runner, _ = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo)
    conv_id = await held_person(repo, task_id, S.WAITING_OWNER)
    approval_id = (
        await repo.create_approval(
            task_id, conv_id, ApprovalKind.QUESTION, {"question": "Брать?"}
        )
    ).id
    await repo.set_approval_msg(approval_id, 77)

    await runner.recover()

    assert (await repo.get_conversation(conv_id)).state is S.WAITING_OWNER
    assert (await repo.get_approval(approval_id)).status is ApprovalStatus.PENDING
    assert await repo.open_events(task_id) == []


@pytest.mark.parametrize("status", [TaskStatus.ACTIVE, TaskStatus.PAUSED])
async def test_recover_sends_the_missing_hand_back_card(
    repo, fake_clock, settings, status
):
    runner, ui = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo, status)
    conv_id = await held_person(repo, task_id, S.HANDED_TO_OWNER)

    await runner.recover()

    [card] = ui.sent
    assert HANDOFF_SELF in card.text
    [handoff] = await repo.pending_approvals(conv_id)
    assert (handoff.kind, handoff.control_msg_id) == (
        ApprovalKind.HANDOFF,
        ui.sent_ids[0],
    )
    assert (await repo.get_conversation(conv_id)).state is S.HANDED_TO_OWNER
    await runner.recover()  # a second start sends no second card
    assert len(ui.sent) == 1


async def test_recover_adds_one_recovered_event_per_task(repo, fake_clock, settings):
    runner, _ = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo)
    for kind in (EventKind.CREATED, EventKind.REMINDER):
        await repo.add_event(task_id, kind, payload={"about": "x"})
        await repo.claim_events(task_id, fake_clock.now())  # two wakes left unfinished
    await held_person(repo, task_id, S.WAITING_OWNER)

    assert await runner.recover() == 2

    assert kinds_of(await repo.open_events(task_id)).count(EventKind.RECOVERED) == 1


# -- A15 (2.6 M4): the money caps --

TASHKENT = ZoneInfo("Asia/Tashkent")


async def spend(repo: Repo, usd: float) -> None:
    await repo.log_llm_call(None, "task", "claude-sonnet-5-5", Usage(1, 1), usd, "x")


async def test_no_wake_starts_once_the_month_s_money_is_spent(
    repo, fake_clock, settings
):
    runner, ui = make_runner(repo, fake_clock, settings)
    fake_clock.set(datetime(2026, 11, 3, 12, 0, tzinfo=TASHKENT))
    await spend(repo, settings.limits.monthly_usd)  # earlier this month, not today
    fake_clock.set(datetime(2026, 11, 10, 12, 0, tzinfo=TASHKENT))
    task_id = await make_task(repo)
    await repo.add_event(task_id, EventKind.CREATED)

    assert await runner.wake(task_id) is None

    assert ui.notes == [MONTH_CAP]
    assert [e.kind for e in await repo.open_events(task_id)] == [EventKind.CREATED]


async def test_the_day_starts_at_midnight_in_the_owner_s_zone(
    repo, fake_clock, settings
):
    llm = FakeLLM()
    llm.steps["task"].append(END_TURN)
    runner, ui = make_runner(repo, fake_clock, settings, llm)
    # 23:00 in Tashkent, 18:00 UTC: yesterday for the owner
    fake_clock.set(datetime(2026, 11, 9, 23, 0, tzinfo=TASHKENT))
    await spend(repo, settings.limits.daily_usd)
    # 00:30 in Tashkent is 19:30 UTC of the same UTC day as the spend
    fake_clock.set(datetime(2026, 11, 10, 0, 30, tzinfo=TASHKENT))
    task_id = await make_task(repo)
    await repo.add_event(task_id, EventKind.CREATED)

    assert await runner.wake(task_id) == "done"
    assert ui.notes == []


async def test_the_cap_notice_goes_once_a_day_and_again_the_next_day(
    repo, fake_clock, settings
):
    runner, ui = make_runner(repo, fake_clock, settings)
    fake_clock.set(datetime(2026, 11, 10, 9, 0, tzinfo=TASHKENT))
    await spend(repo, settings.limits.daily_usd)
    task_id = await make_task(repo)
    await repo.add_event(task_id, EventKind.CREATED)

    await runner.tick()
    fake_clock.advance(timedelta(hours=14))  # 23:00, the same day
    await runner.tick()
    assert ui.notes == [DAY_CAP]
    fake_clock.advance(timedelta(hours=2))  # 01:00 the next day
    await spend(repo, settings.limits.daily_usd)
    await runner.tick()

    assert ui.notes == [DAY_CAP, DAY_CAP]


# -- A15 (2.6 M3): a tool or the context that raises reaches the wake's failure path --


@pytest.mark.parametrize("where", ["tool", "context"])
async def test_a_raising_tool_or_context_releases_the_wake_and_counts_it(
    repo, fake_clock, settings, monkeypatch, where
):
    llm = FakeLLM()
    llm.steps["task"].append(
        [
            {
                "type": "tool_use",
                "id": "t1",
                "name": "update_notes",
                "input": {"notes": "план", "headline": "Работаю", "person_notes": []},
            }
        ]
    )
    runner, _ = make_runner(repo, fake_clock, settings, llm)
    task_id = await make_task(repo)
    await repo.add_event(task_id, EventKind.CREATED)

    async def broken(*args: Any, **kw: Any) -> Any:
        raise RuntimeError("bug")

    if where == "tool":
        monkeypatch.setattr(repo, "set_task_notes", broken)
    else:
        monkeypatch.setattr("tg_agent.agent.runner.build_context", broken)

    assert await runner.wake(task_id) == "error"

    [event] = await repo.open_events(task_id)
    assert (event.kind, event.wake_id, event.attempts) == (EventKind.CREATED, None, 1)
    assert event.due_at == fake_clock.now() + timedelta(minutes=5)
    assert await repo.get_kv(f"task_failures:{task_id}") == "1"
    assert await repo.events_of("wake_failed") == [
        {"task_id": task_id, "wake_id": 1, "error": "RuntimeError"}
    ]
    assert len(llm.requests) == (1 if where == "tool" else 0)


# -- fix pass C4: one unreadable chat does not stop the round --


async def test_a_chat_that_cannot_be_read_counts_as_a_failed_read_and_the_round_goes_on(
    repo, fake_clock, settings
):
    llm = FakeLLM()
    runner, ui = make_runner(repo, fake_clock, settings, llm)
    task_id = await make_task(repo)
    conv_id = await make_conv(repo, task_id, S.CONTACTED)
    item_id = await unclear_item(repo, fake_clock, conv_id, "Здравствуйте")
    other_task = await make_task(repo)
    await repo.add_event(other_task, EventKind.CREATED)
    llm.steps["task"].append(END_TURN)
    fake_clock.advance(timedelta(seconds=61))

    for n in range(5):
        runner._gateway.fail_next = ConnectionError("link down")  # type: ignore[assignment]
        await runner.tick()  # does not raise
        if n == 0:
            assert len(llm.requests) == 1  # the other task's wake ran in this round
        fake_clock.advance(timedelta(minutes=1))

    assert (await repo.get_outbox(item_id)).status is OutboxStatus.FAILED
    [notice] = ui.notes
    assert notice.startswith("Не могу проверить, ушло ли сообщение")


# -- fix pass C8: «Напишу сам» is safe across a crash --


async def test_write_myself_interrupted_after_the_move_keeps_the_chat_with_the_owner(
    repo, fake_clock, settings, monkeypatch
):
    runner, _ = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo)
    conv_id = await held_person(repo, task_id, S.WAITING_OWNER)
    approval = await repo.create_approval(
        task_id, conv_id, ApprovalKind.CONFIRM_ORDER, {"text": "Брать?", "amount": 1}
    )
    await repo.set_approval_msg(approval.id, 77)
    resolve = repo.resolve_approval
    calls = []

    async def dies_once(*args: Any, **kw: Any) -> bool:
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("the program died here")
        return await resolve(*args, **kw)

    monkeypatch.setattr(repo, "resolve_approval", dies_once)
    with pytest.raises(RuntimeError):
        await runner.approvals.owner_decision(approval.id, approval.nonce, "self")

    runner, _ = make_runner(repo, fake_clock, settings)  # the restart
    await runner.recover()
    assert (await repo.get_conversation(conv_id)).state is S.HANDED_TO_OWNER

    toast = await runner.approvals.owner_decision(approval.id, approval.nonce, "self")

    assert toast == "Чат за тобой"
    assert (await repo.get_approval(approval.id)).status is ApprovalStatus.ANSWERED
    assert (await repo.get_conversation(conv_id)).state is S.HANDED_TO_OWNER
    handoffs = [
        a for a in await repo.pending_approvals(conv_id) if a.kind is ApprovalKind.HANDOFF
    ]
    assert len(handoffs) == 1


# -- Task 6.1 (R76): a task's own money --


async def spend_on(repo: Repo, task_id: int, usd: float) -> None:
    await repo.log_llm_call(
        None, "task", "claude-sonnet-5-5", Usage(1, 1), usd, "x", task_id=task_id
    )


async def test_a_wake_may_cost_at_most_what_is_left_of_the_task_s_money(
    repo, fake_clock, settings, monkeypatch
):
    import tg_agent.agent.runner as runner_module

    caps: list[float] = []
    real = runner_module.run_loop

    async def capped(*args: Any, **kw: Any) -> Any:
        caps.append(kw["max_cost"])
        return await real(*args, **kw)

    monkeypatch.setattr(runner_module, "run_loop", capped)
    llm = FakeLLM()
    llm.steps["task"].append(END_TURN)
    runner, ui = make_runner(repo, fake_clock, settings, llm)
    task_id = await make_task(repo)
    await spend_on(repo, task_id, 2.99)
    await repo.add_event(task_id, EventKind.CREATED)

    assert await runner.wake(task_id) == "done"

    assert caps == [pytest.approx(0.01)]
    assert ui.sent == []


async def test_a_task_that_spent_its_money_waits_with_one_card(
    repo, fake_clock, settings
):
    runner, ui = make_runner(repo, fake_clock, settings)
    task_id = await make_task(repo)
    await spend_on(repo, task_id, 3.0)
    await repo.add_event(task_id, EventKind.CREATED)

    await runner.tick()
    fake_clock.advance(timedelta(minutes=5))
    await runner.tick()

    assert await repo._all("SELECT id FROM wakes") == []
    assert [e.kind for e in await repo.open_events(task_id)] == [EventKind.CREATED]
    (card,) = ui.sent
    assert card.text == (
        f"❓ Задача #{task_id} «Дрель» потратила $3.00. Продолжить ещё на $3?"
    )
    assert (card.kind, card.key) == ("task_money", str(task_id))
    ((label, data),) = card.buttons[0]
    assert label == "Продолжить"
    assert data.startswith(f"mn:{task_id}:more:")
    assert ui.notes == []


async def test_the_owner_s_task_limit_by_words_holds_for_the_runner(
    repo, fake_clock, settings
):
    # Ruling R89: a limit set by words (kv) wins over config.toml.
    from tg_agent.agent.tools import limit_key

    await repo.set_kv(limit_key("task"), "4.5")
    llm = FakeLLM()
    llm.steps["task"].append(END_TURN)
    runner, ui = make_runner(repo, fake_clock, settings, llm)
    task_id = await make_task(repo)
    await spend_on(repo, task_id, 4.0)  # past config's $3, under the new $4.5
    await repo.add_event(task_id, EventKind.CREATED)
    assert await runner.wake(task_id) == "done"
    await spend_on(repo, task_id, 0.5)
    await repo.add_event(task_id, EventKind.CREATED)

    await runner.tick()

    (card,) = ui.sent
    assert card.text.endswith("потратила $4.50. Продолжить ещё на $4.5?")


async def test_the_owner_s_day_limit_by_words_holds_for_the_runner(
    repo, fake_clock, settings
):
    from tg_agent.agent.tools import limit_key

    await repo.set_kv(limit_key("day"), "1")
    runner, ui = make_runner(repo, fake_clock, settings)
    await spend(repo, 1.0)
    task_id = await make_task(repo)
    await repo.add_event(task_id, EventKind.CREATED)

    assert await runner.wake(task_id) is None
    assert ui.notes == [DAY_CAP]
