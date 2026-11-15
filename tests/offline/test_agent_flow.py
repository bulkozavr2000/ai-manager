"""The agent core end to end in the test world (`agent_world.py`): one story per test, asserted
on the database, on what the fake Telegram sent and on what the owner's fake bot showed."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from itertools import pairwise

import pytest
from agent_world import BURST, DELIVERY, END, AgentWorld, ask, call, send

from tg_agent.agent import people
from tg_agent.agent.approvals import NOT_SENT_SUGGESTED, OK_TOLD, SENT_SUGGESTED
from tg_agent.agent.outgoing import REASONS, STALE
from tg_agent.agent.runner import (
    DAY_CAP,
    GAVE_UP,
    PAUSED,
    REFUSED,
    STALLED,
    STALLED_ABOUT,
)
from tg_agent.brain.llm import AgentStep, FakeLLM, LLMError, ModelRefusal
from tg_agent.control.base import FakeUI
from tg_agent.gateway.base import ChatMessage, Incoming, TgActionError, TgSendUncertain
from tg_agent.gateway.fake import FakeGateway
from tg_agent.models import (
    ApprovalStatus,
    ConvState,
    Direction,
    EventKind,
    OutboxStatus,
    TaskStatus,
)
from tg_agent.policy.postgates import contains_ai_admission
from tg_agent.policy.tg_safety import TgErrorKind

S = ConvState
K = EventKind
MINUTE = timedelta(minutes=1)
HOUR = timedelta(hours=1)
DAY = timedelta(hours=24)


@pytest.fixture
async def world(repo, fake_clock, settings):
    world = AgentWorld(repo, fake_clock, settings)
    yield world
    await world.assert_clean()


# 1 ------------------------------------------------------------------------------------------


async def test_01_the_first_message_goes_out_with_the_introduction(world):
    task_id, [person] = await world.task("drill_shop")
    world.script([send(person.conv_id, "Здравствуйте! Есть дрель Bosch?")], END)

    await world.round()

    assert await world.state(person) is S.CONTACTED  # queued, not sent yet
    assert world.gateway.sent == []
    assert [k for k in await world.kinds(task_id)] == []

    await world.deliver()

    [text] = world.sent_to(person)
    assert text != "Здравствуйте! Есть дрель Bosch?"  # the name lacks «ИИ»: introduced
    assert contains_ai_admission(text)
    assert "Есть дрель Bosch?" in text
    [message] = await world.repo.last_messages(person.conv_id, 10)
    assert (message.direction, message.text) == (Direction.OUT, text)
    sent_at = world.clock.now()
    [silence] = await world.open_events(task_id)
    assert (silence.kind, silence.conversation_id, silence.due_at) == (
        K.SILENCE,
        person.conv_id,
        sent_at + DAY,
    )
    assert world.owner_messages() == 0


# 2 ------------------------------------------------------------------------------------------


async def test_02_a_blocked_draft_is_rewritten(world):
    _, [person] = await world.task("drill_shop")
    blocked = send(person.conv_id, "Здравствуйте! Вот модель: https://bosch.uz/drill")
    world.script(
        [blocked],
        [send(person.conv_id, "Здравствуйте! Есть дрель Bosch GSB 13?")],
        END,
    )

    await world.round()
    await world.deliver()

    refusal = world.result(blocked)
    assert refusal["is_error"] is True
    assert REASONS["link"] in refusal["content"]
    [text] = world.sent_to(person)
    assert "Bosch GSB 13" in text and "https://" not in text


# 11 -----------------------------------------------------------------------------------------


async def test_11_progress_goes_to_the_status_not_to_the_chat(world):
    task_id, [person] = await world.task("drill_shop")
    world.script(
        [
            call(
                "update_notes",
                notes="Написать магазину, узнать цену",
                headline="Ищу дрель: пишу одному магазину",
                person_notes=[{"person_id": person.conv_id, "note": "ещё не писали"}],
            )
        ],
        END,
    )

    await world.round()

    task = await world.repo.get_task(task_id)
    assert (task.notes, task.headline) == (
        "Написать магазину, узнать цену",
        "Ищу дрель: пишу одному магазину",
    )
    assert (await world.repo.get_conversation(person.conv_id)).note == "ещё не писали"
    assert world.ui.sent == [] and world.ui.notes == []
    assert world.gateway.sent == []


# 23 -----------------------------------------------------------------------------------------


async def test_23_the_person_answers_the_agent_replies_and_the_silence_starts_again(
    world,
):
    task_id, person = await world.replied("Есть, 550 000")
    [first_silence] = await world.repo.handled_events(task_id, [K.SILENCE], 5)
    assert first_silence.conversation_id == person.conv_id  # his answer ended it
    world.script([send(person.conv_id, "Спасибо! А доставка есть?")], END)

    await world.round(BURST)
    await world.deliver()

    assert world.sent_to(person)[-1] == "Спасибо! А доставка есть?"
    assert await world.state(person) is S.TALKING
    [silence] = await world.open_events(task_id)
    assert (silence.kind, silence.due_at) == (K.SILENCE, world.clock.now() + DAY)
    assert silence.id != first_silence.id


# 3, 4 ---------------------------------------------------------------------------------------


async def _promise_waits_for_the_owner(world):
    task_id, person = await world.replied("Есть, 550 000. Берёте?")
    world.promise = send(person.conv_id, "Да, берём за 550 000", commits=True)
    world.script(
        [world.promise],
        [
            ask(
                "approval",
                "Подтвердить покупку дрели за 550 000?",
                person.conv_id,
                amount=550_000,
            )
        ],
        END,
    )
    await world.round(BURST)
    return task_id, person


async def test_03_a_promise_needs_the_owner(world):
    _, person = await _promise_waits_for_the_owner(world)

    refusal = world.result(world.promise)
    assert refusal["is_error"] is True
    assert "ask_owner, kind = approval" in refusal["content"]
    assert len(world.ui.sent) == 1  # the card
    [approval] = await world.repo.pending_approvals()
    assert approval.payload["amount"] == 550_000
    assert await world.state(person) is S.WAITING_OWNER
    await world.deliver()
    assert len(world.sent_to(person)) == 1  # only the greeting


async def test_04_the_owner_says_yes_and_the_confirmation_goes_out(world):
    task_id, person = await _promise_waits_for_the_owner(world)

    assert await world.press("yes") == "Заказ подтверждён"

    [event] = await world.open_events(task_id)
    assert (event.kind, event.payload["decision"]) == (K.OWNER_ANSWER, "approved")
    assert await world.state(person) is S.TALKING
    world.script(
        [send(person.conv_id, "Подтверждаю: берём за 550 000", commits=True)], END
    )
    await world.round()

    trusted = world.trusted()
    assert "подтверждение — можно: до 550 000" in trusted
    assert "владелец разрешил подтвердить — до 550 000" in trusted
    await world.deliver()
    assert world.sent_to(person)[-1] == "Подтверждаю: берём за 550 000"


# 5 ------------------------------------------------------------------------------------------


async def test_05_ok_on_a_question_sends_the_suggested_reply_and_tells_the_agent(world):
    _task_id, person = await world.replied("А доставка нужна?")
    world.script(
        [
            ask(
                "question",
                "Нужна ли доставка?",
                person.conv_id,
                suggested=["Да, доставка нужна"],
            )
        ],
        END,
    )
    await world.round(BURST)
    assert await world.state(person) is S.WAITING_OWNER
    calls = len(world.requests())

    assert await world.press("ok") == "Ок, отправляю"
    assert len(world.requests()) == calls  # the reply goes with no model call
    world.script(END)
    await world.deliver()
    await world.round(HOUR)

    assert world.sent_to(person)[-1] == "Да, доставка нужна"
    # Once it went, the agent hears of his «Ок» (fix round A): one wake, nothing sent again.
    assert len(world.requests()) == calls + 1
    assert "«Ок: твой предложенный ответ ушёл его словами»: «Да, доставка нужна»" in (
        world.trusted()
    )
    assert world.sent_to(person).count("Да, доставка нужна") == 1
    assert await world.state(person) is S.TALKING


async def test_05_after_ok_the_agent_asks_once_and_finishes_with_the_time(world):
    # Paid eval arrange_call_no_window: told nothing, the agent asked again and the same
    # reply went twice; the task never finished.
    text = "Хорошо, тогда в четверг в 11:00"
    task_id, person = await world.replied("Давайте в четверг в 11:00?")
    world.script([ask("question", "Как ответить?", person.conv_id, suggested=[text])], END)
    await world.round(BURST)

    assert await world.press("ok") == "Ок, отправляю"
    world.script([call("finish", result="Звонок в четверг в 11:00.")])
    for _ in range(40):
        await world.round(timedelta(seconds=10))

    assert world.sent_to(person).count(text) == 1
    assert await world.repo.get_kv(f"task_result_waits:{task_id}") == "1"  # R109
    assert any("Звонок в четверг в 11:00." in card.text for card in world.ui.sent)
    await world.assert_clean()


# R98 (review D2): the agent hears of «Ок» when the reply has really gone, or that it did not.

THURSDAY = "Хорошо, тогда в четверг в 11:00"


async def _ok_pressed(world):
    task_id, person = await world.replied("Давайте в четверг в 11:00?")
    world.script(
        [ask("question", "Как ответить?", person.conv_id, suggested=[THURSDAY])], END
    )
    await world.round(BURST)
    assert await world.press("ok") == "Ок, отправляю"
    return task_id, person


async def _ok_events(world, task_id):
    """Every «Ок» event of the task, waiting or handled."""
    events = await world.repo.open_events(task_id)
    events += await world.repo.handled_events(task_id, [K.OWNER_ANSWER], 100)
    return [e.payload["text"] for e in events if e.kind is K.OWNER_ANSWER]


async def test_05_the_ok_event_comes_once_after_the_reply_went(world):
    task_id, person = await _ok_pressed(world)
    assert await _ok_events(world, task_id) == []  # nothing at the press

    await world.sender.tick()
    assert world.sent_to(person)[-1] != THURSDAY  # the human pause is not over
    assert await _ok_events(world, task_id) == []
    world.clock.advance(DELIVERY)
    await world.sender.tick()

    assert world.sent_to(person)[-1] == THURSDAY
    assert await _ok_events(world, task_id) == [SENT_SUGGESTED]
    world.script(END)
    await world.round(HOUR)
    await world.round(HOUR)
    assert await _ok_events(world, task_id) == [SENT_SUGGESTED]
    assert world.sent_to(person).count(THURSDAY) == 1
    # Once told, never again: the approval keeps `ok_told` (review 6b, the uncaught mutation).
    for conv in await world.repo.task_conversations(task_id):
        await world.runner.approvals.tell_ok(conv.id)
    assert await _ok_events(world, task_id) == [SENT_SUGGESTED]


async def test_05_while_the_sender_is_behind_the_agent_is_not_told_it_went(world):
    # Scenario 1 of D2: a flood wait holds the Sender; the core runs; nothing wakes the agent
    # with «ушёл», so no line of its own replaces the owner's reply in the queue.
    task_id, person = await _ok_pressed(world)
    calls = len(world.requests())

    for _ in range(5):
        world.clock.advance(MINUTE)
        await world.runner.tick()

    assert len(world.requests()) == calls
    assert await _ok_events(world, task_id) == []
    world.script(END)
    await world.deliver()
    await world.round(BURST)
    assert world.sent_to(person)[-1] == THURSDAY
    assert len(world.requests()) == calls + 1
    assert "ушёл его словами" in world.trusted()


async def test_05_the_ok_reply_dropped_as_stale_is_told_with_the_owners_text(world):
    # Scenario 2 of D2: «Или лучше в пятницу?» 20 s after «Ок»: the reply is dropped, and
    # the agent hears the owner agreed to it, that it did not go, and why — once.
    task_id, person = await _ok_pressed(world)
    world.clock.advance(timedelta(seconds=20))
    await world.write(person, "Или лучше в пятницу?")
    world.script([send(person.conv_id, "Уточню у Саида и вернусь")], END)

    for _ in range(4):
        await world.deliver()

    assert THURSDAY not in world.sent_to(person)
    told = NOT_SENT_SUGGESTED.format(text=THURSDAY)
    assert await _ok_events(world, task_id) == [told]
    assert told in world.trusted()
    assert "ушёл его словами" not in world.trusted()
    assert world.sent_to(person)[-1] == "Уточню у Саида и вернусь"


async def test_05_the_sender_dropping_the_ok_reply_as_stale_tells_the_agent(world):
    # The same when the Sender's turn comes before any wake: its own stale drop tells.
    task_id, person = await _ok_pressed(world)
    world.clock.advance(DELIVERY)
    await world.write(person, "Или лучше в пятницу?")

    await world.sender.tick()

    assert THURSDAY not in world.sent_to(person)
    told = NOT_SENT_SUGGESTED.format(text=THURSDAY)
    assert await _ok_events(world, task_id) == [told]
    await world.sender.tick()
    assert await _ok_events(world, task_id) == [told]


async def test_05_a_crash_between_the_send_and_the_event_is_healed_at_the_start(world):
    task_id, person = await _ok_pressed(world)

    async def died(*args, **kwargs):  # the program dies right after `mark_sent`
        return None

    world.runner.on_sent = died
    await world.deliver()
    assert world.sent_to(person)[-1] == THURSDAY
    assert await _ok_events(world, task_id) == []

    world.build()  # the restart
    await world.runner.recover()
    assert await _ok_events(world, task_id) == [SENT_SUGGESTED]
    world.build()  # and another
    await world.runner.recover()
    assert await _ok_events(world, task_id) == [SENT_SUGGESTED]


async def test_05_a_crash_after_the_event_and_before_its_mark_does_not_double_it(world):
    task_id, person = await _ok_pressed(world)
    await world.deliver()
    assert await _ok_events(world, task_id) == [SENT_SUGGESTED]
    [approval] = [
        a
        for a in await world.repo.approvals_of(person.conv_id)
        if a.payload.get(OK_TOLD)
    ]
    # As if the program died after the event and before the approval was marked told.
    payload = {k: v for k, v in approval.payload.items() if k != OK_TOLD}
    await world.repo.set_approval_payload(approval.id, payload)

    world.build()
    await world.runner.recover()

    assert await _ok_events(world, task_id) == [SENT_SUGGESTED]


async def test_05_an_ok_reply_whose_send_broke_off_is_told_once_found(world):
    task_id, person = await _ok_pressed(world)
    world.gateway.deliver_then_fail = TgSendUncertain("ConnectionError")

    await world.deliver()
    assert await _ok_events(world, task_id) == []
    world.script(END)
    await world.round(timedelta(seconds=61))
    await world.round(timedelta(seconds=61))

    assert await _ok_events(world, task_id) == [SENT_SUGGESTED]
    assert world.sent_to(person).count(THURSDAY) == 1


# 6 ------------------------------------------------------------------------------------------


async def test_06_the_owner_answers_in_words_and_the_next_wake_shows_it_as_new(world):
    task_id, person = await world.replied("А доставка нужна?")
    world.script([ask("question", "Нужна ли доставка?", person.conv_id)], END)
    await world.round(BURST)

    assert await world.press("answer", "Доставка не нужна, заберу сам") == (
        "Ответ передан агенту"
    )
    [event] = await world.open_events(task_id)
    assert event.kind is K.OWNER_ANSWER
    world.script(END)
    await world.round()

    instructions = world.trusted().split("# Указания", 1)[1]
    [line] = [ln for ln in instructions.splitlines() if "Доставка не нужна" in ln]
    assert ", новое" in line and f"про [{person.conv_id}]" in line
    assert "«Доставка не нужна, заберу сам»" in line
    assert f"владелец ответил по [{person.conv_id}]" in world.context()


# 7 ------------------------------------------------------------------------------------------


class Interrupted(FakeLLM):
    """The person writes again while the model works on its step (when `armed`)."""

    def __init__(self) -> None:
        super().__init__()
        self.world: AgentWorld | None = None
        self.person = None
        self.armed = False

    async def agent_step(self, **kw):
        step = await super().agent_step(**kw)
        if self.armed and self.world is not None:
            self.armed = False
            await self.world.write(self.person, "И ещё: нужен кейс")
        return step


async def test_07_the_person_writes_while_the_agent_thinks(repo, fake_clock, settings):
    llm = Interrupted()
    world = AgentWorld(repo, fake_clock, settings, llm)
    llm.world = world
    task_id, person = await world.replied("Есть GSB 13")
    llm.person, llm.armed = person, True
    late = send(person.conv_id, "Отлично, берём GSB 13")
    world.script([late], END)

    await world.round(BURST)

    refusal = world.result(late)
    assert refusal["is_error"] is True and refusal["content"] == STALE
    assert await _pending(world, person) == []  # nothing was queued
    assert await world.kinds(task_id) == [K.INCOMING]  # his new message waits
    await world.sender.tick()
    assert len(world.sent_to(person)) == 1  # only the greeting

    world.script([send(person.conv_id, "Хорошо: GSB 13 с кейсом")], END)
    await world.round(BURST)
    await world.deliver()

    context = world.context()
    assert "Есть GSB 13" in context and "И ещё: нужен кейс" in context
    assert world.sent_to(person)[1:] == ["Хорошо: GSB 13 с кейсом"]
    await world.assert_clean()


# 8 ------------------------------------------------------------------------------------------


async def _pending(world, person):
    return [
        item
        for item in await world.repo.outbox_of(person.conv_id)
        if item.status is OutboxStatus.PENDING
    ]


async def test_08_a_queued_reply_made_stale_by_a_newer_message_never_goes(world):
    _task_id, person = await world.replied("Есть GSB 13")
    world.script([send(person.conv_id, "Сколько стоит?")], END)
    await world.round(BURST)
    [stale] = await _pending(world, person)
    world.clock.advance(timedelta(seconds=5))
    await world.write(person, "Цена 550 000")

    world.clock.set(stale.not_before)
    await (
        world.sender.tick()
    )  # the Sender alone: the wake for the new message comes later

    assert (await world.repo.get_outbox(stale.id)).status is OutboxStatus.CANCELLED
    assert await world.repo.events_of("outbox_cancelled", person.conv_id) == [
        {"reason": "stale", "items": 1}
    ]
    world.script([send(person.conv_id, "Спасибо! Доставка есть?")], END)
    await world.round(BURST)
    await world.deliver()

    assert "Сколько стоит?" not in world.sent_to(person)
    assert world.sent_to(person)[-1] == "Спасибо! Доставка есть?"


async def test_08_a_reply_queued_before_a_card_is_not_stale_and_still_goes(world):
    _task_id, person = await world.replied("Есть GSB 13")
    world.script(
        [
            send(person.conv_id, "Уточню и вернусь"),
            ask("question", "Брать GSB 13?", person.conv_id),
        ],
        END,
    )
    await world.round(BURST)
    [holding] = await _pending(world, person)
    assert await world.state(person) is S.WAITING_OWNER
    world.clock.advance(timedelta(seconds=5))
    await world.write(person, "Хорошо, жду")  # newer than the queued words

    world.clock.set(holding.not_before)
    await world.sender.tick()

    assert (await world.repo.get_outbox(holding.id)).status is OutboxStatus.SENT
    assert world.sent_to(person)[-1] == "Уточню и вернусь"


# 9 ------------------------------------------------------------------------------------------


async def test_09_the_owner_writes_himself(world):
    _task_id, person = await world.replied("Есть GSB 13")
    world.script(
        [
            send(person.conv_id, "Сколько стоит?"),
            call("remind_me", hours=1, about="проверить ответ", person_id=None),
        ],
        END,
    )
    await world.round(BURST)
    [queued] = await _pending(world, person)

    await world.inbox.on_owner_manual(person.peer_id, "Здравствуйте, это я сам")

    assert (await world.repo.get_outbox(queued.id)).status is OutboxStatus.CANCELLED
    paused = send(person.conv_id, "Так сколько?")
    world.script([paused], END)
    await world.round(HOUR)  # the reminder wakes the agent during the pause
    await world.deliver()

    refusal = world.result(paused)
    assert refusal["is_error"] is True and "владелец сам пишет" in refusal["content"]
    assert len(world.sent_to(person)) == 1  # only the greeting
    calls = len(world.requests())
    await world.round(timedelta(hours=10))
    assert len(world.requests()) == calls  # nothing wakes it before the pause ends

    world.script(END)
    await world.round(timedelta(hours=1, minutes=5))

    assert len(world.requests()) == calls + 1
    assert f"владелец сам написал [{person.conv_id}]" in world.context()


# 10 -----------------------------------------------------------------------------------------


async def test_10_silence_two_reminders_then_the_person_is_closed(world):
    task_id, [person] = await world.contacted()

    for n in (0, 1):
        world.script([send(person.conv_id, f"Напоминаю о дрели ({n + 1})")], END)
        await world.round(DAY)
        assert (
            f"[{person.conv_id}] молчит 24 ч, напоминаний было: {n}" in world.context()
        )
        await world.deliver()
        assert world.sent_to(person)[-1] == f"Напоминаю о дрели ({n + 1})"

    third = send(person.conv_id, "Напоминаю ещё раз")
    world.script(
        [third],
        [call("close_person", person_id=person.conv_id, reason="не отвечает")],
        END,
    )
    await world.round(DAY)

    refusal = world.result(third)
    assert refusal["is_error"] is True
    assert "напоминаний без ответа уже 2" in refusal["content"]
    conv = await world.repo.get_conversation(person.conv_id)
    assert (conv.state, conv.close_reason) == (S.CLOSED, "не отвечает")
    assert len(world.sent_to(person)) == 3
    # nothing of his is left; the task, with nobody open and no `finish`, gets the nudge
    [nudge] = await world.open_events(task_id)
    assert (nudge.kind, nudge.payload) == (K.REMINDER, {"about": STALLED_ABOUT})


# 12 -----------------------------------------------------------------------------------------


async def test_12_the_result_ends_the_task(world):
    task_id, [person] = await world.task("drill_shop")
    world.script(
        [
            send(person.conv_id, "Здравствуйте! Есть дрель Bosch?"),
            call("remind_me", hours=2, about="подвести итог", person_id=None),
        ],
        END,
    )
    await world.round()
    await world.deliver()
    assert await world.kinds(task_id) == [K.REMINDER, K.SILENCE]
    world.script([call("finish", result="Дрель Bosch GSB 13 — 550 000 у @drill_shop")])

    await world.round(2 * HOUR)

    assert await world.repo.get_kv(f"task_result_waits:{task_id}") == "1"  # R109
    [card] = world.ui.sent
    assert "Дрель Bosch GSB 13 — 550 000 у @drill_shop" in card.text
    assert world.ui.notes == []
    # R109: only the owner ends the task, with «Завершить»
    await _ops(world).acknowledge(task_id)
    assert (await world.repo.get_task(task_id)).status is TaskStatus.DONE
    assert await world.open_events(task_id) == []
    calls = len(world.requests())

    # A person of a DONE task is not heard (ruling R104 a): nothing is stored, nothing goes
    # to him, nothing wakes the agent, the owner gets nothing.
    stored = await world.repo.last_messages(person.conv_id, 50)
    await world.write(person, "Ну что, берёте?")
    await world.round(HOUR)

    assert await world.repo.last_messages(person.conv_id, 50) == stored
    assert await world.open_events(task_id) == []
    assert len(world.requests()) == calls
    assert [card.kind for card in world.ui.sent] == ["result"]


# 13 -----------------------------------------------------------------------------------------


class Crash(BaseException):
    """The program dies in the middle of a wake."""


async def test_13_a_crash_in_the_middle_of_a_wake(world):
    task_id, [person] = await world.task("drill_shop")
    world.script([send(person.conv_id, "Здравствуйте! Есть дрель Bosch?")], Crash())

    with pytest.raises(Crash):
        await world.runner.tick()

    [queued] = await _pending(world, person)
    [wake] = await world.wakes()
    assert wake.finished_at is None
    world.build()  # the program starts again: the database is the state
    assert await world.sender.recover() == 0
    assert await world.runner.recover() == 1
    assert await world.kinds(task_id) == [K.CREATED, K.RECOVERED]

    world.script(END)
    await world.round()
    await world.deliver()
    await world.round(HOUR)

    context = world.context()
    assert f": «{queued.text}»)" in context and "(в очереди, уйдёт " in context
    assert "программа перезапускалась посреди работы" in context
    assert world.sent_to(person) == [queued.text]  # exactly once
    dead, woke = await world.wakes()
    assert (dead.id, dead.outcome) == (wake.id, "interrupted")
    assert woke.outcome == "done"
    # the dead wake's `created` and the `recovered` event are handled: only his silence waits
    assert await world.kinds(task_id) == [K.SILENCE]


# 14 -----------------------------------------------------------------------------------------


async def test_14_the_model_fails_three_times_and_the_task_stops(world):
    task_id, _ = await world.task("drill_shop")
    title = (await world.repo.get_task(task_id)).title

    world.script(LLMError("overloaded"))
    await world.round()
    [event] = await world.open_events(task_id)
    assert (event.attempts, event.due_at) == (1, world.clock.now() + 5 * MINUTE)
    calls = len(world.requests())
    await world.round(4 * MINUTE)
    assert len(world.requests()) == calls  # not before the five minutes

    world.script(LLMError("overloaded"))
    await world.round(MINUTE)
    world.script(LLMError("overloaded"))
    await world.round(5 * MINUTE)

    assert len(world.requests()) == calls + 2
    assert (await world.repo.get_task(task_id)).status is TaskStatus.PAUSED
    assert world.ui.notes == [PAUSED.format(id=task_id, title=title, error="LLMError")]
    await world.round(HOUR)
    assert len(world.requests()) == calls + 2
    assert world.owner_messages() == 1


# 15 -----------------------------------------------------------------------------------------


async def test_15_the_model_refuses_three_times_and_the_task_stops(world):
    task_id, _ = await world.task("drill_shop")
    title = (await world.repo.get_task(task_id)).title
    # what the real client returns for a refused call: a step marked so, with its usage
    refused = AgentStep(
        [{"type": "text", "text": ""}],
        world.llm.usage,
        0,
        "claude-sonnet-5-5",
        "refusal",
    )
    world.script(refused)

    await world.round()

    [event] = await world.open_events(task_id)  # released, not lost
    assert (event.kind, event.due_at) == (K.CREATED, world.clock.now() + 5 * MINUTE)
    [wake] = await world.wakes()
    assert wake.outcome == "refusal"
    assert await world.repo.task_cost(task_id) > 0  # the refused call is paid for
    assert world.ui.notes == []  # one refusal is no news for the owner
    assert await world.repo.get_kv(f"task_failures:{task_id}") == "1"
    calls = len(world.requests())
    await world.round(4 * MINUTE)
    assert len(world.requests()) == calls  # not before the five minutes

    world.script(ModelRefusal("declined"))
    await world.round(MINUTE)
    world.script(refused)
    await world.round(5 * MINUTE)

    assert (await world.repo.get_task(task_id)).status is TaskStatus.PAUSED
    assert world.ui.notes == [REFUSED.format(id=task_id, title=title)]
    assert "три раза отказалась" in world.ui.notes[0]
    assert f"напиши «продолжи задачу {task_id}»" in world.ui.notes[0]
    await world.round(HOUR)
    assert len(world.requests()) == calls + 2


# 16 -----------------------------------------------------------------------------------------


async def test_16_stop_holds_wakes_and_sending(world):
    _task_id, [person] = await world.task("drill_shop")
    await world.repo.set_stopped(True)

    await world.round()

    assert world.requests() == []
    await world.repo.set_stopped(False)
    world.script([send(person.conv_id, "Здравствуйте! Есть дрель Bosch?")], END)
    await world.round()
    assert len(world.requests()) == 2
    await world.repo.set_stopped(True)
    await world.deliver()
    assert world.gateway.sent == []

    await world.repo.set_stopped(False)
    await world.round()
    assert len(world.sent_to(person)) == 1


# 17 -----------------------------------------------------------------------------------------


async def test_17_a_model_that_keeps_calling_tools_is_cut_at_the_step_cap(world):
    task_id, _ = await world.task("drill_shop")
    cap = world.settings.limits.wake_max_steps
    world.script(
        *[
            [
                call(
                    "update_notes",
                    notes=f"шаг {n}",
                    headline="Работаю",
                    person_notes=[],
                )
            ]
            for n in range(cap)
        ]
    )

    await world.round()

    assert len(world.requests()) == cap
    [wake] = await world.wakes()
    assert (wake.outcome, wake.steps, wake.finished_at is not None) == (
        "steps",
        cap,
        True,
    )
    [capped] = await world.repo.events_of("wake_capped")
    assert (capped["reason"], capped["steps"]) == ("steps", cap)
    # its `created` event is handled; the cut wake left nothing scheduled: the nudge
    [nudge] = await world.open_events(task_id)
    assert (nudge.kind, nudge.payload) == (K.REMINDER, {"about": STALLED_ABOUT})


async def test_17_no_wake_starts_once_the_day_s_money_is_spent(world):
    day = world.settings.limits.daily_usd
    await world.repo.log_llm_call(
        None, "task", "claude-sonnet-5-5", world.llm.usage, day, "end_turn"
    )
    task_id, _ = await world.task("drill_shop")

    await world.round()
    await world.round(HOUR)

    assert world.requests() == []
    assert world.ui.notes == [DAY_CAP]
    assert await world.kinds(task_id) == [K.CREATED]  # the event waits
    # the next day (in the owner's zone) is spent too: the owner is told again, once
    world.clock.advance(15 * HOUR)  # 02:00 in Tashkent
    await world.repo.log_llm_call(
        None, "task", "claude-sonnet-5-5", world.llm.usage, day, "end_turn"
    )
    await world.round()
    await world.round(HOUR)
    assert world.requests() == []
    assert world.ui.notes == [DAY_CAP, DAY_CAP]


# 18 -----------------------------------------------------------------------------------------


async def test_18_telegram_refuses_the_chat(world):
    task_id, [person] = await world.task("drill_shop")
    world.script([send(person.conv_id, "Здравствуйте! Есть дрель Bosch?")], END)
    await world.round()
    world.gateway.fail_next = TgActionError(
        TgErrorKind.PRIVACY, "USER_PRIVACY_RESTRICTED"
    )

    await world.deliver()

    reason = "его настройки приватности не дают написать"
    conv = await world.repo.get_conversation(person.conv_id)
    assert (conv.state, conv.close_reason) == (S.CLOSED, reason)
    [event] = await world.open_events(task_id)
    assert (event.kind, event.payload) == (
        K.SEND_REFUSED,
        {"reason": reason, "kind": "privacy", "closed": True},
    )
    world.script(END)
    await world.round()
    assert (
        f"сообщение для [{person.conv_id}] не отправлено: {reason}; с ним закончили"
        in world.context()
    )
    assert world.owner_messages() == 0


# 19, 20 -------------------------------------------------------------------------------------


async def _queued_greeting(world):
    task_id, [person] = await world.task("drill_shop")
    world.script([send(person.conv_id, "Здравствуйте! Есть дрель Bosch?")], END)
    await world.round()
    [item] = await _pending(world, person)
    return task_id, person, item


async def test_19_a_send_that_broke_off_but_arrived_is_found_in_the_chat(world):
    task_id, person, item = await _queued_greeting(world)
    world.gateway.deliver_then_fail = TgSendUncertain("ConnectionError")

    await world.deliver()
    tried = world.clock.now()

    assert (await world.repo.get_outbox(item.id)).status is OutboxStatus.UNKNOWN
    await world.round(timedelta(seconds=30))
    assert (await world.repo.get_outbox(item.id)).status is OutboxStatus.UNKNOWN
    await world.round(timedelta(seconds=31))

    found = await world.repo.get_outbox(item.id)
    assert found.status is OutboxStatus.SENT
    assert found.tg_msg_id == world.gateway.chats[person.peer_id][-1].tg_msg_id
    [message] = await world.repo.last_messages(person.conv_id, 5)
    assert (message.direction, message.text) == (Direction.OUT, item.text)
    [silence] = await world.open_events(task_id)
    assert (silence.kind, silence.due_at) == (K.SILENCE, tried + DAY)
    await world.round(HOUR)
    assert world.sent_to(person) == [item.text]  # nothing sent twice
    assert world.owner_messages() == 0


async def test_20_a_send_that_broke_off_and_did_not_arrive_goes_again_once(world):
    _task_id, person, item = await _queued_greeting(world)
    world.gateway.fail_next = TgSendUncertain("ConnectionError")

    await world.deliver()

    assert (await world.repo.get_outbox(item.id)).status is OutboxStatus.UNKNOWN
    assert world.gateway.sent == []
    world.clock.advance(timedelta(seconds=61))
    await world.runner.tick()
    back = await world.repo.get_outbox(item.id)
    assert (back.status, back.not_before) == (OutboxStatus.PENDING, world.clock.now())

    await world.round()
    await world.round(HOUR)

    assert world.sent_to(person) == [item.text]
    assert (await world.repo.get_outbox(item.id)).status is OutboxStatus.SENT
    assert world.owner_messages() == 0


def _in_chat(world, person, text, at):
    """An outgoing message in the chat that the database does not know (an earlier task's,
    or one that is not this item)."""
    message = ChatMessage(world.gateway._new_msg_id(), True, text, at)
    world.gateway.chats.setdefault(person.peer_id, []).append(message)


async def test_20_other_outgoing_messages_in_the_chat_are_not_the_unclear_one(world):
    _task_id, person, item = await _queued_greeting(world)
    # the same greeting, sent to him in an earlier task long ago
    _in_chat(world, person, item.text, item.created_at - DAY)
    world.gateway.fail_next = TgSendUncertain("ConnectionError")

    await world.deliver()
    _in_chat(world, person, "Совсем другое сообщение", world.clock.now())
    world.clock.advance(timedelta(seconds=61))
    await world.runner.tick()

    back = await world.repo.get_outbox(item.id)
    assert (back.status, back.not_before) == (OutboxStatus.PENDING, world.clock.now())
    await world.round()
    assert world.sent_to(person) == [item.text]  # it did go, now


async def test_20_the_person_answers_within_the_minute_and_our_message_is_found(world):
    _task_id, person, item = await _queued_greeting(world)
    world.gateway.deliver_then_fail = TgSendUncertain("ConnectionError")
    await world.deliver()
    world.clock.advance(timedelta(seconds=20))
    await world.write(person, "Здравствуйте, есть")  # his reply to the message we sent

    await world.round(timedelta(seconds=41))

    found = await world.repo.get_outbox(item.id)
    assert found.status is OutboxStatus.SENT
    assert world.sent_to(person) == [item.text]
    texts = [m.text for m in await world.repo.last_messages(person.conv_id, 5)]
    assert texts == [item.text, "Здравствуйте, есть"]
    world.script(END)
    await world.round(BURST)
    assert world.sent_to(person) == [item.text]


async def test_20_a_held_persons_broken_off_message_is_found_not_sent_twice(world):
    _task_id, person = await world.replied("Есть GSB 13")
    world.script(
        [
            send(person.conv_id, "Уточню и вернусь"),
            ask("question", "Брать GSB 13?", person.conv_id),
        ],
        END,
    )
    await world.round(BURST)
    assert await world.state(person) is S.WAITING_OWNER
    world.gateway.deliver_then_fail = TgSendUncertain("ConnectionError")
    await world.deliver()
    world.clock.advance(timedelta(seconds=20))
    await world.write(person, "Хорошо, жду")

    await world.round(timedelta(seconds=41))
    await world.round(HOUR)

    assert world.sent_to(person)[1:] == ["Уточню и вернусь"]  # once


# 21 -----------------------------------------------------------------------------------------


async def test_21_a_file_that_is_gone_from_the_disk(world):
    task_id, [person] = await world.task("drill_shop", attachments=["files/spec.pdf"])
    world.script([send(person.conv_id, "Здравствуйте! ТЗ ниже", attach=True)], END)
    await world.round()

    await world.deliver()

    assert len(world.sent_to(person)) == 1  # the text went out
    assert world.gateway.files == []
    assert world.owner_messages() == 0
    assert await world.kinds(task_id) == [K.SILENCE, K.SEND_REFUSED]
    world.script(END)
    await world.round()
    assert (
        f"сообщение для [{person.conv_id}] не отправлено: файла «spec.pdf» нет на диске"
        in world.context()
    )
    assert "с ним закончили" not in world.context()
    assert await world.state(person) is S.CONTACTED
    assert world.owner_messages() == 0


# 22 -----------------------------------------------------------------------------------------


async def test_22_ask_about_a_person_who_never_answered(world):
    task_id, [person] = await world.contacted()
    world.script(
        [
            ask(
                "question",
                "Магазин молчит сутки: ждать или искать другой?",
                person.conv_id,
            )
        ],
        END,
    )
    await world.round(DAY)
    assert await world.state(person) is S.WAITING_OWNER

    await world.press("answer", "Подожди ещё день")

    assert await world.state(person) is S.CONTACTED
    # his silence was used up by the wake that asked: a new one is armed at once
    assert await world.kinds(task_id) == [K.OWNER_ANSWER, K.SILENCE]
    world.script(END)
    await world.round()
    instructions = world.trusted().split("# Указания", 1)[1]
    [line] = [ln for ln in instructions.splitlines() if "Подожди ещё день" in ln]
    assert ", новое" in line and "«Подожди ещё день»" in line
    # the wake waited, as told: he is not left with nothing that wakes the agent
    [silence] = await world.open_events(task_id)
    assert (silence.kind, silence.conversation_id, silence.due_at) == (
        K.SILENCE,
        person.conv_id,
        world.clock.now() + DAY,
    )


# 24: nothing is left with no way to wake up ------------------------------------------------


async def test_24_a_silence_wake_that_only_waits_arms_the_next_silence(world):
    task_id, [person] = await world.contacted()
    world.script(END)  # "let's wait"

    await world.round(DAY)

    [silence] = await world.open_events(task_id)
    assert (silence.kind, silence.conversation_id, silence.due_at) == (
        K.SILENCE,
        person.conv_id,
        world.clock.now() + DAY,
    )


async def test_24_words_sent_while_a_card_waited_get_a_silence_once_it_is_answered(
    world,
):
    task_id, person = await world.replied("Есть GSB 13")
    world.script(
        [
            send(person.conv_id, "Уточню и вернусь"),
            ask("question", "Брать GSB 13?", person.conv_id),
        ],
        END,
    )
    await world.round(BURST)
    await world.deliver()
    sent_at = world.clock.now()
    assert world.sent_to(person)[-1] == "Уточню и вернусь"
    assert await world.kinds(task_id) == []  # no silence while the card waits
    world.clock.advance(HOUR)

    await world.press("answer", "Бери")

    events = await world.open_events(task_id)
    assert [e.kind for e in events] == [K.OWNER_ANSWER, K.SILENCE]
    assert events[1].due_at == sent_at + DAY


async def test_24_a_message_sent_while_the_task_was_paused_gets_a_silence_on_resume(
    world,
):
    task_id, person = await world.replied("Есть GSB 13")
    world.script([send(person.conv_id, "Сколько стоит?")], END)
    await world.round(BURST)
    await world.repo.set_task_status(task_id, TaskStatus.PAUSED)
    await world.deliver()  # it still goes; a paused task arms no silence
    sent_at = world.clock.now()
    assert world.sent_to(person)[-1] == "Сколько стоит?"
    assert await world.kinds(task_id) == []
    # the owner resumes it (Task 3.2): ACTIVE again, with a `resumed` event
    await world.repo.set_task_status(task_id, TaskStatus.ACTIVE)
    await world.repo.add_event(task_id, K.RESUMED)
    world.script(END)

    await world.round(HOUR)

    [silence] = await world.open_events(task_id)
    assert (silence.kind, silence.due_at) == (K.SILENCE, sent_at + DAY)


async def test_24_a_person_silent_after_the_last_reminder_is_closed_by_the_code(world):
    task_id, [person] = await world.contacted()
    for n in (1, 2):
        world.script([send(person.conv_id, f"Напоминаю о дрели ({n})")], END)
        await world.round(DAY)
        await world.deliver()
    world.script(END)  # the model does not close him

    await world.round(DAY)

    conv = await world.repo.get_conversation(person.conv_id)
    assert (conv.state, conv.close_reason) == (S.CLOSED, GAVE_UP)
    assert len(world.sent_to(person)) == 3
    [nudge] = await world.open_events(task_id)  # nobody is left: the task is stalled
    assert (nudge.kind, nudge.payload) == (K.REMINDER, {"about": STALLED_ABOUT})


async def test_24_a_wake_that_wrote_nothing_gets_a_nudge_then_the_owner_is_told(world):
    task_id, _person = await world.replied("Есть GSB 13, 550 000")
    title = (await world.repo.get_task(task_id)).title
    # the model writes its result as text instead of calling a tool
    world.script([{"type": "text", "text": "Итог: GSB 13 за 550 000"}])

    await world.round(BURST)

    [nudge] = await world.open_events(task_id)
    assert (nudge.kind, nudge.payload, nudge.due_at) == (
        K.REMINDER,
        {"about": STALLED_ABOUT},
        world.clock.now() + 10 * MINUTE,
    )
    assert world.owner_messages() == 0
    world.script(END)

    await world.round(10 * MINUTE)

    assert STALLED_ABOUT in world.context()
    assert await world.open_events(task_id) == []
    [card] = world.ui.sent
    assert card.text == STALLED.format(id=task_id, title=title)
    assert card.task_id == task_id
    assert await world.repo.get_kv(f"task_card:{world.ui.sent_ids[0]}") == str(task_id)
    calls = len(world.requests())
    await world.round(DAY)
    assert len(world.requests()) == calls
    assert world.owner_messages() == 1


async def test_24_two_closed_and_one_never_written_to_gets_the_nudge(world):
    task_id, [a, b, _c] = await world.task("shop_a", "shop_b", "shop_c")
    world.script(
        [
            call("close_person", person_id=a.conv_id, reason="не продаёт"),
            call("close_person", person_id=b.conv_id, reason="нет в наличии"),
        ],
        END,
    )

    await world.round()

    [nudge] = await world.open_events(task_id)
    assert (nudge.kind, nudge.payload) == (K.REMINDER, {"about": STALLED_ABOUT})


async def test_24_the_stall_count_starts_again_once_something_is_scheduled(world):
    task_id, _ = await world.task("shop_a")
    world.script(END)
    await world.round()
    assert await world.repo.get_kv(f"stalled:{task_id}") == "1"
    world.script(
        [call("remind_me", hours=2, about="написать позже", person_id=None)], END
    )

    await world.round(10 * MINUTE)

    assert await world.repo.get_kv(f"stalled:{task_id}") == "0"
    world.script(END)
    await world.round(2 * HOUR)  # the reminder's wake does nothing either
    [nudge] = await world.open_events(task_id)
    assert nudge.payload == {"about": STALLED_ABOUT}  # the nudge again, not the owner
    assert world.owner_messages() == 0


# 25: /stop and a pause during a wake ----------------------------------------------------------


class StoppingUI(FakeUI):
    """The owner sends /stop right after the first card of a wake reaches him."""

    def __init__(self, repo):
        super().__init__()
        self.repo = repo

    async def send(self, card):
        msg_id = await super().send(card)
        await self.repo.set_stopped(True)
        return msg_id


async def test_25_a_stop_between_two_tool_calls_ends_the_wake_and_keeps_its_events(
    world,
):
    world.ui = StoppingUI(world.repo)
    world.build()
    task_id, [person] = await world.task("drill_shop")
    late = send(person.conv_id, "Здравствуйте! Есть дрель Bosch?")
    world.script([ask("question", "Какую модель искать?", None), late])

    await world.round()

    # the call after the stop was answered «работа остановлена» and ended the run: the
    # model was not asked again, and nothing was queued
    assert len(world.requests()) == 1
    assert await _pending(world, person) == []
    [event] = await world.open_events(task_id)  # released for the resume, not failed
    assert (event.kind, event.attempts, event.wake_id) == (K.CREATED, 0, None)
    [wake] = await world.wakes()
    assert wake.finished_at is not None
    assert await world.repo.get_kv(f"task_failures:{task_id}") in (None, "0")
    await world.repo.set_stopped(False)
    world.script(END)
    await world.round()
    assert "задача запущена" in world.context()


class PausedMeanwhile(FakeLLM):
    """The owner pauses the task while the model thinks over its second step."""

    def __init__(self, status=TaskStatus.PAUSED) -> None:
        super().__init__()
        self.repo = None
        self.task_id = 0
        self.calls = 0
        self.status = status

    async def agent_step(self, **kw):
        self.calls += 1
        if self.calls == 2:
            await self.repo.set_task_status(self.task_id, self.status)
        return await super().agent_step(**kw)


async def test_25_a_task_paused_between_two_steps_does_nothing_more(
    repo, fake_clock, settings
):
    llm = PausedMeanwhile()
    world = AgentWorld(repo, fake_clock, settings, llm)
    task_id, [person] = await world.task("drill_shop")
    llm.repo, llm.task_id = repo, task_id
    late = send(person.conv_id, "Здравствуйте! Есть дрель Bosch?")
    world.script(
        [call("update_notes", notes="план", headline="Начинаю", person_notes=[])],
        [late],
    )

    await world.round()

    assert len(world.requests()) == 2  # the refused call ended the run
    assert await _pending(world, person) == []
    # Fix pass C5: the wake is released without a failure; its events wait for the resume
    # and the messages it saw stay new.
    [created] = await world.open_events(task_id)
    assert (created.kind, created.wake_id, created.attempts) == (K.CREATED, None, 0)
    [wake] = await world.wakes()
    assert wake.finished_at is not None and wake.outcome == "paused"
    assert await world.repo.get_kv(f"task_seen:{task_id}") is None
    assert await world.repo.get_kv(f"task_failures:{task_id}") in (None, "0")
    await world.assert_clean()


@pytest.mark.parametrize("status", [TaskStatus.CANCELLED, TaskStatus.DONE])
async def test_25_a_task_closed_between_two_steps_finishes_the_wake(
    repo, fake_clock, settings, status
):
    llm = PausedMeanwhile(status)
    world = AgentWorld(repo, fake_clock, settings, llm)
    task_id, [person] = await world.task("drill_shop")
    llm.repo, llm.task_id = repo, task_id
    world.script(
        [call("update_notes", notes="план", headline="Начинаю", person_notes=[])],
        [send(person.conv_id, "Здравствуйте! Есть дрель Bosch?")],
    )

    await world.round()

    assert await world.open_events(task_id) == []  # the wake is finished
    [wake] = await world.wakes()
    assert wake.finished_at is not None and wake.outcome == "finished"
    await world.assert_clean()


# 26: a confirmation is tied to the approved amount (fix pass B1) --------------------------


async def _approved_at(world, amount=None, terms=""):
    task_id, person = await world.replied("Есть, 550 000. Берёте?")
    world.script(
        [ask("approval", "Подтвердить?", person.conv_id, amount=amount, terms=terms)],
        END,
    )
    await world.round(BURST)
    assert await world.press("yes") == "Заказ подтверждён"
    world.clock.advance(MINUTE)
    return task_id, person


async def test_26_a_confirmation_may_name_only_the_approved_amount(world):
    _task_id, person = await _approved_at(world, amount=550_000)
    await world.write(person, "Извините, уже 750 000. Подтверждаете?")
    above = send(person.conv_id, "Подтверждаю заказ за 590 000", commits=True)
    bare = send(person.conv_id, "Да, подтверждаю, оформляйте", commits=True)
    named = send(person.conv_id, "Подтверждаю заказ за 550 000", commits=True)
    world.script([above], [bare], [named], END)

    await world.round(BURST)

    # under the task's hidden ceiling of 600 000, but above what the owner approved
    assert world.result(above).get("is_error") is True
    assert REASONS["over_ceiling"] in world.result(above)["content"]
    # a confirmation with no number would confirm the 750 000 he asked for
    refusal = world.result(bare)
    assert refusal["is_error"] is True
    assert refusal["content"].startswith(
        "подтверждая, назови сумму, которую разрешил владелец: 550 000"
    )
    assert world.result(named).get("is_error") is not True
    await world.deliver()
    assert world.sent_to(person)[-1] == "Подтверждаю заказ за 550 000"


async def test_26_an_approval_of_terms_only_keeps_the_task_s_ceiling(world):
    _task_id, person = await _approved_at(world, terms="доставка завтра")
    await world.write(person, "Так оформляем?")
    bare = send(person.conv_id, "Да, подтверждаю, оформляйте", commits=True)
    world.script([bare], END)

    await world.round(BURST)

    assert world.result(bare).get("is_error") is not True
    await world.deliver()
    assert world.sent_to(person)[-1] == "Да, подтверждаю, оформляйте"


# 27: the owner answers in the chat instead of on the card (fix pass B2) --------------------


async def test_27_the_owner_answers_in_the_chat_and_the_cards_let_the_person_go(world):
    task_id, person = await world.replied("Какой цвет нужен? И телефон дадите?")
    world.script(
        [send(person.conv_id, "Уточню и вернусь")],
        [ask("question", "Спрашивают цвет", person.conv_id)],
        [ask("phone", "Просит телефон", person.conv_id)],
        END,
    )
    await world.round(BURST)
    await world.deliver()
    question, phone = await world.approvals()
    assert await world.state(person) is S.WAITING_OWNER

    await world.inbox.on_owner_manual(person.peer_id, "Чёрный нужен", 9001)

    assert await world.approvals() == []
    answered = await world.repo.get_approval(question.id)
    assert (answered.status, answered.answer) == (
        ApprovalStatus.ANSWERED,
        "владелец ответил сам в чате",
    )
    assert (await world.repo.get_approval(phone.id)).status is ApprovalStatus.EXPIRED
    assert await world.repo.granted_kinds(person.conv_id) == set()
    assert await world.state(person) is S.TALKING
    assert await world.kinds(task_id) == [K.OWNER_WROTE]  # the pause stays as it was
    late = send(person.conv_id, "Спасибо, тогда жду цену")
    world.script([late], END)

    await world.round(timedelta(hours=12, minutes=1))  # his pause is over

    assert world.result(late).get("is_error") is not True
    assert "Чёрный нужен" in world.context()


# 28: a send that keeps failing stops waking the agent (fix pass B3) -----------------------

PHONE_TEXT = "Пишите: {PHONE}"  # approved, but the number is not in the keyring


async def _phone_approved(world):
    task_id, person = await world.replied("Дайте телефон")
    world.script([ask("phone", "Просит телефон", person.conv_id)], END)
    await world.round(BURST)
    await world.press("yes")
    return task_id, person


async def _phone_fails(world, person, advance=None):
    """The agent writes the number; the Sender cannot fill it in."""
    world.script([send(person.conv_id, PHONE_TEXT)], END)
    await world.round(advance)
    await world.deliver()


def _refused_line(world, person):
    prefix = f"- сообщение для [{person.conv_id}] не отправлено"
    return next(x for x in world.context().splitlines() if x.startswith(prefix))


async def test_28_the_third_failure_in_a_row_goes_to_the_owner(world):
    task_id, person = await _phone_approved(world)
    await _phone_fails(world, person)
    assert K.SEND_REFUSED in await world.kinds(task_id)
    await _phone_fails(world, person)
    assert "(" not in _refused_line(world, person).split(":")[0]  # the first time
    assert world.ui.notes == []

    await _phone_fails(world, person)

    assert _refused_line(world, person).startswith(
        f"- сообщение для [{person.conv_id}] не отправлено (2-й раз подряд): "
    )
    assert K.SEND_REFUSED not in await world.kinds(task_id)  # nothing wakes the agent
    assert len(world.ui.notes) == 1  # the Sender told the owner
    assert len(world.sent_to(person)) == 1  # only the greeting


async def test_28_the_count_starts_again_once_a_message_went_out(world):
    task_id, person = await _phone_approved(world)
    await _phone_fails(world, person)
    await _phone_fails(world, person)
    world.script([send(person.conv_id, "Номер пришлю чуть позже")], END)
    await world.round()
    await world.deliver()
    assert world.sent_to(person)[-1] == "Номер пришлю чуть позже"
    await world.write(person, "Жду номер")

    await _phone_fails(world, person, BURST)

    assert K.SEND_REFUSED in await world.kinds(task_id)  # the agent hears of it again
    assert world.ui.notes == []
    world.script(END)
    await world.round()
    assert "раз подряд" not in _refused_line(world, person)


async def test_28_a_task_file_missing_from_the_disk_is_not_queued_again(world):
    _task_id, [person] = await world.task("drill_shop", attachments=["files/spec.pdf"])
    world.script([send(person.conv_id, "Здравствуйте! ТЗ ниже", attach=True)], END)
    await world.round()
    await world.deliver()
    again = send(person.conv_id, "Повторю ТЗ", attach=True)
    world.script([again], END)

    await world.round()

    assert world.result(again)["content"].endswith(
        "Файл «spec.pdf» не найден на диске — отправить его не получится:"
        " спроси владельца"
    )
    files = [i for i in await world.repo.outbox_of(person.conv_id) if i.attach_path]
    assert [i.status for i in files] == [OutboxStatus.FAILED]


# 29: nothing is queued or finished for a task that is not in work (fix pass B4) ------------


class PausedWhileJudged(FakeLLM):
    """The owner pauses the task while the judge reads the draft (after the tools' check)."""

    def __init__(self) -> None:
        super().__init__()
        self.hook = None

    async def judge(self, messages, allowed, hint):
        if self.hook is not None:
            hook, self.hook = self.hook, None
            await hook()
        return await super().judge(messages, allowed, hint)


async def test_29_a_send_judged_while_the_task_was_paused_is_not_queued(
    repo, fake_clock, settings
):
    llm = PausedWhileJudged()
    world = AgentWorld(repo, fake_clock, settings, llm)
    task_id, person = await world.replied("Есть, 550 000")

    async def pause():  # as the owner's pause does it
        await world.repo.set_task_status(task_id, TaskStatus.PAUSED)
        await world.repo.cancel_pending(person.conv_id)

    llm.hook = pause
    reply = send(person.conv_id, "Спасибо! А доставка есть?")
    world.script([reply], END)

    await world.round(BURST)
    await world.deliver()

    refusal = world.result(reply)
    assert refusal["is_error"] is True
    assert refusal["content"] == "задача остановлена владельцем"
    assert await _pending(world, person) == []
    assert len(world.sent_to(person)) == 1  # only the greeting
    await world.assert_clean()


async def test_29_ok_under_a_card_of_a_paused_task_falls_back_to_its_event(world):
    task_id, person = await world.replied("А доставка нужна?")
    world.script(
        [ask("question", "Доставка?", person.conv_id, suggested=["Да, нужна"])], END
    )
    await world.round(BURST)
    await world.repo.set_task_status(task_id, TaskStatus.PAUSED)

    assert await world.press("ok") == "Ответ передан агенту"

    assert await _pending(world, person) == []
    [event] = await world.open_events(task_id)
    assert (event.kind, event.payload["suggested"]) == (K.OWNER_ANSWER, ["Да, нужна"])


class ClosedWhileSent(FakeUI):
    """The owner closes the task while the result card is on its way to him."""

    def __init__(self) -> None:
        super().__init__()
        self.hook = None

    async def send(self, card):
        if self.hook is not None:
            hook, self.hook = self.hook, None
            await hook()
        return await super().send(card)


async def test_29_finish_changes_nothing_for_a_task_closed_meanwhile(world):
    world.ui = ClosedWhileSent()
    world.build()
    task_id, _person = await world.replied("Есть, 550 000")
    world.script([ask("question", "Брать?", None)], END)
    await world.round(BURST)
    [question] = await world.approvals()

    async def close():  # as the owner's close does it
        await world.repo.set_task_status(task_id, TaskStatus.CANCELLED)

    world.ui.hook = close
    done = call("finish", result="Дрель есть за 550 000.")
    world.script([done])
    await world.repo.add_event(task_id, K.REMINDER, payload={"about": "итог"})

    await world.round()

    # the result card was on its way already; nothing else of `finish` happened
    assert (await world.repo.get_task(task_id)).status is TaskStatus.CANCELLED
    assert (await world.repo.get_approval(question.id)).status is ApprovalStatus.PENDING


# 30: a decision that cannot be carried out is not confirmed (fix pass B5) ------------------


async def test_30_finish_waits_for_the_owner_s_card_about_a_person(world):
    task_id, person = await world.replied("Есть, 550 000. Берёте?")
    early = call("finish", result="Дрель есть за 550 000, жду решения по заказу.")
    world.script(
        [ask("approval", "Подтвердить за 550 000?", person.conv_id, amount=550_000)],
        [early],
        END,
    )

    await world.round(BURST)

    refusal = world.result(early)
    assert refusal["is_error"] is True
    assert refusal["content"] == (
        f"сначала дождись ответа владельца по карточке о [{person.conv_id}]"
        " или закончи с этим человеком"
    )
    assert (await world.repo.get_task(task_id)).status is TaskStatus.ACTIVE
    assert len(world.ui.sent) == 1  # the card, no result
    assert await world.press("yes") == "Заказ подтверждён"
    world.script(
        [send(person.conv_id, "Подтверждаю заказ за 550 000", commits=True)], END
    )
    await world.round()
    await world.deliver()
    assert world.sent_to(person)[-1] == "Подтверждаю заказ за 550 000"


async def test_30_finish_keeps_the_card_that_returns_a_chat(world):
    task_id, [a, b] = await world.contacted("shop_a", "shop_b")
    world.clock.advance(timedelta(minutes=10))
    await world.write(a, "Есть, 550 000. Берёте?")
    world.script(
        [
            ask("approval", "Подтвердить за 550 000?", a.conv_id, amount=550_000),
            ask("question", "Искать ещё?", None),
        ],
        END,
    )
    await world.round(BURST)
    order, _question = await world.approvals()
    # the owner takes the chat with A
    await world.runner.approvals.owner_decision(order.id, order.nonce, "self")
    await world.write(b, "Нет в наличии")
    world.script([call("finish", result="B: нет. A: чат у владельца.")])
    await world.round(BURST)
    assert await world.repo.get_kv(f"task_result_waits:{task_id}") == "1"  # R109
    [handoff] = await world.approvals()  # the task-level question expired, this did not
    assert handoff.kind.value == "handoff"

    # stage 3: an instruction takes the finished task up again
    await world.repo.set_task_status(task_id, TaskStatus.ACTIVE)
    toast = await world.runner.approvals.owner_decision(
        handoff.id, handoff.nonce, "back"
    )

    assert toast == "Агент снова ведёт чат"
    assert await world.state(a) is S.TALKING
    late = send(a.conv_id, "Подскажите, какой срок?")
    world.script([late], END)
    await world.round()
    assert world.result(late).get("is_error") is not True


async def test_30_a_person_closed_while_his_card_waits_takes_the_card_with_him(world):
    task_id, person = await world.replied("Есть, 550 000. Берёте?")
    world.script(
        [ask("approval", "Подтвердить за 550 000?", person.conv_id, amount=550_000)],
        END,
    )
    await world.round(BURST)
    [card] = await world.approvals()

    # Telegram refused the chat for good meanwhile: the person is closed
    await people.close(world.repo, world.gateway, person.conv_id, "он заблокировал нас")

    assert (await world.repo.get_approval(card.id)).status is ApprovalStatus.EXPIRED
    toast = await world.runner.approvals.owner_decision(card.id, card.nonce, "yes")
    assert toast == "Уже решено"
    assert await world.repo.granted_kinds(person.conv_id) == set()
    assert K.OWNER_ANSWER not in await world.kinds(task_id)


# 31: an edit is news (fix pass B7) ---------------------------------------------------------


class EditedWhileThinking(FakeLLM):
    """The person edits his last message while the model works on its step (when armed)."""

    def __init__(self) -> None:
        super().__init__()
        self.world = None
        self.person = None
        self.armed = False

    async def agent_step(self, **kw):
        step = await super().agent_step(**kw)
        if self.armed:
            self.armed = False
            self.world.clock.advance(timedelta(seconds=5))  # the model takes its time
            msg_id = self.world.gateway.chats[self.person.peer_id][-1].tg_msg_id
            await self.world.inbox.on_edit(self.person.peer_id, msg_id, "Цена 740 000")
        return step


async def test_31_an_edit_while_the_agent_thinks_makes_its_reply_stale(
    repo, fake_clock, settings
):
    llm = EditedWhileThinking()
    world = AgentWorld(repo, fake_clock, settings, llm)
    llm.world = world
    _task_id, person = await world.replied("Цена 540 000")
    llm.person, llm.armed = person, True
    reply = send(person.conv_id, "Отлично, 540 000 подходит")
    world.script([reply], END)

    await world.round(BURST)

    refusal = world.result(reply)
    assert refusal["is_error"] is True and refusal["content"] == STALE
    assert await _pending(world, person) == []
    world.script(END)
    await world.round(BURST)  # the edit's own wake
    lines = world.context().splitlines()
    [edited] = [line for line in lines if "Цена 740 000" in line]
    assert edited.endswith("Цена 740 000 (изменено)")
    assert lines.index(edited) > lines.index("— новое —")
    await world.assert_clean()


# 32: only a reminder counts as a reminder (fix pass B10) -----------------------------------


async def test_32_relaying_the_owner_s_answer_is_not_a_reminder(world):
    _task_id, person = await world.replied("Какой цвет нужен?")
    world.script(
        [send(person.conv_id, "Уточню и вернусь")],
        [ask("question", "Спрашивают цвет", person.conv_id)],
        END,
    )
    await world.round(BURST)
    await world.deliver()
    await world.press("answer", "Чёрный")
    world.script([send(person.conv_id, "Нужен чёрный")], END)
    await world.round()
    await world.deliver()

    assert (await world.repo.get_conversation(person.conv_id)).followups_sent == 0
    world.script([send(person.conv_id, "Напомню: чёрный есть?")], END)
    await world.round(DAY)
    assert f"[{person.conv_id}] молчит 24 ч, напоминаний было: 0" in world.context()
    await world.deliver()
    # a send in the wake for his silence is a reminder
    assert (await world.repo.get_conversation(person.conv_id)).followups_sent == 1


# 33: a text that already went is not queued again (fix pass B11) ---------------------------


class SentWhileThinking(FakeLLM):
    """The Sender sends what was queued while the model works on its step (when armed)."""

    def __init__(self) -> None:
        super().__init__()
        self.world = None
        self.armed = False

    async def agent_step(self, **kw):
        step = await super().agent_step(**kw)
        if self.armed:
            self.armed = False
            self.world.clock.advance(timedelta(seconds=5))
            await self.world.sender.tick()
        return step


async def test_33_a_wake_does_not_repeat_what_went_out_while_it_thought(
    repo, fake_clock, settings
):
    llm = SentWhileThinking()
    world = AgentWorld(repo, fake_clock, settings, llm)
    llm.world = world
    # the account says it is an AI: no introduction, the text goes as written
    world.gateway.name = "ИИ помощник Саида"
    task_id, [person] = await world.task("drill_shop")
    world.script([send(person.conv_id, "Здравствуйте! Есть дрель Bosch?")], END)
    await world.runner.tick()  # queued, not sent yet
    await world.repo.add_event(task_id, K.RECOVERED)  # a wake that sees it queued
    world.clock.advance(DELIVERY)
    llm.armed = True
    again = send(person.conv_id, "Здравствуйте! Есть дрель Bosch?")
    other = send(person.conv_id, "Здравствуйте! Есть дрель Bosch?", "И кейс к ней?")
    world.script([again], [other], END)

    await world.round()

    assert "(в очереди, уйдёт " in world.context(-3)
    refusal = world.result(again)
    assert (
        refusal["is_error"] is True and refusal["content"] == "это сообщение уже ушло"
    )
    assert world.result(other).get("is_error") is not True
    await world.deliver()
    assert world.sent_to(person) == ["Здравствуйте! Есть дрель Bosch?", "И кейс к ней?"]
    await world.assert_clean()


# 34: words (fix pass B12) ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("approved", "reason"),
    [
        (True, "нужен телефон владельца, но его нет в хранилище"),
        (False, "в тексте телефон владельца, а разрешения его давать нет"),
    ],
)
async def test_34_the_sender_s_reasons_name_the_owner_as_the_owner(
    world, approved, reason
):
    _task_id, person = await world.replied("Дайте телефон")
    if approved:
        world.script([ask("phone", "Просит телефон", person.conv_id)], END)
        await world.round(BURST)
        await world.press("yes")
        await _phone_fails(world, person)
    else:  # nothing in the core queues this; the Sender's own check still words it
        await world.repo.enqueue(person.conv_id, PHONE_TEXT, world.clock.now())
        world.script(END)  # the wake for his message
        await world.deliver()
    world.script(END)

    await world.round(BURST)

    assert _refused_line(world, person).endswith(f": {reason}")


async def test_34_the_rules_say_to_ask_the_owner_for_more_people(world):
    _task_id, [_person] = await world.task("drill_shop")
    world.script(END)
    await world.round()
    rules = world.requests()[-1]["system"][0]
    # Stage 4 (Task 4.3): the agent may also find more people itself.
    assert (
        "Потом закончи с ним и, если задаче нужны ещё люди, найди их (research) или"
        " спроси владельца." in rules
    )
    assert "найди замену" not in rules


# 35: «Нет» cancels, then reopens (fix pass B13) --------------------------------------------


async def test_35_no_for_a_person_whose_only_message_was_queued_leaves_him_planned(
    world,
):
    _task_id, [person] = await world.task("drill_shop")
    world.script(
        [
            send(
                person.conv_id,
                "Здравствуйте! Возьмём дрель за 550 000, если привезёте?",
            )
        ],
        [ask("approval", "Подтвердить 550 000?", person.conv_id, amount=550_000)],
        END,
    )
    await world.round()

    assert await world.press("no") == "Ок, не заказываем"

    assert await _pending(world, person) == []
    assert await world.state(person) is S.PLANNED
    world.script(END)
    await world.round()
    assert (
        f"[{person.conv_id}] Drill_Shop (@drill_shop) — ещё не писали"
        in world.context()
    )


# 36: first messages to new people are paced (Task 3.1) ------------------------------------


async def test_36_five_new_people_with_a_quota_of_three(repo, fake_clock, settings):
    world = AgentWorld(repo, fake_clock, settings)
    _task_id, found = await world.task("shop_a", "shop_b", "shop_c", "shop_d", "shop_e")
    greetings = [send(p.conv_id, "Здравствуйте! Есть дрель Bosch?") for p in found]
    world.script(greetings, END)
    start = world.clock.now()  # 10:00, the default quota: 3 new people a day
    zone = world.settings.schedule.zone

    await world.round()

    slots = []
    for person in found:
        [item] = await world.repo.outbox_of(person.conv_id)
        slots.append(item.not_before.astimezone(zone))
    today, tomorrow = start.date(), start.date() + timedelta(days=1)
    assert [slot.date() for slot in slots] == [today] * 3 + [tomorrow] * 2
    assert slots[3] == slots[3].replace(hour=9, minute=0, second=0, microsecond=0)
    for earlier, later in pairwise(slots):
        if later.date() == earlier.date():
            assert MINUTE <= later - earlier <= 2 * MINUTE  # ruling R70
    for block, slot in zip(greetings, slots, strict=True):
        day = "сегодня" if slot.date() == today else "завтра"
        content = world.result(block)["content"]
        words = f"Это первое сообщение этому человеку: уйдёт {day} около {slot:%H:%M}."
        assert words in content

    # The Scheduler's rounds, a minute apart: today until noon, tomorrow from 08:55.
    while world.clock.now() < start.replace(hour=12):
        await world.round(MINUTE)
    world.clock.set(start.replace(day=start.day + 1, hour=8, minute=55))
    while world.clock.now() < start.replace(day=start.day + 1, hour=9, minute=45):
        await world.round(MINUTE)

    sent = []
    for person, slot in zip(found, slots, strict=True):
        [item] = await world.repo.outbox_of(person.conv_id)
        assert item.status is OutboxStatus.SENT
        at = item.sent_at.astimezone(zone)
        assert slot <= at <= slot + MINUTE
        sent.append(at)
    assert [at.date() for at in sent] == [today] * 3 + [tomorrow] * 2
    for earlier, later in pairwise(sent):
        if later.date() == earlier.date():
            assert MINUTE <= later - earlier <= 3 * MINUTE
    await world.assert_clean()


async def test_36_slots_gone_stale_after_a_stop_still_go_a_minute_apart(
    repo, fake_clock, settings
):
    world = AgentWorld(repo, fake_clock, settings)
    _task_id, found = await world.task("shop_a", "shop_b", "shop_c")
    world.script(
        [send(p.conv_id, "Здравствуйте! Есть дрель Bosch?") for p in found], END
    )
    await world.round()
    await world.repo.set_stopped(True)  # /stop: every slot passes while nothing is sent
    for _ in range(60):
        await world.round(MINUTE)
    assert world.gateway.sent == []

    await world.repo.set_stopped(False)
    for _ in range(40):
        await world.round(MINUTE)

    sent = []
    for person in found:
        [item] = await world.repo.outbox_of(person.conv_id)
        assert item.status is OutboxStatus.SENT
        sent.append(item.sent_at)
    sent.sort()
    assert all(later - earlier >= MINUTE for earlier, later in pairwise(sent))
    deferred = await world.repo.events_of("first_contact_deferred")
    assert {event["reason"] for event in deferred} == {"gap"}
    await world.assert_clean()


# 37: the reminder cap holds in every wake; only a relay is exempt (fix pass C1) ----------

AT_CAP = "напоминаний без ответа уже 2 — закончи с этим человеком или спроси владельца"


@pytest.mark.parametrize("woken_by", ["another_person", "remind_me", "nudge"])
async def test_37_at_the_cap_a_send_from_any_wake_is_refused(world, woken_by):
    task_id, [a, b] = await world.contacted("shop_a", "shop_b")
    await world.repo.update_conversation(a.conv_id, followups_sent=2)
    world.clock.advance(HOUR)
    if woken_by == "another_person":
        await world.write(b, "Есть, 550 000")
    elif woken_by == "remind_me":
        await world.repo.add_event(
            task_id, K.REMINDER, conv_id=a.conv_id, payload={"about": "напомнить"}
        )
    else:
        await world.repo.add_event(task_id, K.REMINDER, payload={"about": STALLED_ABOUT})
    again = send(a.conv_id, "Напомню: есть дрель Bosch?")
    world.script([again], END)

    await world.round(BURST)

    refusal = world.result(again)
    assert refusal["is_error"] is True and refusal["content"] == AT_CAP
    assert await _pending(world, a) == []
    assert (await world.repo.get_conversation(a.conv_id)).followups_sent == 2


@pytest.mark.parametrize("kind", ["question", "handoff"])
async def test_37_a_relay_of_the_owner_s_word_goes_out_at_the_cap(world, kind):
    task_id, [person] = await world.contacted()
    await world.repo.update_conversation(person.conv_id, followups_sent=2)
    world.clock.advance(HOUR)
    await world.repo.add_event(
        task_id, K.REMINDER, conv_id=person.conv_id, payload={"about": "спросить"}
    )
    world.script([ask(kind, "Что ему ответить?", person.conv_id)], END)
    await world.round()
    if kind == "question":
        await world.press("answer", "Скажи, что берём чёрный")
    else:
        await world.press("back")
    relay = send(person.conv_id, "Владелец выбрал чёрный")
    world.script([relay], END)

    await world.round()

    assert world.result(relay).get("is_error") is not True
    assert (await world.repo.get_conversation(person.conv_id)).followups_sent == 2


async def test_37_ok_under_a_suggested_reply_relays_it_at_the_cap(world):
    _task_id, person = await world.replied("Какой цвет нужен?")
    world.script(
        [send(person.conv_id, "Уточню и вернусь")],
        [ask("question", "Цвет?", person.conv_id, suggested=["Нужен чёрный"])],
        END,
    )
    await world.round(BURST)
    await world.deliver()
    await world.repo.update_conversation(person.conv_id, followups_sent=2)

    assert await world.press("ok") == "Ок, отправляю"
    world.script(END)  # the agent hears of his «Ок» once it went

    await world.deliver()
    assert world.sent_to(person)[-1] == "Нужен чёрный"
    assert (await world.repo.get_conversation(person.conv_id)).followups_sent == 2
    await world.round()  # the event came with the send: the next round's wake reads it


async def test_37_the_rules_say_to_answer_everyone_who_wrote(world):
    _task_id, [_person] = await world.task("drill_shop")
    world.script(END)
    await world.round()
    rules = world.requests()[-1]["system"][0]
    assert (
        "Каждому, кто написал, ответь в этом же пробуждении или запиши в заметках,"
        " почему ответ не нужен." in rules
    )


# 38: the keep-alive acts under the conversation's lock (fix pass C3) ---------------------


async def test_38_a_reply_that_lands_during_the_keep_alive_does_not_close_him(world):
    task_id, [person] = await world.contacted()
    await world.repo.update_conversation(person.conv_id, followups_sent=2)
    await world.repo.drop_task_events(task_id)  # his silence is gone: the keep-alive acts
    world.clock.advance(DAY + HOUR)
    store = world.repo.add_message
    jobs = []

    async def keep_alive_meanwhile(*args, **kw):
        # The inbox holds his lock and is about to store the reply: the keep-alive runs now.
        if not jobs:
            jobs.append(asyncio.create_task(world.runner._keep_alive(task_id)))
            for _ in range(100):
                if jobs[0].done():
                    break
                await asyncio.sleep(0.01)
        return await store(*args, **kw)

    world.repo.add_message = keep_alive_meanwhile
    await world.write(person, "Есть, 550 000")
    await jobs[0]

    assert await world.state(person) is not S.CLOSED
    assert await world.repo.events_of("person_gave_up") == []


async def test_38_two_keep_alives_at_once_tell_the_owner_once(world, monkeypatch):
    task_id, [_person] = await world.task("drill_shop", start=False)
    await world.repo.set_task_status(task_id, TaskStatus.ACTIVE)
    await world.repo.set_kv(f"stalled:{task_id}", "1")  # the nudge was given already
    read = world.repo.get_kv

    async def slow_read(key):
        value = await read(key)
        if key.startswith("stalled:"):
            await asyncio.sleep(0.05)  # both calls read the count before either writes it
        return value

    monkeypatch.setattr(world.repo, "get_kv", slow_read)

    await asyncio.gather(
        world.runner._keep_alive(task_id), world.runner._keep_alive(task_id)
    )

    stalled = [card for card in world.ui.sent if "остановилась" in card.text]
    assert len(stalled) == 1


# 39: the hand-back card is repaired while the program runs (fix pass C6) ------------------


@pytest.mark.parametrize("next_step", ["wake", "decision"])
async def test_39_a_hand_back_card_that_failed_is_sent_after_the_next_step(
    world, next_step
):
    task_id, person = await world.replied("Есть, 550 000. Берёте?")
    world.script(
        [
            ask("approval", "Подтвердить за 550 000?", person.conv_id, amount=550_000),
            ask("question", "Искать ещё?", None),
        ],
        END,
    )
    await world.round(BURST)
    order, question = await world.approvals()
    send_card = world.ui.send
    failing = True

    async def flaky_send(card):
        if failing:
            raise ConnectionError("bot is down")
        return await send_card(card)

    world.ui.send = flaky_send
    toast = await world.runner.approvals.owner_decision(order.id, order.nonce, "self")
    assert toast == "Чат за тобой. Карточку для возврата пришлю позже"
    assert [a.kind.value for a in await world.approvals()] == ["question"]
    failing = False

    if next_step == "wake":
        await world.repo.add_event(task_id, K.REMINDER, payload={"about": "проверить"})
        world.script(END)
        await world.round()
    else:
        await world.runner.approvals.owner_decision(
            question.id, question.nonce, "answer", "Нет, хватит"
        )

    [handoff] = [a for a in await world.approvals() if a.kind.value == "handoff"]
    assert handoff.conversation_id == person.conv_id
    assert handoff.control_msg_id is not None
    assert await world.state(person) is S.HANDED_TO_OWNER


# 40: what the owner does to a task (Task 3.2) ----------------------------------------------


def _ops(world):
    import random

    from tg_agent.agent.ops import TaskOps

    return TaskOps(
        world.repo, world.gateway, world.clock, world.settings, world.runner,
        random.Random(11),
    )  # fmt: skip


class OwnerMeanwhile(FakeLLM):
    """The owner acts (`act`) while the model thinks over the second step after `arm`."""

    def __init__(self) -> None:
        super().__init__()
        self.act = None
        self.calls = 0

    def arm(self, act) -> None:
        self.act, self.calls = act, 0

    async def agent_step(self, **kw):
        self.calls += 1
        if self.act is not None and self.calls == 2:
            act, self.act = self.act, None
            await act()
        return await super().agent_step(**kw)


async def test_40_the_goodbye_of_a_closed_task_goes_out_although_he_wrote_after_it(
    world,
):
    from tg_agent.tasks.templates import DECLINE

    task_id, person = await world.replied("Есть GSB 13, 550 000")
    assert await world.state(person) is S.TALKING
    calls = len(world.requests())

    line = await _ops(world).close(task_id)

    assert line == f"Задача #{task_id} закрыта. Попрощался: @drill_shop"
    [bye] = await _pending(world, person)
    world.clock.advance(timedelta(seconds=5))
    await world.write(person, "Так берёте или нет?")
    # Even a message stored after the goodbye was queued does not make it stale: a closed
    # person gets no other answer (`Runner.can_reanswer` is false for him).
    await world.repo.update_conversation(person.conv_id, last_in_at=world.clock.now())
    fresh = await world.repo.get_conversation(person.conv_id)
    assert world.runner.can_reanswer(fresh) is False
    await world.deliver()

    assert world.sent_to(person)[-1] == DECLINE["ru"]
    assert (await world.repo.get_outbox(bye.id)).status is OutboxStatus.SENT
    assert len(world.requests()) == calls  # nobody woke the agent
    assert await world.open_events(task_id) == []


async def test_40_a_goodbye_queued_before_the_exclusion_never_goes(world):
    # R97 (b), review D1 (a): «Закрыть задачу» at night queues the goodbye; «не работай
    # больше с @drill_shop» follows; in the morning the Sender drops the goodbye.
    from tg_agent.agent import memory

    task_id, person = await world.replied("Есть, 500 000")
    await _ops(world).close(task_id)
    [bye] = await _pending(world, person)
    await memory.remember(
        world.repo, "не работать с @drill_shop", "drill_shop", world.clock.now()
    )
    before = len(world.sent_to(person))

    for _ in range(4):
        await world.deliver()

    assert len(world.sent_to(person)) == before
    assert (await world.repo.get_outbox(bye.id)).status is OutboxStatus.CANCELLED
    assert await world.repo.events_of("outbox_cancelled", person.conv_id) == [
        {"reason": "excluded", "items": 1}
    ]


async def test_40_a_task_closed_between_two_tool_calls_does_nothing_more(
    repo, fake_clock, settings
):
    from tg_agent.tasks.templates import DECLINE

    llm = OwnerMeanwhile()
    world = AgentWorld(repo, fake_clock, settings, llm)
    task_id, person = await world.replied("Есть GSB 13, 550 000")
    ops = _ops(world)
    later = call("remind_me", hours=2, about="проверить", person_id=None)
    world.script([send(person.conv_id, "Отлично, а кейс есть?")], [later])
    lines = []

    async def close():
        lines.append(await ops.close(task_id))

    llm.arm(close)
    calls = len(world.requests())
    await world.round(BURST)

    assert lines == [f"Задача #{task_id} закрыта. Попрощался: @drill_shop"]
    # the call after the close was answered «работа остановлена» and ended the run: the
    # model was not asked again, and no reminder was set
    assert len(world.requests()) == calls + 2
    assert not await world.repo.events_of("wake_failed")
    # the reply queued before the close was cancelled; only the goodbye waits
    assert [item.text for item in await _pending(world, person)] == [DECLINE["ru"]]
    assert await world.open_events(task_id) == []
    [*_, wake] = await world.wakes()
    assert wake.finished_at is not None
    await world.deliver()
    assert world.sent_to(person)[1:] == [DECLINE["ru"]]
    await world.round(DAY)
    assert world.sent_to(person)[1:] == [DECLINE["ru"]]
    assert await world.open_events(task_id) == []
    await world.assert_clean()


async def test_40_a_task_paused_between_two_tool_calls_waits_for_the_resume(
    repo, fake_clock, settings
):
    llm = OwnerMeanwhile()
    world = AgentWorld(repo, fake_clock, settings, llm)
    task_id, person = await world.replied("Есть GSB 13, 550 000")
    ops = _ops(world)
    later = call("remind_me", hours=2, about="проверить", person_id=None)
    world.script([send(person.conv_id, "Отлично, а кейс есть?")], [later])

    async def pause():
        await ops.pause(task_id)

    llm.arm(pause)
    calls = len(world.requests())
    await world.round(BURST)

    assert len(world.requests()) == calls + 2  # the refused call ended the run
    assert (await world.wakes())[-1].outcome == "paused"
    assert await _pending(world, person) == []  # the reply queued before was cancelled
    # released for the resume, and the cancelled reply is news for the agent then
    incoming, refused = await world.open_events(task_id)
    assert (incoming.kind, incoming.wake_id, incoming.attempts) == (K.INCOMING, None, 0)
    assert (refused.kind, refused.payload["closed"]) == (K.SEND_REFUSED, False)
    await world.deliver()
    await world.round(DAY)
    assert len(world.sent_to(person)) == 1  # only the greeting
    kinds = [K.INCOMING, K.SEND_REFUSED]
    assert [e.kind for e in await world.open_events(task_id)] == kinds

    assert await ops.resume(task_id) == f"Задача #{task_id} продолжена"
    world.script([send(person.conv_id, "Отлично, а кейс есть?")], END)
    await world.round()
    await world.deliver()

    assert "Есть GSB 13, 550 000" in world.context()
    assert "задача продолжена после паузы" in world.context()
    assert world.sent_to(person)[1:] == ["Отлично, а кейс есть?"]
    await world.assert_clean()


async def test_40_a_finished_task_taken_up_again_lets_go_who_is_busy_elsewhere(world):
    task_id, [free, busy] = await world.contacted("drill_shop", "tool_shop")
    await world.repo.add_event(task_id, K.REMINDER, payload={"about": "итог"})
    world.script([call("finish", result="Дрель есть у @drill_shop")])
    await world.round()
    # finished before R109: DONE, its people still open
    await world.repo.set_task_status(task_id, TaskStatus.DONE)
    # tool_shop is in work in another task by now
    other = await world.repo.create_task(
        "Шуруповёрт", "Купить шуруповёрт", {}, None, "UZS", [], []
    )
    contact = await world.repo.contact_by_peer(busy.peer_id)
    assert contact is not None
    await world.repo.create_conversation(other, contact.id, state=S.PLANNED)
    await world.repo.set_task_status(other, TaskStatus.ACTIVE)

    line = await _ops(world).instruct(task_id, "Спроси у них про кейс")

    assert line == f"Задача #{task_id} снова в работе: передал агенту"
    assert (await world.repo.get_task(task_id)).status is TaskStatus.ACTIVE
    let_go = await world.repo.get_conversation(busy.conv_id)
    assert (let_go.state, let_go.close_reason) == (S.CLOSED, f"занят в задаче #{other}")
    assert await world.state(free) is S.CONTACTED
    # tool_shop is still in work in #other
    assert world.gateway.allowed == {free.peer_id, busy.peer_id}
    world.script(END)
    await world.round()
    assert "Спроси у них про кейс" in world.trusted()


# 41: the owner's decision is never overwritten by a running wake (Task 3.2, fix round 1) -----


async def test_41_a_close_between_finish_s_card_and_its_status_write_stands(
    world, monkeypatch
):
    task_id, _ = await world.contacted()
    ops = _ops(world)
    real_set_kv = world.repo.set_kv
    lines = []

    async def closing_set_kv(key, value):
        if key.startswith("task_card:") and not lines:
            lines.append(await ops.close(task_id))  # the owner closes it right now
        await real_set_kv(key, value)

    monkeypatch.setattr(world.repo, "set_kv", closing_set_kv)
    await world.repo.add_event(task_id, K.REMINDER, payload={"about": "итог"})
    world.script([call("finish", result="Дрель есть у @drill_shop")])

    await world.round()

    assert (await world.repo.get_task(task_id)).status is TaskStatus.CANCELLED
    assert await world.open_events(task_id) == []


async def test_41_a_wake_that_fails_after_a_close_leaves_no_open_event(
    world, monkeypatch
):
    task_id, _ = await world.replied("Есть GSB 13")
    ops = _ops(world)
    real_log = world.repo.log_event

    async def closing_log(type, conv_id=None, **payload):
        if type == "wake_failed":
            await ops.close(task_id)  # closed after the wake read the task's status
        await real_log(type, conv_id, **payload)

    monkeypatch.setattr(world.repo, "log_event", closing_log)
    world.script(LLMError("overloaded"))

    await world.round(BURST)

    assert (await world.repo.get_task(task_id)).status is TaskStatus.CANCELLED
    assert await world.open_events(task_id) == []
    await world.round(DAY)
    assert await world.open_events(task_id) == []


async def test_41_a_stop_and_then_a_close_during_a_wake_leave_no_open_event(
    repo, fake_clock, settings
):
    llm = OwnerMeanwhile()
    world = AgentWorld(repo, fake_clock, settings, llm)
    task_id, person = await world.replied("Есть GSB 13")
    ops = _ops(world)
    world.script(
        [call("update_notes", notes="план", headline="Отвечаю", person_notes=[])],
        [send(person.conv_id, "Отлично")],
    )

    async def stop_then_close():
        await ops.stop_all()
        await ops.close(task_id)

    llm.arm(stop_then_close)
    await world.round(BURST)

    assert (await world.repo.get_task(task_id)).status is TaskStatus.CANCELLED
    assert await world.open_events(task_id) == []
    [*_, wake] = await world.wakes()
    assert wake.finished_at is not None
    await ops.resume_all()
    await world.round(DAY)
    assert await world.open_events(task_id) == []
    await world.assert_clean()


async def test_41_a_reply_cancelled_by_a_pause_is_written_again_after_the_resume(world):
    task_id, person = await world.replied("Есть GSB 13, 550 000")
    world.script([send(person.conv_id, "Сколько стоит доставка?")], END)
    await world.round(BURST)  # the wake ended; its reply waits its human pause
    assert [item.text for item in await _pending(world, person)] == [
        "Сколько стоит доставка?"
    ]
    ops = _ops(world)

    await ops.pause(task_id)
    await world.deliver()
    assert len(world.sent_to(person)) == 1  # only the greeting
    await ops.resume(task_id)
    world.script([send(person.conv_id, "Подскажите, сколько стоит доставка?")], END)
    await world.round()
    await world.deliver()

    context = world.context()
    assert "сообщение для" in context
    assert "задача была на паузе: сообщение не ушло" in context
    assert world.sent_to(person)[1:] == ["Подскажите, сколько стоит доставка?"]


async def test_41_an_instruction_given_while_the_result_goes_wakes_the_agent(world):
    task_id, _ = await world.contacted()
    ops = _ops(world)
    send_card = world.ui.send
    told = []

    async def send_and_instruct(card):
        msg_id = await send_card(card)
        if card.kind == "result":
            told.append(await ops.instruct(task_id, "Спроси ещё про кейс"))
        return msg_id

    world.ui.send = send_and_instruct
    await world.repo.add_event(task_id, K.REMINDER, payload={"about": "итог"})
    world.script([call("finish", result="Дрель есть у @drill_shop")])

    await world.round()

    # his words on the result continue the task (R109)
    assert told == [f"Задача #{task_id} снова в работе: передал агенту"]
    assert [card.kind for card in world.ui.sent] == ["result"]
    assert (await world.repo.get_task(task_id)).status is TaskStatus.ACTIVE
    assert K.OWNER_INSTRUCTION in await world.kinds(task_id)
    world.script(END)
    await world.round()
    assert "Спроси ещё про кейс" in world.trusted()


# 42: a queued message dies with its person (fix pass D1.1) --------------------------------


async def test_42_a_night_launch_called_off_by_the_owner_sends_nothing_at_9(world):
    world.clock.set(world.clock.now().replace(hour=0, minute=40))
    task_id, [person] = await world.task("drill_shop")
    world.script([send(person.conv_id, "Здравствуйте! Есть дрель Bosch?")], END)
    await world.round()
    [queued] = await _pending(world, person)
    zone = world.settings.schedule.zone
    assert queued.not_before.astimezone(zone).strftime("%H:%M") == "09:00"

    assert await _ops(world).instruct(task_id, "Не надо, я уже сам узнал") == (
        "Передал агенту"
    )
    close = call("close_person", person_id=person.conv_id, reason="владелец узнал сам")
    done = call("finish", result="Владелец узнал сам, никому не писал")
    world.script([close], [done])
    await world.round()

    assert world.result(close)["content"] == (
        f"С [{person.conv_id}] закончили. Не ушедшие сообщения ему отменены: 1."
    )
    # R109: the result waits for «Завершить»; the task is the owner's to end
    assert await world.repo.get_kv(f"task_result_waits:{task_id}") == "1"
    # one result, after the close
    assert [card.kind for card in world.ui.sent] == ["result"]
    assert await _pending(world, person) == []

    world.clock.set(world.clock.now().replace(hour=8, minute=55))
    for _ in range(120):
        await world.round(MINUTE)
    assert world.gateway.sent == []


async def test_42_a_person_the_agent_closes_never_gets_his_queued_first_message(world):
    names = ("vendor_1", "vendor_2", "vendor_3", "vendor_4")
    task_id, found = await world.task(*names)
    world.script(
        [send(p.conv_id, "Здравствуйте! Есть дрель Bosch?") for p in found], END
    )
    await world.round()
    fourth = found[3]
    assert len(await _pending(world, fourth)) == 1  # tomorrow at 09:00, over the quota

    await _ops(world).instruct(task_id, "vendor_4 не пиши, это мошенник")
    close = call(
        "close_person", person_id=fourth.conv_id, reason="владелец просил не писать"
    )
    world.script([close], END)
    await world.round()

    assert world.result(close)["content"] == (
        f"С [{fourth.conv_id}] закончили. Не ушедшие сообщения ему отменены: 1."
    )
    start = world.clock.now()
    while world.clock.now() < start.replace(hour=12):
        await world.round(MINUTE)
    world.clock.set(start.replace(day=start.day + 1, hour=8, minute=55))
    while world.clock.now() < start.replace(day=start.day + 1, hour=9, minute=45):
        await world.round(MINUTE)

    assert all(len(world.sent_to(p)) == 1 for p in found[:3])
    assert world.sent_to(fourth) == []
    await world.assert_clean()


async def test_42_a_goodbye_queued_with_close_person_in_one_wake_still_goes(world):
    _task_id, person = await world.replied("Нет, дрелей не держим")
    bye = send(person.conv_id, "Понял, спасибо!")
    close = call("close_person", person_id=person.conv_id, reason="дрелей нет")
    world.script([bye, close], END)

    await world.round(BURST)

    assert world.result(close)["content"] == f"С [{person.conv_id}] закончили."
    await world.deliver()
    assert world.sent_to(person)[-1] == "Понял, спасибо!"


async def test_42_a_reply_and_finish_in_one_wake_still_deliver_the_reply(world):
    task_id, person = await world.replied("Есть GSB 13, 550 000")
    world.script(
        [
            send(person.conv_id, "Спасибо, передам владельцу!"),
            call("finish", result="Дрель GSB 13 — 550 000 у @drill_shop"),
        ]
    )

    await world.round(BURST)

    assert await world.repo.get_kv(f"task_result_waits:{task_id}") == "1"  # R109
    await world.deliver()
    assert world.sent_to(person)[-1] == "Спасибо, передам владельцу!"


async def test_42_the_sender_never_sends_a_first_message_of_a_closed_person_or_task(
    world,
):
    closed_task, [closed] = await world.task("shop_a", start=False)
    done_task, [done] = await world.task("shop_b", start=False)
    open_task, [known] = await world.task("shop_c", start=False)
    for task_id in (closed_task, done_task, open_task):
        await world.repo.set_task_status(task_id, TaskStatus.ACTIVE)
    await world.repo.set_state(closed.conv_id, S.CLOSED)
    await world.repo.set_state(done.conv_id, S.CONTACTED)
    await world.repo.set_task_status(done_task, TaskStatus.DONE)
    # a goodbye to a closed person who was written to before still goes
    await world.repo.set_state(known.conv_id, S.CLOSED)
    await world.repo.update_conversation(known.conv_id, first_sent=True)
    now = world.clock.now()
    for person in (closed, done, known):
        await world.repo.enqueue(person.conv_id, "Здравствуйте!", now)

    await world.sender.tick()

    assert world.sent_to(closed) == [] and world.sent_to(done) == []
    assert world.sent_to(known) == ["Здравствуйте!"]
    for person in (closed, done):
        [item] = await world.repo.outbox_of(person.conv_id)
        assert item.status is OutboxStatus.CANCELLED
    dropped = await world.repo._all(
        "SELECT conversation_id FROM events WHERE type = 'first_message_dropped'"
    )
    assert sorted(row["conversation_id"] for row in dropped) == sorted(
        [closed.conv_id, done.conv_id]
    )


# 43: a person the account already wrote to is not a new contact (fix pass D1.2) ----------


async def _known_in_a_new_task(world, *strangers: str):
    """drill_shop was written to in a task that is finished by now; three hours later a new
    task names him again, with these strangers."""
    from tg_agent.gateway.base import PeerInfo

    first, [known] = await world.contacted("drill_shop")
    await world.repo.add_event(first, K.REMINDER, payload={"about": "итог"})
    world.script([call("finish", result="Дрель есть у @drill_shop")])
    await world.round()
    await _ops(world).acknowledge(first)  # «Завершить» (R109)
    world.clock.advance(3 * HOUR)
    task_id = await world.repo.create_task(
        "Шуруповёрт", "Купить шуруповёрт", {}, None, "UZS", ["name"], []
    )
    people_ = []
    for username in ("drill_shop", *strangers):
        if username not in world.gateway.peers:
            world._peers += 1
            world.gateway.peers[username] = PeerInfo(
                world._peers, username, username.title(), False
            )
        attached = await people.attach(world.repo, world.gateway, task_id, username)
        assert isinstance(attached, people.Attached), attached
        peer = world.gateway.peers[username].peer_id
        people_.append(type(known)(attached.conv_id, peer, username))
    await world.runner.start(task_id)
    return task_id, people_


async def test_43_a_known_person_is_written_with_the_human_pause_outside_the_quota(
    world,
):
    from tg_agent.db import FIRST_SENT

    task_id, [known, stranger] = await _known_in_a_new_task(world, "tool_shop")
    hello = send(known.conv_id, "Здравствуйте! Есть шуруповёрт?")
    world.script([hello], END)
    await world.round()

    queued_at = world.clock.now()
    [item] = await _pending(world, known)
    assert item.not_before - queued_at <= DELIVERY
    assert world.result(hello)["content"].startswith("В очереди: 1 сообщ., уйдёт ")
    await world.deliver()
    assert len(world.sent_to(known)) == 2
    # no new contact counted, and no gap started for the next stranger
    assert len(await world.repo.events_of(FIRST_SENT)) == 1

    await world.repo.add_event(task_id, K.REMINDER, payload={"about": "второй"})
    world.script([send(stranger.conv_id, "Здравствуйте! Есть шуруповёрт?")], END)
    await world.round()
    [item] = await _pending(world, stranger)
    assert item.not_before - world.clock.now() <= DELIVERY
    await world.deliver()
    assert len(world.sent_to(stranger)) == 1
    assert len(await world.repo.events_of(FIRST_SENT)) == 2
    await world.assert_clean()


async def test_43_a_known_person_goes_when_the_day_s_quota_is_full(world):
    from tg_agent.db import FIRST_SENT

    _task_id, [known, stranger] = await _known_in_a_new_task(world, "tool_shop")
    for _ in range(2):  # with drill_shop's, the day's three new people are used up
        await world.repo.log_event(FIRST_SENT, None)
    world.script(
        [
            send(stranger.conv_id, "Здравствуйте! Есть шуруповёрт?"),
            send(known.conv_id, "Здравствуйте! Есть шуруповёрт?"),
        ],
        END,
    )
    await world.round()

    [waits] = await _pending(world, stranger)
    zone = world.settings.schedule.zone
    assert waits.not_before.astimezone(zone).date() > world.clock.now().date()
    # the known person's message is not held behind the stranger's slot
    [item] = await _pending(world, known)
    assert item.not_before - world.clock.now() <= DELIVERY
    await world.deliver()
    assert len(world.sent_to(known)) == 2
    assert world.sent_to(stranger) == []
    assert len(await world.repo.events_of(FIRST_SENT)) == 3


# 44: a person who refused to talk to an AI is not written to again (fix pass D1.4) -------


async def _attach_elsewhere(world, person):
    other = await world.repo.create_task(
        "Шуруповёрт", "Купить шуруповёрт", {}, None, "UZS", [], []
    )
    return await people.attach(world.repo, world.gateway, other, person.username)


async def test_44_after_a_handoff_the_person_cannot_be_attached_to_a_new_task(world):
    _task_id, person = await world.replied("Я не буду говорить с ботом, позовите Саида")
    world.script(
        [ask("handoff", "Не хочет говорить с ИИ, просит тебя", person.conv_id)], END
    )
    await world.round(BURST)
    assert await world.state(person) is S.HANDED_TO_OWNER

    result = await _attach_elsewhere(world, person)

    assert result == people.NotAttached("просил не писать")


async def test_44_back_to_the_agent_lets_him_be_written_to_again(world):
    task_id, person = await world.replied("Я не буду говорить с ботом, позовите Саида")
    world.script(
        [ask("handoff", "Не хочет говорить с ИИ, просит тебя", person.conv_id)], END
    )
    await world.round(BURST)

    await world.press("back")

    contact = await world.repo.contact_by_peer(person.peer_id)
    assert contact is not None and contact.do_not_contact is False
    await _ops(world).close(task_id)
    assert isinstance(await _attach_elsewhere(world, person), people.Attached)


async def test_44_the_owner_taking_a_chat_himself_is_no_refusal_of_the_person(world):
    task_id, person = await world.replied("Какой цвет нужен?")
    world.script([ask("question", "Какой цвет?", person.conv_id)], END)
    await world.round(BURST)

    assert await world.press("self") == "Чат за тобой"

    contact = await world.repo.contact_by_peer(person.peer_id)
    assert contact is not None and contact.do_not_contact is False
    await _ops(world).close(task_id)
    assert isinstance(await _attach_elsewhere(world, person), people.Attached)


# 45: the owner's new instruction lifts the reminder cap for its wake (fix pass D1.6) -----


@pytest.mark.parametrize("about_him", [False, True])
async def test_45_an_instruction_lets_the_agent_write_once_to_a_person_at_the_cap(
    world, about_him
):
    task_id, [person] = await world.contacted()
    await world.repo.update_conversation(person.conv_id, followups_sent=2)
    world.clock.advance(HOUR)
    conv_id = person.conv_id if about_him else None
    await _ops(world).instruct(task_id, "Спроси ещё про доставку", conv_id=conv_id)
    ask_more = send(person.conv_id, "Подскажите, а доставка у вас есть?")
    world.script([ask_more], END)

    await world.round()

    assert world.result(ask_more).get("is_error") is not True
    assert len(await _pending(world, person)) == 1
    await world.deliver()

    # the next wake carries no instruction: the cap holds again
    await world.repo.add_event(
        task_id, K.REMINDER, conv_id=person.conv_id, payload={"about": "напомнить"}
    )
    again = send(person.conv_id, "Напомню про доставку")
    world.script([again], END)
    await world.round(HOUR)
    refusal = world.result(again)
    assert refusal["is_error"] is True and refusal["content"] == AT_CAP


# 46: a finished task taken up again sees what its people wrote meanwhile (fix pass D1.8) --


class CatchingUp(FakeGateway):
    """The fake userbot's catch-up (`replay`) hands the inbox what reached a chat after a
    message, as the real one does; a chat in `unreadable` cannot be read."""

    def __init__(self, clock) -> None:
        super().__init__(clock)
        self.inbox = None
        self.unreadable: set[int] = set()

    async def replay(self, peer_id, after_msg_id, known_out_ids):
        await super().replay(peer_id, after_msg_id, known_out_ids)
        if peer_id in self.unreadable:
            raise TgActionError(TgErrorKind.OTHER, "CHANNEL_INVALID")
        count = 0
        for message in self.chats.get(peer_id, []):
            if message.tg_msg_id > after_msg_id and not message.out:
                await self.inbox.on_incoming(
                    Incoming(
                        peer_id, message.tg_msg_id, message.text, "text", message.date
                    )
                )
                count += 1
        return count


def _catching_up(world) -> CatchingUp:
    world.gateway = CatchingUp(world.clock)
    world.build()
    world.gateway.inbox = world.inbox
    return world.gateway


async def test_46_what_a_person_wrote_after_the_result_is_new_in_the_next_wake(world):
    gateway = _catching_up(world)
    task_id, [first, second] = await world.contacted("drill_shop", "tool_shop")
    await world.repo.add_event(task_id, K.REMINDER, payload={"about": "итог"})
    world.script([call("finish", result="Дрель есть у @drill_shop")])
    await world.round()
    # finished before R109: DONE, its people still open
    await world.repo.set_task_status(task_id, TaskStatus.DONE)
    world.clock.advance(HOUR)
    # neither is stored: nobody of a finished task is heard (R104), and the take-up reads
    # the chats
    world.gateway.add_incoming(first.peer_id, "Кстати, есть ещё кейс к дрели")
    world.gateway.add_incoming(second.peer_id, "А у нас есть GSB 13 за 520 000")
    gateway.unreadable = {first.peer_id}  # one chat cannot be read, the other still is
    assert len(await world.repo.last_messages(second.conv_id, 10)) == 1

    line = await _ops(world).instruct(task_id, "Спроси у них про кейс")

    assert line == f"Задача #{task_id} снова в работе: передал агенту"
    world.script(END)
    await world.round(BURST)
    context = world.context()
    assert "А у нас есть GSB 13 за 520 000" in context
    transcript = context.split(f'<переписка id="{second.conv_id}">')[1]
    assert transcript.index("— новое —") < transcript.index("GSB 13 за 520 000")
    assert "Спроси у них про кейс" in world.trusted()
    assert "кейс к дрели" not in context


# 47: the owner's instruction reopens a person the agent finished with (ruling R72) --------


async def test_47_the_owner_s_bargain_reopens_the_first_person(world):
    task_id, [first, second] = await world.contacted("shop_a", "shop_b")
    world.clock.advance(10 * MINUTE)
    await world.write(first, "250 000, срок 4 дня")
    await world.write(second, "230 000, срок 3 дня")
    thanks = send(first.conv_id, "Спасибо!")
    done = call("close_person", person_id=first.conv_id, reason="Цена и срок получены")
    world.script([thanks], [done], END)
    await world.round(BURST)
    await world.deliver()
    assert await world.state(first) is S.CLOSED
    tried = send(first.conv_id, "А за 200 000 сделаете?")
    world.script([tried], END)
    await world.repo.add_event(task_id, K.REMINDER, payload={"about": "проверить"})
    await world.round(HOUR)
    assert world.result(tried)["content"] == (
        "С этим человеком уже закончили. Написать ему снова можно, когда об этом"
        " попросит владелец"
    )

    await _ops(world).instruct(task_id, "поторгуйся с первым до 200")
    bargain = send(first.conv_id, "А за 200 000 сделаете?")
    world.script([bargain], END)
    await world.round()
    await world.deliver()

    assert world.result(bargain).get("is_error") is not True, world.result(bargain)
    assert await world.state(first) is S.TALKING
    assert world.sent_to(first)[-1].endswith("А за 200 000 сделаете?")
    assert first.peer_id in world.gateway.allowed
