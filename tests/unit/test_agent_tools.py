"""The task agent's tools and their handler (`agent/tools.py`)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from tg_agent.agent import chats
from tg_agent.agent.approvals import Approvals
from tg_agent.agent.context import WakeContext
from tg_agent.agent.loop import ToolResult
from tg_agent.agent.outgoing import REASONS, Outgoing
from tg_agent.agent.tools import TOOLS, ToolBox, when_it_goes
from tg_agent.brain.llm import FakeLLM
from tg_agent.clock import FakeClock
from tg_agent.config import Settings
from tg_agent.control.base import FakeUI
from tg_agent.db import Repo
from tg_agent.gateway.fake import FakeGateway
from tg_agent.models import (
    ApprovalKind,
    ApprovalStatus,
    ConvState,
    Direction,
    EventKind,
    MsgSource,
    OutboxStatus,
    Task,
    TaskStatus,
)

S = ConvState
TASHKENT = ZoneInfo("Asia/Tashkent")
NAMES = [
    "send_message",
    "ask_owner",
    "update_notes",
    "finish",
    "close_person",
    "remind_me",
    "research",
    "add_person",
]
HINT = "Хватит попыток: спроси владельца или оставь до следующего раза."
LIMIT_WORDS = {"maxItems", "minItems", "maxLength", "minLength", "minimum", "maximum"}


@dataclass
class World:
    repo: Repo
    ui: FakeUI
    gateway: FakeGateway
    clock: FakeClock
    settings: Settings
    outgoing: Outgoing
    approvals: Approvals
    task: Task


@pytest.fixture
async def world(repo: Repo, fake_clock: FakeClock, settings: Settings) -> World:
    gateway = FakeGateway()
    gateway.name = "ИИ помощник Саида"  # no introduction: the texts go out as written
    llm = FakeLLM()
    outgoing = Outgoing(
        repo,
        gateway,
        llm,
        fake_clock,
        settings,
        random.Random(3),
        canary="ZX-CANARY-7731",
        owner_secrets=[],
    )
    ui = FakeUI()
    approvals = Approvals(repo, ui, fake_clock, settings, outgoing)
    task_id = await repo.create_task(
        "Футболки", "Напечатать 100 футболок", {}, None, "UZS", ["name"], []
    )
    await repo.set_task_status(task_id, TaskStatus.ACTIVE)
    return World(
        repo,
        ui,
        gateway,
        fake_clock,
        settings,
        outgoing,
        approvals,
        await repo.get_task(task_id),
    )


async def person(
    world: World,
    *path: ConvState,
    peer_id: int = 501,
    username: str = "print_lab",
    task_id: int | None = None,
) -> int:
    repo = world.repo
    contact_id = await repo.upsert_contact(peer_id, username, username, False)
    conv_id = await repo.create_conversation(
        world.task.id if task_id is None else task_id, contact_id, state=S.PLANNED
    )
    for state in path:
        await repo.set_state(conv_id, state)
    return conv_id


async def talking(world: World, **kw: Any) -> int:
    """A person who answered our first message."""
    conv_id = await person(world, S.CONTACTED, S.TALKING, **kw)
    now = world.clock.now()
    await world.repo.update_conversation(conv_id, first_sent=True, last_out_at=now)
    await world.repo.add_message(
        conv_id, Direction.OUT, MsgSource.AGENT, "text", "Здравствуйте!", None, now
    )
    world.clock.advance(timedelta(minutes=1))
    now = world.clock.now()
    await world.repo.add_message(
        conv_id, Direction.IN, MsgSource.COUNTERPART, "text", "Да, слушаю", None, now
    )
    await world.repo.update_conversation(conv_id, last_in_at=now)
    world.clock.advance(timedelta(minutes=1))
    return conv_id


async def new_wake(world: World) -> int:
    """A real `wakes` row (the outbox names its wake); its one event is handled at once."""
    await world.repo.add_event(world.task.id, EventKind.CREATED)
    wake_id, _ = await world.repo.claim_events(world.task.id, world.clock.now())
    await world.repo.finish_wake(wake_id, "test", 0)
    return wake_id


async def toolbox(world: World, wake_id: int | None = None) -> ToolBox:
    if wake_id is None:
        wake_id = await new_wake(world)
    context = WakeContext(
        system=("правила", "задача"),
        content=[],
        seen_message_id=await world.repo.max_message_id(world.task.id),
        instructions=(),
    )
    return ToolBox(
        world.repo,
        world.ui,
        world.gateway,
        world.clock,
        world.settings,
        world.outgoing,
        world.approvals,
        task=world.task,
        wake_id=wake_id,
        context=context,
    )


def send(person_id: Any, *messages: str, **changes: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "person_id": person_id,
        "messages": list(messages),
        "language": "ru",
        "attach_task_files": False,
        "commits_owner": False,
    }
    return args | changes


def ask(person_id: int | None, **changes: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "kind": "question",
        "question": "Какой цвет футболок?",
        "person_id": person_id,
        "suggested_reply": ["Белые"],
        "amount": None,
        "terms": "",
    }
    return args | changes


def notes(**changes: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "notes": "План: спросить цену",
        "headline": "Пишем двум типографиям",
        "person_notes": [],
    }
    return args | changes


def remind(**changes: Any) -> dict[str, Any]:
    args: dict[str, Any] = {"hours": 24, "about": "проверить заказ", "person_id": None}
    return args | changes


async def assert_nothing_written(world: World) -> None:
    task = await world.repo.get_task(world.task.id)
    assert task.status is TaskStatus.ACTIVE
    assert (task.notes, task.headline) == ("", "")
    assert await world.repo.open_events(world.task.id) == []
    for conv in await world.repo.task_conversations(world.task.id):
        assert await world.repo.outbox_of(conv.id) == []
        assert conv.note == ""
    assert world.ui.sent == []


# 1. The definitions


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _keys(v)}
    return set()


def _objects(schema: Any) -> list[dict[str, Any]]:
    """Every object schema inside `schema`, itself included."""
    found = []
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            found.append(schema)
        for value in schema.values():
            found += _objects(value)
    elif isinstance(schema, list):
        for value in schema:
            found += _objects(value)
    return found


def test_strict_tools_with_every_property_required() -> None:
    assert [tool["name"] for tool in TOOLS] == NAMES
    for tool in TOOLS:
        assert tool["strict"] is True
        assert tool["description"]
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        for obj in _objects(schema):
            assert obj["additionalProperties"] is False
            assert obj["required"] == list(obj["properties"])
        assert not LIMIT_WORDS & _keys(schema)
    json.dumps(TOOLS)  # plain data the API can take


def test_the_fields_and_types_of_the_tools() -> None:
    props = {t["name"]: t["input_schema"]["properties"] for t in TOOLS}
    assert list(props["send_message"]) == [
        "person_id",
        "messages",
        "language",
        "attach_task_files",
        "commits_owner",
    ]
    assert props["send_message"]["language"]["enum"] == ["ru", "uz_latn", "other"]
    assert props["send_message"]["messages"]["items"] == {"type": "string"}
    assert props["ask_owner"]["kind"]["enum"] == [
        "question",
        "approval",
        "phone",
        "address",
        "handoff",
    ]
    assert props["ask_owner"]["person_id"]["type"] == ["integer", "null"]
    assert props["ask_owner"]["amount"]["type"] == ["number", "null"]
    assert props["remind_me"]["person_id"]["type"] == ["integer", "null"]
    assert props["remind_me"]["hours"]["type"] == "number"
    assert list(props["update_notes"]["person_notes"]["items"]["properties"]) == [
        "person_id",
        "note",
    ]
    assert list(props["finish"]) == ["result"]
    assert list(props["close_person"]) == ["person_id", "reason"]
    descriptions = {t["name"]: t["description"] for t in TOOLS}
    assert descriptions["send_message"].startswith(
        "Написать человеку из задачи в Telegram."
    )
    assert descriptions["send_message"].endswith(
        "Одному человеку — один вызов за пробуждение."
    )
    assert descriptions["remind_me"].endswith(
        "если человек не ответит сутки, тебя разбудят."
    )
    # Fix pass D1.5: the model is told the program introduces it.
    assert (
        "В первое сообщение человеку программа, когда это нужно, сама добавит после"
        " приветствия «Я ИИ-помощник …»: сам не представляйся."
    ) in descriptions["send_message"]


# 2. Happy paths


async def test_send_message_queues_and_says_when(world: World) -> None:
    conv_id = await talking(world)
    tools = await toolbox(world)

    result = await tools("send_message", send(conv_id, "Сколько стоит?", "И срок?"))

    items = await world.repo.outbox_of(conv_id)
    assert [item.text for item in items] == ["Сколько стоит?", "И срок?"]
    assert all(item.wake_id == tools._wake_id for item in items)
    when = items[0].not_before.astimezone(world.settings.schedule.zone)
    assert result == ToolResult(f"В очереди: 2 сообщ., уйдёт около {when:%H:%M}.")


# Task 3.1: a first message to a new person has a paced slot, and the answer names it.
PACED = (
    " Новым людям программа пишет не чаще чем раз в 1–2 минуты и в пределах дневной"
    " квоты."
)


async def test_a_first_contact_says_it_goes_today_at_its_slot(world: World) -> None:
    conv_id = await person(world)
    tools = await toolbox(world)

    result = await tools(
        "send_message", send(conv_id, "Здравствуйте! Печатаете футболки?")
    )

    [item] = await world.repo.outbox_of(conv_id)
    when = item.not_before.astimezone(TASHKENT)
    assert result == ToolResult(
        "В очереди: 1 сообщ. Это первое сообщение этому человеку: уйдёт сегодня около"
        f" {when:%H:%M}." + PACED
    )
    assert (await world.repo.get_conversation(conv_id)).state is S.CONTACTED


async def test_a_first_contact_at_night_says_tomorrow(world: World) -> None:
    world.clock.set(datetime(2026, 11, 10, 22, 0, tzinfo=TASHKENT))
    conv_id = await person(world)
    tools = await toolbox(world)

    result = await tools("send_message", send(conv_id, "Здравствуйте!"))

    assert result == ToolResult(
        "В очереди: 1 сообщ. Это первое сообщение этому человеку: уйдёт завтра около"
        " 09:00." + PACED
    )


async def test_a_first_contact_past_tomorrows_quota_names_the_date(
    world: World,
) -> None:
    world.clock.set(datetime(2026, 11, 10, 22, 0, tzinfo=TASHKENT))
    for peer, minute in ((601, 0), (602, 15), (603, 30)):  # tomorrow's quota of 3
        other = await person(world, peer_id=peer, username=f"shop_{peer}")
        morning = datetime(2026, 11, 11, 9, minute, tzinfo=TASHKENT)
        await world.repo.enqueue(other, "Здравствуйте!", morning)
    conv_id = await person(world)
    tools = await toolbox(world)

    result = await tools("send_message", send(conv_id, "Здравствуйте!"))

    assert result == ToolResult(
        "В очереди: 1 сообщ. Это первое сообщение этому человеку: уйдёт 12.11 около"
        " 09:00." + PACED
    )


# Fix pass B12: the answer names the day when the message does not go today.


async def test_a_reply_that_goes_tomorrow_says_so(world: World) -> None:
    world.clock.set(datetime(2026, 11, 10, 21, 30, tzinfo=TASHKENT))
    conv_id = await talking(world)
    tools = await toolbox(world)

    result = await tools("send_message", send(conv_id, "Уточню и вернусь"))

    assert result == ToolResult("В очереди: 1 сообщ., уйдёт завтра около 09:00.")


def test_the_day_is_named_unless_it_is_today() -> None:
    now = datetime(2026, 11, 10, 21, 30, tzinfo=TASHKENT)
    today = datetime(2026, 11, 10, 21, 32, tzinfo=TASHKENT)
    tomorrow = datetime(2026, 11, 11, 9, 0, tzinfo=TASHKENT)
    later = datetime(2026, 11, 12, 9, 0, tzinfo=TASHKENT)
    assert when_it_goes(today, now, TASHKENT) == "около 21:32"
    assert when_it_goes(tomorrow, now, TASHKENT) == "завтра около 09:00"
    assert when_it_goes(later, now, TASHKENT) == "12.11 около 09:00"
    # the day is the owner's: 23:30 in Tashkent is still the 10th in UTC
    assert when_it_goes(later.astimezone(UTC), now, TASHKENT) == "12.11 около 09:00"


def test_no_tool_promises_a_daily_quota() -> None:
    assert not [t["name"] for t in TOOLS if "квот" in t["description"]]


async def test_ask_owner_sends_a_card(world: World) -> None:
    conv_id = await talking(world)
    tools = await toolbox(world)

    result = await tools("ask_owner", ask(conv_id))

    assert result == ToolResult(
        "Карточка отправлена владельцу. Этому человеку больше ничего не пиши, пока"
        " владелец не ответит."
    )
    [approval] = await world.repo.pending_approvals(conv_id)
    assert approval.kind is ApprovalKind.QUESTION
    assert approval.payload["suggested"] == ["Белые"]
    assert len(world.ui.sent) == 1


async def test_ask_owner_passes_an_error_on(world: World) -> None:
    tools = await toolbox(world)
    result = await tools("ask_owner", ask(None, kind="phone"))
    assert result == ToolResult("укажи человека: этот вопрос о собеседнике", True)


async def test_update_notes_writes_the_notes_and_the_status(world: World) -> None:
    conv_id = await talking(world)
    tools = await toolbox(world)

    result = await tools(
        "update_notes",
        notes(person_notes=[{"person_id": conv_id, "note": "спросили цену"}]),
    )

    assert result == ToolResult("Записано.")
    task = await world.repo.get_task(world.task.id)
    assert (task.notes, task.headline) == (
        "План: спросить цену",
        "Пишем двум типографиям",
    )
    assert (await world.repo.get_conversation(conv_id)).note == "спросили цену"


async def test_close_person(world: World) -> None:
    conv_id = await talking(world)
    tools = await toolbox(world)

    result = await tools("close_person", {"person_id": conv_id, "reason": "дорого"})

    assert result == ToolResult(f"С [{conv_id}] закончили.")
    conv = await world.repo.get_conversation(conv_id)
    assert (conv.state, conv.close_reason) == (S.CLOSED, "дорого")
    again = await tools("close_person", {"person_id": conv_id, "reason": "дорого"})
    assert again.is_error


async def test_remind_me_adds_a_reminder(world: World) -> None:
    conv_id = await talking(world)
    tools = await toolbox(world)

    result = await tools("remind_me", remind(person_id=conv_id))

    assert result == ToolResult("Разбужу через 24 ч.")
    [event] = await world.repo.open_events(world.task.id)
    assert event.kind is EventKind.REMINDER
    assert event.conversation_id == conv_id
    assert event.payload == {"about": "проверить заказ"}
    assert event.due_at == world.clock.now() + timedelta(hours=24)
    half = await tools("remind_me", remind(hours=1.5))
    assert half == ToolResult("Разбужу через 1,5 ч.")


# 3. Input errors


async def test_wrong_inputs_are_errors_and_write_nothing(world: World) -> None:
    conv_id = await talking(world)
    other_task = await world.repo.create_task(
        "Другая", "Другое", {}, None, "UZS", [], []
    )
    stranger = await person(world, peer_id=777, username="stranger", task_id=other_task)
    tools = await toolbox(world)

    calls = [
        ("send_message", send(999, "Привет")),
        ("send_message", send(stranger, "Привет")),
        ("send_message", send(conv_id)),
        ("send_message", send(conv_id, "1", "2", "3", "4")),
        ("send_message", send(conv_id, "д" * 401)),
        ("send_message", send(conv_id, "Привет", language="en")),
        ("send_message", send(True, "Привет")),
        ("ask_owner", ask(999)),
        (
            "update_notes",
            notes(person_notes=[{"person_id": conv_id, "note": "н" * 151}]),
        ),
        ("update_notes", notes(notes="н" * 2001)),
        ("update_notes", notes(headline="н" * 201)),
        ("update_notes", notes(person_notes=[{"person_id": stranger, "note": "x"}])),
        ("finish", {"result": "   "}),
        ("finish", {"result": "и" * 1501}),
        ("close_person", {"person_id": 999, "reason": "дорого"}),
        ("close_person", {"person_id": conv_id, "reason": " "}),
        ("close_person", {"person_id": conv_id, "reason": "п" * 151}),
        ("remind_me", remind(hours=0)),
        ("remind_me", remind(hours=337)),
        ("remind_me", remind(hours="24")),
        ("remind_me", remind(about="")),
        ("remind_me", remind(person_id=stranger)),
    ]
    for name, args in calls:
        result = await tools(name, args)
        assert result.is_error, (name, args, result)
        assert result.text, (name, args)
        assert not result.stop

    assert (await world.repo.get_conversation(conv_id)).state is S.TALKING
    await assert_nothing_written(world)


async def test_the_errors_say_what_to_fix(world: World) -> None:
    conv_id = await talking(world)
    tools = await toolbox(world)
    assert (await tools("send_message", send(999, "x"))).text == (
        "в этой задаче нет человека [999] — возьми номер из списка «Люди»"
    )
    assert "от 1 до 3" in (await tools("send_message", send(conv_id))).text
    assert "400" in (await tools("send_message", send(conv_id, "д" * 401))).text
    assert (
        "150"
        in (
            await tools(
                "update_notes",
                notes(person_notes=[{"person_id": conv_id, "note": "н" * 151}]),
            )
        ).text
    )
    assert "от 1 до 336" in (await tools("remind_me", remind(hours=0))).text
    assert "итог" in (await tools("finish", {"result": ""})).text


# 4. send_message refusals


async def test_a_refusal_comes_back_as_an_error_with_its_reason(world: World) -> None:
    conv_id = await talking(world)
    tools = await toolbox(world)

    result = await tools("send_message", send(conv_id, "Вот сайт https://shop.uz"))

    assert result == ToolResult(REASONS["link"], True)
    assert await world.repo.outbox_of(conv_id) == []


async def test_one_queued_send_per_person_per_wake(world: World) -> None:
    conv_id = await talking(world)
    tools = await toolbox(world)

    assert not (await tools("send_message", send(conv_id, "Сколько стоит?"))).is_error
    second = await tools("send_message", send(conv_id, "И срок?"))

    assert second == ToolResult("этому человеку в этом пробуждении уже написал", True)
    items = await world.repo.outbox_of(conv_id)
    assert [i.text for i in items if i.status is OutboxStatus.PENDING] == [
        "Сколько стоит?"
    ]
    # Another wake may write again.
    later = await toolbox(world)
    assert not (await later("send_message", send(conv_id, "И срок?"))).is_error


async def test_the_third_refusal_adds_the_hint(world: World) -> None:
    conv_id = await talking(world)
    other = await talking(world, peer_id=502, username="other")
    tools = await toolbox(world)
    link = send(conv_id, "Вот сайт https://shop.uz")

    first = await tools("send_message", link)
    await tools("send_message", send(other, "Вот сайт https://shop.uz"))
    second = await tools("send_message", link)
    third = await tools("send_message", link)

    assert HINT not in first.text and HINT not in second.text
    assert third.is_error and third.text == f"{REASONS['link']} {HINT}"


# 5. finish


async def test_finish_sends_one_result_card_and_leaves_the_task_in_work(
    world: World,
) -> None:
    conv_id = await talking(world)
    # (fix pass B5: a card about a person refuses `finish`; a task-level question expires)
    approval = await world.repo.create_approval(
        world.task.id, None, ApprovalKind.QUESTION, {"question": "Искать ещё?"}
    )
    await world.repo.add_event(
        world.task.id,
        EventKind.REMINDER,
        payload={"about": "позже"},
        due_at=world.clock.now() + timedelta(hours=5),
    )
    tools = await toolbox(world)

    result = await tools("finish", {"result": "PrintLab — 240 000, 3 дня"})

    assert result == ToolResult(
        "Итог отправлен владельцу. Задача открыта, пока он её не завершит: ответы людей"
        " придут тебе, как обычно.",
        stop=True,
    )
    # Only the owner ends a task (R109): it stays in work, and the reminder still wakes.
    task = await world.repo.get_task(world.task.id)
    assert task.status is TaskStatus.ACTIVE
    assert (await world.repo.get_approval(approval.id)).status is ApprovalStatus.EXPIRED
    assert [e.kind for e in await world.repo.open_events(world.task.id)] == [
        EventKind.REMINDER
    ]
    [card] = world.ui.sent
    title = "Готово — Футболки"
    # Stage 5 (R54, R56): the result carries its marker and waits for «Завершить».
    assert card.text == f"✅ {title}\nPrintLab — 240 000, 3 дня"
    assert card.bold == [(2, 2 + len(title))]
    assert card.buttons == [[("Завершить", f"fn:{world.task.id}:ack:-")]]
    assert (card.kind, card.task_id) == ("result", world.task.id)
    [msg_id] = world.ui.sent_ids
    assert await world.repo.get_kv(f"task_card:{msg_id}") == str(world.task.id)
    # Kept, so that the card can be drawn again; it waits for the owner.
    task_id = world.task.id
    assert await world.repo.get_kv(f"task_result:{task_id}") == "PrintLab — 240 000, 3 дня"
    assert await world.repo.get_kv(f"task_result_waits:{task_id}") == "1"
    # The people keep their states: a later instruction can reopen the task where it stood.
    assert (await world.repo.get_conversation(conv_id)).state is S.TALKING

    later = await tools("update_notes", notes())
    assert later == ToolResult("итог уже отправлен: закончи это пробуждение", True)
    assert len(world.ui.sent) == 1


# Fix round 1: terms are bounded. Fix pass B5: finish waits for a card about a person and
# keeps the hand-back card.


async def test_finish_waits_for_a_card_about_a_person_and_keeps_the_hand_back_card(
    world: World,
) -> None:
    held = await talking(world)
    handed = await talking(world, peer_id=502, username="handed")
    await world.repo.set_state(handed, S.HANDED_TO_OWNER)
    handoff = await world.repo.create_approval(
        world.task.id, handed, ApprovalKind.HANDOFF, {"text": "Чат у тебя"}
    )
    other_task = await world.repo.create_task(
        "Другая", "Другое", {}, None, "UZS", [], []
    )
    stranger = await person(world, peer_id=777, username="stranger", task_id=other_task)
    foreign = await world.repo.create_approval(
        other_task, stranger, ApprovalKind.QUESTION, {"question": "?"}
    )
    tools = await toolbox(world)
    assert not (await tools("ask_owner", ask(held))).is_error
    assert (await world.repo.get_conversation(held)).state is S.WAITING_OWNER
    [card] = await world.repo.pending_approvals(held)

    refused = await tools("finish", {"result": "Итог"})

    assert refused == ToolResult(
        f"сначала дождись ответа владельца по карточке о [{held}]"
        " или закончи с этим человеком",
        True,
    )
    assert (await world.repo.get_task(world.task.id)).status is TaskStatus.ACTIVE
    assert (await world.repo.get_approval(card.id)).status is ApprovalStatus.PENDING
    assert len(world.ui.sent) == 1  # the question card, no result
    await world.repo.resolve_approval(card.id, card.nonce, ApprovalStatus.ANSWERED)
    await world.repo.set_state(held, S.TALKING)

    result = await tools("finish", {"result": "Итог"})

    assert result.stop and not result.is_error
    assert (await world.repo.get_approval(handoff.id)).status is ApprovalStatus.PENDING
    assert (await world.repo.get_conversation(handed)).state is S.HANDED_TO_OWNER
    assert (await world.repo.get_approval(foreign.id)).status is ApprovalStatus.PENDING


class FailingUI(FakeUI):
    """Shows approval cards, but the result card does not reach the owner."""

    async def send(self, card: Any) -> int:
        if card.kind == "result":
            raise ConnectionError("telegram is down")
        return await super().send(card)


async def test_finish_whose_card_fails_changes_nothing(world: World) -> None:
    held = await talking(world)
    world.ui = FailingUI()
    world.approvals = Approvals(
        world.repo, world.ui, world.clock, world.settings, world.outgoing
    )
    tools = await toolbox(world)
    # (fix pass B5: a task-level question, since a card about a person refuses `finish`)
    assert not (await tools("ask_owner", ask(None))).is_error
    await world.repo.add_event(
        world.task.id,
        EventKind.REMINDER,
        payload={"about": "позже"},
        due_at=world.clock.now() + timedelta(hours=5),
    )

    result = await tools("finish", {"result": "Итог"})

    assert result.is_error and not result.stop
    assert "ConnectionError" in result.text
    assert (await world.repo.get_task(world.task.id)).status is TaskStatus.ACTIVE
    [card] = await world.repo.pending_approvals()
    assert card.status is ApprovalStatus.PENDING
    assert (await world.repo.get_conversation(held)).state is S.TALKING
    assert len(await world.repo.open_events(world.task.id)) == 1
    assert len(world.ui.sent) == 1  # only the question card
    assert await world.repo.get_kv("task_card:2") is None
    # No result waits for the owner: none reached him.
    assert not await world.repo.get_kv(f"task_result_waits:{world.task.id}")
    # The tool is still usable: the task is not finished.
    assert not (await tools("update_notes", notes())).is_error


async def test_finish_whose_card_fails_keeps_an_earlier_result_waiting(
    world: World,
) -> None:
    await talking(world)
    task_id = world.task.id
    # An earlier result already waits for «Завершить».
    await world.repo.set_kv(f"task_result:{task_id}", "Старый итог")
    await world.repo.set_kv(f"task_result_waits:{task_id}", "1")
    await world.repo.set_kv(f"task_result_wake:{task_id}", "7")
    world.ui = FailingUI()
    world.approvals = Approvals(
        world.repo, world.ui, world.clock, world.settings, world.outgoing
    )
    tools = await toolbox(world)

    result = await tools("finish", {"result": "Новый итог"})

    assert result.is_error and "ConnectionError" in result.text
    assert await world.repo.get_kv(f"task_result_waits:{task_id}") == "1"
    assert await world.repo.get_kv(f"task_result:{task_id}") == "Старый итог"
    assert await world.repo.get_kv(f"task_result_wake:{task_id}") == "7"


class ClosingUI(FakeUI):
    """The owner closes the task while the result card is on its way to him."""

    def __init__(self, repo: Repo, task_id: int) -> None:
        super().__init__()
        self.repo, self.task_id = repo, task_id

    async def send(self, card: Any) -> int:
        if card.kind == "result":
            await self.repo.set_task_status(self.task_id, TaskStatus.CANCELLED)
        return await super().send(card)


async def test_finish_of_a_task_closed_meanwhile_changes_nothing(world: World) -> None:
    world.ui = ClosingUI(world.repo, world.task.id)
    question = await world.repo.create_approval(
        world.task.id, None, ApprovalKind.QUESTION, {"question": "Брать?"}
    )
    tools = await toolbox(world)

    result = await tools("finish", {"result": "Итог"})

    assert result == ToolResult("задача остановлена владельцем", True, stop=True)
    assert tools.halted
    assert (await world.repo.get_task(world.task.id)).status is TaskStatus.CANCELLED
    assert (await world.repo.get_approval(question.id)).status is ApprovalStatus.PENDING
    [msg_id] = world.ui.sent_ids
    assert await world.repo.get_kv(f"task_card:{msg_id}") is None


async def test_terms_longer_than_300_characters_are_refused(world: World) -> None:
    held = await talking(world)
    tools = await toolbox(world)

    result = await tools(
        "ask_owner",
        ask(
            held,
            kind="approval",
            question="Подтвердить заказ?",
            amount=240_000.0,
            terms="т" * 301,
        ),
    )

    assert result.is_error and "300" in result.text
    assert await world.repo.pending_approvals(held) == []
    ok = await tools(
        "ask_owner",
        ask(
            held,
            kind="approval",
            question="Подтвердить заказ?",
            amount=240_000.0,
            terms="т" * 300,
        ),
    )
    assert not ok.is_error


# -- A11: /stop and a task stopped during a wake --


@pytest.mark.parametrize("how", ["stopped", "paused", "cancelled"])
async def test_no_tool_runs_after_a_stop_or_once_the_task_is_not_active(
    world: World, how: str
) -> None:
    conv_id = await talking(world)
    tools = await toolbox(world)
    assert not (await tools("update_notes", notes())).is_error  # the first call ran
    if how == "stopped":
        await world.repo.set_stopped(True)
    else:
        await world.repo.set_task_status(world.task.id, TaskStatus(how))

    for name, args in (
        ("send_message", send(conv_id, "Сколько стоит?")),
        ("ask_owner", ask(conv_id)),
        ("remind_me", remind()),
        ("finish", {"result": "Готово"}),
    ):
        result = await tools(name, args)
        assert (result.text, result.is_error, result.stop) == (
            "работа остановлена",
            True,
            True,
        )
    assert await world.repo.outbox_of(conv_id) == []
    assert await world.repo.pending_approvals() == []
    assert world.ui.sent == []
    assert await world.repo.open_events(world.task.id) == []


# Fix pass B14: an empty headline keeps the old one.


async def test_update_notes_with_an_empty_headline_keeps_the_old_one(
    world: World,
) -> None:
    tools = await toolbox(world)
    await tools("update_notes", notes())

    result = await tools("update_notes", notes(notes="Новый план", headline="  "))

    assert result == ToolResult("Записано.")
    task = await world.repo.get_task(world.task.id)
    assert (task.notes, task.headline) == ("Новый план", "Пишем двум типографиям")


# Fix pass B17: reminders have a ceiling.


async def test_remind_me_is_refused_once_ten_reminders_wait(world: World) -> None:
    tools = await toolbox(world)
    for n in range(10):
        assert not (await tools("remind_me", remind(about=f"дело {n}"))).is_error

    result = await tools("remind_me", remind(about="ещё одно"))

    assert result == ToolResult("слишком много напоминаний: дождись прежних", True)
    reminders = [
        e
        for e in await world.repo.open_events(world.task.id)
        if e.kind is EventKind.REMINDER
    ]
    assert len(reminders) == 10


@pytest.mark.parametrize(
    "kind",
    [
        EventKind.OWNER_ANSWER,
        EventKind.OWNER_INSTRUCTION,
        EventKind.RESUMED,  # fix pass C9: he gave a chat back to the agent
    ],
)
async def test_finish_waits_for_the_owners_words_this_wake_has_not_seen(
    world: World, kind: EventKind
) -> None:
    """He pressed «Да» (or gave an instruction) while the model was writing `finish`: the
    card is no longer pending, and finishing now would drop his words unread."""
    conv_id = await talking(world)
    tools = await toolbox(world)
    await world.repo.add_event(
        world.task.id, kind, conv_id=conv_id, payload={"text": "Да"}
    )

    refused = await tools("finish", {"result": "Итог"})

    assert refused.is_error and not refused.stop
    assert "владелец" in refused.text
    assert (await world.repo.get_task(world.task.id)).status is TaskStatus.ACTIVE
    assert [e.kind for e in await world.repo.open_events(world.task.id)] == [kind]
    assert world.ui.sent == []


async def test_finish_is_not_held_by_the_owners_words_of_its_own_wake(
    world: World,
) -> None:
    conv_id = await talking(world)
    await world.repo.add_event(
        world.task.id,
        EventKind.OWNER_ANSWER,
        conv_id=conv_id,
        payload={"text": "Да"},
    )
    wake_id, _ = await world.repo.claim_events(world.task.id, world.clock.now())
    tools = await toolbox(world, wake_id)

    done = await tools("finish", {"result": "Итог"})

    assert done.stop and not done.is_error
    assert await world.repo.get_kv(f"task_result_waits:{world.task.id}") == "1"


class InstructingUI(FakeUI):
    """The owner gives an instruction while the result card is on its way to him."""

    def __init__(self, repo: Repo, task_id: int) -> None:
        super().__init__()
        self.repo, self.task_id = repo, task_id

    async def send(self, card: Any, *, silent: bool = False) -> int:
        if card.kind == "result":
            await self.repo.add_event(
                self.task_id, EventKind.OWNER_INSTRUCTION, payload={"text": "ещё"}
            )
        return await super().send(card, silent=silent)


async def test_a_result_taken_up_at_once_by_the_owners_words_does_not_wait(
    world: World,
) -> None:
    # Stage 5 (R56): the task goes back to work for his words, so «Завершить» has nothing
    # to finish: the result card stops waiting.
    await talking(world)
    world.ui = InstructingUI(world.repo, world.task.id)
    tools = await toolbox(world)

    done = await tools("finish", {"result": "Итог"})

    assert done.stop and not done.is_error
    assert (await world.repo.get_task(world.task.id)).status is TaskStatus.ACTIVE
    assert await world.repo.get_kv(f"task_result:{world.task.id}") == "Итог"
    assert not await world.repo.get_kv(f"task_result_waits:{world.task.id}")


async def muted_by_the_program(world: World, conv_id: int) -> int:
    """His chat existed and the program muted it at the first message (R105)."""
    contact_id = (await world.repo.get_conversation(conv_id)).contact_id
    peer_id = (await world.repo.get_contact(contact_id)).peer_id
    world.gateway.muted[peer_id] = True
    await world.repo.set_kv(chats.muted_key(contact_id), "1")
    return peer_id


async def test_a_result_keeps_the_chats_muted(world: World) -> None:
    # R109: the result is no end; the sound comes back when the owner ends the task.
    conv_id = await talking(world)
    peer_id = await muted_by_the_program(world, conv_id)
    tools = await toolbox(world)

    await tools("finish", {"result": "PrintLab — 240 000, 3 дня"})

    assert world.gateway.muted[peer_id] is True
    assert world.gateway.mute_calls == []


def test_close_person_says_that_the_goodbye_belongs_to_the_same_wake() -> None:
    # What earlier wakes queued for him is cancelled by the close (fix D1.1).
    (tool,) = [tool for tool in TOOLS if tool["name"] == "close_person"]
    assert "в этом же пробуждении" in tool["description"]
    assert "отменяются" in tool["description"]
