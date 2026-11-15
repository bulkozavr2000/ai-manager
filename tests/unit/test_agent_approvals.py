"""Cards to the owner and his decisions in the agent core (`agent/approvals.py`)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pytest

from tg_agent.agent.approvals import OK_ITEMS, Approvals
from tg_agent.agent.outgoing import Outgoing, Queued, Refused
from tg_agent.brain.llm import FakeLLM
from tg_agent.brain.schemas import JudgeVerdict
from tg_agent.clock import FakeClock
from tg_agent.config import Settings
from tg_agent.control.base import Card, FakeUI
from tg_agent.db import Repo
from tg_agent.gateway.fake import FakeGateway
from tg_agent.models import (
    Approval,
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
from tg_agent.tasks.texts import HANDOFF_SELF

S = ConvState
K = ApprovalKind
SENT_TO_OWNER = (
    "Карточка отправлена владельцу. Этому человеку больше ничего не пиши, пока"
    " владелец не ответит."
)
ORDER_BUTTONS = [[["Да", "yes"], ["Нет", "no"]], [["Напишу сам", "self"]]]


@dataclass
class World:
    repo: Repo
    ui: FakeUI
    clock: FakeClock
    llm: FakeLLM
    approvals: Approvals
    task: Task
    outgoing: Outgoing


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
    task_id = await repo.create_task(
        "Дрель", "купить дрель до 500 000 сум", {}, 500_000.0, "UZS", ["name"], []
    )
    await repo.set_task_status(task_id, TaskStatus.ACTIVE)
    approvals = Approvals(repo, ui, fake_clock, settings, outgoing)
    return World(
        repo, ui, fake_clock, llm, approvals, await repo.get_task(task_id), outgoing
    )


async def person(
    world: World,
    *path: ConvState,
    peer_id: int = 501,
    username: str = "ivan_tools",
    task_id: int | None = None,
) -> int:
    repo = world.repo
    contact_id = await repo.upsert_contact(peer_id, username, username.title(), False)
    conv_id = await repo.create_conversation(
        world.task.id if task_id is None else task_id, contact_id, state=S.PLANNED
    )
    for state in path:
        await repo.set_state(conv_id, state)
    return conv_id


async def say(
    world: World, conv_id: int, text: str, direction: Direction = Direction.IN
) -> None:
    now = world.clock.now()
    source = MsgSource.COUNTERPART if direction is Direction.IN else MsgSource.AGENT
    await world.repo.add_message(conv_id, direction, source, "text", text, None, now)
    field = "last_in_at" if direction is Direction.IN else "last_out_at"
    await world.repo.update_conversation(conv_id, **{field: now})
    world.clock.advance(timedelta(minutes=1))


async def talking(world: World, **kw: Any) -> int:
    """A person who answered our message and wrote twice."""
    conv_id = await person(world, S.CONTACTED, S.TALKING, **kw)
    await world.repo.update_conversation(conv_id, first_sent=True)
    await say(world, conv_id, "Здравствуйте, есть дрель?", Direction.OUT)
    await say(world, conv_id, "Есть, 450 000")
    await say(world, conv_id, "Берёте?")
    return conv_id


async def ask(world: World, conv_id: int | None, **kw: Any) -> tuple[str, bool]:
    arguments: dict[str, Any] = {
        "kind": "question",
        "question": "Брать за 450 000?",
        "conv_id": conv_id,
        "suggested": [],
        "amount": None,
        "terms": "",
    } | kw
    return await world.approvals.ask(world.task, **arguments)


async def the_approval(world: World) -> Approval:
    [approval] = await world.repo.pending_approvals()
    return approval


async def events(world: World) -> list[tuple[EventKind, int | None, dict]]:
    found = await world.repo.open_events(world.task.id)
    for event in found:
        assert event.due_at == world.clock.now()
    return [(event.kind, event.conversation_id, event.payload) for event in found]


async def decide(
    world: World, approval: Approval, choice: str, text: str | None = None
) -> str:
    return await world.approvals.owner_decision(
        approval.id, approval.nonce, choice, text
    )


# -- ask --


async def test_a_question_about_a_person(world: World) -> None:
    conv_id = await talking(world)
    pending = await world.repo.enqueue(conv_id, "Подождите", world.clock.now())

    result = await ask(world, conv_id, suggested=["Беру, когда доставите?"])

    assert result == (SENT_TO_OWNER, False)
    approval = await the_approval(world)
    assert (approval.kind, approval.conversation_id, approval.task_id) == (
        K.QUESTION,
        conv_id,
        world.task.id,
    )
    assert approval.payload == {
        "task": "Дрель",
        "about": "",
        "question": "Брать за 450 000?",
        "received": ["Есть, 450 000", "Берёте?"],
        "suggested": ["Беру, когда доставите?"],
        "seen_message_id": await world.repo.max_incoming_id(conv_id),
    }
    [card] = world.ui.sent
    assert card.approval_id == approval.id
    assert "👤 @ivan_tools" in card.text and "Берёте?" in card.text
    assert approval.control_msg_id == world.ui.sent_ids[0]
    assert await world.repo.events_of("approval", conv_id) == [
        {"approval_id": approval.id, "kind": "question"}
    ]
    assert (await world.repo.get_conversation(conv_id)).state is S.WAITING_OWNER
    # queued before the card, it passed every gate: it still goes
    assert (await world.repo.get_outbox(pending)).status is OutboxStatus.PENDING


async def test_the_card_shows_at_most_five_messages_cut_short(world: World) -> None:
    conv_id = await talking(world)
    for number in range(4):
        await say(world, conv_id, f"Ещё {number}")
    await say(world, conv_id, "я" * 400)

    await ask(world, conv_id)

    received = (await the_approval(world)).payload["received"]
    assert received == ["Ещё 0", "Ещё 1", "Ещё 2", "Ещё 3", "я" * 299 + "…"]


async def test_an_order_card(world: World) -> None:
    conv_id = await talking(world)

    result = await ask(
        world,
        conv_id,
        kind="approval",
        question="Заказать за 450 000?",
        amount=450_000,
        terms="доставка завтра",
    )

    assert result == (SENT_TO_OWNER, False)
    approval = await the_approval(world)
    assert approval.kind is K.CONFIRM_ORDER
    assert approval.payload == {
        "text": "Заказать за 450 000?",
        "amount": 450_000,
        "terms": "доставка завтра",
        "currency": "UZS",
        "last_incoming": "Берёте?",
        "buttons": ORDER_BUTTONS,
    }
    [card] = world.ui.sent
    assert "Сумма: 450 000 сум" in card.text
    assert "Условия: доставка завтра" in card.text
    assert [[label for label, _ in row] for row in card.buttons] == [
        ["Да — 450 000 сум", "Нет"],
        ["Напишу сам"],
    ]
    assert card.buttons[0][0][1] == f"ap:{approval.id}:yes:{approval.nonce}"


@pytest.mark.parametrize(
    ("kind", "approval_kind"), [("phone", K.SHARE_PHONE), ("address", K.SHARE_ADDRESS)]
)
async def test_a_card_for_the_phone_or_the_address(
    world: World, kind: str, approval_kind: ApprovalKind
) -> None:
    conv_id = await talking(world)

    assert await ask(world, conv_id, kind=kind, question="Дать?") == (
        SENT_TO_OWNER,
        False,
    )

    approval = await the_approval(world)
    assert approval.kind is approval_kind
    assert approval.payload == {"text": "Дать?", "last_incoming": "Берёте?"}
    assert (await world.repo.get_conversation(conv_id)).state is S.WAITING_OWNER


async def test_a_handoff_gives_the_chat_to_the_owner(world: World) -> None:
    conv_id = await talking(world)
    pending = await world.repo.enqueue(conv_id, "Подождите", world.clock.now())

    assert await ask(world, conv_id, kind="handoff", question="Он хочет человека") == (
        SENT_TO_OWNER,
        False,
    )

    approval = await the_approval(world)
    assert (approval.kind, approval.payload) == (
        K.HANDOFF,
        {"text": "Он хочет человека"},
    )
    assert (await world.repo.get_conversation(conv_id)).state is S.HANDED_TO_OWNER
    assert (await world.repo.get_outbox(pending)).status is OutboxStatus.PENDING
    [card] = world.ui.sent
    assert card.buttons[-1] == [
        ("Вернуть агенту", f"ap:{approval.id}:back:{approval.nonce}")
    ]


async def test_a_question_about_the_task_as_a_whole(world: World) -> None:
    result = await ask(world, None, question="Какой бюджет?", suggested=["Уточню"])

    assert result == ("Вопрос отправлен владельцу.", False)
    approval = await the_approval(world)
    assert (approval.conversation_id, approval.task_id) == (None, world.task.id)
    assert approval.payload == {
        "task": "Дрель",
        "about": "",
        "question": "Какой бюджет?",
        "received": [],
        "suggested": ["Уточню"],
    }
    [card] = world.ui.sent
    assert "👤" not in card.text
    assert "❓ Какой бюджет?" in card.text


@pytest.mark.parametrize(
    ("path", "state"), [((), S.PLANNED), ((S.CONTACTED,), S.CONTACTED)]
)
async def test_a_card_about_a_person_who_has_not_answered_and_back(
    world: World, path: tuple[ConvState, ...], state: ConvState
) -> None:
    conv_id = await person(world, *path)
    if state is S.CONTACTED:  # our first message went, no answer yet
        item = await world.repo.enqueue(conv_id, "Здравствуйте", world.clock.now())
        await world.repo.mark_outbox(item, OutboxStatus.SENT, at=world.clock.now())
        await world.repo.update_conversation(conv_id, first_sent=True)

    await ask(world, conv_id, kind="phone", question="Дать?")
    assert (await world.repo.get_conversation(conv_id)).state is S.WAITING_OWNER

    await decide(world, await the_approval(world), "no")
    assert (await world.repo.get_conversation(conv_id)).state is state


async def nothing_asked(world: World, result: tuple[str, bool], text: str) -> None:
    assert result == (text, True)
    assert await world.repo.pending_approvals() == []
    assert world.ui.sent == []


async def test_an_unknown_kind_is_an_error(world: World) -> None:
    conv_id = await talking(world)
    await nothing_asked(
        world,
        await ask(world, conv_id, kind="money"),
        "неизвестный kind: money — можно question, approval, phone, address, handoff",
    )


@pytest.mark.parametrize("kind", ["approval", "phone", "address", "handoff"])
async def test_a_card_about_a_person_needs_the_person(world: World, kind: str) -> None:
    await nothing_asked(
        world,
        await ask(world, None, kind=kind, amount=1.0),
        "укажи человека: этот вопрос о собеседнике",
    )


async def test_the_person_must_be_of_this_task_and_open(world: World) -> None:
    other = await world.repo.create_task("Ключи", "ключи", {}, None, "UZS", [], [])
    stranger = await person(world, task_id=other)
    closed = await person(world, S.CLOSED, peer_id=502, username="petr_master")

    await nothing_asked(
        world, await ask(world, stranger), "в этой задаче нет такого человека"
    )
    await nothing_asked(
        world, await ask(world, stranger + 100), "в этой задаче нет такого человека"
    )
    await nothing_asked(
        world,
        await ask(world, closed),
        "С этим человеком уже закончили. Написать ему снова можно, когда об этом"
        " попросит владелец",
    )


@pytest.mark.parametrize(
    ("options", "text"),
    [
        ({"question": "   "}, "пустой вопрос"),
        ({"question": "я" * 501}, "вопрос длиннее 500 символов — сократи"),
        (
            {"suggested": ["а", "б", "в", "г"]},
            "не больше 3 предложенных ответов, каждый до 400 символов",
        ),
        (
            {"suggested": ["я" * 401]},
            "не больше 3 предложенных ответов, каждый до 400 символов",
        ),
        (
            {"kind": "approval", "amount": None, "terms": "  "},
            "для approval нужна сумма (amount) или условия (terms)",
        ),
    ],
)
async def test_what_is_asked_is_checked(
    world: World, options: dict[str, Any], text: str
) -> None:
    conv_id = await talking(world)
    await nothing_asked(world, await ask(world, conv_id, **options), text)


async def test_the_same_card_is_not_sent_twice(world: World) -> None:
    conv_id = await talking(world)
    await ask(world, conv_id, kind="phone", question="Дать?")

    assert await ask(world, conv_id, kind="phone", question="Дать телефон?") == (
        "уже спрошено, ответа пока нет",
        True,
    )
    assert len(world.ui.sent) == 1
    # another kind about the same person is another question
    assert await ask(world, conv_id, kind="address", question="Дать?") == (
        SENT_TO_OWNER,
        False,
    )


async def test_the_same_question_about_the_task_is_not_asked_twice(
    world: World,
) -> None:
    await ask(world, None, question="Какой бюджет?")
    assert await ask(world, None, question="Какой бюджет?") == (
        "уже спрошено, ответа пока нет",
        True,
    )
    assert await ask(world, None, question="Какой цвет?") == (
        "Вопрос отправлен владельцу.",
        False,
    )


# -- owner_decision --


async def asked(world: World, conv_id: int | None, **kw: Any) -> Approval:
    await ask(world, conv_id, **kw)
    return (await world.repo.pending_approvals())[-1]


async def test_ok_sends_the_suggested_reply(world: World) -> None:
    conv_id = await talking(world)
    approval = await asked(world, conv_id, suggested=["Беру", "Когда доставите?"])

    assert await decide(world, approval, "ok") == "Ок, отправляю"

    settled = await world.repo.get_approval(approval.id)
    assert (settled.status, settled.answer) == (
        ApprovalStatus.ANSWERED,
        "Беру\nКогда доставите?",
    )
    assert (await world.repo.get_conversation(conv_id)).state is S.TALKING
    items = await world.repo.outbox_of(conv_id)
    assert [(item.text, item.status) for item in items] == [
        ("Беру", OutboxStatus.PENDING),
        ("Когда доставите?", OutboxStatus.PENDING),
    ]
    await told_it_went(world, conv_id, approval, ["Беру", "Когда доставите?"])


async def told_it_went(
    world: World, conv_id: int, approval: Approval, suggested: list[str]
) -> None:
    """Ruling R98: nothing is told at the press; the approval keeps the reply's outbox
    items, and the agent hears of its «Ок» once they went (or did not: `tell_ok`)."""
    assert await world.repo.open_events(world.task.id) == []
    items = await world.repo.outbox_of(conv_id)
    assert [item.text for item in items] == suggested
    settled = await world.repo.get_approval(approval.id)
    assert settled.payload[OK_ITEMS] == [item.id for item in items]


async def test_ok_sends_a_reply_the_judge_would_call_a_promise(world: World) -> None:
    # The owner read the text on the card and pressed «Ок»: it is his word. Before, the judge
    # refused it, the agent asked him again, and he pressed «Ок» again for nothing.
    conv_id = await talking(world)
    reply = "Саид пришлёт файл и напишет вам сам"
    approval = await asked(world, conv_id, suggested=[reply])
    world.llm.verdicts.append(
        JudgeVerdict(ok=False, issues=["commitment", "invented_fact"], note="")
    )

    assert await decide(world, approval, "ok") == "Ок, отправляю"

    assert [item.text for item in await world.repo.outbox_of(conv_id)] == [reply]
    await told_it_went(world, conv_id, approval, [reply])


async def test_ok_falls_back_to_the_agent_when_the_gates_refuse(world: World) -> None:
    conv_id = await talking(world)
    suggested = ["Смотрите https://example.com"]
    approval = await asked(world, conv_id, suggested=suggested)

    assert await decide(world, approval, "ok") == "Ответ передан агенту"

    assert await world.repo.outbox_of(conv_id) == []
    assert (
        await world.repo.get_approval(approval.id)
    ).status is ApprovalStatus.ANSWERED
    assert await events(world) == [
        (
            EventKind.OWNER_ANSWER,
            conv_id,
            {
                "approval_id": approval.id,
                "kind": "question",
                "text": "отправь предложенный ответ",
                "suggested": suggested,
            },
        )
    ]


async def test_ok_without_a_suggestion_changes_nothing(world: World) -> None:
    conv_id = await talking(world)
    approval = await asked(world, conv_id)

    assert await decide(world, approval, "ok") == (
        "Готового ответа нет — ответь на карточку своими словами"
    )
    assert (await world.repo.get_approval(approval.id)).status is ApprovalStatus.PENDING


async def test_the_owners_answer_to_a_question(world: World) -> None:
    conv_id = await talking(world)
    approval = await asked(world, conv_id)

    assert await decide(world, approval, "answer", "бери, но не дороже 500 000") == (
        "Ответ передан агенту"
    )

    settled = await world.repo.get_approval(approval.id)
    assert (settled.status, settled.answer) == (
        ApprovalStatus.ANSWERED,
        "бери, но не дороже 500 000",
    )
    assert (await world.repo.get_conversation(conv_id)).state is S.TALKING
    assert await events(world) == [
        (
            EventKind.OWNER_ANSWER,
            conv_id,
            {
                "approval_id": approval.id,
                "kind": "question",
                "text": "бери, но не дороже [скрыто]",
            },
        )
    ]


async def test_an_answer_needs_its_text(world: World) -> None:
    conv_id = await talking(world)
    approval = await asked(world, conv_id)
    assert await decide(world, approval, "answer", "  ") == "Нужен текст ответа"
    assert (await world.repo.get_approval(approval.id)).status is ApprovalStatus.PENDING


async def test_the_answer_to_a_question_about_the_task(world: World) -> None:
    approval = await asked(world, None, question="Какой бюджет?", suggested=["Уточню"])

    assert await decide(world, approval, "ok") == "Неизвестная кнопка"
    assert await decide(world, approval, "self") == "Неизвестная кнопка"
    assert await decide(world, approval, "answer", "до 500 000") == (
        "Ответ передан агенту"
    )

    assert await events(world) == [
        (
            EventKind.OWNER_ANSWER,
            None,
            {"approval_id": approval.id, "kind": "question", "text": "до [скрыто]"},
        )
    ]


async def order(world: World, conv_id: int) -> Approval:
    return await asked(
        world,
        conv_id,
        kind="approval",
        question="Заказать?",
        amount=450_000,
        terms="доставка завтра",
    )


async def test_yes_approves_the_order(world: World) -> None:
    conv_id = await talking(world)
    approval = await order(world, conv_id)

    assert await decide(world, approval, "yes") == "Заказ подтверждён"

    settled = await world.repo.get_approval(approval.id)
    assert settled.status is ApprovalStatus.APPROVED
    assert settled.payload["approved_amount"] == 450_000
    assert settled.payload["terms"] == "доставка завтра"
    assert (await world.repo.get_conversation(conv_id)).state is S.TALKING
    assert await events(world) == [
        (
            EventKind.OWNER_ANSWER,
            conv_id,
            {
                "approval_id": approval.id,
                "kind": "approval",
                "decision": "approved",
                "amount": 450_000,
                "terms": "доставка завтра",
            },
        )
    ]


async def test_no_rejects_the_order(world: World) -> None:
    conv_id = await talking(world)
    approval = await order(world, conv_id)

    assert await decide(world, approval, "no") == "Ок, не заказываем"

    assert (
        await world.repo.get_approval(approval.id)
    ).status is ApprovalStatus.REJECTED
    assert (await world.repo.get_conversation(conv_id)).state is S.TALKING
    assert await events(world) == [
        (
            EventKind.OWNER_ANSWER,
            conv_id,
            {"approval_id": approval.id, "kind": "approval", "decision": "rejected"},
        )
    ]


async def test_words_on_an_order_card_are_no_approval(world: World) -> None:
    conv_id = await talking(world)
    approval = await order(world, conv_id)

    assert await decide(world, approval, "answer", "поторгуйся") == (
        "Ответ передан агенту"
    )

    settled = await world.repo.get_approval(approval.id)
    assert (settled.status, settled.answer) == (ApprovalStatus.ANSWERED, "поторгуйся")
    assert K.CONFIRM_ORDER not in await world.repo.granted_kinds(conv_id)
    assert await events(world) == [
        (
            EventKind.OWNER_ANSWER,
            conv_id,
            {"approval_id": approval.id, "kind": "approval", "text": "поторгуйся"},
        )
    ]


@pytest.mark.parametrize("words", ["да", "Да.", "ок", "нет"])
async def test_a_yes_or_no_in_words_on_an_order_card_is_only_an_answer(
    world: World, words: str
) -> None:
    # Review x01: only the button confirms an order; «да» in words is the agent's to read.
    conv_id = await talking(world)
    approval = await order(world, conv_id)

    assert await decide(world, approval, "answer", words) == "Ответ передан агенту"

    settled = await world.repo.get_approval(approval.id)
    assert settled.status is ApprovalStatus.ANSWERED
    assert K.CONFIRM_ORDER not in await world.repo.granted_kinds(conv_id)


@pytest.mark.parametrize(
    ("kind", "name", "choice", "status", "decision", "toast"),
    [
        ("phone", "phone", "yes", ApprovalStatus.APPROVED, "approved", "Ок, телефон дадим"),
        ("phone", "phone", "no", ApprovalStatus.REJECTED, "rejected", "Ок, телефон не дадим"),
        ("address", "address", "yes", ApprovalStatus.APPROVED, "approved", "Ок, адрес дадим"),
        ("address", "address", "no", ApprovalStatus.REJECTED, "rejected", "Ок, адрес не дадим"),
    ],
)  # fmt: skip
async def test_the_phone_and_the_address(
    world: World,
    kind: str,
    name: str,
    choice: str,
    status: ApprovalStatus,
    decision: str,
    toast: str,
) -> None:
    conv_id = await talking(world)
    approval = await asked(world, conv_id, kind=kind, question="Дать?")

    assert await decide(world, approval, choice) == toast

    assert (await world.repo.get_approval(approval.id)).status is status
    assert (await world.repo.get_conversation(conv_id)).state is S.TALKING
    assert await events(world) == [
        (
            EventKind.OWNER_ANSWER,
            conv_id,
            {"approval_id": approval.id, "kind": name, "decision": decision},
        )
    ]


@pytest.mark.parametrize(
    ("kind", "options"),
    [
        ("question", {}),
        ("approval", {"amount": 450_000}),
        ("phone", {}),
        ("address", {}),
    ],
)
async def test_self_hands_the_chat_to_the_owner(
    world: World, kind: str, options: dict[str, Any]
) -> None:
    conv_id = await talking(world)
    approval = await asked(world, conv_id, kind=kind, **options)
    pending = await world.repo.enqueue(conv_id, "Подождите", world.clock.now())

    assert await decide(world, approval, "self") == "Чат за тобой"

    assert (
        await world.repo.get_approval(approval.id)
    ).status is ApprovalStatus.ANSWERED
    assert (await world.repo.get_conversation(conv_id)).state is S.HANDED_TO_OWNER
    assert (await world.repo.get_outbox(pending)).status is OutboxStatus.CANCELLED
    [handoff] = await world.repo.pending_approvals(conv_id)
    assert (handoff.kind, handoff.payload) == (K.HANDOFF, {"text": HANDOFF_SELF})
    card = world.ui.sent[-1]
    assert card.approval_id == handoff.id and HANDOFF_SELF in card.text
    assert (await world.repo.get_approval(handoff.id)).control_msg_id == (
        world.ui.sent_ids[-1]
    )
    assert await events(world) == []


async def test_back_returns_the_chat_to_the_agent(world: World) -> None:
    conv_id = await talking(world)
    approval = await asked(world, conv_id, kind="handoff", question="Он хочет человека")

    assert await decide(world, approval, "back") == "Агент снова ведёт чат"

    assert (
        await world.repo.get_approval(approval.id)
    ).status is ApprovalStatus.APPROVED
    assert (await world.repo.get_conversation(conv_id)).state is S.TALKING
    assert await events(world) == [
        (EventKind.RESUMED, conv_id, {"approval_id": approval.id})
    ]


async def test_a_second_press_or_a_wrong_nonce_is_already_decided(
    world: World,
) -> None:
    conv_id = await talking(world)
    approval = await asked(world, conv_id, kind="phone", question="Дать?")

    assert (
        await world.approvals.owner_decision(approval.id, "000000", "yes")
        == "Уже решено"
    )
    assert (await world.repo.get_approval(approval.id)).status is ApprovalStatus.PENDING
    assert await world.approvals.owner_decision(approval.id + 50, "x", "yes") == (
        "Уже решено"
    )

    await decide(world, approval, "yes")
    assert await decide(world, approval, "no") == "Уже решено"
    assert (
        await world.repo.get_approval(approval.id)
    ).status is ApprovalStatus.APPROVED
    assert len(await events(world)) == 1


async def test_an_unknown_button_changes_nothing(world: World) -> None:
    conv_id = await talking(world)
    approval = await asked(world, conv_id, kind="phone", question="Дать?")

    assert await decide(world, approval, "order") == "Неизвестная кнопка"
    assert await decide(world, approval, "back") == "Неизвестная кнопка"

    assert (await world.repo.get_approval(approval.id)).status is ApprovalStatus.PENDING
    assert (await world.repo.get_conversation(conv_id)).state is S.WAITING_OWNER
    assert await events(world) == []


async def test_the_person_waits_while_another_card_about_him_is_open(
    world: World,
) -> None:
    conv_id = await talking(world)
    phone = await asked(world, conv_id, kind="phone", question="Дать?")
    address = await asked(world, conv_id, kind="address", question="Дать?")

    await decide(world, phone, "yes")
    assert (await world.repo.get_conversation(conv_id)).state is S.WAITING_OWNER

    await decide(world, address, "yes")
    assert (await world.repo.get_conversation(conv_id)).state is S.TALKING


# -- fix round 1 --


async def test_the_amount_on_the_card_is_the_amount_approved(world: World) -> None:
    conv_id = await talking(world)
    approval = await order(world, conv_id)
    [card] = world.ui.sent

    await decide(world, approval, "yes")

    approved = (await world.repo.get_approval(approval.id)).payload["approved_amount"]
    assert approved == 450_000
    assert "Сумма: 450 000 сум" in card.text
    assert card.buttons[0][0][0] == "Да — 450 000 сум"


async def test_ok_after_a_newer_message_lets_the_agent_decide(world: World) -> None:
    conv_id = await talking(world)
    approval = await asked(world, conv_id, suggested=["Беру"])
    await say(world, conv_id, "Уже продал, извините")

    assert await decide(world, approval, "ok") == "Ответ передан агенту"

    assert await world.repo.outbox_of(conv_id) == []
    assert await events(world) == [
        (
            EventKind.OWNER_ANSWER,
            conv_id,
            {
                "approval_id": approval.id,
                "kind": "question",
                "text": "отправь предложенный ответ",
                "suggested": ["Беру"],
            },
        )
    ]


class FailingUI(FakeUI):
    """A control bot whose next card does not go."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_next = True

    async def send(self, card: Card) -> int:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("telegram is down")
        return await super().send(card)


async def test_a_card_that_did_not_reach_the_owner_holds_nobody(
    world: World, settings: Settings
) -> None:
    world.ui = FailingUI()
    world.approvals = Approvals(
        world.repo, world.ui, world.clock, settings, world.outgoing
    )
    conv_id = await talking(world)

    result = await ask(world, conv_id, kind="phone", question="Дать?")

    assert result == (
        "Карточка не отправилась (RuntimeError). Попробуй ещё раз позже",
        True,
    )
    [failed] = await world.repo.approvals_of(conv_id)
    assert failed.status is ApprovalStatus.EXPIRED
    assert (await world.repo.get_conversation(conv_id)).state is S.TALKING
    assert await world.repo.events_of("approval_failed", conv_id) == [
        {"approval_id": failed.id, "error": "RuntimeError"}
    ]
    assert await ask(world, conv_id, kind="phone", question="Дать?") == (
        SENT_TO_OWNER,
        False,
    )


@pytest.mark.parametrize(
    ("kind", "choice", "text"),
    [
        ("question", "ok", None),
        ("question", "answer", "Бери"),
        ("question", "self", None),
        ("approval", "yes", None),
        ("approval", "no", None),
        ("approval", "answer", "Бери"),
        ("approval", "self", None),
        ("phone", "yes", None),
        ("phone", "no", None),
        ("handoff", "back", None),
    ],
)
async def test_a_card_about_a_person_closed_meanwhile_carries_out_nothing(
    world: World, kind: str, choice: str, text: str | None
) -> None:
    conv_id = await talking(world)
    approval = await asked(
        world, conv_id, kind=kind, suggested=["Беру"], amount=450_000.0
    )
    await world.repo.set_state(conv_id, S.CLOSED)

    assert await decide(world, approval, choice, text) == (
        "С этим человеком уже закончили"
    )

    assert (await world.repo.get_approval(approval.id)).status is ApprovalStatus.EXPIRED
    assert await world.repo.granted_kinds(conv_id) == set()
    assert await world.repo.outbox_of(conv_id) == []
    assert await events(world) == []
    assert (await world.repo.get_conversation(conv_id)).state is S.CLOSED


@pytest.mark.parametrize(
    ("kind", "choice"),
    [("phone", "answer"), ("handoff", "answer"), ("handoff", "self")],
)
async def test_a_choice_the_kind_does_not_take(
    world: World, kind: str, choice: str
) -> None:
    conv_id = await talking(world)
    approval = await asked(world, conv_id, kind=kind, question="Дать?")

    assert await decide(world, approval, choice, "текст") == "Неизвестная кнопка"
    assert (await world.repo.get_approval(approval.id)).status is ApprovalStatus.PENDING


async def test_a_handoff_asked_while_the_person_waits_for_the_owner(
    world: World,
) -> None:
    conv_id = await talking(world)
    await ask(world, conv_id, kind="phone", question="Дать?")

    await ask(world, conv_id, kind="handoff", question="Он хочет человека")

    assert (await world.repo.get_conversation(conv_id)).state is S.HANDED_TO_OWNER


async def test_words_queued_before_the_card_still_go_and_nothing_new(
    world: World,
) -> None:
    conv_id = await talking(world)
    sent = await world.outgoing.send(
        world.task,
        conv_id,
        ["Уточню у владельца и вернусь"],
        language="ru",
        commitment=False,
        attach_files=False,
        seen_message_id=await world.repo.max_incoming_id(conv_id),
    )
    assert isinstance(sent, Queued)

    await ask(world, conv_id)

    [item] = await world.repo.outbox_of(conv_id)
    assert item.status is OutboxStatus.PENDING
    assert (await world.repo.get_conversation(conv_id)).state is S.WAITING_OWNER
    again = await world.outgoing.send(
        world.task,
        conv_id,
        ["И ещё"],
        language="ru",
        commitment=False,
        attach_files=False,
        seen_message_id=await world.repo.max_incoming_id(conv_id),
    )
    assert isinstance(again, Refused)


async def test_a_first_message_on_its_way_brings_him_back_contacted(
    world: World,
) -> None:
    conv_id = await person(world)
    await world.outgoing.send(
        world.task,
        conv_id,
        ["Здравствуйте! Есть дрель?"],
        language="ru",
        commitment=False,
        attach_files=False,
        seen_message_id=0,
    )
    await ask(world, conv_id, kind="phone", question="Дать?")

    await decide(world, await the_approval(world), "no")

    assert (await world.repo.get_conversation(conv_id)).state is S.CONTACTED


# -- fix round 2 --


@pytest.mark.parametrize("other_card", [False, True])
async def test_a_handoff_card_that_failed_leaves_the_person_where_he_was(
    world: World, settings: Settings, other_card: bool
) -> None:
    conv_id = await talking(world)
    if other_card:
        await ask(world, conv_id, kind="phone", question="Дать?")
    before = (await world.repo.get_conversation(conv_id)).state
    world.ui = FailingUI()
    world.approvals = Approvals(
        world.repo, world.ui, world.clock, settings, world.outgoing
    )

    result = await ask(world, conv_id, kind="handoff", question="Он хочет человека")

    assert result[1] is True
    assert before is (S.WAITING_OWNER if other_card else S.TALKING)
    assert (await world.repo.get_conversation(conv_id)).state is before


async def queued_proposal(world: World, conv_id: int) -> int:
    return await world.repo.enqueue(conv_id, "Берём за 450 000", world.clock.now())


async def test_no_on_an_order_card_drops_the_queued_proposal(world: World) -> None:
    conv_id = await talking(world)
    item = await queued_proposal(world, conv_id)
    approval = await order(world, conv_id)

    await decide(world, approval, "no")

    assert (await world.repo.get_outbox(item)).status is OutboxStatus.CANCELLED


@pytest.mark.parametrize(
    ("kind", "choice", "text"),
    [
        ("approval", "yes", None),
        ("approval", "answer", "поторгуйся"),
        ("phone", "no", None),
    ],
)
async def test_other_decisions_leave_the_queue_alone(
    world: World, kind: str, choice: str, text: str | None
) -> None:
    conv_id = await talking(world)
    item = await queued_proposal(world, conv_id)
    approval = await asked(world, conv_id, kind=kind, question="Дать?", amount=450_000)

    await decide(world, approval, choice, text)

    assert (await world.repo.get_outbox(item)).status is OutboxStatus.PENDING


# -- A13: the hand-back card that cannot be sent --


async def test_self_when_the_hand_back_card_cannot_be_sent(
    world: World, settings: Settings
) -> None:
    conv_id = await talking(world)
    approval = await asked(world, conv_id, kind="phone", question="Дать?")
    ui = FailingUI()  # the next card (the hand-back one) does not go
    world.approvals = Approvals(world.repo, ui, world.clock, settings, world.outgoing)

    result = await decide(world, approval, "self")  # raises nothing into the bot

    assert result == "Чат за тобой. Карточку для возврата пришлю позже"
    assert (await world.repo.get_conversation(conv_id)).state is S.HANDED_TO_OWNER
    assert await world.repo.pending_approvals(conv_id) == []  # recover sends it later
    [handoff] = [
        a for a in await world.repo.approvals_of(conv_id) if a.kind is K.HANDOFF
    ]
    assert handoff.status is ApprovalStatus.EXPIRED


# Fix pass B15: only a failed send counts as "the card was not sent".


async def test_a_bookkeeping_error_after_the_card_reached_the_owner_is_raised(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    conv_id = await talking(world)

    async def broken(approval_id: int, msg_id: int) -> None:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(world.repo, "set_approval_msg", broken)

    with pytest.raises(RuntimeError, match="database is locked"):
        await ask(world, conv_id)

    assert len(world.ui.sent) == 1  # it reached the owner
    [approval] = await world.repo.approvals_of(conv_id)
    assert approval.status is ApprovalStatus.PENDING  # not expired while he holds it


async def test_a_card_that_cannot_be_drawn_is_raised(
    world: World, settings: Settings
) -> None:
    conv_id = await talking(world)

    def broken(*args: Any) -> Any:
        raise KeyError("buttons")

    world.approvals = Approvals(
        world.repo, world.ui, world.clock, settings, world.outgoing, render=broken
    )

    with pytest.raises(KeyError):
        await ask(world, conv_id)
