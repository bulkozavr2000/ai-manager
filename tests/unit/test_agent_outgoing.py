"""The send pipeline of the agent core (`Outgoing.send`) and what goes out on a first reply:
the AI introduction (`with_own_intro`, `must_introduce`)."""

from __future__ import annotations

import asyncio
import inspect
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, get_args
from zoneinfo import ZoneInfo

import pytest

from tg_agent.agent import people
from tg_agent.agent.outgoing import (
    OWNER_REPLY_WAITS,
    REASONS,
    Outgoing,
    Queued,
    Refused,
    must_introduce,
    reason_for,
    with_own_intro,
)
from tg_agent.brain.context import load_prompt
from tg_agent.brain.llm import FakeLLM, LLMError, LLMResult, ModelRefusal
from tg_agent.brain.schemas import JudgeVerdict
from tg_agent.clock import FakeClock
from tg_agent.config import Settings
from tg_agent.db import Repo
from tg_agent.gateway.base import TgActionError
from tg_agent.gateway.fake import FakeGateway
from tg_agent.models import (
    ApprovalKind,
    ApprovalStatus,
    Conversation,
    ConvState,
    Direction,
    EventKind,
    MsgSource,
    OutboxStatus,
    Task,
    TaskStatus,
)
from tg_agent.policy import postgates
from tg_agent.policy.tg_safety import TgErrorKind
from tg_agent.tasks import templates

S = ConvState
TASHKENT = ZoneInfo("Asia/Tashkent")
INTRO = templates.INTRO["ru"]


def test_the_introduction_goes_in_after_the_greeting_of_the_first_message() -> None:
    texts = with_own_intro(
        ["Ассаляму алейкум! Есть дрель?", "Сколько стоит?"], "ru", introduce=True
    )
    assert texts == [f"Ассаляму алейкум! {INTRO}. Есть дрель?", "Сколько стоит?"]


def test_the_very_phrase_already_written_gets_no_introduction() -> None:
    messages = ["Здравствуйте! Я ИИ-помощник Саида, есть дрель?"]
    assert with_own_intro(messages, "ru", introduce=True) == messages


def test_the_very_phrase_in_a_later_message_gets_no_introduction_either() -> None:
    # The greeting alone first, as the style asks: the phrase must not be said twice.
    messages = ["Ассаляму алейкум!", "Я ИИ-помощник Саида. Подскажите, есть дрель?"]
    assert with_own_intro(messages, "ru", introduce=True) == messages


def test_no_introduction_when_not_asked() -> None:
    messages = ["Здравствуйте! Есть дрель?"]
    assert with_own_intro(messages, "ru", introduce=False) == messages


def test_blank_messages_are_dropped() -> None:
    assert with_own_intro(["  Есть дрель? ", "", "   "], "ru", introduce=False) == [
        "Есть дрель?"
    ]


async def conversation(repo: Repo) -> Conversation:
    task_id = await repo.create_task(
        "Дрель",
        "самое дешёвое предложение",
        {"товар": "дрель"},
        500_000.0,
        "UZS",
        [],
        [],
    )
    contact_id = await repo.upsert_contact(1001, "seller", "Продавец", False)
    return await repo.get_conversation(
        await repo.create_conversation(task_id, contact_id)
    )


async def test_no_introduction_once_something_was_sent(repo: Repo) -> None:
    conv = await conversation(repo)
    await repo.update_conversation(conv.id, first_sent=True)
    gateway = FakeGateway()
    gateway.name = "Саид"
    assert await must_introduce(gateway, await repo.get_conversation(conv.id)) is False


async def test_no_introduction_when_the_name_says_it(repo: Repo) -> None:
    gateway = FakeGateway()
    gateway.name = "ИИ помощник Саида"
    assert await must_introduce(gateway, await conversation(repo)) is False


async def test_an_introduction_when_the_name_does_not_say_it(repo: Repo) -> None:
    gateway = FakeGateway()
    gateway.name = "Саид"
    assert await must_introduce(gateway, await conversation(repo)) is True


@pytest.mark.parametrize(
    "name",
    [
        "Said SI",
        "Саид si",
        "Said Axmedov Si",
        "SI-yordamchi Said",
        "Said (SI)",
        "Said_SI",
    ],
)
async def test_no_introduction_when_the_name_holds_the_uzbek_si(
    repo: Repo, name: str
) -> None:
    # Ruling R82 (the owner, 2026-11-12): «SI» — sun'iy intellekt — as a word of its own, in
    # any case, says that an AI writes; in Uzbekistan everyone reads it so.
    gateway = FakeGateway()
    gateway.name = name
    assert await must_introduce(gateway, await conversation(repo)) is False


@pytest.mark.parametrize(
    "name",
    ["Said", "Sirojiddin", "Saidning yordamchisi", "SaidSI", "Kasimov", "SI2 Said"],
)
async def test_si_inside_a_word_is_not_the_abbreviation(repo: Repo, name: str) -> None:
    gateway = FakeGateway()
    gateway.name = name
    assert await must_introduce(gateway, await conversation(repo)) is True


async def test_an_introduction_when_the_name_cannot_be_read(repo: Repo) -> None:
    gateway = FakeGateway()
    gateway.name = "ИИ помощник Саида"
    gateway.fail_next = TgActionError(TgErrorKind.OTHER, "no name")
    assert await must_introduce(gateway, await conversation(repo)) is True


# -- the send pipeline (`Outgoing.send`) --

CANARY = "ZX-CANARY-7731"
OWNER_PHONE = "+998 90 111 22 33"
FILE = "/files/чертёж.pdf"


@dataclass
class World:
    repo: Repo
    gateway: FakeGateway
    llm: FakeLLM
    clock: FakeClock
    settings: Settings
    outgoing: Outgoing
    task: Task


def outgoing_of(world: World) -> Outgoing:
    return Outgoing(
        world.repo,
        world.gateway,
        world.llm,
        world.clock,
        world.settings,
        random.Random(7),
        canary=CANARY,
        owner_secrets=[OWNER_PHONE],
    )


@pytest.fixture
async def world(repo: Repo, fake_clock: FakeClock, settings: Settings) -> World:
    gateway = FakeGateway()
    gateway.name = "Саид"  # the account's name does not say it is an AI
    task_id = await repo.create_task(
        "Дрель",
        "купить дрель до 500 000 сум",
        {"товар": "дрель"},
        500_000.0,
        "UZS",
        ["name"],
        [FILE],
    )
    await repo.set_task_status(task_id, TaskStatus.ACTIVE)
    world = World(
        repo,
        gateway,
        FakeLLM(),
        fake_clock,
        settings,
        None,  # type: ignore[arg-type]
        await repo.get_task(task_id),
    )
    world.outgoing = outgoing_of(world)
    return world


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


async def incoming(world: World, conv_id: int, text: str) -> int:
    """The person wrote (as the inbox stores it)."""
    now = world.clock.now()
    message_id = await world.repo.add_message(
        conv_id, Direction.IN, MsgSource.COUNTERPART, "text", text, None, now
    )
    await world.repo.update_conversation(conv_id, last_in_at=now, followups_sent=0)
    world.clock.advance(timedelta(minutes=1))
    return message_id


async def sent_before(world: World, conv_id: int, text: str = "Есть дрель?") -> None:
    """Our message went out a while ago."""
    now = world.clock.now()
    item = await world.repo.enqueue(conv_id, text, now)
    await world.repo.mark_outbox(item, OutboxStatus.SENT, at=now)
    await world.repo.add_message(
        conv_id, Direction.OUT, MsgSource.AGENT, "text", text, None, now
    )
    await world.repo.update_conversation(conv_id, first_sent=True, last_out_at=now)
    world.clock.advance(timedelta(hours=1))


async def send(
    world: World, conv_id: int, messages: list[str], **options: Any
) -> Queued | Refused:
    seen = options.pop("seen_message_id", None)
    if seen is None:
        seen = await world.repo.max_incoming_id(conv_id)
    arguments: dict[str, Any] = {
        "language": "ru",
        "commitment": False,
        "attach_files": False,
        "seen_message_id": seen,
    } | options
    return await world.outgoing.send(world.task, conv_id, messages, **arguments)


async def refused(
    world: World, conv_id: int, messages: list[str], **options: Any
) -> Refused:
    """`send` refuses, and nothing is queued."""
    before = await world.repo.outbox_of(conv_id)
    result = await send(world, conv_id, messages, **options)
    assert isinstance(result, Refused), result
    assert await world.repo.outbox_of(conv_id) == before
    return result


async def approve(
    world: World, conv_id: int, kind: ApprovalKind, payload: dict | None = None
) -> None:
    approval = await world.repo.create_approval(
        world.task.id, conv_id, kind, payload or {}
    )
    await world.repo.resolve_approval(
        approval.id, approval.nonce, ApprovalStatus.APPROVED
    )


async def a_wake(world: World) -> int:
    await world.repo.add_event(world.task.id, EventKind.CREATED)
    wake_id, _ = await world.repo.claim_events(world.task.id, world.clock.now())
    return wake_id


# 1. the person


async def test_a_person_of_another_task_is_refused(world: World) -> None:
    other_task = await world.repo.create_task("Ключи", "ключи", {}, None, "UZS", [], [])
    stranger = await person(world, task_id=other_task)

    result = await refused(world, stranger, ["Здравствуйте!"])
    assert result == Refused("в этой задаче нет такого человека")

    missing = await send(world, stranger + 100, ["Здравствуйте!"], seen_message_id=0)
    assert missing == Refused("в этой задаче нет такого человека")


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        (
            [S.CLOSED],
            (
                "С этим человеком уже закончили. Написать ему снова можно, когда об"
                " этом попросит владелец"
            ),
        ),
        ([S.HANDED_TO_OWNER], "этот чат у владельца: он пишет сам"),
        (
            [S.WAITING_OWNER],
            "по этому человеку ждём ответа владельца — пока ему ничего не пишем",
        ),
    ],
)
async def test_a_person_the_agent_may_not_write_to_is_refused(
    world: World, path: list[ConvState], reason: str
) -> None:
    conv_id = await person(world, *path)
    assert await refused(world, conv_id, ["Здравствуйте!"]) == Refused(reason)


# Ruling R72: the owner's instruction reopens a person the agent had finished with.


async def owner_wake(
    world: World, kind: EventKind = EventKind.OWNER_INSTRUCTION
) -> int:
    """A wake that carries the owner's word about the task."""
    await world.repo.add_event(
        world.task.id, kind, payload={"text": "поторгуйся с Иваном до 200 000"}
    )
    wake_id, _ = await world.repo.claim_events(world.task.id, world.clock.now())
    return wake_id


async def closed_after_talk(world: World, *, wrote: bool = True) -> int:
    conv_id = await person(world, S.CONTACTED)
    await sent_before(world, conv_id)
    if wrote:
        await incoming(world, conv_id, "450 000, срок 3 дня")
        await world.repo.set_state(conv_id, S.TALKING)
        await sent_before(world, conv_id, "Спасибо!")  # thanked, then finished
    await world.repo.set_state(conv_id, S.CLOSED)
    return conv_id


@pytest.mark.parametrize(
    ("kind", "wrote", "state"),
    [
        (EventKind.OWNER_INSTRUCTION, True, S.TALKING),
        (EventKind.OWNER_ANSWER, True, S.TALKING),
        (EventKind.OWNER_INSTRUCTION, False, S.CONTACTED),
    ],
)
async def test_the_owner_s_word_reopens_a_person_the_agent_finished_with(
    world: World, kind: EventKind, wrote: bool, state: ConvState
) -> None:
    conv_id = await closed_after_talk(world, wrote=wrote)
    stale = await world.repo.create_approval(
        world.task.id, conv_id, ApprovalKind.QUESTION, {"question": "?"}
    )
    await world.repo.resolve_approval(stale.id, stale.nonce, ApprovalStatus.EXPIRED)
    wake_id = await owner_wake(world, kind)
    now = world.clock.now()

    result = await send(world, conv_id, ["А за 200 000 сделаете?"], wake_id=wake_id)

    assert isinstance(result, Queued), result
    assert (await world.repo.get_conversation(conv_id)).state is state
    assert 501 in await people.listened(world.repo)  # his task is in work: he is heard
    assert now + timedelta(seconds=3) <= result.send_at <= now + timedelta(seconds=8)
    [item] = [
        i for i in await world.repo.outbox_of(conv_id) if i.status.value == "pending"
    ]
    assert item.text.endswith("А за 200 000 сделаете?")
    # A card about him that ended stays ended.
    assert (await world.repo.get_approval(stale.id)).status is ApprovalStatus.EXPIRED


async def test_without_the_owner_s_word_a_finished_person_stays_closed(
    world: World,
) -> None:
    conv_id = await closed_after_talk(world)
    wake_id = await a_wake(world)

    result = await refused(world, conv_id, ["А за 200 000?"], wake_id=wake_id)

    assert result == Refused(
        "С этим человеком уже закончили. Написать ему снова можно, когда об этом"
        " попросит владелец"
    )
    assert (await world.repo.get_conversation(conv_id)).state is S.CLOSED


async def test_one_who_refused_an_ai_or_was_handed_to_the_owner_stays_closed(
    world: World,
) -> None:
    refusing = await closed_after_talk(world)
    contact = (await world.repo.get_conversation(refusing)).contact_id
    await world.repo.set_do_not_contact(contact, True)
    handed = await person(
        world, S.CONTACTED, S.HANDED_TO_OWNER, peer_id=502, username="drill_shop"
    )
    card = await world.repo.create_approval(
        world.task.id, handed, ApprovalKind.HANDOFF, {"question": "?"}
    )
    await world.repo.resolve_approval(card.id, card.nonce, ApprovalStatus.EXPIRED)
    await world.repo.set_state(handed, S.CLOSED)
    wake_id = await owner_wake(world)

    for conv_id in (refusing, handed):
        result = await refused(world, conv_id, ["А за 200 000?"], wake_id=wake_id)
        assert result.reason.startswith("С этим человеком уже закончили")
        assert (await world.repo.get_conversation(conv_id)).state is S.CLOSED


async def test_nothing_goes_while_the_owner_writes_in_the_chat(world: World) -> None:
    conv_id = await person(world)
    await world.repo.update_conversation(
        conv_id, paused_until=world.clock.now() + timedelta(minutes=30)
    )
    result = await refused(world, conv_id, ["Здравствуйте!"])
    assert result == Refused("владелец сам пишет в этом чате — подожди")


async def test_a_pause_that_is_over_holds_nothing_back(world: World) -> None:
    conv_id = await person(world)
    await world.repo.update_conversation(
        conv_id, paused_until=world.clock.now() - timedelta(minutes=1)
    )
    assert isinstance(await send(world, conv_id, ["Здравствуйте!"]), Queued)


# 2. stale


async def test_a_reply_drafted_before_a_newer_message_is_stale(world: World) -> None:
    conv_id = await person(world, S.CONTACTED)
    seen = await incoming(world, conv_id, "Есть, 450 000")
    await incoming(world, conv_id, "А вам какая нужна?")

    result = await refused(world, conv_id, ["Беру!"], seen_message_id=seen)
    assert result == Refused(
        "собеседник написал после начала этого пробуждения — ответ устарел;"
        " новое сообщение увидишь в следующем пробуждении"
    )


# 3. the texts


async def test_an_empty_message_is_refused(world: World) -> None:
    conv_id = await person(world)
    assert await refused(world, conv_id, ["  ", ""]) == Refused("пустое сообщение")


# 4. reminders


async def test_no_more_reminders_than_the_limit(world: World) -> None:
    conv_id = await person(world, S.CONTACTED)
    await sent_before(world, conv_id)
    await world.repo.update_conversation(conv_id, followups_sent=2)

    result = await refused(world, conv_id, ["Напомню о себе: есть дрель?"])
    assert result == Refused(
        "напоминаний без ответа уже 2 — закончи с этим человеком или спроси владельца"
    )


async def test_a_reply_is_no_reminder_however_many_went_before(world: World) -> None:
    conv_id = await person(world, S.CONTACTED)
    await sent_before(world, conv_id)
    await incoming(world, conv_id, "Есть")
    await world.repo.update_conversation(conv_id, followups_sent=2)

    queued = await send(world, conv_id, ["Сколько стоит?"])
    assert isinstance(queued, Queued)
    assert (await world.repo.get_conversation(conv_id)).followups_sent == 2


# 6. other people of the task


async def test_one_person_never_hears_the_contact_of_another(world: World) -> None:
    conv_id = await person(world)
    await person(world, peer_id=502, username="petr_master")

    result = await refused(world, conv_id, ["Здравствуйте! Вам советовал @Petr_Master"])
    assert result == Refused("не называй одному собеседнику контакты другого")


@pytest.mark.parametrize(
    "text",
    ["Здравствуйте! Ещё есть @maker_uz", "Позвоните Абдулхамиду: +998 90 765 43 21"],
)
async def test_one_person_never_hears_the_contact_of_a_candidate(
    world: World, text: str
) -> None:
    # Fix pass E1.6, M12: a candidate the search found is another person too.
    conv_id = await person(world)
    await world.repo.add_candidate(world.task.id, "maker_uz", "Maker", "", "")
    await world.repo.add_candidate(world.task.id, "+998907654321", "Абдулхамид", "", "")

    result = await refused(world, conv_id, [text])

    assert result == Refused("не называй одному собеседнику контакты другого")


async def test_a_candidate_who_is_this_person_may_be_named_to_him(
    world: World,
) -> None:
    conv_id = await person(world, username="maker_uz")
    await world.repo.add_candidate(world.task.id, "maker_uz", "Maker", "", "")

    assert isinstance(await send(world, conv_id, ["Это @maker_uz?"]), Queued)


# 7. the gates


async def test_a_link_is_refused_with_the_words_of_the_table(world: World) -> None:
    conv_id = await person(world)
    result = await refused(
        world, conv_id, ["Здравствуйте! Вот такая: https://example.com"]
    )
    assert result == Refused(REASONS["link"])


async def test_a_commitment_without_an_approval_asks_for_one(world: World) -> None:
    conv_id = await person(world)
    await incoming(world, conv_id, "Дрель 450 000")
    result = await refused(world, conv_id, ["Хорошо, берём"], commitment=True)
    assert result == Refused(REASONS["commitment"], ApprovalKind.CONFIRM_ORDER)


async def test_the_owners_phone_without_an_approval_asks_for_one(world: World) -> None:
    conv_id = await person(world)
    await incoming(world, conv_id, "Ваш номер?")
    result = await refused(world, conv_id, ["Звоните: {PHONE}"])
    assert result == Refused(REASONS["placeholder_phone"], ApprovalKind.SHARE_PHONE)


async def test_the_owners_address_without_an_approval_asks_for_one(
    world: World,
) -> None:
    conv_id = await person(world)
    await incoming(world, conv_id, "Куда везти?")
    result = await refused(world, conv_id, ["Везите сюда: {ADDRESS}"])
    assert result == Refused(REASONS["placeholder_address"], ApprovalKind.SHARE_ADDRESS)


async def test_two_rules_are_told_together(world: World) -> None:
    conv_id = await person(world)
    await incoming(world, conv_id, "Сколько штук?")
    result = await refused(world, conv_id, ["Вот: https://example.com", "х" * 401])
    assert result == Refused(f"{REASONS['link']}; {REASONS['too_long']}")


async def test_a_leak_is_refused_and_logged(world: World) -> None:
    conv_id = await person(world)
    await incoming(world, conv_id, "Ваш номер?")

    result = await refused(world, conv_id, ["Мой номер +998 90 111 22 33"])
    assert result.reason.startswith(REASONS["owner_secret"])
    assert result.ask is None
    result = await refused(world, conv_id, [f"Код {CANARY}"])
    assert result == Refused(REASONS["canary"])

    leaks = await world.repo.events_of("leak_blocked", conv_id)
    assert [event["reasons"] for event in leaks] == [
        ["owner_secret", "pii"],
        ["canary"],
    ]
    assert all(event["task_id"] == world.task.id for event in leaks)


async def test_asked_if_it_is_a_bot_the_helper_s_answer_is_not_refused(
    world: World,
) -> None:
    # Ruling R82 (the owner, 2026-11-12): to «вы бот?» the agent answers «помощник Саида»;
    # the account's name («Said SI») says the rest. No code demands an admission any more.
    conv_id = await person(world, S.CONTACTED)
    await sent_before(world, conv_id)
    await incoming(world, conv_id, "Вы бот?")

    assert isinstance(await send(world, conv_id, ["Я помощник Саида"]), Queued)


async def test_asked_if_it_is_a_bot_a_denial_is_still_refused(world: World) -> None:
    conv_id = await person(world, S.CONTACTED)
    await sent_before(world, conv_id)
    await incoming(world, conv_id, "Вы бот?")

    assert await refused(world, conv_id, ["Нет, я не бот, я Саид"]) == Refused(
        REASONS["ai_denial"]
    )


async def test_a_question_answered_before_our_last_message_is_old(
    world: World,
) -> None:
    conv_id = await person(world, S.CONTACTED)
    await incoming(world, conv_id, "Вы бот?")
    await sent_before(world, conv_id, "Да, я ИИ-помощник")
    await incoming(world, conv_id, "Понятно, дрель есть")

    assert isinstance(await send(world, conv_id, ["Сколько стоит?"]), Queued)


async def test_the_gate_reads_the_language_the_person_writes(world: World) -> None:
    conv_id = await person(world)
    await incoming(world, conv_id, "Здравствуйте, что вы ищете?")  # «ы»: Russian
    result = await refused(world, conv_id, ["Rahmat, дрель ищу"], language="uz_latn")
    assert result == Refused(REASONS["not_russian"])


async def test_without_a_clue_the_declared_language_holds(world: World) -> None:
    conv_id = await person(world, S.CONTACTED)
    await world.repo.update_conversation(conv_id, first_sent=True)  # no introduction
    result = await refused(
        world, conv_id, ["Ассалому алайкум, дрель борми?"], language="uz_latn"
    )
    assert result == Refused(REASONS["not_uzbek_latin"])


def test_every_rule_of_the_gates_has_its_words() -> None:
    rules = set(re.findall(r'_Fired\("(\w+)"', inspect.getsource(postgates)))
    assert len(rules) == 12  # «ai_not_admitted» is gone (ruling R82)
    assert rules <= REASONS.keys()
    issues = get_args(get_args(JudgeVerdict.model_fields["issues"].annotation)[0])
    assert {f"judge_{issue}" for issue in issues} <= REASONS.keys()
    assert {"judge", "judge_refused"} <= REASONS.keys()


def test_a_rule_without_words_is_named() -> None:
    assert reason_for(["new_rule"]) == "текст не прошёл проверку: new_rule"
    assert reason_for(["link", "link"]) == REASONS["link"]


# 8. the judge


async def test_the_judges_objection_refuses(world: World) -> None:
    conv_id = await person(world)
    world.llm.verdicts.append(
        JudgeVerdict(ok=False, issues=["invented_fact", "leak"], note="")
    )
    result = await refused(world, conv_id, ["Здравствуйте! Есть дрель?"])
    assert result == Refused(
        f"{REASONS['judge_invented_fact']}; {REASONS['judge_leak']}"
    )


async def test_a_judge_that_says_no_without_an_issue_refuses(world: World) -> None:
    conv_id = await person(world)
    world.llm.verdicts.append(JudgeVerdict(ok=False, issues=[], note=""))
    assert await refused(world, conv_id, ["Здравствуйте!"]) == Refused(REASONS["judge"])


async def test_an_issue_is_an_objection_even_when_ok(world: World) -> None:
    conv_id = await person(world)
    world.llm.verdicts.append(JudgeVerdict(ok=True, issues=["commitment"], note=""))
    assert await refused(world, conv_id, ["Здравствуйте!"]) == Refused(
        REASONS["judge_commitment"]
    )


@pytest.mark.parametrize("error", [ModelRefusal("no"), LLMError("down")])
async def test_a_judge_that_fails_refuses(world: World, error: Exception) -> None:
    conv_id = await person(world)
    world.llm.verdicts.append(error)
    assert await refused(world, conv_id, ["Здравствуйте!"]) == Refused(
        REASONS["judge_refused"]
    )
    [failed] = await world.repo.events_of("judge_failed", conv_id)
    assert failed == {"task_id": world.task.id, "error": type(error).__name__}


async def test_the_judge_call_is_logged_with_the_task_and_the_wake(
    world: World,
) -> None:
    conv_id = await person(world)
    wake_id = await a_wake(world)

    await send(world, conv_id, ["Здравствуйте! Есть дрель?"], wake_id=wake_id)

    rows = await world.repo._all(
        "SELECT purpose, conversation_id, task_id, wake_id FROM llm_calls", ()
    )
    assert [tuple(row) for row in rows] == [("judge", conv_id, world.task.id, wake_id)]


class HintLLM(FakeLLM):
    """Keeps what the judge was told."""

    def __init__(self) -> None:
        super().__init__()
        self.judged: list[tuple[list[str], list[str], str]] = []

    async def judge(
        self, messages: list[str], allowed: list[str], hint: str
    ) -> LLMResult[JudgeVerdict]:
        self.judged.append((messages, allowed, hint))
        return await super().judge(messages, allowed, hint)


async def test_the_judge_knows_what_the_owner_allowed_but_never_the_ceiling(
    world: World,
) -> None:
    world.llm = llm = HintLLM()
    world.outgoing = outgoing_of(world)
    conv_id = await person(world)
    await incoming(world, conv_id, "Это бот?")
    await approve(
        world,
        conv_id,
        ApprovalKind.CONFIRM_ORDER,
        {"approved_amount": 450_000, "terms": "доставка завтра"},
    )

    await send(
        world,
        conv_id,
        ["Я ИИ-помощник Саида. Берём за 450 000"],
        commitment=True,
        instructions=["бюджет 500 000, не больше"],
    )

    [(messages, allowed, hint)] = llm.judged
    assert messages == ["Я ИИ-помощник Саида. Берём за 450 000"]
    assert allowed == ["имя — Саид Ахмедов"]
    assert hint.splitlines() == [
        "Задача: купить дрель до [скрыто] сум",
        "Телефон владельца ({PHONE}): не разрешён",
        "Адрес владельца ({ADDRESS}): не разрешён",
        "Заказ: владелец разрешил кнопкой — подтверждать можно в этих пределах:",
        "• сумма до 450 000; условия: доставка завтра",
        "Указание владельца: бюджет [скрыто], не больше",
    ]  # «Это бот?» adds no line any more (ruling R82)


async def test_an_approved_amount_equal_to_the_ceiling_is_shown_to_the_judge(
    world: World,
) -> None:
    # Fix pass E1.3: what the owner approved by button is no secret from the judge.
    world.llm = llm = HintLLM()
    world.outgoing = outgoing_of(world)
    conv_id = await person(world)
    await approve(
        world, conv_id, ApprovalKind.CONFIRM_ORDER, {"approved_amount": 500_000}
    )

    await send(
        world,
        conv_id,
        ["Берём за 500 000"],
        commitment=True,
        instructions=["владелец разрешил подтвердить — до 500 000", "бюджет 500 000"],
    )

    [(_, _, hint)] = llm.judged
    lines = hint.splitlines()
    assert "Указание владельца: владелец разрешил подтвердить — до 500 000" in lines
    assert "Указание владельца: бюджет [скрыто]" in lines


async def test_the_judge_hears_when_nothing_is_allowed(world: World) -> None:
    world.llm = llm = HintLLM()
    world.outgoing = outgoing_of(world)
    conv_id = await person(world)
    await approve(world, conv_id, ApprovalKind.SHARE_PHONE)

    await send(world, conv_id, ["Здравствуйте! Есть дрель?"])

    [(_, _, hint)] = llm.judged
    assert hint.splitlines() == [
        "Задача: купить дрель до [скрыто] сум",
        "Телефон владельца ({PHONE}): разрешён",
        "Адрес владельца ({ADDRESS}): не разрешён",
        "Заказ: владелец не разрешал подтверждать заказ, оплату и условия",
    ]


# 9-10. the queue


async def test_a_first_message_is_introduced_and_queued(world: World) -> None:
    conv_id = await person(world)
    wake_id = await a_wake(world)
    now = world.clock.now()

    result = await send(world, conv_id, ["Здравствуйте! Есть дрель?"], wake_id=wake_id)

    assert isinstance(result, Queued)
    assert (result.count, result.first_contact) == (1, True)
    # Ruling R70: nobody waits for it, so it goes at once (a few seconds).
    assert now + timedelta(seconds=3) <= result.send_at <= now + timedelta(seconds=8)
    [item] = await world.repo.outbox_of(conv_id)
    assert item.text == f"Здравствуйте! {INTRO}. Есть дрель?"
    assert (item.status, item.not_before, item.wake_id) == (
        OutboxStatus.PENDING,
        result.send_at,
        wake_id,
    )
    conv = await world.repo.get_conversation(conv_id)
    assert conv.state is S.CONTACTED
    assert (conv.agent_messages, conv.followups_sent) == (1, 0)


async def test_no_introduction_when_the_account_name_says_it(world: World) -> None:
    world.gateway.name = "ИИ помощник Саида"
    conv_id = await person(world)
    await send(world, conv_id, ["Здравствуйте! Есть дрель?"])
    [item] = await world.repo.outbox_of(conv_id)
    assert item.text == "Здравствуйте! Есть дрель?"


async def test_out_of_hours_the_message_waits_for_the_morning(world: World) -> None:
    world.clock.set(datetime(2026, 11, 10, 22, 0, tzinfo=TASHKENT))
    conv_id = await person(world)
    result = await send(world, conv_id, ["Здравствуйте! Есть дрель?"])
    assert isinstance(result, Queued)
    assert result.send_at == datetime(2026, 11, 11, 9, 0, tzinfo=TASHKENT)


# Task 3.1: a first message to a new person gets a paced slot (`tasks/first_contacts`).


async def test_first_sends_to_two_new_people_are_1_to_2_minutes_apart(
    world: World,
) -> None:
    first = await person(world)
    second = await person(world, peer_id=502, username="drill_shop")

    one = await send(world, first, ["Здравствуйте! Есть дрель?"])
    two = await send(world, second, ["Здравствуйте! Есть дрель?"])

    assert isinstance(one, Queued) and isinstance(two, Queued)
    assert (one.first_contact, two.first_contact) == (True, True)
    gap = two.send_at - one.send_at
    assert timedelta(minutes=1) <= gap <= timedelta(minutes=2)  # ruling R70
    [item] = await world.repo.outbox_of(second)
    assert item.not_before == two.send_at


async def test_every_text_and_file_of_a_first_send_gets_the_slot(world: World) -> None:
    await send(world, await person(world), ["Здравствуйте! Есть дрель?"])
    conv_id = await person(world, peer_id=502, username="drill_shop")

    result = await send(
        world, conv_id, ["Здравствуйте!", "Есть дрель?"], attach_files=True
    )

    assert isinstance(result, Queued)
    items = await world.repo.outbox_of(conv_id)
    assert [item.attach_path for item in items] == [None, None, FILE]
    assert {item.not_before for item in items} == {result.send_at}


async def test_a_reply_to_a_person_written_to_is_not_delayed_by_new_ones(
    world: World,
) -> None:
    known = await person(world, S.CONTACTED, peer_id=503, username="old_shop")
    await sent_before(world, known)
    await incoming(world, known, "Есть, 450 000")
    for peer in (501, 502):
        new = await person(world, peer_id=peer, username=f"shop_{peer}")
        await send(world, new, ["Здравствуйте! Есть дрель?"])
    now = world.clock.now()

    result = await send(world, known, ["А со скидкой?"])

    assert isinstance(result, Queued) and result.first_contact is False
    assert now + timedelta(seconds=60) <= result.send_at <= now + timedelta(seconds=180)


async def test_a_second_send_to_the_same_new_person_keeps_his_place(
    world: World,
) -> None:
    conv_id = await person(world)
    first = await send(world, conv_id, ["Здравствуйте! Есть дрель?"])
    world.clock.advance(timedelta(seconds=30))

    second = await send(world, conv_id, ["Здравствуйте! Есть дрель Bosch?"])

    assert isinstance(first, Queued) and isinstance(second, Queued)
    assert second.send_at <= first.send_at + timedelta(seconds=180)
    items = await world.repo.outbox_of(conv_id)
    assert [item.status for item in items] == [
        OutboxStatus.CANCELLED,
        OutboxStatus.PENDING,
    ]


async def test_a_re_send_keeps_the_new_person_s_place_before_a_later_one(
    world: World,
) -> None:
    """Fix pass C10: A's text is replaced while B waits behind him."""
    a = await person(world)
    b = await person(world, peer_id=502, username="drill_shop")
    one = await send(world, a, ["Здравствуйте! Есть дрель?"])
    two = await send(world, b, ["Здравствуйте! Есть дрель?"])
    # His slot (3-8 s, ruling R70) still waits.
    world.clock.advance(timedelta(seconds=2))

    again = await send(world, a, ["Здравствуйте! Есть дрель Bosch?"])

    assert isinstance(one, Queued) and isinstance(two, Queued)
    assert isinstance(again, Queued) and again.first_contact is True
    assert again.send_at == one.send_at < two.send_at
    [pending] = [
        i for i in await world.repo.outbox_of(a) if i.status.value == "pending"
    ]
    assert pending.not_before == one.send_at


async def test_a_re_send_whose_old_slot_is_past_gets_a_new_one(world: World) -> None:
    a = await person(world)
    b = await person(world, peer_id=502, username="drill_shop")
    one = await send(world, a, ["Здравствуйте! Есть дрель?"])
    two = await send(world, b, ["Здравствуйте! Есть дрель?"])
    assert isinstance(one, Queued) and isinstance(two, Queued)
    world.clock.set(one.send_at + timedelta(seconds=5))  # his slot is past (a stop)

    again = await send(world, a, ["Здравствуйте! Есть дрель Bosch?"])

    assert isinstance(again, Queued)
    assert again.send_at >= two.send_at + timedelta(minutes=1)


async def test_a_reply_to_a_person_who_wrote_first_is_not_paced(world: World) -> None:
    """Fix pass C12: he wrote to us first; he is no stranger to the account."""
    for peer in (601, 602, 603):  # the default quota: 3 new people a day
        await send(
            world,
            await person(world, peer_id=peer, username=f"shop_{peer}"),
            ["Здравствуйте! Есть дрель?"],
        )
    conv_id = await person(world)
    await incoming(world, conv_id, "Здравствуйте, вы искали дрель?")
    now = world.clock.now()

    result = await send(world, conv_id, ["Да, есть Bosch?"])

    assert isinstance(result, Queued) and result.first_contact is False
    assert now + timedelta(seconds=60) <= result.send_at <= now + timedelta(seconds=180)


async def test_two_first_sends_at_once_get_slots_a_minute_apart(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fix pass C13: both read the waiting list before either queued its message."""
    first = await person(world)
    second = await person(world, peer_id=502, username="drill_shop")
    read = world.repo.waiting_first_contacts

    async def slow_read(exclude: int | None = None) -> list[datetime]:
        found = await read(exclude)
        await asyncio.sleep(0.05)
        return found

    monkeypatch.setattr(world.repo, "waiting_first_contacts", slow_read)

    one, two = await asyncio.gather(
        send(world, first, ["Здравствуйте! Есть дрель?"]),
        send(world, second, ["Здравствуйте! Есть дрель?"]),
    )

    assert isinstance(one, Queued) and isinstance(two, Queued)
    assert abs(two.send_at - one.send_at) >= timedelta(minutes=1)


# Ruling R70: the 1-3 minute pause is for answering a person who waits, nothing else.


async def test_an_answer_to_a_fresh_message_waits_one_to_three_minutes(
    world: World,
) -> None:
    conv_id = await person(world, S.CONTACTED)
    await sent_before(world, conv_id)
    await incoming(world, conv_id, "Есть, 450 000")
    now = world.clock.now()

    result = await send(world, conv_id, ["А со скидкой?"])

    assert isinstance(result, Queued)
    assert now + timedelta(seconds=60) <= result.send_at <= now + timedelta(seconds=180)


async def test_a_reminder_to_one_who_is_silent_goes_at_once(world: World) -> None:
    conv_id = await person(world, S.CONTACTED)
    await sent_before(world, conv_id)
    now = world.clock.now()

    result = await send(world, conv_id, ["Подскажите, пожалуйста?"])

    assert isinstance(result, Queued)
    assert now + timedelta(seconds=3) <= result.send_at <= now + timedelta(seconds=8)


async def test_a_message_after_the_owner_s_answer_goes_at_once(world: World) -> None:
    # He wrote, we answered; now the owner's answer brings more: nobody waits for it.
    conv_id = await person(world, S.CONTACTED)
    await incoming(world, conv_id, "Есть, 450 000")
    await sent_before(world, conv_id, "Уточню у заказчика")
    now = world.clock.now()

    result = await send(world, conv_id, ["Заказчик спрашивает, есть ли доставка?"])

    assert isinstance(result, Queued)
    assert now + timedelta(seconds=3) <= result.send_at <= now + timedelta(seconds=8)


async def test_with_the_quota_used_a_first_send_goes_tomorrow(world: World) -> None:
    for peer in (601, 602, 603):  # the default quota: 3 new people a day
        await send(
            world,
            await person(world, peer_id=peer, username=f"shop_{peer}"),
            ["Здравствуйте! Есть дрель?"],
        )
    conv_id = await person(world)

    result = await send(world, conv_id, ["Здравствуйте! Есть дрель?"])

    assert isinstance(result, Queued)
    assert result.send_at == datetime(2026, 11, 11, 9, 0, tzinfo=TASHKENT)


async def test_a_newer_reply_replaces_one_that_has_not_gone(world: World) -> None:
    conv_id = await person(world, S.CONTACTED)
    await sent_before(world, conv_id)
    await incoming(world, conv_id, "Есть, 450 000")
    await world.repo.set_state(conv_id, S.TALKING)
    await send(world, conv_id, ["А со скидкой?"])

    result = await send(world, conv_id, ["А со скидкой?", "Доставка есть?"])

    assert isinstance(result, Queued)
    assert (result.count, result.first_contact) == (2, False)
    items = await world.repo.outbox_of(conv_id)
    assert [(item.text, item.status) for item in items[1:]] == [
        ("А со скидкой?", OutboxStatus.CANCELLED),
        ("А со скидкой?", OutboxStatus.PENDING),
        ("Доставка есть?", OutboxStatus.PENDING),
    ]
    conv = await world.repo.get_conversation(conv_id)
    assert conv.state is S.TALKING
    assert (conv.agent_messages, conv.followups_sent) == (3, 0)


async def test_the_agent_s_own_line_does_not_replace_the_owner_s_ok_reply(
    world: World,
) -> None:
    # The owner pressed «Ок»: his reply (queued by no wake) waits its human pause. A wake —
    # say, for another person of the task — must not replace it unseen (review 6b, D4).
    conv_id = await person(world, S.CONTACTED)
    await sent_before(world, conv_id)
    await incoming(world, conv_id, "Давайте в четверг в 11:00")
    await world.repo.set_state(conv_id, S.TALKING)
    reply = "Хорошо, тогда в четверг в 11:00"
    assert isinstance(await send(world, conv_id, [reply], owner_text=True), Queued)

    wake = await a_wake(world)
    result = await refused(world, conv_id, ["Спасибо, жду"], wake_id=wake)

    assert result == Refused(OWNER_REPLY_WAITS)
    waiting = [
        item.text
        for item in await world.repo.outbox_of(conv_id)
        if item.status is OutboxStatus.PENDING
    ]
    assert waiting == [reply]


async def test_the_task_files_are_queued_once(world: World) -> None:
    conv_id = await person(world)

    result = await send(world, conv_id, ["Здравствуйте! Вот чертёж"], attach_files=True)

    assert isinstance(result, Queued) and result.count == 1
    text, file = await world.repo.outbox_of(conv_id)
    assert (file.text, file.attach_path, file.not_before) == ("", FILE, result.send_at)
    await world.repo.mark_outbox(text.id, OutboxStatus.SENT, at=world.clock.now())
    await world.repo.mark_outbox(file.id, OutboxStatus.SENT, at=world.clock.now())
    await world.repo.update_conversation(conv_id, first_sent=True)

    await incoming(world, conv_id, "Получил")
    await send(world, conv_id, ["Сможете сделать?"], attach_files=True)

    items = await world.repo.outbox_of(conv_id)
    assert [item.attach_path for item in items] == [None, FILE, None]


async def test_a_reminder_is_counted(world: World) -> None:
    conv_id = await person(world, S.CONTACTED)
    await sent_before(world, conv_id)

    queued = await send(world, conv_id, ["Напомню: есть дрель?"])
    assert isinstance(queued, Queued)
    assert (await world.repo.get_conversation(conv_id)).followups_sent == 1


async def test_a_commitment_passes_after_the_owner_approved_the_order(
    world: World,
) -> None:
    conv_id = await person(world)
    await incoming(world, conv_id, "Дрель 600 000")
    await approve(
        world, conv_id, ApprovalKind.CONFIRM_ORDER, {"approved_amount": 600_000}
    )

    over = await refused(world, conv_id, ["Берём за 700 000"], commitment=True)
    assert over == Refused(REASONS["over_ceiling"], ApprovalKind.CONFIRM_ORDER)
    result = await send(world, conv_id, ["Берём за 600 000"], commitment=True)
    assert isinstance(result, Queued)


async def test_the_owners_phone_passes_after_he_allowed_it(world: World) -> None:
    conv_id = await person(world)
    await incoming(world, conv_id, "Ваш номер?")
    await approve(world, conv_id, ApprovalKind.SHARE_PHONE)

    assert isinstance(await send(world, conv_id, ["Звоните: {PHONE}"]), Queued)
    [item] = await world.repo.outbox_of(conv_id)
    assert "{PHONE}" in item.text  # the Sender fills it in


# -- fix round 1 --


class MeddlingLLM(FakeLLM):
    """A judge during whose call something else happens to the conversation."""

    def __init__(self, during: Callable[[], Awaitable[None]]) -> None:
        super().__init__()
        self.during = during

    async def judge(
        self, messages: list[str], allowed: list[str], hint: str
    ) -> LLMResult[JudgeVerdict]:
        await self.during()
        return await super().judge(messages, allowed, hint)


async def in_talk_with_a_reply_queued(world: World) -> tuple[int, int]:
    """A person who answered, and our older reply to him still waits in the outbox."""
    conv_id = await person(world, S.CONTACTED)
    await sent_before(world, conv_id)
    await incoming(world, conv_id, "Есть, 450 000")
    await world.repo.set_state(conv_id, S.TALKING)
    older = await world.repo.enqueue(conv_id, "Хорошо", world.clock.now())
    return conv_id, older


async def meddle(world: World, conv_id: int, what: str) -> None:
    repo = world.repo
    if what == "waiting":
        await repo.set_state(conv_id, S.WAITING_OWNER)
    elif what == "handed":
        await repo.set_state(conv_id, S.HANDED_TO_OWNER)
    elif what == "wrote":
        await repo.add_message(
            conv_id, Direction.IN, MsgSource.COUNTERPART, "text", "Алло?", None,
            world.clock.now(),
        )  # fmt: skip
    else:
        await repo.update_conversation(
            conv_id, paused_until=world.clock.now() + timedelta(minutes=30)
        )


@pytest.mark.parametrize(
    ("what", "reason"),
    [
        (
            "waiting",
            "по этому человеку ждём ответа владельца — пока ему ничего не пишем",
        ),
        ("handed", "этот чат у владельца: он пишет сам"),
        (
            "wrote",
            (
                "собеседник написал после начала этого пробуждения — ответ устарел;"
                " новое сообщение увидишь в следующем пробуждении"
            ),
        ),
        ("paused", "владелец сам пишет в этом чате — подожди"),
    ],
)
async def test_what_happens_while_the_judge_thinks_is_seen_before_queueing(
    world: World, what: str, reason: str
) -> None:
    conv_id, older = await in_talk_with_a_reply_queued(world)
    world.llm = MeddlingLLM(lambda: meddle(world, conv_id, what))
    world.outgoing = outgoing_of(world)

    result = await send(world, conv_id, ["А со скидкой?"])

    assert result == Refused(reason)
    [item] = [item for item in await world.repo.outbox_of(conv_id) if item.id >= older]
    assert (item.id, item.status) == (older, OutboxStatus.PENDING)


async def test_a_file_cancelled_with_a_replaced_reply_is_queued_again(
    world: World,
) -> None:
    conv_id = await person(world)
    await send(world, conv_id, ["Здравствуйте! Вот чертёж"], attach_files=True)
    await send(world, conv_id, ["Здравствуйте! Есть дрель?"])  # replaces both, no file

    await send(world, conv_id, ["Здравствуйте! Вот чертёж"], attach_files=True)

    files = [
        item
        for item in await world.repo.outbox_of(conv_id)
        if item.attach_path == FILE and item.status is OutboxStatus.PENDING
    ]
    assert len(files) == 1


async def test_a_username_counts_only_written_as_a_contact(world: World) -> None:
    conv_id = await person(world)
    await person(world, peer_id=502, username="tashkent")

    for text in [
        "Здравствуйте! Напишите @Tashkent",
        "Здравствуйте! Вот: t.me/TASHKENT",
    ]:
        assert await refused(world, conv_id, [text]) == Refused(
            "не называй одному собеседнику контакты другого"
        )
    result = await send(world, conv_id, ["Здравствуйте! Мы в Ташкенте, in Tashkent"])
    assert isinstance(result, Queued)


async def test_the_gates_see_the_introduction(world: World) -> None:
    text = "Здравствуйте! " + "а" * 380  # fits alone, not with the introduction
    conv_id = await person(world)
    assert await refused(world, conv_id, [text]) == Refused(REASONS["too_long"])

    world.gateway.name = "ИИ помощник Саида"  # no introduction now
    assert isinstance(await send(world, conv_id, [text]), Queued)


async def test_a_refused_send_keeps_the_older_pending_reply(world: World) -> None:
    conv_id, older = await in_talk_with_a_reply_queued(world)

    await refused(world, conv_id, ["Смотрите https://example.com"])

    assert (await world.repo.get_outbox(older)).status is OutboxStatus.PENDING


async def test_of_two_failures_the_earlier_step_is_told(world: World) -> None:
    conv_id = await person(world, S.CONTACTED)
    seen = await incoming(world, conv_id, "Есть")
    await incoming(world, conv_id, "Алло?")
    stale = {"seen_message_id": seen}

    assert await refused(world, conv_id, ["  "], **stale) == Refused(
        "собеседник написал после начала этого пробуждения — ответ устарел;"
        " новое сообщение увидишь в следующем пробуждении"
    )
    await world.repo.set_state(conv_id, S.CLOSED)
    assert await refused(world, conv_id, ["Хорошо"], **stale) == Refused(
        "С этим человеком уже закончили. Написать ему снова можно, когда об этом"
        " попросит владелец"
    )


async def test_an_empty_text_is_told_before_the_reminder_limit(world: World) -> None:
    conv_id = await person(world, S.CONTACTED)
    await sent_before(world, conv_id)
    await world.repo.update_conversation(conv_id, followups_sent=2)
    empty = await refused(world, conv_id, [""])
    assert empty == Refused("пустое сообщение")


# Fix pass B6: the language the agent writes in is remembered.


@pytest.mark.parametrize(
    ("language", "text", "kept"),
    [
        ("uz_latn", "Salom! Qora rang bormi?", "uz_latn"),
        ("ru", "Здравствуйте! Есть дрель?", "ru"),
        ("other", "Hello! Do you have a drill?", None),
    ],
)
async def test_the_language_of_a_queued_send_is_remembered(
    world: World, language: str, text: str, kept: str | None
) -> None:
    conv_id = await person(world)

    assert isinstance(await send(world, conv_id, [text], language=language), Queued)

    assert (await world.repo.get_conversation(conv_id)).language == kept


# Fix pass B9: the introduction is skipped only for a real admission.


@pytest.mark.parametrize(
    "first",
    [
        "Здравствуйте! Я ИИ, помогаю Саиду. Есть дрель?",
        "Здравствуйте, я бот Саида. Есть дрель?",
        "Добрый день! Я виртуальный помощник. Есть дрель?",
        "Salom! Men sun'iy intellekt yordamchisiman. Drel bormi?",
        "Salom, men botman. Drel bormi?",
        # Fix pass C2: the subject directly followed by the predicate, a dash at most.
        "Здравствуйте! Я робот, пишу за Саида. Есть дрель?",
        "Здравствуйте! Я нейросеть, помогаю Саиду. Есть дрель?",
        "Здравствуйте! Я - искусственный интеллект Саида. Есть дрель?",
        "Hello! Я AI-ассистент Саида. Есть дрель?",
        "Здравствуйте! Это ИИ-помощник Саида. Есть дрель?",
        "Здравствуйте! С вами виртуальный помощник Саида. Есть дрель?",
        "Здравствуйте! Вам пишет бот Саида. Есть дрель?",
        "Salom! Men suniy intellekt yordamchisiman. Drel bormi?",
        "Salom! Bu sun'iy intellekt yordamchisi. Drel bormi?",
        "Salom! Men – sun’iy intellekt. Drel bormi?",
    ],
)
def test_an_admission_in_other_words_still_gets_the_program_s_phrase(first: str) -> None:
    # Ruling R82: one rule — the program's own phrase, unless that very phrase is there.
    [text] = with_own_intro([first], "ru", introduce=True)
    assert INTRO in text


@pytest.mark.parametrize(
    "messages",
    [
        ["Ассаляму алейкум!", "Подскажите, вы делаете чат-ботов с ИИ?"],
        ["Ассаляму алейкум! Вы делаете чат-ботов с ИИ?"],
        ["Ассаляму алейкум! Есть дрель?", "Кстати, я ИИ-помощник"],
        # Fix pass C2: a word between, or a noun that is not the predicate.
        ["Здравствуйте! Я ищу чат-бота для магазина"],
        ["Добрый день, я ищу ИИ-ассистента для магазина"],
        ["Здравствуйте, я про виртуальную АТС"],
        ["Здравствуйте! Вы делаете ботов?"],
        ["Здравствуйте! Я ботаник, ищу микроскоп"],
        ["Здравствуйте! Это робот-пылесос Xiaomi?"],
        ["Здравствуйте! Я бота ищу для магазина"],
        ["Salom! Men do'konim uchun bot qidiryapman"],
    ],
)
def test_a_mere_mention_of_an_ai_gets_the_introduction(messages: list[str]) -> None:
    texts = with_own_intro(messages, "ru", introduce=True)
    assert INTRO in texts[0]
    assert texts[1:] == messages[1:]


# Fix pass D1.5: the program's own sentences, and «ИИ-помощник Саида» in the first person or
# after «Вас беспокоит» / «Вам пишет», are admissions: the introduction is not said twice.


@pytest.mark.parametrize(
    ("first", "language"),
    [
        ("Ассаляму алейкум! Я ИИ-помощник Саида. Есть дрель?", "ru"),
        ("Я ИИ-помощник Саида. Подскажите, есть дрель?", "ru"),
        (
            "Assalomu alaykum! Men Saidning sun'iy intellekt yordamchisiman. Drel bormi?",
            "uz_latn",
        ),
        ("Salom! Men Saidning sun’iy intellekt yordamchisiman, drel bormi?", "uz_latn"),
        ("Здравствуйте! Я, ИИ-помощник Саида, хотел спросить: есть дрель?", "ru"),
        ("Здравствуйте! Я — ИИ-помощник Саида. Есть дрель?", "ru"),
        ("ЗДРАВСТВУЙТЕ! я ии помощник саида. Есть дрель?", "ru"),
    ],
)
def test_the_program_s_own_introduction_written_by_the_model_is_not_doubled(
    first: str, language: str
) -> None:
    assert with_own_intro([first], language, introduce=True) == [first]


@pytest.mark.parametrize(
    "first",
    [
        "Здравствуйте! Я помощник Саида, есть дрель?",
        "Здравствуйте! Вас беспокоит помощник Саида. Есть дрель?",
        "Здравствуйте! Саид ищет ИИ-помощника для магазина",
        # not the very phrase, though they say the same
        "Здравствуйте! Вас беспокоит ИИ-помощник Саида, подскажите, есть дрель?",
        "Здравствуйте! Вам пишет ИИ-помощник Саида. Есть дрель?",
        "Добрый день, я его ИИ помощник Саида. Есть дрель?",
    ],
)
def test_a_helper_without_the_ai_still_gets_the_program_s_sentence(first: str) -> None:
    [text] = with_own_intro([first], "ru", introduce=True)
    assert INTRO in text


async def test_the_pipeline_introduces_a_first_message_that_only_mentions_an_ai(
    world: World,
) -> None:
    conv_id = await person(world)

    queued = await send(
        world, conv_id, ["Ассаляму алейкум! Вы делаете чат-ботов с ИИ?"]
    )

    assert isinstance(queued, Queued)
    [item] = await world.repo.outbox_of(conv_id)
    assert INTRO in item.text


# Fix pass C1 (corrects B10): a second message after ours is a reminder in any wake; only
# relaying the owner's word is not one.


async def test_a_second_message_after_ours_is_capped_in_any_wake(world: World) -> None:
    conv_id = await person(world, S.CONTACTED)
    await sent_before(world, conv_id)
    await world.repo.update_conversation(conv_id, followups_sent=2)

    result = await refused(world, conv_id, ["Владелец ответил: чёрный"])
    assert result == Refused(
        "напоминаний без ответа уже 2 — закончи с этим человеком или спроси владельца"
    )


async def test_a_relay_of_the_owner_s_word_is_no_reminder_even_at_the_cap(
    world: World,
) -> None:
    conv_id = await person(world, S.CONTACTED)
    await sent_before(world, conv_id)
    await world.repo.update_conversation(conv_id, followups_sent=2)

    queued = await send(world, conv_id, ["Владелец ответил: чёрный"], relay=True)

    assert isinstance(queued, Queued)
    assert (await world.repo.get_conversation(conv_id)).followups_sent == 2


# Fix pass B16: a person who wrote first is in talk.


async def test_the_first_message_to_a_person_who_wrote_first_puts_him_in_talk(
    world: World,
) -> None:
    conv_id = await person(world)
    await incoming(world, conv_id, "Здравствуйте, вы искали дрель?")

    assert isinstance(await send(world, conv_id, ["Да, есть Bosch?"]), Queued)

    assert (await world.repo.get_conversation(conv_id)).state is S.TALKING


async def test_the_first_message_to_a_person_who_never_wrote_makes_him_contacted(
    world: World,
) -> None:
    conv_id = await person(world)

    assert isinstance(await send(world, conv_id, ["Здравствуйте! Есть дрель?"]), Queued)

    assert (await world.repo.get_conversation(conv_id)).state is S.CONTACTED


# Fix pass C7: words that commit, whatever the model's flag says.

AGREEING = [
    "Хорошо, берём за 590 000",
    "Да, давайте за 590 000",
    "Согласен на 590 000",
    "Ладно, беру за 590 000 сум",
    "Договорились на 590 000",
    "Bo'pti, olamiz 590 000 so'mga",
    "Kelishdik, 590 000",
]


@pytest.mark.parametrize("text", AGREEING)
async def test_agreeing_to_a_sum_is_a_commitment_without_the_flag(
    world: World, text: str
) -> None:
    conv_id = await person(world)
    await incoming(world, conv_id, "Отдам за 590 000")

    result = await refused(world, conv_id, [text])

    assert result == Refused(REASONS["commitment"], ApprovalKind.CONFIRM_ORDER)


@pytest.mark.parametrize("text", AGREEING)
async def test_agreeing_above_the_approved_sum_is_refused_without_the_flag(
    world: World, text: str
) -> None:
    conv_id = await person(world)
    await incoming(world, conv_id, "Отдам за 590 000")
    await approve(
        world, conv_id, ApprovalKind.CONFIRM_ORDER, {"approved_amount": 550_000}
    )

    result = await refused(world, conv_id, [text])

    assert result == Refused(REASONS["over_ceiling"], ApprovalKind.CONFIRM_ORDER)


@pytest.mark.parametrize("separator", ["'", "’"])
async def test_an_apostrophe_between_digits_does_not_split_a_number(
    world: World, separator: str
) -> None:
    conv_id = await person(world)
    await incoming(world, conv_id, "Отдам за 590 000")
    await approve(
        world, conv_id, ApprovalKind.CONFIRM_ORDER, {"approved_amount": 550_000}
    )

    result = await refused(
        world, conv_id, [f"Подтверждаю 550 000, итого 590{separator}000"]
    )

    assert result == Refused(REASONS["over_ceiling"], ApprovalKind.CONFIRM_ORDER)


@pytest.mark.parametrize(
    "text", ["Сколько будет за 590 000?", "Берёте ли вы оплату картой?"]
)
async def test_a_question_about_a_sum_is_no_commitment(world: World, text: str) -> None:
    conv_id = await person(world)
    await incoming(world, conv_id, "Отдам за 590 000")

    assert isinstance(await send(world, conv_id, [text]), Queued)


# The gate's language: what the person writes decides, never what we wrote before.


async def test_an_uzbek_answer_to_a_russian_opening_is_answered_in_uzbek(
    world: World,
) -> None:
    """The first message goes out in Russian before the person has said a word; when he
    answers in Uzbek (Latin letters carry no clue for the code), the agent's declared
    language holds, not the remembered language of our own opening."""
    conv_id = await person(world)
    opening = await send(world, conv_id, ["Здравствуйте! Есть дрель?"], language="ru")
    assert isinstance(opening, Queued)
    assert (await world.repo.get_conversation(conv_id)).language == "ru"
    await incoming(world, conv_id, "Salom! Ha, bor. Narxi 500 ming so'm")

    reply = await send(
        world, conv_id, ["Rahmat! Yetkazib berish bormi?"], language="uz_latn"
    )

    assert isinstance(reply, Queued), reply


async def test_a_russian_answer_after_an_uzbek_opening_is_answered_in_russian(
    world: World,
) -> None:
    conv_id = await person(world)
    opening = await send(world, conv_id, ["Salom! Drel bormi?"], language="uz_latn")
    assert isinstance(opening, Queued)
    await incoming(world, conv_id, "Да, есть")  # Cyrillic, no decisive letter

    reply = await send(world, conv_id, ["Спасибо! А доставка есть?"], language="ru")

    assert isinstance(reply, Queued), reply


# -- the order path after the owner's «Да» (paid run of order_after_button, 2026-11-11) --


async def test_the_judge_is_told_every_order_the_owner_approved(world: World) -> None:
    # A second card (a prepayment of 120 000) must not narrow what the first one allowed:
    # the judge was told «до 120 000» and refused the confirmation of the 240 000 order.
    world.llm = llm = HintLLM()
    world.outgoing = outgoing_of(world)
    conv_id = await person(world)
    await approve(
        world,
        conv_id,
        ApprovalKind.CONFIRM_ORDER,
        {
            "approved_amount": 240_000,
            "terms": "срок 3 дня",
            "text": "Оформляем заказ за 240 000?",
        },
    )
    await approve(
        world,
        conv_id,
        ApprovalKind.CONFIRM_ORDER,
        {
            "approved_amount": 120_000,
            "terms": "предоплата 50%",
            "text": "Подтвердить предоплату\n120 000?",
        },
    )

    await send(
        world,
        conv_id,
        ["Здравствуйте! Подтверждаем заказ: 240 000, срок 3 дня"],
        commitment=True,
    )

    [(_, _, hint)] = llm.judged
    assert hint.splitlines()[3:6] == [
        "Заказ: владелец разрешил кнопкой — подтверждать можно в этих пределах:",
        (
            "• сумма до 240 000; условия: срок 3 дня;"
            " на карточке было: «Оформляем заказ за 240 000?»"
        ),
        (
            "• сумма до 120 000; условия: предоплата 50%;"
            " на карточке было: «Подтвердить предоплату 120 000?»"
        ),
    ]


def test_the_judge_s_rules_take_the_owner_s_button_for_a_permission() -> None:
    # The rules named «Саид согласен» as a problem with no condition, so the judge refused
    # the confirmation right after the owner's «Да».
    rules = load_prompt("judge")
    assert "владелец разрешил кнопкой" in rules
    assert "на которое в подсказке нет разрешения" in rules


def test_the_judge_s_rules_do_not_take_the_counterpart_s_words_for_an_invention() -> (
    None
):
    # The judge never sees what the counterpart wrote, so «Понял, с 9 до 18. Спасибо!» read
    # to it as a made-up fact and the thanks never went (paid run of 2026-11-11, asks_why).
    rules = load_prompt("judge")
    assert "Слова собеседника тебе не показаны" in rules
    assert "повторяет или подтверждает" in rules


def test_the_judge_s_rules_let_the_agent_say_by_whose_request_it_writes() -> None:
    # Second paid run of asks_why: «Саид попросил уточнить, поэтому и спросил» — the honest
    # answer the agent's rules ask for — was refused as a made-up fact about the owner.
    rules = load_prompt("judge")
    assert "по просьбе Саида" in rules
    assert "это сама задача" in rules


async def test_a_text_the_owner_approved_is_not_refused_for_what_it_promises(
    world: World,
) -> None:
    # «Ок» under a suggested reply: he read this very text, so what it promises or says
    # about him is his own word.
    conv_id = await person(world)
    world.llm.verdicts.append(
        JudgeVerdict(ok=False, issues=["commitment", "invented_fact"], note="")
    )
    result = await send(
        world,
        conv_id,
        ["Здравствуйте! Саид пришлёт файл сам и напишет вам"],
        owner_text=True,
    )
    assert isinstance(result, Queued)


async def test_a_text_the_owner_approved_is_still_refused_for_anything_else(
    world: World,
) -> None:
    conv_id = await person(world)
    world.llm.verdicts.append(
        JudgeVerdict(ok=False, issues=["commitment", "leak"], note="")
    )
    assert await refused(
        world, conv_id, ["Здравствуйте! Саид пришлёт файл сам"], owner_text=True
    ) == Refused(REASONS["judge_leak"])
    # and a judge that objects without naming an issue still stops it
    world.llm.verdicts.append(JudgeVerdict(ok=False, issues=[], note=""))
    assert await refused(
        world, conv_id, ["Здравствуйте! Саид пришлёт файл сам"], owner_text=True
    ) == Refused(REASONS["judge"])


CALL_AGREED = "Договорились, созвонимся в 16:00"


@pytest.mark.parametrize(
    "reply",
    [
        CALL_AGREED,
        "Оплачу 240 000 при получении",
        "Переведу предоплату 500 000 сегодня",
        "Оформляйте заказ, оплачу при получении",
    ],
)
async def test_commitment_words_the_owner_pressed_ok_on_still_need_the_order_card(
    world: World, reply: str
) -> None:
    # Fix round A (H): «Ок» on a question card is no order card and checks no ceiling.
    conv_id = await person(world)
    assert await refused(world, conv_id, [reply], owner_text=True) == Refused(
        REASONS["commitment"], ApprovalKind.CONFIRM_ORDER
    )


async def test_the_same_words_written_by_the_agent_are_still_refused(
    world: World,
) -> None:
    conv_id = await person(world)
    assert await refused(world, conv_id, [CALL_AGREED]) == Refused(
        REASONS["commitment"], ApprovalKind.CONFIRM_ORDER
    )


async def test_an_accepted_reply_that_agrees_to_a_sum_still_needs_the_order_card(
    world: World,
) -> None:
    # Only the bare words pass: a sum with them is an order, and an order needs its card.
    conv_id = await person(world)
    assert await refused(
        world, conv_id, ["Договорились, берём за 240 000"], owner_text=True
    ) == Refused(REASONS["commitment"], ApprovalKind.CONFIRM_ORDER)


async def test_the_model_s_own_text_is_still_refused_for_a_promise(
    world: World,
) -> None:
    conv_id = await person(world)
    world.llm.verdicts.append(JudgeVerdict(ok=False, issues=["commitment"], note=""))
    assert await refused(
        world, conv_id, ["Здравствуйте! Саид пришлёт файл сам"]
    ) == Refused(REASONS["judge_commitment"])


def test_the_judge_s_rules_let_a_call_time_inside_the_window_through() -> None:
    # Ruling R74: «Хорошо, тогда в 16:00» inside «завтра после 15:00» is no promise.
    rules = load_prompt("judge")
    assert "Время звонка или встречи, которое укладывается в то, что разрешают" in rules
    assert "не commitment и не invented_fact" in rules
