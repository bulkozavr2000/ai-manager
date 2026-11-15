"""Fix round B2 of stage 6: the owner's address never enters the memory (ruling R91), and
«не работай с @x» also ends a talk already going (ruling R92)."""

from __future__ import annotations

from datetime import timedelta
from random import Random

import pytest
from agent_world import END, AgentWorld, call, send
from test_bot_flow import Program, say

from tg_agent.agent import memory, outgoing, people
from tg_agent.agent.ops import TaskOps
from tg_agent.brain.llm import FakeLLM
from tg_agent.clock import FakeClock
from tg_agent.config import Settings
from tg_agent.db import Repo
from tg_agent.models import ConvState
from tg_agent.secrets import Key, set_secret


@pytest.fixture
def program(repo: Repo, fake_clock: FakeClock, settings: Settings) -> Program:
    return Program(repo, fake_clock, settings, FakeLLM())


@pytest.fixture
def world(repo: Repo, fake_clock: FakeClock, settings: Settings) -> AgentWorld:
    return AgentWorld(repo, fake_clock, settings)


PHONE = "+998907770011"


def tool_results(program: Program) -> list[str]:
    """What the secretary's tools answered its model, in order."""
    found: list[str] = []
    for kind, request in program.llm.requests:
        if kind != "secretary":
            continue
        for message in request["messages"]:
            if message["role"] != "user" or isinstance(message["content"], str):
                continue
            for item in message["content"]:
                if item.get("type") == "tool_result" and item["content"] not in found:
                    found.append(item["content"])
    return found


# -- R91 --


async def test_his_address_said_in_words_is_not_remembered_and_the_model_is_told_why(
    program: Program,
) -> None:
    set_secret(Key.OWNER_ADDRESS, "Чиланзар 5, дом 12, кв. 30")
    words = "запомни: доставка всегда на Чиланзар 5, дом 12, кв. 30"
    program.secretary(
        [call("remember", fact="Доставка всегда на Чиланзар 5, дом 12, кв. 30",
              quote=words, exclude_contact=None)],
        say("Адрес я не запоминаю."),
    )  # fmt: skip

    await program.write(words)

    assert await memory.load(program.repo) == []
    assert memory.OWNER_DATA in tool_results(program)


# -- R92 --


async def test_excluding_a_person_mid_talk_closes_his_talk_and_cancels_the_queue(
    program: Program,
) -> None:
    await program.talking("Есть, 240 000 сум, 3 дня")
    program.script([send(program.person, "Спасибо, уточню и вернусь")], END)
    await program.round(timedelta(minutes=1))  # the reply waits in the queue
    queued = len(program.sent())
    words = "не работай больше с @print_lab"
    program.secretary(
        [call("remember", fact="не работать с @print_lab", quote=words,
              exclude_contact="@print_lab")],
        say("Запомнил: не работать с @print_lab."),
    )  # fmt: skip

    await program.write(words)

    conv = await program.repo.get_conversation(program.person)
    assert conv.state is ConvState.CLOSED
    assert conv.close_reason == memory.EXCLUDED
    for _ in range(6):
        await program.round(timedelta(minutes=10))
    assert len(program.sent()) == queued  # neither the queued reply nor a goodbye
    assert memory.EXCLUDED in program.board()


async def test_the_send_door_refuses_an_excluded_person(world: AgentWorld) -> None:
    _, person = await world.replied("Есть, 500 000")
    await memory.remember(world.repo, "не работать", person.username, world.clock.now())
    block = send(person.conv_id, "Спасибо, уточню и вернусь")
    world.script([block], END)

    await world.round(timedelta(minutes=2))

    assert world.result(block)["is_error"] is True
    assert outgoing.EXCLUDED in world.result(block)["content"]
    for _ in range(4):
        await world.deliver()
    assert "Спасибо, уточню и вернусь" not in world.sent_to(person)


async def test_excluded_by_phone_he_is_refused_by_the_phone_his_record_holds(
    world: AgentWorld,
) -> None:
    _, person = await world.replied("Есть, 500 000")
    conv = await world.repo.get_conversation(person.conv_id)
    contact = await world.repo.get_contact(conv.contact_id)
    await world.repo.update_contact_card(  # as `add_person` keeps the contact he came by
        contact.id, {**contact.card, people.CONTACT_KEY: PHONE}, contact.language
    )
    await memory.remember(world.repo, "не работать", PHONE, world.clock.now())
    block = send(person.conv_id, "Спасибо, уточню и вернусь")
    world.script([block], END)

    await world.round(timedelta(minutes=2))

    assert outgoing.EXCLUDED in world.result(block)["content"]
    # …and no task attaches him by his @username either.
    task_id, _ = await world.task(start=False)
    refused = await people.attach(world.repo, world.gateway, task_id, person.username)
    assert refused == people.NotAttached(memory.NOT_WITH)


async def test_the_exclusion_closes_only_his_open_talks(world: AgentWorld) -> None:
    task_id, [bad, good] = await world.contacted("bad_shop", "good_shop")
    for person in (bad, good):
        await world.repo.enqueue(person.conv_id, "Ещё вопрос", world.clock.now())
    ops = TaskOps(
        world.repo, world.gateway, world.clock, world.settings, world.runner, Random(1)
    )

    assert await ops.exclude("bad_shop") == 1

    assert (await world.repo.get_conversation(bad.conv_id)).state is ConvState.CLOSED
    assert (await world.repo.get_conversation(good.conv_id)).state is not ConvState.CLOSED
    for _ in range(4):
        await world.deliver()
    assert "Ещё вопрос" not in world.sent_to(bad)
    assert "Ещё вопрос" in world.sent_to(good)


# -- fix round C of stage 6: R97 (a) — the exclusion empties every queue to him --


async def test_a_goodbye_queued_before_the_exclusion_does_not_go(
    world: AgentWorld,
) -> None:
    """The owner closes the task (the goodbye waits its human pause), then says «не работай
    больше с @x»: his conversation is CLOSED already, and its queue is cancelled too."""
    task_id, person = await world.replied("Есть, 500 000")
    ops = TaskOps(
        world.repo, world.gateway, world.clock, world.settings, world.runner, Random(1)
    )
    await ops.close(task_id)
    assert await world.repo.outbox_of(person.conv_id)  # the goodbye waits
    await memory.remember(world.repo, "не работать с ним", person.username, world.clock.now())
    before = len(world.sent_to(person))

    assert await ops.exclude(person.username) == 0  # nothing open to close…

    for _ in range(4):
        await world.deliver()
    assert len(world.sent_to(person)) == before  # …and the goodbye never goes


# -- fix round C of stage 6: R102 — the number in the fact does not undo the exclusion --


async def test_an_exclusion_by_phone_with_the_number_in_the_fact_is_kept(
    program: Program,
) -> None:
    words = "не работай больше с +998 90 777 00 11, не отвечает на звонки"
    program.secretary(
        [call("remember", fact="не работать с +998 90 777 00 11: не отвечает на звонки",
              quote=words, exclude_contact="+998 90 777 00 11")],
        say("Запомнил."),
    )  # fmt: skip

    await program.write(words)

    assert await memory.excluded(program.repo) == {PHONE}
    assert memory.PRIVATE not in tool_results(program)
