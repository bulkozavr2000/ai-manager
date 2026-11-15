"""Stage 9.2, ruling R104: the agent talks to people only while their task is in work.

The whole program on fakes (`test_bot_flow.Program`). (a) After the task ends — «Завершить»,
«Закрыть» — its people are not listened to at all: nothing is stored, nothing is queued for
them, the owner gets no card. (b) In an ACTIVE task a person whose talk is closed and who
writes reopens the talk, whatever closed it, and the agent has his message as any incoming
one — except where the owner's own words or the kept guards hold him shut.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from agent_world import END, call
from test_bot_flow import PEER, Program, draft

from tg_agent.agent import memory, people
from tg_agent.agent.people import listened
from tg_agent.agent.ops import TASK_CLOSED, TASK_DONE
from tg_agent.agent.runner import GAVE_UP
from tg_agent.brain.llm import FakeLLM
from tg_agent.clock import FakeClock
from tg_agent.config import Settings
from tg_agent.db import Repo
from tg_agent.models import (
    ApprovalKind,
    ApprovalStatus,
    ConvState,
    EventKind,
    TaskStatus,
)
from tg_agent.tasks.recovery import replay_missed, replay_starts


@pytest.fixture
def program(repo: Repo, fake_clock: FakeClock, settings: Settings) -> Program:
    return Program(repo, fake_clock, settings, FakeLLM())


async def pass_time(program: Program, minutes: int = 10) -> None:
    for _ in range(minutes):
        await program.round(timedelta(minutes=1))


async def incoming_events(program: Program) -> int:
    rows = await program.repo._all(
        "SELECT id FROM agent_events WHERE kind = ? AND conversation_id = ?",
        (EventKind.INCOMING.value, program.person),
    )
    return len(rows)


async def stored(program: Program) -> list[str]:
    return [m.text for m in await program.repo.last_messages(program.person, 50)]


async def late_kv(program: Program) -> list[str]:
    rows = await program.repo._all("SELECT key FROM kv WHERE key GLOB 'late_*'")
    return [row["key"] for row in rows]


async def nothing_happens(program: Program) -> None:
    """The person writes twice after the end; nothing about it anywhere."""
    assert PEER not in program.telegram.allowed
    assert PEER not in await listened(program.repo)
    messages, cards = await stored(program), len(program.cards())
    queued = len(await program.repo.outbox_of(program.person))
    before, wakes = len(program.sent()), await program.wakes()

    await program.person_writes("Кто это?")
    program.clock.advance(timedelta(hours=1))
    await program.person_writes("Ещё актуально?")
    await pass_time(program)

    assert await stored(program) == messages  # nothing stored
    assert len(await program.repo.outbox_of(program.person)) == queued  # nothing queued
    assert program.sent()[before:] == []
    assert len(program.cards()) == cards  # no card for the owner
    assert await late_kv(program) == []
    assert await program.wakes() == wakes


async def test_after_zavershit_its_people_are_not_heard(program: Program) -> None:
    await program.talking("Есть, 240 000 сум")
    program.script([call("finish", result="PrintLab: 240 000 сум.")])
    await program.round(timedelta(minutes=2))
    await program.app.ops.acknowledge(program.task_id)  # «Завершить»

    await nothing_happens(program)


async def test_after_zakryt_its_people_are_not_heard(program: Program) -> None:
    await program.contacted()
    await program.app.ops.close(program.task_id)  # «Закрыть»

    await nothing_happens(program)


async def test_a_done_task_before_zavershit_does_not_hear_them_either(
    program: Program,
) -> None:
    # A task the owner ended (R109); if he continues it by answering its result (R104 c),
    # the person's words of meanwhile are read then, from the chat (`ops._catch_up`).
    await program.contacted()
    await program.repo.set_task_status(program.task_id, TaskStatus.DONE)
    messages, cards = await stored(program), len(program.cards())

    await program.person_writes("Кто это?")
    await pass_time(program)

    assert await stored(program) == messages
    assert len(program.cards()) == cards


@pytest.mark.parametrize(
    "reason",
    [GAVE_UP, "просил больше не писать", "отказался", TASK_DONE, TASK_CLOSED],
)
async def test_a_closed_person_of_an_active_task_who_writes_reopens_the_talk(
    program: Program, reason: str
) -> None:
    # No list of «which close reasons count» (R104 b): the model decides what to do.
    await program.contacted()
    program.clock.advance(timedelta(hours=1))
    await people.close(program.repo, program.telegram, program.person, reason)
    assert PEER in program.telegram.allowed  # his task is in work: he is heard
    before, wakes = len(program.sent()), await program.wakes()

    program.script(END)
    await program.person_writes("Извините, был в отъезде. 240 000 сум, 3 дня")
    await pass_time(program, 5)

    assert (await program.repo.get_conversation(program.person)).state is (
        ConvState.TALKING
    )
    assert await program.wakes() == wakes + 1
    assert "240 000 сум, 3 дня" in program.context()
    assert program.sent()[before:] == []  # no fixed line: the agent decides
    await program.assert_clean()


async def test_a_closed_person_who_wrote_while_the_bot_was_down_is_heard_at_start_up(
    program: Program,
) -> None:
    # Fix round 9.2 (D3): the start-up replay reads every listened person, closed ones too.
    await program.contacted()
    program.clock.advance(timedelta(hours=1))
    await people.close(program.repo, program.telegram, program.person, GAVE_UP)
    program.telegram.add_incoming(PEER, "Извините, был в отъезде. 240 000 сум")  # down

    await replay_missed(program.repo, program.telegram, await replay_starts(program.repo))

    assert "Извините, был в отъезде. 240 000 сум" in await stored(program)
    assert (await program.repo.get_conversation(program.person)).state is (
        ConvState.TALKING
    )


async def test_an_excluded_person_is_not_replayed_at_start_up(program: Program) -> None:
    await program.contacted()
    await memory.remember(
        program.repo, "Не работать с @print_lab", "print_lab", program.clock.now()
    )
    await program.app.ops.exclude("print_lab")
    program.telegram.add_incoming(PEER, "Ещё актуально?")

    await replay_missed(program.repo, program.telegram, await replay_starts(program.repo))

    assert "Ещё актуально?" not in await stored(program)


async def stays_closed(program: Program, text: str = "Ещё актуально?") -> None:
    heard, before = await incoming_events(program), len(program.sent())
    program.script(END, END)  # the task's own stall nudge, which has nothing to do with him

    await program.person_writes(text)
    await pass_time(program, 5)

    assert (await program.repo.get_conversation(program.person)).state is (
        ConvState.CLOSED
    )
    assert await incoming_events(program) == heard  # nothing wakes the agent for him
    assert program.sent()[before:] == []
    assert text in await stored(program)  # his task is in work: his message is kept


async def test_a_person_the_owner_excluded_reaches_nobody(program: Program) -> None:
    # «если я запретил работать, ты же можешь просто удалить его из отслеживаемых»: one
    # mechanism, the gateway's listening set; his message never reaches the inbox.
    await program.contacted()
    await people.close(program.repo, program.telegram, program.person, GAVE_UP)
    assert PEER in program.telegram.allowed
    _, kept = await memory.remember(
        program.repo, "Не работать с @print_lab", "print_lab", program.clock.now()
    )
    assert kept and await program.app.ops.exclude("print_lab") == 0  # closed already
    assert PEER not in program.telegram.allowed
    assert PEER not in await listened(program.repo)
    messages, heard = await stored(program), await incoming_events(program)

    # The userbot drops the updates of a chat it does not listen to before the inbox; he
    # writes, and nothing of it reaches the program.
    program.telegram.add_incoming(PEER, "Ещё актуально?")
    await pass_time(program, 5)

    conv = await program.repo.get_conversation(program.person)
    assert (conv.state, conv.close_reason) == (ConvState.CLOSED, GAVE_UP)  # no reopen
    assert await incoming_events(program) == heard
    assert await stored(program) == messages

    # «забудь»: he is heard again.
    (fact,) = await memory.load(program.repo)
    await memory.forget(program.repo, fact.id)
    await program.app.ops.listen()
    assert PEER in program.telegram.allowed


async def test_a_chat_handed_to_the_owner_stays_closed(program: Program) -> None:
    await program.contacted()
    await program.repo.create_approval(
        program.task_id, program.person, ApprovalKind.HANDOFF, {"text": "Чат у тебя"}
    )
    await people.close(program.repo, program.telegram, program.person, GAVE_UP)

    await stays_closed(program)


async def test_a_chat_handed_to_the_owner_and_given_back_reopens_later(
    program: Program,
) -> None:
    # Fix round 9.2 (D2 ii): only a hand-over still pending keeps him shut.
    await program.contacted()
    card = await program.repo.create_approval(
        program.task_id, program.person, ApprovalKind.HANDOFF, {"text": "Чат у тебя"}
    )
    await program.repo.resolve_approval(card.id, card.nonce, ApprovalStatus.APPROVED)
    program.clock.advance(timedelta(days=2))
    await people.close(program.repo, program.telegram, program.person, GAVE_UP)
    wakes = await program.wakes()

    program.script(END)
    await program.person_writes("Извините, вот цена: 240 000 сум")
    await pass_time(program, 5)

    assert (await program.repo.get_conversation(program.person)).state is (
        ConvState.TALKING
    )
    assert await program.wakes() == wakes + 1
    assert "240 000 сум" in program.context()


async def test_a_closed_person_of_a_paused_task_is_heard_after_the_pause(
    program: Program,
) -> None:
    # Fix round 9.2 (D2 i): his talk reopens in a PAUSED task too; nothing wakes while it
    # is paused, and the wake after «Продолжить» has his message.
    await program.contacted()
    program.clock.advance(timedelta(hours=1))
    await people.close(program.repo, program.telegram, program.person, GAVE_UP)
    await program.app.ops.pause(program.task_id)
    wakes, before = await program.wakes(), len(program.sent())

    await program.person_writes("Извините, был в отъезде. 240 000 сум, 3 дня")
    await pass_time(program, 5)

    assert (await program.repo.get_conversation(program.person)).state is (
        ConvState.TALKING
    )
    assert await program.wakes() == wakes  # paused: nothing wakes
    assert program.sent()[before:] == []

    program.script(END)
    await program.app.ops.resume(program.task_id)
    await pass_time(program, 3)

    assert await program.wakes() == wakes + 1
    assert "240 000 сум, 3 дня" in program.context()


async def test_the_owners_own_quiet_keeps_the_agent_out_then_he_is_heard(
    program: Program, settings: Settings
) -> None:
    await program.talking("Подумаю")
    program.script(END)
    await pass_time(program, 2)
    await people.close(program.repo, program.telegram, program.person, GAVE_UP)
    program.clock.advance(timedelta(hours=1))
    own = program.telegram._log(PEER, True, "Это Саид. Давайте я сам позвоню вам завтра")
    await program.app.events.on_owner_manual(PEER, own.text, own.tg_msg_id)
    pause = timedelta(hours=settings.limits.owner_pause_hours)
    conv = await program.repo.get_conversation(program.person)
    assert conv.paused_until == program.clock.now() + pause
    assert (await stored(program)).count(own.text) == 1
    program.clock.advance(timedelta(minutes=5))

    await stays_closed(program, "Хорошо, Саид, жду звонка")

    # Once his quiet is over, the person is heard again.
    heard = await incoming_events(program)
    program.clock.advance(pause)
    await program.person_writes("Так вы позвоните?")
    assert (await program.repo.get_conversation(program.person)).state is (
        ConvState.TALKING
    )
    assert await incoming_events(program) == heard + 1


async def test_a_person_busy_in_another_task_is_heard_there(program: Program) -> None:
    await program.contacted()
    await people.close(program.repo, program.telegram, program.person, GAVE_UP)
    first = program.person
    card = await program.drafted(
        "Узнай у @print_lab про печать ещё 10 корпусов",
        [draft("print_lab", title="Ещё корпуса")],
    )
    launch = next(d for row in card.buttons for _, d in row if ":launch:" in d)
    assert await program.press(launch) == "Запущено"
    (conv,) = await program.repo.task_conversations(card.task_id)

    await program.person_writes("Здравствуйте, по 10 корпусам?")

    assert (await program.repo.last_messages(conv.id, 1))[0].text == (
        "Здравствуйте, по 10 корпусам?"
    )
    assert (await program.repo.get_conversation(first)).state is ConvState.CLOSED
    assert "по 10 корпусам" not in await stored(program)
