"""Stage 9.3, ruling R109: only the owner ends a task.

The whole program on fakes (`test_bot_flow.Program`). The agent's `finish` is the result
notification «✅ Готово — …» with «Завершить»; the task stays in work: its people are heard,
their chats stay muted, and what they write wakes the agent, who decides what follows. A new
result replaces the old one, and only one waits. «Завершить» (or «Закрыть») ends the task:
nobody of it is heard after that, and the chats the program muted get their sound back.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from agent_world import END, call, send
from test_bot_flow import PEER, TITLE, Program

from tg_agent.agent import chats
from tg_agent.brain.llm import FakeLLM
from tg_agent.brain.schemas import Usage
from tg_agent.clock import FakeClock
from tg_agent.config import Settings
from tg_agent.control.cards import result_key, result_waits_key
from tg_agent.db import Repo
from tg_agent.main import start_up
from tg_agent.models import OutboxStatus, TaskStatus


@pytest.fixture
def program(repo: Repo, fake_clock: FakeClock, settings: Settings) -> Program:
    return Program(repo, fake_clock, settings, FakeLLM())


def waiting_results(program: Program) -> list[str]:
    """The result cards on the owner's screen that still offer «Завершить»."""
    return [
        card.text
        for _, card in program.bot.chat()
        if card.kind == "result" and card.buttons
    ]


async def reported(program: Program, result: str = "PrintLab: 240 000 сум.") -> None:
    """The person answered and the agent reported: the result waits for «Завершить»."""
    await program.talking("Есть, 240 000 сум")
    program.script([call("finish", result=result)])
    await program.round(timedelta(minutes=2))
    await program.app.logic.layout.settle()


async def pass_time(program: Program, minutes: int = 10) -> None:
    for _ in range(minutes):
        await program.round(timedelta(minutes=1))


async def test_the_result_leaves_the_task_in_work_and_its_people_heard(
    program: Program,
) -> None:
    await reported(program)

    task = await program.repo.get_task(program.task_id)
    assert task.status is TaskStatus.ACTIVE
    assert await program.repo.get_kv(result_waits_key(program.task_id)) == "1"
    assert PEER in program.telegram.allowed
    assert waiting_results(program) == [f"✅ Готово — {TITLE}\nPrintLab: 240 000 сум."]
    # The pinned status shows it as ready: the owner reads the result and ends it.
    assert f"#{program.task_id} {TITLE} — готова" in program.board()
    await program.assert_clean()


async def test_a_late_answer_wakes_the_agent_and_a_new_result_replaces_the_old(
    program: Program,
) -> None:
    await reported(program)
    wakes = await program.wakes()

    program.script([call("finish", result="PrintLab: уже 220 000 сум.")])
    await program.person_writes("Могу уступить: 220 000 сум")
    await pass_time(program, 3)
    await program.app.logic.layout.settle()

    assert await program.wakes() == wakes + 1
    assert "Могу уступить: 220 000 сум" in program.context()
    # The agent knows what it reported and that the owner has not ended the task.
    assert "Итог уже отправлен владельцу и ждёт «Завершить»" in program.context()
    assert "PrintLab: 240 000 сум." in program.context()
    task_id = program.task_id
    assert await program.repo.get_kv(result_key(task_id)) == "PrintLab: уже 220 000 сум."
    assert await program.repo.get_kv(result_waits_key(task_id)) == "1"
    # One result waits: the old card lost its button.
    assert waiting_results(program) == [f"✅ Готово — {TITLE}\nPrintLab: уже 220 000 сум."]
    assert (await program.repo.get_task(task_id)).status is TaskStatus.ACTIVE
    await program.assert_clean()


async def test_a_thank_you_after_the_result_needs_nothing(program: Program) -> None:
    await reported(program)
    cards, wakes = len(program.cards()), await program.wakes()

    program.script(END)  # the model decides that a courtesy needs nothing
    await program.person_writes("Спасибо!")
    await pass_time(program, 3)

    assert await program.wakes() == wakes + 1
    assert len(program.cards()) == cards  # no new card
    assert await program.repo.get_kv(result_waits_key(program.task_id)) == "1"
    assert len(waiting_results(program)) == 1
    # Nothing is scheduled and the result waits: no nudge «заверши задачу», no alarm.
    await pass_time(program, 60)
    assert await program.repo.events_of("task_nudged") == []
    assert await program.repo.events_of("task_stalled") == []
    await program.assert_clean()


async def test_zavershit_ends_the_listening_and_gives_the_sound_back(
    program: Program,
) -> None:
    await program.talking("Есть, 240 000 сум")
    # His chat existed before: the program muted it and remembers that it did (R105).
    contact_id = (await program.repo.get_conversation(program.person)).contact_id
    program.telegram.muted[PEER] = True
    await program.repo.set_kv(chats.muted_key(contact_id), "1")
    program.script([call("finish", result="PrintLab: 240 000 сум.")])
    await program.round(timedelta(minutes=2))
    assert program.telegram.muted[PEER] is True  # the result is no end: still muted

    toast = await program.press(f"fn:{program.task_id}:ack:-")  # «Завершить»

    assert toast == f"Задача #{program.task_id} завершена"
    assert (await program.repo.get_task(program.task_id)).status is TaskStatus.DONE
    assert not await program.repo.get_kv(result_waits_key(program.task_id))
    assert PEER not in program.telegram.allowed
    assert program.telegram.muted[PEER] is False
    assert waiting_results(program) == []
    messages = await program.repo.last_messages(program.person, 50)
    await program.person_writes("Ещё актуально?")
    await pass_time(program, 3)
    assert await program.repo.last_messages(program.person, 50) == messages


async def test_the_owners_words_on_the_result_continue_the_task(
    program: Program,
) -> None:
    await reported(program)
    (card,) = [c for c in program.cards() if c.kind == "result"]

    program.secretary([])
    program.script(END)
    await program.write("А дешевле можно?", reply_to=program.msg_id(card))

    assert (await program.repo.get_task(program.task_id)).status is TaskStatus.ACTIVE
    assert not await program.repo.get_kv(result_waits_key(program.task_id))
    assert waiting_results(program) == []  # «Завершить» has nothing to end now
    assert await program.press(f"fn:{program.task_id}:ack:-") == (
        f"Задача #{program.task_id} не завершена агентом — завершать нечего"
    )


async def test_a_restart_replays_the_people_of_a_task_whose_result_waits(
    program: Program,
) -> None:
    await reported(program)
    program.telegram.add_incoming(PEER, "Кстати, доставка бесплатно")  # while down

    program.app = program.build()
    await start_up(program.app)
    program.script(END)
    await pass_time(program, 3)

    texts = [m.text for m in await program.repo.last_messages(program.person, 50)]
    assert "Кстати, доставка бесплатно" in texts
    assert "Кстати, доставка бесплатно" in program.context()
    await program.assert_clean()


async def test_a_task_an_older_version_finished_stays_done_and_zavershit_still_works(
    program: Program,
) -> None:
    await reported(program)
    # An older `finish` moved the task to DONE; its result was never acknowledged.
    await program.repo.set_task_status(program.task_id, TaskStatus.DONE)

    program.app = program.build()
    await start_up(program.app)
    assert (await program.repo.get_task(program.task_id)).status is TaskStatus.DONE

    toast = await program.press(f"fn:{program.task_id}:ack:-")

    assert toast == f"Задача #{program.task_id} завершена"
    assert (await program.repo.get_task(program.task_id)).status is TaskStatus.DONE
    assert not await program.repo.get_kv(result_waits_key(program.task_id))


async def test_the_owners_words_on_an_older_versions_result_take_the_task_up(
    program: Program,
) -> None:
    await reported(program)
    await program.repo.set_task_status(program.task_id, TaskStatus.DONE)
    (card,) = [c for c in program.cards() if c.kind == "result"]

    program.secretary([])
    program.script(END)
    await program.write("А дешевле можно?", reply_to=program.msg_id(card))

    assert (await program.repo.get_task(program.task_id)).status is TaskStatus.ACTIVE
    assert not await program.repo.get_kv(result_waits_key(program.task_id))


async def test_the_task_money_cap_still_stops_wakes_after_the_result(
    program: Program,
) -> None:
    await reported(program)
    wakes = await program.wakes()
    await program.repo.log_llm_call(
        None, "task", "m", Usage(1, 1), 3.0, "end_turn", task_id=program.task_id
    )

    await program.person_writes("Могу уступить: 220 000 сум")
    await pass_time(program, 5)

    assert await program.wakes() == wakes
    assert [c for c in program.cards() if c.kind == "task_money"]


# -- what is queued when the owner ends the task (R109) --

CONFIRM = "Хорошо, тогда завтра в 18:00"
QUESTION = "Подтвердите, пожалуйста, 18:00?"


async def test_a_question_queued_by_a_later_wake_is_cancelled_at_zavershit(
    program: Program,
) -> None:
    await reported(program)
    # The person writes again; the agent's next wake asks a question that waits in the queue.
    await program.person_writes("А в 18:00 можно?")
    program.script([send(program.person, QUESTION)], END)
    await program.round(timedelta(seconds=50))
    pending = await program.repo.outbox_in(OutboxStatus.PENDING)
    assert [item.text for item in pending] == [QUESTION]

    await program.press(f"fn:{program.task_id}:ack:-")
    await program.round(timedelta(hours=1))

    assert QUESTION not in program.sent()
    item = (await program.repo.outbox_of(program.person))[-1]
    assert item.status is OutboxStatus.CANCELLED


async def test_the_confirmation_of_the_results_own_wake_still_goes_after_zavershit(
    program: Program,
) -> None:
    await program.talking("Есть, 240 000 сум")
    program.script([send(program.person, CONFIRM), call("finish", result="PrintLab: 240 000 сум.")])
    await program.round(timedelta(minutes=2))
    await program.app.logic.layout.settle()
    assert CONFIRM not in program.sent()  # still waits its one to three minutes

    await program.press(f"fn:{program.task_id}:ack:-")  # «Завершить» at once
    await program.until(lambda: CONFIRM in program.sent(), timedelta(minutes=1))

    assert program.sent().count(CONFIRM) == 1


async def test_an_ok_reply_without_a_wake_still_goes_after_zavershit(
    program: Program,
) -> None:
    await reported(program)
    # A reply the owner approved with «Ок»: queued by no wake.
    await program.repo.enqueue(
        program.person, "Договорились, до завтра",
        program.clock.now() + timedelta(minutes=2), wake_id=None,
    )

    await program.press(f"fn:{program.task_id}:ack:-")
    await program.until(lambda: "Договорились, до завтра" in program.sent(), timedelta(minutes=1))
