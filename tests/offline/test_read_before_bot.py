"""Read before the bot (ruling R79): the program never looks at read marks.

The owner opens a chat on the work account and reads a person's message before the agent
answered: Telegram marks it read, and nothing of it reaches the program. The core is told of
new messages, edits, deletions and the owner's own messages (`ports.ChatEvents`), never of a
read; the agent answers on the next tick all the same. His own message in the chat is another
thing: the agent keeps quiet there for 12 hours and is woken when they end.
"""

from __future__ import annotations

import inspect
from datetime import timedelta

import pytest
from agent_world import BURST, END, AgentWorld, send

from tg_agent.agent.inbox import Inbox
from tg_agent.models import EventKind
from tg_agent.ports import ChatEvents

REPLY = "Сколько стоит GSB 13?"


@pytest.fixture
async def world(repo, fake_clock, settings):
    world = AgentWorld(repo, fake_clock, settings)
    yield world
    await world.assert_clean()


def test_a_read_is_nothing_the_core_is_ever_told_of() -> None:
    told = {
        name for name, _ in inspect.getmembers(ChatEvents) if name.startswith("on_")
    }
    assert told == {"on_incoming", "on_edit", "on_owner_manual", "on_delete"}
    assert not [name for name in dir(Inbox) if "read" in name.casefold()]


async def test_a_message_the_owner_read_first_is_answered_on_the_next_tick(world):
    _task_id, person = await world.replied("Есть GSB 13")
    # The owner opens the chat on the work account and reads it: Telegram's business only,
    # nothing reaches the program (there is no such report to make).
    assert world.gateway.read == []
    world.script([send(person.conv_id, REPLY)], END)

    await world.round(BURST)  # the next tick: the agent's turn
    await world.deliver()

    assert world.sent_to(person)[-1] == REPLY
    # the only read is the program's own, just before it answers
    assert [peer for peer, _ in world.gateway.read] == [person.peer_id]


async def test_the_owner_s_own_message_keeps_the_agent_quiet_for_12_hours(world):
    task_id, person = await world.replied("Есть GSB 13")
    world.script([send(person.conv_id, REPLY)], END)
    await world.round(BURST)
    await world.deliver()
    sent, asked = len(world.sent_to(person)), len(world.requests())

    await world.inbox.on_owner_manual(person.peer_id, "Сам отвечу", 9001)
    assert await world.kinds(task_id) == [EventKind.OWNER_WROTE]
    for _ in range(11):
        await world.round(timedelta(hours=1))
    await world.round(timedelta(minutes=59))  # 11 h 59 min after his message

    assert len(world.requests()) == asked  # the agent was not woken
    assert len(world.sent_to(person)) == sent  # nothing went to the person

    after = send(person.conv_id, "Хорошо, тогда жду цену")
    world.script([after], END)
    await world.round(timedelta(minutes=2))  # the 12 hours are over
    await world.deliver()

    assert len(world.requests()) == asked + 2  # one wake: the message, then the end
    assert f"владелец сам написал [{person.conv_id}]" in world.context()
    assert world.result(after).get("is_error") is not True  # queued, not refused
    # 12 hours after a morning message it is past 21:00: it goes out when work starts
    assert world.clock.now().hour >= 21
    morning = world.clock.now().replace(hour=9, minute=5) + timedelta(days=1)
    await world.round(morning - world.clock.now())
    await world.deliver()
    assert world.sent_to(person)[-1] == "Хорошо, тогда жду цену"
