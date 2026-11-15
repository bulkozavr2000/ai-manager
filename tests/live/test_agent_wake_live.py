"""The task agent on the real model, in the test world. Run explicitly:

    TG_AGENT_LIVE=1 uv run pytest tests/live/test_agent_wake_live.py -q -s

Everything but the model is fake (Telegram, the bot, the clock, a temporary database), so this
shows what no scripted test can: that the real model, given the real rules, context and tools,
carries a small task from its launch to its result. It costs a few cents.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "offline"))

from agent_world import BURST, AgentWorld, Person

from tg_agent.agent import people
from tg_agent.brain.llm import AgentStep, AnthropicLLM
from tg_agent.gateway.base import PeerInfo
from tg_agent.models import OutboxStatus, TaskStatus
from tg_agent.secrets import Key, require_secret

pytestmark = pytest.mark.live


class Recording(AnthropicLLM):
    """The real client, noting every tool call of the task agent for the failure messages."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def agent_step(self, **kwargs: Any) -> AgentStep:
        step = await super().agent_step(**kwargs)
        for block in step.content:
            if block.type == "tool_use":
                self.calls.append((block.name, dict(block.input)))
        return step


async def test_live_a_small_task_from_launch_to_result(repo, fake_clock, settings):
    llm = Recording(require_secret(Key.ANTHROPIC_API_KEY), settings.models)
    world = AgentWorld(repo, fake_clock, settings, llm)  # type: ignore[arg-type]
    task_id = await repo.create_task(
        "Часы работы мастерской",
        "Узнать, работает ли мастерская в воскресенье и до скольки, и можно ли прийти"
        " без записи",
        {},
        None,
        "UZS",
        ["name"],
        [],
        source_text="Узнай у @print_lab, работают ли они в воскресенье и до скольки,"
        " можно ли прийти без записи",
    )
    world.gateway.peers["print_lab"] = PeerInfo(901, "print_lab", "Print Lab", False)
    attached = await people.attach(repo, world.gateway, task_id, "print_lab")
    assert isinstance(attached, people.Attached), attached
    person = Person(attached.conv_id, 901, "print_lab")
    await world.runner.start(task_id)

    # The first wake: the agent greets the workshop and asks.
    await world.round()

    queued = [
        item.text
        for item in await repo.outbox_of(person.conv_id)
        if item.status is OutboxStatus.PENDING
    ]
    assert queued, f"no first message was queued; tool calls: {llm.calls}"
    await world.deliver()
    first = world.sent_to(person)
    assert first, f"the first message did not go out; tool calls: {llm.calls}"
    assert "ИИ" in " ".join(first), first
    assert world.owner_messages() == 0, (world.ui.sent, world.ui.notes)
    print("\nfirst message:", first)

    # The workshop answers everything; the agent has what the owner asked for.
    world.clock.advance(BURST)
    await world.write(
        person,
        "Здравствуйте. В воскресенье работаем с 10:00 до 16:00, приходите без записи",
    )
    await world.round(BURST)
    await world.deliver()

    task = await repo.get_task(task_id)
    results = [card.text for card in world.ui.sent if card.kind == "result"]
    print("tool calls:", llm.calls)
    print("result:", results, "| headline:", task.headline)
    assert task.status is TaskStatus.DONE, f"not finished; tool calls: {llm.calls}"
    assert len(results) == 1, results
    assert "16" in results[0] and "10" in results[0], results
    assert not world.ui.notes, world.ui.notes
