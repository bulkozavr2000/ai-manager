"""The eval harness of stage 6 (ruling R80), offline: the clock jumps to the task's next event
(`jump_hours`).

FakeLLM plays the agent and the counterpart, as in `test_eval_runner_dry.py`.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest
from agent_world import END, send

from tg_agent.brain.llm import FakeLLM
from tg_agent.evals.checks import Speaker
from tg_agent.evals.runner import (
    MAX_JUMP_HOURS,
    START,
    Scenario,
    ScenarioError,
    ScenarioResult,
    fake_owner_keyring,
    parse_scenario,
    run_scenario,
)

PERSON = 1
OPENING = "Здравствуйте! Я ИИ-помощник Саида. Подскажите, готов ли заказ 123?"
REMINDER = "Напоминаю: готов ли заказ 123?"
SILENT = "[молчит]"

pytestmark = pytest.mark.usefixtures("owner_keyring")


@pytest.fixture
def owner_keyring():
    with fake_owner_keyring():
        yield


def toml_list(texts: tuple[str, ...]) -> str:
    return "[" + ", ".join(f'"{text}"' for text in texts) + "]"


def scenario_text(
    *,
    jump: int | None = None,
    lines: tuple[str, ...] = (),
    max_turns: int = 3,
    extra: str = "",
) -> str:
    return (
        f'id = "dry_time"\nmax_turns = {max_turns}\n'
        + (f"jump_hours = {jump}\n" if jump is not None else "")
        + '[task]\ngoal = "Узнай у @ali_print, готов ли заказ 123"\n'
        'allowed_to_share = ["name"]\n'
        '[counterpart]\ndisplay_name = "Али"\nusername = "ali_print"\n'
        'role = "Ты мастер, заказ 123 у тебя"\nreply_delay_seconds = 60\n'
        f"lines = {toml_list(lines)}\n{extra}"
        '[owner]\nshare_phone = "deny"\nconfirm_order = "deny"\n'
        "[expect]\nintroduced_first = true\n"
    )


def timed(**kw: Any) -> Scenario:
    return parse_scenario(scenario_text(**kw), "dry_time.toml")


def scripted(*wakes: list[list[dict[str, Any]]]) -> FakeLLM:
    llm = FakeLLM()
    for wake in wakes:
        llm.steps["task"].extend(wake)
    return llm


def opening() -> list[list[dict[str, Any]]]:
    return [[send(PERSON, OPENING)], END]


def says(text: str) -> list[list[dict[str, Any]]]:
    return [[send(PERSON, text)], END]


def turns(result: ScenarioResult, who: Speaker) -> list[str]:
    return [turn.text for turn in result.chat if turn.speaker is who]


def wakes(llm: FakeLLM) -> list[str]:
    """What the task agent was shown at each step, as text."""
    return [
        json.dumps(request, ensure_ascii=False)
        for purpose, request in llm.requests
        if purpose == "task"
    ]


def simulated(llm: FakeLLM) -> int:
    return sum(purpose == "simulate" for purpose, _ in llm.requests)


# --- the scenario file -----------------------------------------------------------------


def test_the_new_keys_are_off_unless_the_scenario_sets_them() -> None:
    plain = timed()
    assert plain.jump_hours is None


def test_the_new_keys_are_read() -> None:
    assert timed(jump=72).jump_hours == 72


@pytest.mark.parametrize(
    ("changes", "key"),
    [
        ({"jump": 0}, "jump_hours"),
        ({"jump": MAX_JUMP_HOURS + 1}, "jump_hours"),
        # Late replies are gone (ruling R104): nobody of a finished task is heard.
        ({"extra": 'late_lines = ["Кто это?"]\n'}, "late_lines"),
    ],
)
def test_a_wrong_new_key_is_refused(changes: dict[str, Any], key: str) -> None:
    with pytest.raises(ScenarioError, match=key):
        timed(**changes)


# --- jump_hours ------------------------------------------------------------------------


async def test_a_silence_of_a_day_is_reached_by_a_jump() -> None:
    llm = scripted(opening(), says(REMINDER))
    llm.sim_replies.extend([SILENT, SILENT])  # he keeps quiet, before and after it

    result = await run_scenario(timed(jump=30), llm, 0)

    assert result.error is None, result.error
    assert turns(result, Speaker.AGENT) == [OPENING, REMINDER]
    first, reminder = result.chat
    assert reminder.at - first.at >= timedelta(hours=24)
    assert "молчит 24 ч" in wakes(llm)[-2]  # the wake the silence woke
    # asked again after the reminder, though he had kept quiet before it
    assert simulated(llm) == 2
    assert not llm.steps["task"] and not llm.sim_replies


async def test_a_jump_never_goes_past_jump_hours() -> None:
    # The day's silence is 24 hours off: 20 hours do not reach it.
    llm = scripted(opening())
    llm.sim_replies.append(SILENT)

    result = await run_scenario(timed(jump=20), llm, 0)

    assert result.error is None, result.error
    assert turns(result, Speaker.AGENT) == [OPENING]
    assert len(wakes(llm)) == 2  # the opening wake's two steps, nothing after


async def test_all_the_jumps_together_stay_within_jump_hours() -> None:
    # 30 hours reach the first silence (24 h), not the second (48 h after the start).
    llm = scripted(opening(), says(REMINDER))
    llm.sim_replies.extend([SILENT, SILENT])

    result = await run_scenario(timed(jump=30), llm, 0)

    assert all(turn.at < START + timedelta(hours=30) for turn in result.chat)
    assert len(turns(result, Speaker.AGENT)) == 2


async def test_without_jump_hours_a_run_ends_when_nobody_speaks() -> None:
    llm = scripted(opening())
    llm.sim_replies.append(SILENT)

    result = await run_scenario(timed(), llm, 0)

    assert turns(result, Speaker.AGENT) == [OPENING]
    assert len(wakes(llm)) == 2
