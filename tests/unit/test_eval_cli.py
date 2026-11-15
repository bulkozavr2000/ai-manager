"""The eval command: what it says before it spends money, and what it keeps away from the run.

The model is a FakeLLM handed in through `make_llm`; the keyring is the test's own. Nothing here
reaches the network or the real credential store.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import keyring
import pytest

from tg_agent.brain.llm import LLM, AgentStep, FakeLLM, LLMResult
from tg_agent.brain.pricing import cost_usd
from tg_agent.brain.schemas import Usage
from tg_agent.config import Models
from tg_agent.evals.__main__ import estimate_cost, main
from tg_agent.evals.runner import FAKE_PHONE, load_scenarios
from tg_agent.secrets import Key, get_secret, set_secret

REAL_KEY = "sk-ant-real-0000"
REAL_PHONE = "+998 90 123 45 67"
OPENING = "Ассаляму алейкум! Подскажите, пожалуйста, сколько стоит печать 5 корпусов?"


PERSON = 1  # the run's one conversation: the counterpart, as the agent's tools name him
END = [
    {"type": "text", "text": "Жду ответа."}
]  # a step of the agent that ends its turn


def send(person: int, text: str) -> dict[str, Any]:
    """The agent's tool call that writes to the person (one block of a scripted step)."""
    return {
        "type": "tool_use",
        "id": f"send-{text[:10]}",
        "name": "send_message",
        "input": {
            "person_id": person,
            "messages": [text],
            "language": "ru",
            "attach_task_files": False,
            "commits_owner": False,
        },
    }


def ask(kind: str, question: str, person: int) -> dict[str, Any]:
    """The agent's tool call that puts a card to the owner."""
    return {
        "type": "tool_use",
        "id": f"ask-{kind}",
        "name": "ask_owner",
        "input": {
            "kind": kind,
            "question": question,
            "person_id": person,
            "suggested_reply": [],
            "amount": None,
            "terms": "",
        },
    }


class WatchingLLM(FakeLLM):
    """Scripted for `refuses_ai`; notes which owner phone the keyring holds at each call."""

    def __init__(self) -> None:
        super().__init__()
        self.steps["task"].extend(
            [
                [send(PERSON, OPENING)],
                END,
                [
                    send(PERSON, "Понял, передам Саиду, он напишет сам"),
                    ask("handoff", "Не хочет говорить с ИИ", PERSON),
                ],
                END,
            ]
        )
        self.sim_replies.append("Я с ботами не общаюсь. Пусть Саид напишет сам")
        self.phones: list[str | None] = []

    async def agent_step(self, **kw: Any) -> AgentStep:
        self.phones.append(get_secret(Key.OWNER_PHONE))
        return await super().agent_step(**kw)

    async def simulate(self, role: str, transcript: str) -> LLMResult[str]:
        self.phones.append(get_secret(Key.OWNER_PHONE))
        return await super().simulate(role, transcript)


class Factory:
    """`make_llm` that hands out one LLM and remembers what it was asked for."""

    def __init__(self, llm: LLM | None = None) -> None:
        self.llm = llm
        self.calls: list[tuple[str, Models]] = []

    def __call__(self, api_key: str, models: Models) -> LLM:
        self.calls.append((api_key, models))
        assert self.llm is not None, "no model should have been built"
        return self.llm


class Session:
    """One run of the command with everything it prints and asks collected."""

    def __init__(self, tmp_path: Path) -> None:
        self.config = tmp_path / "config.toml"
        self.config.write_text('data_dir = "data"\n', encoding="utf-8")
        self.data = tmp_path / "data"
        self.events: list[str] = []
        self.answers: list[str] = []

    def say(self, line: str) -> None:
        self.events.append(line)

    def ask(self, prompt: str) -> str:
        self.events.append(f"ASK {prompt}")
        return self.answers.pop(0) if self.answers else "n"

    def run(self, *argv: str, factory: Factory | None = None) -> int:
        return main(
            ["--config", str(self.config), *argv],
            ask=self.ask,
            say=self.say,
            make_llm=factory or Factory(),
        )

    @property
    def output(self) -> str:
        return "\n".join(self.events)

    @property
    def asked(self) -> list[str]:
        return [event for event in self.events if event.startswith("ASK ")]


@pytest.fixture
def session(tmp_path: Path) -> Session:
    set_secret(Key.ANTHROPIC_API_KEY, REAL_KEY)
    set_secret(Key.OWNER_PHONE, REAL_PHONE)
    return Session(tmp_path)


# --- the estimate -----------------------------------------------------------------------------


def test_the_estimate_prices_every_turn_of_every_run() -> None:
    scenarios = load_scenarios(["refuses_ai", "happy_ru"])
    decide = cost_usd("claude-sonnet-5-5", Usage(4_500, 500))
    judge = cost_usd("claude-haiku-5-5", Usage(2_000, 150))
    simulate = cost_usd("claude-haiku-5-5", Usage(1_500, 150))
    turns = sum(scenario.max_turns for scenario in scenarios)
    answers = turns + len(
        scenarios
    )  # every counterpart line is answered, and each chat opens
    expected = 3 * (answers * (decide + judge) + turns * simulate)

    assert estimate_cost(scenarios, 3, Models()) == pytest.approx(expected)
    assert estimate_cost(scenarios, 1, Models()) == pytest.approx(expected / 3)


def test_a_line_the_scenario_gives_costs_no_simulator_call() -> None:
    (scenario,) = load_scenarios(["happy_ru"])
    simulate = cost_usd("claude-haiku-5-5", Usage(1_500, 150))
    scripted = replace(
        scenario,
        counterpart=replace(scenario.counterpart, lines=("Вы бот?", "А цена?")),
    )

    saved = estimate_cost([scenario], 1, Models()) - estimate_cost(
        [scripted], 1, Models()
    )

    assert saved == pytest.approx(2 * simulate)


def test_the_estimate_counts_the_turns_of_every_counterpart() -> None:
    (scenario,) = load_scenarios(["happy_ru"])
    decide = cost_usd("claude-sonnet-5-5", Usage(4_500, 500))
    judge = cost_usd("claude-haiku-5-5", Usage(2_000, 150))
    simulate = cost_usd("claude-haiku-5-5", Usage(1_500, 150))
    second = replace(scenario.counterpart, username="second_one", lines=("Цена 1",))
    third = replace(scenario.counterpart, username="third_one")
    trio = replace(scenario, others=(second, third), several=True)
    turns = scenario.max_turns  # each of the three has max_turns lines at most
    expected = 3 * (turns + 1) * (decide + judge) + (3 * turns - 1) * simulate

    assert estimate_cost([trio], 1, Models()) == pytest.approx(expected)
    assert estimate_cost([trio], 1, Models()) > estimate_cost([scenario], 1, Models())


def test_a_dearer_model_raises_only_the_price_of_the_decisions() -> None:
    scenarios = load_scenarios(["happy_ru"])
    cheap = estimate_cost(scenarios, 1, Models())
    dear = estimate_cost(scenarios, 1, replace(Models(), decide="claude-opus-5-5"))
    answers = scenarios[0].max_turns + 1
    extra = answers * (
        cost_usd("claude-opus-5-5", Usage(4_500, 500))
        - cost_usd("claude-sonnet-5-5", Usage(4_500, 500))
    )
    assert dear - cheap == pytest.approx(extra)


def test_a_model_without_a_price_is_counted_at_the_dearest_rate() -> None:
    scenarios = load_scenarios(["happy_ru"])
    unknown = estimate_cost(scenarios, 1, replace(Models(), decide="claude-mystery-9"))
    assert unknown == pytest.approx(
        estimate_cost(scenarios, 1, replace(Models(), decide="claude-opus-5-5"))
    )


# --- asking first -----------------------------------------------------------------------------


def test_the_cost_is_shown_before_the_question_and_no_means_no(
    session: Session,
) -> None:
    factory = Factory()  # asserts if a model is built

    code = session.run("--only", "refuses_ai", factory=factory)

    assert code == 1
    assert factory.calls == []
    first_ask = next(n for n, e in enumerate(session.events) if e.startswith("ASK "))
    estimate = next(n for n, e in enumerate(session.events) if "$" in e)
    assert estimate < first_ask  # the figure comes before the question
    assert session.asked == ["ASK Запустить? [y/N] "]
    assert not session.data.exists()  # nothing was written


@pytest.mark.parametrize("answer", ["", "n", "N", "нет", "yes please maybe", "ye"])
def test_only_a_clear_yes_starts_the_run(session: Session, answer: str) -> None:
    session.answers.append(answer)
    factory = Factory()

    assert session.run("--only", "refuses_ai", factory=factory) == 1
    assert factory.calls == []


@pytest.mark.parametrize("answer", ["y", "Y", "yes", " да ", "Д"])
def test_a_yes_starts_the_run(session: Session, answer: str) -> None:
    session.answers.append(answer)
    factory = Factory(WatchingLLM())

    assert session.run("--only", "refuses_ai", "--runs", "1", factory=factory) == 0
    assert len(factory.calls) == 1


def test_yes_on_the_command_line_skips_the_question(session: Session) -> None:
    factory = Factory(WatchingLLM())

    code = session.run("--only", "refuses_ai", "--runs", "1", "--yes", factory=factory)

    assert code == 0
    assert session.asked == []
    assert "$" in session.output  # the figure is still shown


def test_the_estimate_names_the_models_and_the_size_of_the_run(
    session: Session,
) -> None:
    session.run("--only", "refuses_ai", "happy_ru", "--runs", "2")

    assert "2 сценария" in session.output
    assert "по 2 прогона" in session.output
    assert (
        "claude-sonnet-5-5" in session.output and "claude-haiku-5-5" in session.output
    )


# --- the models ---------------------------------------------------------------------------------


def test_the_model_option_replaces_the_deciding_model_only(session: Session) -> None:
    factory = Factory(WatchingLLM())

    session.run(
        "--only", "refuses_ai", "--runs", "1", "--yes", "--model", "claude-opus-5-5",
        factory=factory,
    )  # fmt: skip

    ((_, models),) = factory.calls
    assert models == replace(Models(), decide="claude-opus-5-5")
    assert "claude-opus-5-5" in session.output


def test_a_model_without_a_price_is_said_out_loud(session: Session) -> None:
    session.run("--only", "refuses_ai", "--model", "claude-mystery-9")

    assert "claude-mystery-9" in session.output
    assert "самой дорогой" in session.output


# --- what the run is kept away from ---------------------------------------------------------------


def test_the_run_gets_the_real_key_and_a_made_up_owner_and_the_keyring_comes_back(
    session: Session,
) -> None:
    llm = WatchingLLM()
    factory = Factory(llm)
    before = keyring.get_keyring()

    code = session.run("--only", "refuses_ai", "--runs", "1", "--yes", factory=factory)

    assert code == 0
    ((key, _),) = factory.calls
    assert key == REAL_KEY  # read from the real keyring, before it was swapped
    assert llm.phones and set(llm.phones) == {
        FAKE_PHONE
    }  # never the owner's own number
    assert keyring.get_keyring() is before
    assert get_secret(Key.OWNER_PHONE) == REAL_PHONE
    assert REAL_KEY not in session.output and REAL_PHONE not in session.output


def test_the_keyring_comes_back_when_the_run_is_stopped(session: Session) -> None:
    class Stopped(FakeLLM):
        async def agent_step(self, **kw: Any) -> AgentStep:
            raise KeyboardInterrupt

    before = keyring.get_keyring()

    code = session.run("--only", "refuses_ai", "--yes", factory=Factory(Stopped()))

    assert code == 130
    assert keyring.get_keyring() is before
    assert get_secret(Key.OWNER_PHONE) == REAL_PHONE
    assert "Остановлено" in session.output


def test_the_report_goes_to_the_data_folder_of_the_settings(session: Session) -> None:
    code = session.run(
        "--only", "refuses_ai", "--runs", "1", "--yes", factory=Factory(WatchingLLM())
    )

    assert code == 0
    (path,) = (session.data / "evals").glob("*.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["scenarios"][0]["id"] == "refuses_ai"
    assert data["models"]["decide"] == "claude-sonnet-5-5"
    assert str(path) in session.output  # and the owner is told where it is
    assert "pass^1" in session.output


def test_the_exit_code_says_whether_every_scenario_passed(session: Session) -> None:
    llm = FakeLLM()
    # the agent opens, but the counterpart's refusal is not answered by a handoff
    llm.steps["task"].extend(
        [[send(PERSON, OPENING)], END, [send(PERSON, "Понял, продолжим")], END]
    )
    llm.sim_replies.extend(["Я с ботами не общаюсь. Пусть Саид напишет сам", ""])

    code = session.run(
        "--only", "refuses_ai", "--runs", "1", "--yes", factory=Factory(llm)
    )

    assert code == 1  # handed_to_owner failed
    (path,) = (session.data / "evals").glob("*.json")
    (scenario,) = json.loads(path.read_text(encoding="utf-8"))["scenarios"]
    assert [failure["check"] for failure in scenario["failed"]] == ["handed_to_owner"]
    assert scenario["errors"] == []  # the run itself worked: the agent failed the check


# --- mistakes of the owner --------------------------------------------------------------------------


def test_a_missing_key_is_said_before_anything_is_asked(tmp_path: Path) -> None:
    session = Session(tmp_path)  # no key in the keyring

    code = session.run("--only", "refuses_ai")

    assert code == 2
    assert "anthropic_api_key" in session.output
    assert session.asked == []


def test_an_unknown_scenario_is_refused_with_the_known_ones(session: Session) -> None:
    code = session.run("--only", "no_such_scenario")

    assert code == 2
    assert "no_such_scenario" in session.output and "happy_ru" in session.output
    assert session.asked == []


@pytest.mark.parametrize("runs", ["0", "-1", "x"])
def test_the_number_of_runs_must_be_a_positive_whole_number(
    session: Session, runs: str
) -> None:
    with pytest.raises(SystemExit) as stop:
        session.run("--runs", runs)
    assert stop.value.code == 2


def test_help_lists_the_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        main(["--help"])

    assert stop.value.code == 0
    text = capsys.readouterr().out
    for option in ("--runs", "--only", "--model", "--yes"):
        assert option in text
