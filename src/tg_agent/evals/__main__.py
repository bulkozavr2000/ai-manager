"""`uv run python -m tg_agent.evals [--runs 3] [--only ID ...] [--model MODEL] [--yes]`.

Plays the scenarios against the agent with a model as the counterpart, and says how many of each
scenario's runs passed (pass^k: all of them). It spends the owner's money, so the first thing
it does is say what the run is expected to cost and ask; only a yes starts it. The Anthropic key
is read from the real keyring before anything else; from then on the process has a keyring of
made-up owner data until the end (see `runner.fake_owner_keyring`). The report is kept in
`<data folder>/evals/<UTC time>.json`.

Exit code: 0 if every scenario passed all its runs, 1 if not (or if the run was not started),
2 for a mistake in the arguments or the settings, 130 if it was stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from tg_agent.brain.llm import LLM, AnthropicLLM
from tg_agent.brain.pricing import PRICES, price
from tg_agent.brain.schemas import Usage
from tg_agent.config import PROJECT_ROOT, ConfigError, Models, load_settings
from tg_agent.evals.runner import Scenario, ScenarioError, load_scenarios, run_all
from tg_agent.secrets import Key, MissingSecret, require_secret

# What one answer of the agent costs, in tokens: a wake of the agent, the judge's check of the
# draft, and the simulator's next line. Averages over a conversation; the real figures are in the
# report.
DECIDE_USAGE = Usage(4_500, 500)
JUDGE_USAGE = Usage(2_000, 150)
SIMULATE_USAGE = Usage(1_500, 150)

_YES = frozenset({"y", "yes", "д", "да"})
Say = Callable[[str], object]


def estimate_cost(scenarios: Sequence[Scenario], runs: int, models: Models) -> float:
    """What the run is expected to cost at most, in dollars, at the prices of `brain/pricing`.

    Every counterpart line costs a wake of the agent, a judge's check and a line from the
    simulator, up to the scenario's `max_turns` for each of its counterparts (a line the scenario
    gives itself costs no simulator); the opening message of each chat costs a wake and a check.
    Chats that end sooner cost less. A model without a price is counted at the dearest one.
    """
    decide = price(models.decide, DECIDE_USAGE)
    judge = price(models.judge, JUDGE_USAGE)
    simulate = price(models.simulate, SIMULATE_USAGE)
    per_run = sum(
        (scenario.max_turns + 1) * (decide + judge)
        + max(0, scenario.max_turns - len(person.lines)) * simulate
        for scenario in scenarios
        for person in scenario.cast
    )
    return per_run * runs


def _plural(number: int, forms: tuple[str, str, str]) -> str:
    """The Russian word for the number: 1 сценарий, 2 сценария, 5 сценариев."""
    if 11 <= number % 100 <= 14:
        return forms[2]
    return (
        forms[0]
        if number % 10 == 1
        else forms[1]
        if 2 <= number % 10 <= 4
        else forms[2]
    )


def _positive(text: str) -> int:
    try:
        number = int(text)
    except ValueError:
        number = 0
    if number < 1:
        raise argparse.ArgumentTypeError("нужно целое число больше нуля")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tg_agent.evals",
        description="Учения: модель играет собеседников и пытается сломать агента",
    )
    parser.add_argument(
        "--runs",
        type=_positive,
        default=3,
        help="сколько раз гонять каждый сценарий (3)",
    )
    parser.add_argument(
        "--only", nargs="+", default=(), metavar="ID", help="только эти сценарии"
    )
    parser.add_argument(
        "--model", help="модель для решений агента вместо указанной в настройках"
    )
    parser.add_argument(
        "--yes", action="store_true", help="не спрашивать подтверждение стоимости"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.toml",
        help="файл настроек (по умолчанию config.toml в корне проекта)",
    )
    return parser


def _describe(
    scenarios: Sequence[Scenario], runs: int, models: Models, say: Say
) -> None:
    """Say what is about to run and what it is expected to cost."""
    chats = len(scenarios) * runs
    say(
        f"Учения: {len(scenarios)} {_plural(len(scenarios), ('сценарий', 'сценария', 'сценариев'))},"
        f" по {runs} {_plural(runs, ('прогон', 'прогона', 'прогонов'))} каждый"
        f" (всего {chats} {_plural(chats, ('разговор', 'разговора', 'разговоров'))})"
    )
    judge = (
        f"контролёр и собеседник — {models.judge}"
        if models.judge == models.simulate
        else f"контролёр — {models.judge}, собеседник — {models.simulate}"
    )
    say(f"Модели: решения — {models.decide}; {judge}")
    for model in dict.fromkeys((models.decide, models.judge, models.simulate)):
        if not any(model.startswith(name) for name in PRICES):
            say(
                f"Для модели {model} нет цены в brain/pricing.py: считаю по самой дорогой"
            )
    say(
        f"Оценка стоимости: не больше ${estimate_cost(scenarios, runs, models):.2f}"
        " (с первыми сообщениями; разговоры, что кончатся раньше, обойдутся дешевле)"
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    ask: Callable[[str], str] = input,
    say: Say = print,
    make_llm: Callable[[str, Models], LLM] = AnthropicLLM,
) -> int:
    """Run the command; `ask`, `say` and `make_llm` are what tests replace."""
    args = _parser().parse_args(argv)
    for stream in (
        sys.stdout,
        sys.stderr,
    ):  # a console that cannot print a letter must not fail
        with contextlib.suppress(AttributeError, ValueError, OSError):
            stream.reconfigure(errors="backslashreplace")
    try:
        settings = load_settings(args.config)
        scenarios = load_scenarios(args.only)
        api_key = require_secret(
            Key.ANTHROPIC_API_KEY
        )  # from the real keyring, before the swap
    except (ConfigError, ScenarioError, MissingSecret) as error:
        say(str(error))
        return 2
    models = (
        replace(settings.models, decide=args.model) if args.model else settings.models
    )

    _describe(scenarios, args.runs, models, say)
    if not args.yes and ask("Запустить? [y/N] ").strip().casefold() not in _YES:
        say("Отменено: ничего не запущено, ничего не потрачено")
        return 1

    out_dir = settings.data_dir / "evals"
    try:
        report = asyncio.run(
            run_all(
                scenarios,
                make_llm(api_key, models),
                args.runs,
                out_dir=out_dir,
                models=models,
                say=say,
            )
        )
    except KeyboardInterrupt:
        say(f"Остановлено. Что успело закончиться, записано в {out_dir}")
        return 130
    say(f"Отчёт: {report.path}")
    return 0 if all(report.pass_all(name) for name in report.scenario_ids()) else 1


if __name__ == "__main__":
    sys.exit(main())
