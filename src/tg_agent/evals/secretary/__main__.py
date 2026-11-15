"""`uv run python -m tg_agent.evals.secretary [--runs N] [--only ID ...] [--yes] [--config PATH]`.

Gives the owner's words of every case, in the case's state, to the program's secretary on the
real model, and says how many runs of each case passed (pass^k: all of them). It spends the
owner's money, so the first thing it does is say how many model calls it will make at most and
what they can cost at most, and ask; only a yes starts it. The report is kept in
`<data folder>/evals/secretary-<UTC time>.json`.

Exit code: 0 if every case passed all its runs, 1 if not (or if the run was not started),
2 for a mistake in the arguments or the settings, 130 if it was stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from tg_agent.agent.secretary import MAX_COST, MAX_STEPS
from tg_agent.brain.llm import LLM, AnthropicLLM
from tg_agent.brain.pricing import PRICES, price
from tg_agent.brain.schemas import Usage
from tg_agent.config import PROJECT_ROOT, ConfigError, Models, load_settings
from tg_agent.evals.secretary.cases import CaseError, load_cases
from tg_agent.evals.secretary.runner import run_all
from tg_agent.secrets import Key, MissingSecret, require_secret

# What one step of the secretary costs at most, in tokens: the rules and the tools (about five
# thousand), the state and the owner's words, the answers of earlier steps; a short answer or a
# tool call out. The real figures are in the report.
STEP_USAGE = Usage(6_000, 500)
TYPICAL_STEPS = 2  # a tool call and the words after it

_YES = frozenset({"y", "yes", "д", "да"})
Say = Callable[[str], object]


def estimate_cost(turns: int, models: Models) -> float:
    """The most `turns` turns of the secretary can cost, in dollars: every turn at its cap of
    `MAX_STEPS` steps (the secretary also stops a turn once it has spent `MAX_COST`). A model
    without a price is counted at the dearest one."""
    step = price(models.intake, STEP_USAGE)
    return turns * min(MAX_STEPS * step, MAX_COST + step)


def _plural(number: int, forms: tuple[str, str, str]) -> str:
    """The Russian word for the number: 1 кейс, 2 кейса, 5 кейсов."""
    if 11 <= number % 100 <= 14:
        return forms[2]
    if number % 10 == 1:
        return forms[0]
    return forms[1] if 2 <= number % 10 <= 4 else forms[2]


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
        prog="python -m tg_agent.evals.secretary",
        description="Учения секретаря: понимает ли он слова владельца",
    )
    parser.add_argument(
        "--runs", type=_positive, default=1, help="сколько раз гонять каждый кейс (1)"
    )
    parser.add_argument(
        "--only", nargs="+", default=(), metavar="ID", help="только эти кейсы"
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


def _describe(cases: int, runs: int, models: Models, say: Say) -> None:
    """Say what is about to run, how many calls it makes at most and what they can cost."""
    turns = cases * runs
    say(
        f"Учения секретаря: {cases} {_plural(cases, ('кейс', 'кейса', 'кейсов'))},"
        f" по {runs} {_plural(runs, ('прогон', 'прогона', 'прогонов'))}"
        f" (всего {turns} {_plural(turns, ('ход', 'хода', 'ходов'))})"
    )
    say(f"Модель секретаря: {models.intake}")
    if not any(models.intake.startswith(name) for name in PRICES):
        say(
            f"Для модели {models.intake} нет цены в brain/pricing.py: считаю по самой дорогой"
        )
    calls = turns * MAX_STEPS
    typical = turns * TYPICAL_STEPS * price(models.intake, STEP_USAGE)
    say(
        f"Вызовов модели: не больше {calls} {_plural(calls, ('вызова', 'вызовов', 'вызовов'))}"
        f" ({MAX_STEPS} шагов на ход). Стоимость: не больше"
        f" ${estimate_cost(turns, models):.2f}, обычно около ${typical:.2f}"
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
    for stream in (sys.stdout, sys.stderr):  # a console that cannot print a letter
        with contextlib.suppress(AttributeError, ValueError, OSError):
            stream.reconfigure(errors="backslashreplace")
    try:
        settings = load_settings(args.config)
        cases = load_cases(args.only)
        api_key = require_secret(Key.ANTHROPIC_API_KEY)
    except (ConfigError, CaseError, MissingSecret) as error:
        say(str(error))
        return 2

    _describe(len(cases), args.runs, settings.models, say)
    if not args.yes and ask("Запустить? [y/N] ").strip().casefold() not in _YES:
        say("Отменено: ничего не запущено, ничего не потрачено")
        return 1

    out_dir = settings.data_dir / "evals"
    try:
        report = asyncio.run(
            run_all(
                cases,
                make_llm(api_key, settings.models),
                args.runs,
                settings=settings,
                out_dir=out_dir,
                say=say,
            )
        )
    except KeyboardInterrupt:
        say(f"Остановлено. Что успело закончиться, записано в {out_dir}")
        return 130
    say(f"Отчёт: {report.path}")
    return 0 if all(report.pass_all(case_id) for case_id in report.case_ids()) else 1


if __name__ == "__main__":
    sys.exit(main())
