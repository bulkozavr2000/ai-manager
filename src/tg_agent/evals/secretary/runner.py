"""Running a case: the program's secretary reads the owner's turn in the case's state.

The state — drafts, tasks in work, paused and finished with their people and chats, the cards
that wait, the tail of the owner's chat with the bot, a stop — is written into a database in a
temporary folder that is gone when the run ends. The secretary is the program's own
(`agent/secretary.Secretary`: its rules, its tools, its context), on the model the caller
hands in, over a real `TaskOps`, `Runner` and `Approvals` on a fake Telegram and a fake control
bot. Its three callbacks are the eval's: a draft's card is only noted; a launch moves the draft
to work without waking any agent (no task agent runs, so no other model is called); closing a
draft closes it at once, as the program does, and closing a task in work only notes that the
confirmation card would be shown. Nothing reaches Telegram, the owner's chat or the live
database. Code, not a model, then scores the turn (`checks`).
"""

from __future__ import annotations

import json
import random
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from tg_agent.agent import memory, people
from tg_agent.agent.ops import TaskOps
from tg_agent.agent.runner import Runner
from tg_agent.agent.secretary import (
    OwnerMessage,
    Secretary,
    SecretaryReply,
    missing_key,
)
from tg_agent.brain.llm import LLM, AgentStep, LLMResult, SearchStep
from tg_agent.brain.schemas import JudgeVerdict
from tg_agent.clock import FakeClock
from tg_agent.config import Settings
from tg_agent.control.base import FakeUI
from tg_agent.db import Repo
from tg_agent.evals.checks import CheckResult
from tg_agent.evals.runner import START, RunCost, fake_owner_keyring
from tg_agent.evals.secretary.cases import (
    CardSpec,
    DraftSpec,
    PersonSpec,
    SecretaryCase,
)
from tg_agent.evals.secretary.checks import ToolCallRecord, TurnRecord, run_checks
from tg_agent.gateway.base import PeerInfo
from tg_agent.gateway.fake import FakeGateway
from tg_agent.models import ApprovalKind, ConvState, Direction, MsgSource, TaskStatus

__all__ = ["START", "CaseResult", "Office", "Report", "owner_messages", "run_all"]

ANSWER_CHARS = 300  # how much of a tool's answer the report keeps
_STATUS = {
    "active": TaskStatus.ACTIVE,
    "paused": TaskStatus.PAUSED,
    "done": TaskStatus.DONE,
}
# How a person gets to the state the case names (the state machine's legal moves).
_PATH = {
    "contacted": (ConvState.CONTACTED,),
    "talking": (ConvState.CONTACTED, ConvState.TALKING),
    "waiting_owner": (ConvState.CONTACTED, ConvState.TALKING, ConvState.WAITING_OWNER),
    "closed": (ConvState.CONTACTED, ConvState.TALKING, ConvState.CLOSED),
}
_PEER_ID = 9001
NOT_A_DRAFT = "Задача #{id} не черновик"
CLOSED = "Задача #{id} «{title}» закрыта"


def owner_messages(case: SecretaryCase) -> list[OwnerMessage]:
    """The owner's turn as the bot hands it to the secretary; `reply_to_draft` is the first
    message's."""
    return [
        OwnerMessage(
            part.text,
            case.reply_to_draft if number == 0 else None,
            person=part.person,
            forwarded=part.forwarded,
        )
        for number, part in enumerate(case.parts)
    ]


class Office:
    """The program's secretary over a real Repo, Runner, TaskOps and Approvals on fakes, with
    the eval's callbacks. `build` writes a case's state."""

    def __init__(
        self, repo: Repo, clock: FakeClock, settings: Settings, llm: LLM
    ) -> None:
        self.repo, self.clock, self.settings = repo, clock, settings
        self.gateway = FakeGateway(clock)
        self.ui = FakeUI()
        self.runner = Runner(
            repo, llm, self.gateway, self.ui, clock, settings, random.Random(7),
            canary="ZX-SECRETARY-EVAL", owner_secrets=[],
        )  # fmt: skip
        self.ops = TaskOps(
            repo, self.gateway, clock, settings, self.runner, random.Random(11)
        )
        self.shown: list[int] = []
        self.launched: list[int] = []
        self.closing: list[int] = []
        self.asked: list[tuple[int, list[people.Person]]] = []
        self.secretary = Secretary(
            repo, llm, self.ops, self.runner.approvals, clock, settings,
            show_draft=self.show_draft, launch=self.launch,
            confirm_close=self.confirm_close, gateway=self.gateway,
            ask_which=self.ask_which,
        )  # fmt: skip

    # -- the callbacks --

    async def show_draft(self, task_id: int) -> None:
        self.shown.append(task_id)

    async def launch(self, task_id: int) -> tuple[bool, list[str]]:
        """The draft goes to work; no agent wakes (nothing ticks the scheduler)."""
        moved = await self.repo.move_task_if(
            task_id, TaskStatus.DRAFT, TaskStatus.ACTIVE
        )
        if not moved:
            return False, [NOT_A_DRAFT.format(id=task_id)]
        self.launched.append(task_id)
        return True, []

    async def ask_which(self, task_id: int, persons: list[people.Person]) -> None:
        """«Кого ты имел в виду?» would be shown: only noted."""
        self.asked.append((task_id, list(persons)))

    async def confirm_close(self, task_id: int) -> str | None:
        """A draft is closed at once and the line says so; a task in work would get the
        confirmation card, which is only noted."""
        self.closing.append(task_id)
        task = await self.repo.get_task(task_id)
        if task.status is TaskStatus.DRAFT:
            await self.repo.set_task_status(task_id, TaskStatus.CANCELLED)
            return CLOSED.format(id=task_id, title=task.title)
        return None

    # -- the state --

    async def build(self, case: SecretaryCase) -> None:
        """Write the case's state: tasks under their own numbers (a gap is a closed task that
        shows nowhere), then the cards, the memory, the stop and the chat's tail."""
        repo = self.repo
        specs = {spec.id: spec for spec in (*case.drafts, *case.tasks)}
        convs: dict[str, int] = {}
        for number in range(1, max(specs, default=0) + 1):
            spec = specs.get(number)
            if spec is None:
                filler = await repo.create_task("—", "—", {}, None, "UZS", [], [])
                await repo.set_task_status(filler, TaskStatus.CANCELLED)
                continue
            draft = isinstance(spec, DraftSpec)
            task_id = await repo.create_task(
                spec.title, spec.brief, {}, spec.budget,
                spec.currency if draft else "UZS", ["name"], [],
                source_text=spec.words if draft else spec.brief,
            )  # fmt: skip
            if task_id != number:  # a fresh database numbers from 1, one after another
                raise RuntimeError(f"задача #{number} получила номер {task_id}")
            if draft:
                await people.save_targets(repo, task_id, list(spec.people))
                await repo.set_kv(
                    missing_key(task_id),
                    json.dumps(list(spec.missing), ensure_ascii=False),
                )
                continue
            for person in spec.people:
                convs[person.username.casefold()] = await self._person(task_id, person)
            if spec.headline:
                await repo.set_task_notes(task_id, headline=spec.headline)
            await repo.set_task_status(task_id, _STATUS[spec.status])
        for card in case.cards:
            await self._card(card, convs, specs[card.task].title)
        self.gateway.contacts = [
            PeerInfo(_PEER_ID + 500 + n, entry.username, entry.name, False, entry.phone)
            for n, entry in enumerate(case.contacts)
        ]
        for fact in case.memory:
            await memory.remember(repo, fact, None, self.clock.now())
        if case.stopped:
            await repo.set_stopped(True)
        for row in case.chat:
            self.clock.advance(timedelta(minutes=1))
            direction = Direction.IN if row.speaker == "owner" else Direction.OUT
            await repo.add_owner_chat(direction, "text", row.text)
        self.clock.advance(timedelta(minutes=1))

    async def _person(self, task_id: int, person: PersonSpec) -> int:
        peer = _PEER_ID + len(self.gateway.peers)
        self.gateway.peers[person.username] = PeerInfo(
            peer,
            person.username,
            person.name or person.username.replace("_", " ").title(),
            False,
        )
        attached = await people.attach(
            self.repo, self.gateway, task_id, person.username
        )
        if not isinstance(attached, people.Attached):
            raise RuntimeError(f"{person.username}: {attached}")  # noqa: TRY004 - a case that cannot be set up
        for speaker, text in person.messages:
            self.clock.advance(timedelta(minutes=5))
            incoming = speaker == "counterpart"
            await self.repo.add_message(
                attached.conv_id,
                Direction.IN if incoming else Direction.OUT,
                MsgSource.COUNTERPART if incoming else MsgSource.AGENT,
                "text", text, None, self.clock.now(),
            )  # fmt: skip
        for state in _PATH[person.state]:
            await self.repo.set_state(attached.conv_id, state)
        return attached.conv_id

    async def _card(self, card: CardSpec, convs: dict[str, int], title: str) -> None:
        kind = ApprovalKind(card.kind)
        payload = {"question": card.question}
        conv_id = convs[card.about.casefold()] if card.about else None
        await self.repo.create_approval(card.task, conv_id, kind, payload)


# -- the model, recorded --


class _Recorder:
    """The run's model, keeping the secretary's conversation: the loop appends every tool's
    answer to the very list it hands the model, so the calls and the code's answers are read
    from there once the turn is over."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm
        self.thread: list[dict[str, Any]] = []

    def calls(self) -> list[ToolCallRecord]:
        answers: dict[str, tuple[str, bool]] = {}
        uses: list[dict[str, Any]] = []
        for message in self.thread:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for raw in content:
                block = _block(raw)
                if (
                    message.get("role") == "assistant"
                    and block.get("type") == "tool_use"
                ):
                    uses.append(block)
                elif block.get("type") == "tool_result":
                    answers[str(block.get("tool_use_id"))] = (
                        _text(block.get("content")),
                        bool(block.get("is_error")),
                    )
        records = []
        for use in uses:
            answer = answers.get(str(use.get("id")))
            args = use.get("input")
            records.append(
                ToolCallRecord(
                    name=str(use.get("name")),
                    input=dict(args) if isinstance(args, dict) else {},
                    accepted=answer is not None and not answer[1],
                    answer="" if answer is None else answer[0],
                )
            )
        return records

    async def agent_step(self, **kw: Any) -> AgentStep:
        if kw.get("purpose") == "secretary":
            self.thread = kw["messages"]
        return await self._llm.agent_step(**kw)

    async def judge(
        self, messages: list[str], allowed: list[str], hint: str
    ) -> LLMResult[JudgeVerdict]:
        return await self._llm.judge(messages, allowed, hint)

    async def simulate(self, role: str, transcript: str) -> LLMResult[str]:
        return await self._llm.simulate(role, transcript)

    async def search_step(self, **kw: Any) -> SearchStep:
        return await self._llm.search_step(**kw)


def _block(block: Any) -> dict[str, Any]:
    """A content block as a dict: the fake model gives dicts, the real one SDK objects."""
    if isinstance(block, dict):
        return block
    dump = getattr(block, "model_dump", None)
    return dump() if callable(dump) else {}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    return ""


# -- one run --


@dataclass
class CaseResult:
    """One run of one case. `error` is why the run itself did not work, if it did not;
    `stopped` how the secretary's loop ended (None: the model ended its turn)."""

    case_id: str
    run_idx: int
    checks: list[CheckResult]
    record: TurnRecord
    spent: RunCost
    steps: int = 0
    stopped: str | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(check.passed for check in self.checks)


async def run_case(
    case: SecretaryCase,
    llm: LLM,
    run_idx: int,
    *,
    settings: Settings | None = None,
) -> CaseResult:
    """Play the case once in a database of its own and score the turn.

    A model that does not answer is the result's `error`: the run counts as not passed and is
    still scored. A case that cannot even be set up raises.
    """
    recorder = _Recorder(llm)
    base = settings or Settings()
    with tempfile.TemporaryDirectory(
        prefix="tg-agent-secretary-eval-", ignore_cleanup_errors=True
    ) as folder:
        db_path = Path(folder) / "agent.db"
        clock = FakeClock(START)
        world = replace(base, data_dir=Path(folder))
        repo = await Repo.open(db_path, clock=clock)
        try:
            office = Office(repo, clock, world, recorder)
            await office.build(case)
            before = max((task.id for task in await repo.all_tasks()), default=0)
            error = None
            reply: SecretaryReply | None = None
            try:
                reply = await office.secretary.handle_messages(owner_messages(case))
            except Exception as failure:  # noqa: BLE001 - what broke is part of the result
                error = f"секретарь упал: {type(failure).__name__}: {failure}"
            made = sum(task.id > before for task in await repo.all_tasks())
            trace = await repo.events_of("secretary")
            info = trace[-1] if trace else {}
            stopped = info.get("stopped")
            if error is None and not trace:
                # The secretary logs its event only when its loop came to an end; a turn
                # that broke inside is swallowed (the owner reads «Не разобрал…»).
                error = "ход секретаря оборвался (подробности в журнале)"
            elif stopped == "error" and error is None:
                error = "модель не ответила (ошибка API)"
        finally:
            await repo.close()
        spent = await _spent(db_path)
    record = TurnRecord(tuple(recorder.calls()), reply.text if reply else None, made)
    return CaseResult(
        case.id,
        run_idx,
        run_checks(case.expect, record),
        record,
        spent,
        steps=int(info.get("steps", 0)),
        stopped=stopped,
        error=error,
    )


async def _spent(db_path: Path) -> RunCost:
    """The sum of the run's model calls, from the database the secretary wrote them to."""
    async with aiosqlite.connect(db_path) as db:
        [row] = await db.execute_fetchall(
            "SELECT COUNT(*), COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0),"
            " COALESCE(SUM(cache_read), 0), COALESCE(SUM(cache_write), 0),"
            " COALESCE(SUM(cost_usd), 0.0) FROM llm_calls"
        )
    return RunCost(*row)


# -- all of them --


@dataclass
class Report:
    """The runs of a set of cases, and what to say about them."""

    created_at: datetime
    runs: int
    model: str
    results: list[CaseResult] = field(default_factory=list)
    path: Path | None = None

    def case_ids(self) -> list[str]:
        return list(dict.fromkeys(result.case_id for result in self.results))

    def of(self, case_id: str) -> list[CaseResult]:
        return [result for result in self.results if result.case_id == case_id]

    def passes(self, case_id: str) -> int:
        return sum(result.passed for result in self.of(case_id))

    def pass_all(self, case_id: str) -> bool:
        """pass^k: every one of the k runs passed."""
        return self.passes(case_id) == self.runs

    @property
    def total_cost(self) -> float:
        return sum(result.spent.usd for result in self.results)

    def table(self) -> str:
        """The cases one to a line: how many runs passed, whether all did, what failed."""
        rows = [("Кейс", "Прошло", f"pass^{self.runs}", "Не прошло", "$ за прогон")]
        for case_id in self.case_ids():
            results = self.of(case_id)
            failed = dict.fromkeys(
                check.name for r in results for check in r.checks if not check.passed
            )
            if any(result.error for result in results):
                failed["ошибка прогона"] = None
            mean = sum(result.spent.usd for result in results) / len(results)
            rows.append(
                (
                    case_id,
                    f"{self.passes(case_id)}/{len(results)}",
                    "да" if self.pass_all(case_id) else "нет",
                    ", ".join(failed) or "-",
                    f"{mean:.3f}",
                )
            )
        widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
        lines = [
            "  ".join(
                cell.ljust(w) for cell, w in zip(row, widths, strict=True)
            ).rstrip()
            for row in rows
        ]
        every = sum(self.pass_all(case_id) for case_id in self.case_ids())
        lines.append(
            f"Кейсов, где прошли все прогоны (pass^{self.runs}): {every} из"
            f" {len(self.case_ids())}. Потрачено ${self.total_cost:.2f}"
        )
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        cases = []
        for case_id in self.case_ids():
            results = self.of(case_id)
            cases.append(
                {
                    "id": case_id,
                    "runs": len(results),
                    "passes": self.passes(case_id),
                    "pass_k": self.pass_all(case_id),
                    "failed": [
                        {"run": r.run_idx + 1, "check": c.name, "detail": c.detail}
                        for r in results
                        for c in r.checks
                        if not c.passed
                    ],
                    "errors": [
                        {"run": r.run_idx + 1, "error": r.error}
                        for r in results
                        if r.error
                    ],
                    "results": [_result_json(result) for result in results],
                }
            )
        return {
            "created_at": self.created_at.isoformat(),
            "runs": self.runs,
            "model": self.model,
            "total_cost_usd": round(self.total_cost, 4),
            "cases": cases,
        }

    def save(self) -> None:
        """Write the report where it belongs (nothing without a `path`)."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_json(), ensure_ascii=False, indent=2)
        self.path.write_text(text, encoding="utf-8")


def _result_json(result: CaseResult) -> dict[str, Any]:
    record = result.record
    return {
        "run": result.run_idx + 1,
        "passed": result.passed,
        "error": result.error,
        "steps": result.steps,
        "stopped": result.stopped,
        "cost_usd": round(result.spent.usd, 4),
        "tokens": {
            "input": result.spent.input_tokens,
            "output": result.spent.output_tokens,
            "cache_read": result.spent.cache_read,
            "cache_write": result.spent.cache_write,
            "calls": result.spent.calls,
        },
        "reply": record.reply,
        "drafts_made": record.drafts_made,
        "calls": [
            {
                "name": call.name,
                "input": dict(call.input),
                "accepted": call.accepted,
                "answer": call.answer[:ANSWER_CHARS],
            }
            for call in record.calls
        ],
        "checks": [asdict(check) for check in result.checks],
    }


async def run_all(
    cases: Sequence[SecretaryCase],
    llm: LLM,
    runs: int = 1,
    *,
    settings: Settings | None = None,
    out_dir: Path | None = None,
    say: Callable[[str], object] = print,
) -> Report:
    """Run each case `runs` times, one run after another, and report.

    `say` gets a line as each run ends and then the table. With `out_dir` the report is written
    there as `secretary-<UTC time>.json` after every run, so that a run stopped halfway still
    leaves what was done. The keyring holds a made-up owner meanwhile: nothing of a case can
    meet his real phone or address.
    """
    created = datetime.now(UTC)
    model = (settings or Settings()).models.intake
    report = Report(created, runs, model)
    if out_dir is not None:
        report.path = Path(out_dir) / f"secretary-{created:%Y-%m-%dT%H-%M-%S}Z.json"
    with fake_owner_keyring():
        for case in cases:
            for run_idx in range(runs):
                result = await run_case(case, llm, run_idx, settings=settings)
                report.results.append(result)
                report.save()
                verdict = "прошёл" if result.passed else "НЕ ПРОШЁЛ"
                failed = [check.name for check in result.checks if not check.passed]
                if result.error:
                    failed.append(result.error)
                why = f" ({'; '.join(failed)})" if failed else ""
                say(
                    f"{case.id} #{run_idx + 1}/{runs}: {verdict}{why},"
                    f" ${result.spent.usd:.3f}"
                )
    say(report.table())
    return report
