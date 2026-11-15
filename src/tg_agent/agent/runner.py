"""The runner of the agent core: wakes, what the Sender reports, and recovery after a crash.

`Runner` is the core's `ports.Wakeups` (the Scheduler calls `tick` every round) and its
`ports.SendOutcomes` (the Sender reports what became of each message). A **wake** takes the
task's due events, builds the context from the database, and lets the model act through the
tools until it ends its turn or a cap stops it; then the events are handled, or released for a
retry when the wake failed. Between wakes nothing is kept in memory: the database is the state.

The Inbox (`agent/inbox.Inbox`) is built by the caller with the runner's locks, so that the
inbox, the send pipeline and the cards share one lock per conversation:
`Inbox(repo, clock, settings, outcomes=runner, locks=runner.locks)`.
"""

from __future__ import annotations

import asyncio
import logging
import random
import secrets
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta

from tg_agent.agent import chats, memory, people
from tg_agent.agent.approvals import Approvals
from tg_agent.agent.context import build_context, seen_key
from tg_agent.agent.locks import ConvLocks
from tg_agent.agent.loop import LoopOutcome, run_loop
from tg_agent.agent.outgoing import Outgoing
from tg_agent.agent.tools import (
    TOOLS,
    SignIns,
    TaskResearch,
    ToolBox,
    money_card_key,
    money_limit,
    spend_cap,
    task_allowance,
    task_money_left,
)
from tg_agent.agent.unclear import is_file_item, is_unclear_item
from tg_agent.brain.llm import LLM, AgentStep
from tg_agent.clock import Clock
from tg_agent.config import Settings
from tg_agent.control.base import UI, Card
from tg_agent.control.cards import result_waits_key, task_money_card
from tg_agent.control.layout import UNOWNED_WAIT
from tg_agent.db import Repo
from tg_agent.gateway.base import Gateway, TgActionError
from tg_agent.models import (
    ApprovalKind,
    ApprovalStatus,
    Contact,
    Conversation,
    ConvState,
    Direction,
    EventKind,
    OutboxItem,
    OutboxStatus,
    Task,
    TaskStatus,
    Wake,
)
from tg_agent.policy.tg_safety import TgErrorKind
from tg_agent.tasks.outbox import mark_sent
from tg_agent.tasks.texts import quote, who

_log = logging.getLogger(__name__)

S = ConvState

MAX_PARALLEL_WAKES = 3
RETRY_AFTER = timedelta(minutes=5)  # a failed wake's events wait this long
FAILURES_TO_PAUSE = 3  # failed wakes in a row before the task stops
FAILS_TO_OWNER = 3  # sends failed in a row for one person before the owner is told
UNCLEAR_WAIT = timedelta(
    seconds=60
)  # how long after a broken-off send the chat is read
UNCLEAR_READ = 20  # how many of the chat's newest messages are read
UNCLEAR_FAILED_ROUNDS = 5  # reads of the chat that failed before the owner is told
# A failed read of the chat is tried again this late.
UNCLEAR_RETRY = timedelta(minutes=1)
# A wake not running in this process that started this long ago will never end by itself.
ABANDONED_AFTER = timedelta(minutes=10)

# The conversation states in which the person is to answer us: a stale reply is dropped and a
# silence is armed after each message.
_ANSWERING = frozenset({S.CONTACTED, S.TALKING})
# A wake that ended this way is over: its events are handled.
_FINISHED = {None: "done", "finished": "finished", "steps": "steps", "budget": "budget"}
_OVER = frozenset({TaskStatus.DONE, TaskStatus.CANCELLED})
# An outbox item in one of these is on its way (or may have gone): it will be reported.
_ON_ITS_WAY = frozenset(
    {OutboxStatus.PENDING, OutboxStatus.SENDING, OutboxStatus.UNKNOWN}
)
# A task with nothing scheduled is nudged this soon.
STALLED_NUDGE = timedelta(minutes=10)
_STALLED_KEY = "stalled:{id}"
_FAILURES_KEY = "task_failures:{id}"  # failed wakes of the task in a row

# For the owner.
DAY_CAP = "Дневной лимит денег исчерпан: задачи продолжатся завтра"
MONTH_CAP = (
    "Месячный лимит денег исчерпан: задачи остановлены до следующего месяца или пока ты"
    " не поднимешь лимит"
)
REFUSED = (
    "Задача #{id} «{title}» остановлена: модель три раза отказалась работать над ней."
    " Измени задачу или закрой её. Продолжить — напиши «продолжи задачу {id}»"
)
PAUSED = (
    "Задача #{id} «{title}» остановлена: три сбоя подряд ({error}). Продолжить —"
    " напиши «продолжи задачу {id}»"
)
CANNOT_CHECK = (
    "Не могу проверить, ушло ли сообщение для {who}: «{quote}». Посмотри чат сам"
)
# ... when it has nobody and its last search found nobody (fix pass E1.6, M2).
STALLED_NOBODY = (
    "Задача #{id}: поиск никого не нашёл. Назови, кому написать, ответом на карточку"
    " задачи, или закрой её."
)
STALLED = (
    "Задача #{id} «{title}» остановилась: агент не знает, что делать дальше. Ответь на"
    " это сообщение указанием или закрой задачу"
)
# For the model: the `about` of the nudge (the context shows a reminder's `about`).
STALLED_ABOUT = (
    "у задачи ничего не запланировано: никто не ждёт ответа, вопросов владельцу и"
    " напоминаний нет. Если всё сделано — заверши задачу итогом; если нет — сделай"
    " следующий шаг"
)
# Why the code finished with a person who stayed silent after the last allowed reminder.
GAVE_UP = "не ответил после напоминаний"
# For the model: why a message whose send broke off is given up.
UNCHECKED = "не удалось проверить, ушло ли сообщение"
_CAP_NOTICE_KEY = "cap_notice"


class Runner:
    """Wakes the task agent. `Wakeups` for the Scheduler, `SendOutcomes` for the Sender."""

    def __init__(
        self,
        repo: Repo,
        llm: LLM,
        gateway: Gateway,
        ui: UI,
        clock: Clock,
        settings: Settings,
        rng: random.Random,
        *,
        canary: str,
        owner_secrets: Sequence[str],
        inline: bool = False,
        searcher: TaskResearch | None = None,
        site_logins: SignIns | None = None,
    ) -> None:
        self._repo = repo
        # The owner's sign-ins from his phone: `research` hands them the walls it met.
        self._site_logins = site_logins
        # The search behind the agent's `research` tool; without one the tool says the search
        # is not available. A wake that searches may last minutes: it holds its task's lock
        # (and `_running` holds it) all along, so `_sweep` never takes it for a dead one.
        self._searcher = searcher
        self._llm = llm
        self._gateway = gateway
        self._ui = ui
        self._clock = clock
        self._settings = settings
        self._limits = settings.limits
        self._canary = canary
        self._inline = inline
        # One lock per conversation for the inbox, the send pipeline and the cards.
        self.locks = ConvLocks()
        self._outgoing = Outgoing(
            repo,
            gateway,
            llm,
            clock,
            settings,
            rng,
            canary=canary,
            owner_secrets=owner_secrets,
            locks=self.locks,
        )
        # Public: the bot's buttons call `approvals.owner_decision`.
        self.approvals = Approvals(
            repo,
            ui,
            clock,
            settings,
            self._outgoing,
            locks=self.locks,
            on_decided=self._keep_alive,
        )
        # One wake of a task at a time.
        self._task_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        # One keep-alive of a task at a time (a wake's end and an owner's decision may call
        # it at once). Not the wake's lock: a wake calls it while holding that one.
        self._alive_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        # The wakes started by `tick` (not inline), by task; kept until they end.
        self._running: dict[int, asyncio.Task[None]] = {}
        self._slots = asyncio.Semaphore(MAX_PARALLEL_WAKES)
        # Failed reads of the chat of an UNKNOWN item, and when the last one was; forgotten
        # on a restart.
        self._unread: dict[int, tuple[int, datetime]] = {}

    # -- launching and recovery --

    async def start(self, task_id: int) -> bool:
        """Launch a DRAFT task: ACTIVE, with one `created` event; True if it started. A task
        in any other status is left as it is (a second press of the launch button changes
        nothing, even when both presses read DRAFT)."""
        if not await self._repo.move_task_if(
            task_id, TaskStatus.DRAFT, TaskStatus.ACTIVE
        ):
            _log.info("task %s is not a draft: not started again", task_id)
            return False
        await self._repo.add_event(task_id, EventKind.CREATED)
        return True

    async def recover(self) -> int:
        """At start-up: what a crash left. A wake that had claimed events releases them (no
        attempt counted); one that had claimed nothing had started nothing, and is only
        closed. The cards of ACTIVE and PAUSED tasks are repaired (`_repair_cards`). A task
        with a released wake or a repair gets one `recovered` event (the agent is to look at
        what was done and what is queued), and every ACTIVE task is kept alive. Returns how
        many unfinished wakes there were."""
        repo = self._repo
        wakes = await repo.unfinished_wakes()
        # One `recovered` event per task, however many repairs.
        recovered: set[int] = set()
        for wake in wakes:
            if await self._interrupted(wake) and wake.task_id is not None:
                recovered.add(wake.task_id)
        tasks = await repo.all_tasks()
        for task in tasks:
            live = task.status in (TaskStatus.ACTIVE, TaskStatus.PAUSED)
            if live and await self._repair_cards(task.id):
                recovered.add(task.id)
            for conv in await repo.task_conversations(task.id) if live else ():
                # An «Ок» reply that went just before a crash (R98).
                await self.approvals.tell_ok(conv.id)
        for task_id in sorted(recovered):
            await repo.add_event(task_id, EventKind.RECOVERED)
        for task in tasks:
            if task.status is TaskStatus.ACTIVE:
                await self._keep_alive(task.id)
        return len(wakes)

    async def _repair_cards(self, task_id: int) -> bool:
        """What a crash can leave around the cards of a task: an approval whose card never
        went (it is expired), a person held for the owner by no card (reopened), and a chat
        the owner has with no card to hand it back (the card is sent). True if a person or
        an approval was repaired: the agent is to look again (`recovered`)."""
        repo = self._repo
        repaired = False
        for approval in await repo.pending_approvals():
            if approval.task_id == task_id and approval.control_msg_id is None:
                await repo.resolve_approval(
                    approval.id, approval.nonce, ApprovalStatus.EXPIRED
                )
                await repo.log_event(
                    "approval_expired_at_start",
                    approval.conversation_id,
                    approval_id=approval.id,
                )
                repaired = True
        for conv in await repo.task_conversations(task_id):
            async with self.locks.of(conv.id):
                conv = await repo.get_conversation(conv.id)
                cards = await repo.pending_approvals(conv.id)
                if conv.state is S.WAITING_OWNER and not cards:
                    await people.reopen(repo, conv.id)
                    repaired = True
                elif conv.state is S.HANDED_TO_OWNER and not any(
                    card.kind is ApprovalKind.HANDOFF for card in cards
                ):
                    await self.approvals.send_hand_back(task_id, conv)
        return repaired

    async def _interrupted(self, wake: Wake) -> bool:
        """Close a wake that will never end by itself (a crash, or an error in its own
        bookkeeping). Its events wait again, due now, without an attempt more: that is not
        the model's failure. True if it had claimed any."""
        if not await self._repo.claimed_events(wake.id):
            await self._repo.finish_wake(wake.id, "interrupted", wake.steps)
            return False
        await self._repo.release_wake(
            wake.id, "interrupted", retry_at=self._clock.now(), attempt=False
        )
        return True

    async def forget_failures(self, task_id: int) -> None:
        """The task starts again with no failed wakes counted (the owner resumed it)."""
        await self._repo.set_kv(_FAILURES_KEY.format(id=task_id), "0")

    # -- the rounds (`ports.Wakeups`) --

    async def tick(self) -> None:
        """One round: check the unclear sends, release the wakes an error left unfinished,
        then wake every task that has a due event and no wake running. Nothing while the
        agent is stopped.

        An error in this work of its own is raised to the Scheduler, which reports a failing
        step (once in ten minutes): a broken core must not hide in a log line every round.
        An error inside a wake stays inside the wake's own handling."""
        if await self._repo.is_stopped():
            return
        # A broken check of the unclear sends is raised only after the round's wakes: it
        # must not keep every task asleep.
        error = await self._check_unclear()
        await self._sweep()
        due = await self._repo.due_task_ids(self._clock.now())
        for task_id in due:
            if task_id in self._running or self._task_locks[task_id].locked():
                continue
            if self._inline:
                await self._wake_safely(task_id)
                continue
            job = asyncio.create_task(self._wake_in_slot(task_id))
            self._running[task_id] = job
            job.add_done_callback(lambda _, tid=task_id: self._running.pop(tid, None))
        if error is not None:
            raise error

    async def _sweep(self) -> None:
        """A wake that is not running in this process and started long ago will never end:
        an exception in its own bookkeeping left it. Its events must not wait for a
        restart."""
        now = self._clock.now()
        for wake in await self._repo.unfinished_wakes():
            task_id = wake.task_id
            if task_id is not None and (
                task_id in self._running or self._task_locks[task_id].locked()
            ):
                continue
            if now - wake.started_at > ABANDONED_AFTER:
                _log.warning("wake %s of task %s was left unfinished", wake.id, task_id)
                await self._interrupted(wake)

    async def idle(self) -> None:
        """Wait until no wake started by `tick` is running (tests, a clean stop)."""
        while self._running:
            await asyncio.gather(*list(self._running.values()), return_exceptions=True)

    async def stop(self) -> None:
        """A clean stop: the wakes started by `tick` are cancelled and awaited, so that none
        writes to the database after it is closed. What a cancelled wake leaves (its events
        claimed, its row unfinished) is released by `recover` at the next start."""
        jobs = list(self._running.values())
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)

    async def _wake_in_slot(self, task_id: int) -> None:
        async with self._slots:
            await self._wake_safely(task_id)

    async def _wake_safely(self, task_id: int) -> None:
        try:
            await self.wake(task_id)
        except Exception:  # noqa: BLE001 - one task's failure must not stop the others
            _log.exception("the wake of task %s failed", task_id)

    # -- one wake --

    async def wake(self, task_id: int) -> str | None:
        """One wake of the task; its outcome, None if there was nothing to do."""
        async with self._task_locks[task_id]:
            return await self._wake(task_id)

    async def _wake(self, task_id: int) -> str | None:
        repo = self._repo
        # A wake that waited for its slot may start after a /stop.
        if await repo.is_stopped():
            return None
        task = await repo.get_task(task_id)
        if task.status is not TaskStatus.ACTIVE:
            return None  # its events wait (a PAUSED task sees them when it resumes)
        if await self._out_of_money():
            return None
        left = await task_money_left(repo, self._settings, task_id)
        if left <= 0:
            await self._ask_for_money(task)
            return None  # its events wait for «Продолжить» (R76)
        await self._drop_overtaken_ok(task_id)
        wake_id, events = await repo.claim_events(task_id, self._clock.now())
        if not events:
            return None
        try:
            context = await build_context(
                repo,
                self._gateway,
                self._settings,
                self._clock,
                task=task,
                events=events,
                canary=self._canary,
            )
            toolbox = ToolBox(
                repo,
                self._ui,
                self._gateway,
                self._clock,
                self._settings,
                self._outgoing,
                self.approvals,
                task=task,
                wake_id=wake_id,
                context=context,
                searcher=self._searcher,
                site_logins=self._site_logins,
            )

            async def log_step(step: AgentStep, cost: float) -> None:
                await repo.log_llm_call(
                    None,
                    "task",
                    step.model,
                    step.usage,
                    cost,
                    step.stop_reason,
                    task_id=task_id,
                    wake_id=wake_id,
                )

            outcome = await run_loop(
                self._llm,
                purpose="task",
                model=self._settings.models.decide,
                system=context.system,
                content=context.content,
                tools=TOOLS,
                handler=toolbox,
                effort="medium",
                max_tokens=16000,
                max_steps=self._limits.wake_max_steps,
                max_cost=min(self._limits.wake_max_usd, left),
                on_step=log_step,
            )
        except Exception as error:  # noqa: BLE001 - a broken tool must not lose the events
            _log.exception("wake %s of task %s broke", wake_id, task_id)
            return await self._failed(task, wake_id, "error", type(error).__name__)
        # A task closed or finished meanwhile is looked at first, before the stop flag: the
        # wake is over, and its events must not be released into a task nobody will wake.
        status = (await repo.get_task(task_id)).status
        if toolbox.halted and status not in _OVER and await repo.is_stopped():
            # /stop during the wake: its events wait for the agent to start again, and no
            # failure is counted.
            await self._release(task_id, wake_id, "stopped", self._clock.now(), False)
            await repo.log_event("wake_stopped", None, task_id=task_id, wake_id=wake_id)
            return "stopped"
        if toolbox.halted and status is TaskStatus.PAUSED:
            # Paused during the wake: as for /stop, its events wait for the resume, its
            # messages stay new (no watermark), and no failure is counted.
            await self._release(task_id, wake_id, "paused", self._clock.now(), False)
            await repo.log_event("wake_paused", None, task_id=task_id, wake_id=wake_id)
            return "paused"
        return await self._ended(task, wake_id, outcome, context.seen_message_id)

    async def _drop_overtaken_ok(self, task_id: int) -> None:
        """Ruling R98: a reply the owner sent with «Ок» (queued by no wake) that the person
        wrote before is stale, as the Sender would find it at its turn. It is dropped before
        the wake claims its events, so that the wake answering him hears of it, and a reply
        of the agent's own does not replace the owner's unseen."""
        repo = self._repo
        for conv in await repo.task_conversations(task_id):
            if conv.last_in_at is None or not self.can_reanswer(conv):
                continue
            dropped = 0
            for item in await repo.outbox_of(conv.id):
                if (
                    item.status is OutboxStatus.PENDING
                    and item.wake_id is None
                    and item.created_at < conv.last_in_at
                ):
                    dropped += await repo.cancel_outbox(item.id)
            if dropped:
                await repo.log_event(
                    "outbox_cancelled", conv.id, reason="stale", items=dropped
                )
                await self.approvals.tell_ok(conv.id, stale=True)

    async def _release(
        self,
        task_id: int,
        wake_id: int,
        outcome: str,
        retry_at: datetime,
        attempt: bool,
    ) -> None:
        """Release the wake's events for a later wake. A task the owner closed (or the
        agent finished) while they were released has nobody to wake: they are dropped, never
        left open after the close dropped the rest."""
        await self._repo.release_wake(wake_id, outcome, retry_at, attempt=attempt)
        if (await self._repo.get_task(task_id)).status in _OVER:
            await self._repo.drop_task_events(task_id)

    async def _ended(
        self, task: Task, wake_id: int, outcome: LoopOutcome, seen_message_id: int
    ) -> str:
        repo = self._repo
        stopped = outcome.stopped
        if stopped in _FINISHED:
            word = _FINISHED[stopped]
            await repo.finish_wake(wake_id, word, outcome.steps)
            # Only a wake that ended: a failed one is repeated, its messages still new.
            await repo.set_kv(seen_key(task.id), str(seen_message_id))
            await self.forget_failures(task.id)
            if stopped in ("steps", "budget"):
                await repo.log_event(
                    "wake_capped",
                    None,
                    task_id=task.id,
                    wake_id=wake_id,
                    reason=stopped,
                    steps=outcome.steps,
                    cost=round(outcome.cost, 4),
                )
            await self._keep_alive(task.id, wake_ended=True)
            return word
        # "max_tokens", "refusal" or "error": nothing that step asked for was done; the
        # events wait, and the failure counts.
        word = stopped if stopped in ("max_tokens", "refusal") else "error"
        return await self._failed(
            task, wake_id, word, outcome.error or word, steps=outcome.steps
        )

    async def _failed(
        self, task: Task, wake_id: int, word: str, error: str, *, steps: int = 0
    ) -> str:
        """The wake failed: its events wait five minutes, and the failure is counted for the
        task (kv `task_failures:<id>`, whatever events the wakes had); the third in a row
        stops the task, and the owner is told once. A task finished or closed meanwhile has
        nobody to wake: its events are finished instead."""
        repo = self._repo
        status = (await repo.get_task(task.id)).status
        await repo.log_event(
            "wake_failed", None, task_id=task.id, wake_id=wake_id, error=error
        )
        if status not in (TaskStatus.ACTIVE, TaskStatus.PAUSED):
            await repo.finish_wake(wake_id, word, steps)
            return word
        await self._release(
            task.id, wake_id, word, self._clock.now() + RETRY_AFTER, True
        )
        key = _FAILURES_KEY.format(id=task.id)
        failures = int(await repo.get_kv(key) or 0) + 1
        await repo.set_kv(key, str(failures))
        # Only from ACTIVE as it is now: a task the owner paused or closed meanwhile stays
        # as he left it.
        if failures >= FAILURES_TO_PAUSE and await repo.move_task_if(
            task.id, TaskStatus.ACTIVE, TaskStatus.PAUSED
        ):
            await repo.log_event("task_paused", None, task_id=task.id, error=error)
            if word == "refusal":
                notice = REFUSED.format(id=task.id, title=task.title)
            else:
                notice = PAUSED.format(id=task.id, title=task.title, error=error)
            await self._tell(notice)
        return word

    # -- nothing is left with no way to wake up --

    async def _keep_alive(self, task_id: int, *, wake_ended: bool = False) -> None:
        """Make sure an ACTIVE task has something that will wake the agent or tell the
        owner. Run after every finished wake, after every owner's decision on a card and for
        every ACTIVE task at start-up.

        1. A person who is to answer us (our message is his chat's newest, nothing of ours on
           its way, no silence or reminder of his) gets his silence back; after the last
           allowed reminder and a silent day the code finishes with him instead.
        2. A task with nothing scheduled (no open event, nothing on its way, no pending card)
           gets one nudge (a reminder in ten minutes); the second time in a row the owner is
           told once. The count (kv `stalled:<id>`) starts again when a finished wake leaves
           something scheduled.

        A task whose result waits for «Завершить» (R109) gets neither: it waits for the owner.

        Two calls for one task run one after the other, so the owner is told only once.
        """
        async with self._alive_locks[task_id]:
            await self._keep_alive_now(task_id, wake_ended=wake_ended)

    async def _keep_alive_now(self, task_id: int, *, wake_ended: bool) -> None:
        repo = self._repo
        task = await repo.get_task(task_id)
        if task.status in (TaskStatus.ACTIVE, TaskStatus.PAUSED):
            await self._hand_back_missing(task_id)
        if task.status is not TaskStatus.ACTIVE:
            return
        # A task whose result waits for «Завершить» waits for the owner (R109): the agent
        # hears its people and its own reminders, but no silence is armed and nothing stalls.
        waits = await self._result_waits(task_id)
        for conv in await repo.task_conversations(task_id):
            if conv.state in _ANSWERING and not waits:
                await self._keep_waiting(task_id, conv)
        key = _STALLED_KEY.format(id=task_id)
        if waits or await self._scheduled(task_id):
            if wake_ended:
                await repo.set_kv(key, "0")
            return
        count = int(await repo.get_kv(key) or 0) + 1
        await repo.set_kv(key, str(count))
        if count == 1:
            await repo.add_event(
                task_id,
                EventKind.REMINDER,
                payload={"about": STALLED_ABOUT},
                due_at=self._clock.now() + STALLED_NUDGE,
            )
            await repo.log_event("task_nudged", None, task_id=task_id)
        elif count == 2:
            text = STALLED.format(id=task.id, title=task.title)
            if await self._found_nobody(task_id):
                text = STALLED_NOBODY.format(id=task.id)
            card = Card(text, task_id=task.id)
            try:
                msg_id = await self._ui.send(card)
            except Exception:  # noqa: BLE001 - told again the next time it stalls
                _log.exception("the stalled task's message to the owner failed")
                await repo.set_kv(key, "1")
                return
            # His reply to it is an instruction for this task (stage 3 reads the key).
            await repo.set_kv(f"task_card:{msg_id}", str(task.id))
            await repo.log_event("task_stalled", None, task_id=task_id)

    async def _result_waits(self, task_id: int) -> bool:
        return await self._repo.get_kv(result_waits_key(task_id)) == "1"

    async def _found_nobody(self, task_id: int) -> bool:
        """Has the task nobody, and did its last search find nobody?"""
        if await self._repo.task_conversations(task_id):
            return False
        ends = await self._repo.searches_of(task_id, "research_done")
        return bool(ends) and not ends[-1].get("found")

    async def _hand_back_missing(self, task_id: int) -> None:
        """A chat the owner has with no card to hand it back (the card failed to send on
        «Напишу сам»): the card is sent now, not only at the next start."""
        repo = self._repo
        for conv in await repo.task_conversations(task_id):
            if conv.state is not S.HANDED_TO_OWNER:
                continue
            async with self.locks.of(conv.id):
                conv = await repo.get_conversation(conv.id)
                if conv.state is S.HANDED_TO_OWNER and not any(
                    card.kind is ApprovalKind.HANDOFF
                    for card in await repo.pending_approvals(conv.id)
                ):
                    await self.approvals.send_hand_back(task_id, conv)

    async def _keep_waiting(self, task_id: int, conv: Conversation) -> None:
        """Step 1 of `_keep_alive` for one person in CONTACTED or TALKING. The check and the
        act run under his conversation's lock, on his conversation read again: a reply the
        inbox stores meanwhile must not get him closed as one who did not answer."""
        async with self.locks.of(conv.id):
            conv = await self._repo.get_conversation(conv.id)
            if conv.state in _ANSWERING:
                await self._keep_waiting_locked(task_id, conv)

    async def _keep_waiting_locked(self, task_id: int, conv: Conversation) -> None:
        repo = self._repo
        if any(item.status in _ON_ITS_WAY for item in await repo.outbox_of(conv.id)):
            return  # `on_sent` arms his silence when it goes
        if any(
            event.conversation_id == conv.id
            and event.kind in (EventKind.SILENCE, EventKind.REMINDER)
            for event in await repo.open_events(task_id)
        ):
            return
        newest = await repo.last_messages(conv.id, 1)
        if not newest or newest[0].direction is not Direction.OUT:
            return  # he wrote last (or nothing went yet): not a silence
        now = self._clock.now()
        wait = timedelta(hours=self._limits.followup_after_hours)
        last_out = conv.last_out_at or newest[0].created_at
        if conv.followups_sent >= self._limits.max_followups and now >= last_out + wait:
            # The agent would be woken for him every day for ever.
            if await people.close(repo, self._gateway, conv.id, GAVE_UP):
                await repo.log_event("person_gave_up", conv.id, task_id=task_id)
            return
        due = last_out + wait
        await repo.arm_silence(task_id, conv.id, due if due > now else now + wait)

    async def _scheduled(self, task_id: int) -> bool:
        """Will anything wake the agent for this task or reach the owner about it?"""
        repo = self._repo
        if await repo.open_events(task_id):
            return True
        if any(
            approval.task_id == task_id for approval in await repo.pending_approvals()
        ):
            return True
        for conv in await repo.task_conversations(task_id):
            if any(
                item.status in _ON_ITS_WAY for item in await repo.outbox_of(conv.id)
            ):
                return True
        return False

    async def _out_of_money(self) -> bool:
        """Has today's or this month's spend reached its cap? Then no wake starts (the events
        wait), and the owner is told once a day."""
        local = self._clock.now().astimezone(self._settings.schedule.zone)
        cap = await spend_cap(self._repo, self._clock, self._settings)
        if cap == "month":
            text = MONTH_CAP
        elif cap == "day":
            text = DAY_CAP
        else:
            return False
        today = local.date().isoformat()
        # Marked once it went: a notice that failed is tried again the next round.
        if await self._repo.get_kv(_CAP_NOTICE_KEY) != today and await self._tell(text):
            await self._repo.set_kv(_CAP_NOTICE_KEY, today)
        return True

    async def _ask_for_money(self, task: Task) -> None:
        """The task spent its allowance: a card «Продолжить» (R76), again once a day while
        nobody presses it (an unpressed card settles after `UNOWNED_WAIT`), each with a new
        code. A card that failed to go is tried again the next round."""
        repo, key, now = self._repo, money_card_key(task.id), self._clock.now()
        allowance = await task_allowance(repo, self._settings, task.id)
        stored = (await repo.get_kv(key) or "").split(" ")
        if (
            len(stored) == 3
            and stored[0] == str(allowance)
            and now - datetime.fromisoformat(stored[2]) < UNOWNED_WAIT
        ):
            return
        code = secrets.token_hex(3)
        await repo.set_kv(key, f"{allowance} {code} {now.isoformat()}")
        spent = await repo.task_cost(task.id)
        usd = await money_limit(repo, self._settings, "task")
        try:
            await self._ui.send(task_money_card(task.id, task.title, spent, usd, code))
        except Exception:  # noqa: BLE001 - asked again the next round
            _log.exception("the task's money card failed")
            await repo.set_kv(key, "")

    async def _tell(self, text: str) -> bool:
        """A notice to the owner; one that fails is only logged. True if it went."""
        try:
            await self._ui.notify(text)
        except Exception:  # noqa: BLE001 - the core goes on without the notice
            _log.exception("a notice to the owner failed")
            return False
        return True

    # -- unclear sends --

    async def _check_unclear(self) -> Exception | None:
        """A message whose send broke off (UNKNOWN) may or may not be in the chat: a minute
        after the attempt, the chat is read. Found: it went out. Not found: it goes back to
        the queue. A read that fails is tried again a minute later; after five failed reads
        the owner looks himself, and the agent hears that the message may not have gone.
        Returns the first error of the check's own bookkeeping, for `tick` to raise."""
        now = self._clock.now()
        error: Exception | None = None
        for item in await self._repo.outbox_in(OutboxStatus.UNKNOWN):
            if item.tried_at is None:
                await self._repo.mark_outbox_tried(item.id, now)
                continue
            if now - item.tried_at < UNCLEAR_WAIT:
                continue
            unread = self._unread.get(item.id)
            if unread is not None and now - unread[1] < UNCLEAR_RETRY:
                continue
            try:
                await self._check_one(item)
            except Exception as exc:  # noqa: BLE001 - the others are checked, then it is raised
                _log.exception("could not check the unclear send %s", item.id)
                error = error or exc
        return error

    async def _check_one(self, item: OutboxItem) -> None:
        repo = self._repo
        conv = await repo.get_conversation(item.conversation_id)
        contact = await repo.get_contact(conv.contact_id)
        # After our newest stored message, not the person's: his quick reply to the message
        # that broke off is newer than it, and reading after the reply would miss it.
        after = await repo.last_tg_msg_id(conv.id, Direction.OUT) or 0
        try:
            messages = await self._gateway.recent_messages(
                contact.peer_id, after_msg_id=after, limit=UNCLEAR_READ
            )
        except Exception as error:  # noqa: BLE001 - any failed read is a failed read
            # Telethon's RPC errors and a dropped link are not wrapped in TgActionError:
            # they count toward the five reads too, and the round goes on.
            reads = self._unread.get(item.id, (0, self._clock.now()))[0] + 1
            self._unread[item.id] = (reads, self._clock.now())
            kind = (
                error.kind.value
                if isinstance(error, TgActionError)
                else type(error).__name__
            )
            await repo.log_event(
                "outbox_check_failed", conv.id, outbox_id=item.id, kind=kind
            )
            if reads >= UNCLEAR_FAILED_ROUNDS:
                self._unread.pop(item.id, None)
                await self._give_up(item, conv, contact)
            return
        self._unread.pop(item.id, None)
        is_file = is_file_item(item)
        stored = await repo.out_tg_msg_ids(conv.id)
        found = next(
            (
                message
                for message in messages
                if is_unclear_item(
                    item,
                    out=message.out,
                    text=message.text,
                    date=message.date,
                    tg_msg_id=message.tg_msg_id,
                    stored_out_ids=stored,
                )
            ),
            None,
        )
        async with self.locks.of(conv.id):
            # Afresh under the lock: the inbox may have found it meanwhile (the owner's
            # app reported our message), or the owner settled it.
            if (await repo.get_outbox(item.id)).status is not OutboxStatus.UNKNOWN:
                return
            if found is None:
                await repo.mark_outbox(item.id, OutboxStatus.PENDING)
                await repo.reschedule_outbox(item.id, self._clock.now())
                await repo.log_event("outbox_requeued", conv.id, outbox_id=item.id)
                return
            conv = await repo.get_conversation(conv.id)
            first = await mark_sent(
                repo, item, conv, found.tg_msg_id, found.date, is_file=is_file
            )
            await self.on_sent(item, conv, found.date, first=first)
            await repo.log_event("outbox_found_sent", conv.id, outbox_id=item.id)

    async def _give_up(
        self, item: OutboxItem, conv: Conversation, contact: Contact
    ) -> None:
        """The chat could not be read five times: the item is FAILED, the owner looks at the
        chat himself, and the agent learns that the message may not have gone."""
        repo = self._repo
        async with self.locks.of(conv.id):
            # Afresh under the lock: the inbox may have found it meanwhile.
            if (await repo.get_outbox(item.id)).status is not OutboxStatus.UNKNOWN:
                return
            await repo.mark_outbox(item.id, OutboxStatus.FAILED, error="unclear")
            await repo.log_event("outbox_failed", conv.id, reason="unclear")
            if (await repo.get_task(conv.task_id)).status not in _OVER:
                await repo.add_event(
                    conv.task_id,
                    EventKind.SEND_REFUSED,
                    conv_id=conv.id,
                    payload={"reason": UNCHECKED, "kind": "unclear", "closed": False},
                )
        shown = quote(item.text) if item.text.strip() else item.attach_path
        await self._tell(CANNOT_CHECK.format(who=who(contact), quote=shown))

    # -- what the Sender reports (`ports.SendOutcomes`) --

    def can_reanswer(self, conv: Conversation) -> bool:
        """The agent answers a newer message in these states, so a reply it made stale is
        dropped; elsewhere (a card waits, the chat is the owner's, he is closed) what was
        queued goes out as it is."""
        return conv.state in _ANSWERING

    async def excluded(self, conv: Conversation) -> bool:
        return await memory.is_excluded(
            self._repo, await self._repo.get_contact(conv.contact_id)
        )

    async def on_sent(
        self, item: OutboxItem, conv: Conversation, at: datetime, *, first: bool
    ) -> None:
        """The person is to answer: wake the agent if he stays silent. Never takes the
        conversation's lock: the inbox calls this while holding it."""
        if item.wake_id is None:  # queued by no wake: maybe the owner's «Ок» reply (R98)
            await self.approvals.tell_ok(conv.id)
        if first:  # his chat: no sound, into the folder (R105, R106)
            await chats.first_sent(self._repo, self._gateway, self._settings, conv)
        fresh = await self._repo.get_conversation(conv.id)
        if fresh.state not in _ANSWERING:
            return
        task = await self._repo.get_task(fresh.task_id)
        if task.status is not TaskStatus.ACTIVE or await self._result_waits(task.id):
            return
        due = at + timedelta(hours=self._limits.followup_after_hours)
        await self._repo.arm_silence(fresh.task_id, fresh.id, due)

    async def on_stale(self, conv_id: int) -> None:
        """The message that made the reply stale has its `incoming` event; an «Ок» reply
        dropped with it is told to the agent (R98)."""
        await self.approvals.tell_ok(conv_id, stale=True)

    async def on_unclear(self, item: OutboxItem) -> None:
        """A send broke off: the chat is read a minute later (`tick`). Nothing to the owner."""
        await self._repo.mark_outbox_tried(item.id, self._clock.now())
        await self._repo.log_event(
            "outbox_unknown_pending", item.conversation_id, outbox_id=item.id
        )

    async def on_refused(
        self,
        item: OutboxItem,
        conv: Conversation,
        kind: TgErrorKind,
        reason: str,
        cancelled: int,
    ) -> None:
        """Telegram refused this chat for good: the person is closed, and the agent decides
        what follows (no message to the owner)."""
        await people.close(self._repo, self._gateway, conv.id, reason)
        task = await self._repo.get_task(conv.task_id)
        if task.status not in _OVER:  # a finished task has nobody to wake
            await self._repo.add_event(
                conv.task_id,
                EventKind.SEND_REFUSED,
                conv_id=conv.id,
                payload={"reason": reason, "kind": kind.value, "closed": True},
            )
        await self._repo.log_event(
            "outbox_failed", conv.id, reason=kind.value, cancelled=cancelled
        )

    async def on_failed(
        self, item: OutboxItem, conv: Conversation, code: str, reason: str
    ) -> bool:
        """The Sender failed a message by itself: in an ACTIVE task the agent hears of it and
        decides (the person keeps his state); otherwise the Sender tells the owner. So does
        the third failure in a row for this person (the item is FAILED already): a send that
        keeps failing must not wake the agent, and write to the person, again and again."""
        task = await self._repo.get_task(conv.task_id)
        if task.status is not TaskStatus.ACTIVE:
            return False
        if (
            people.failed_in_a_row(await self._repo.outbox_of(conv.id))
            >= FAILS_TO_OWNER
        ):
            return False
        await self._repo.add_event(
            conv.task_id,
            EventKind.SEND_REFUSED,
            conv_id=conv.id,
            payload={"reason": reason, "kind": code, "closed": False},
        )
        return True
