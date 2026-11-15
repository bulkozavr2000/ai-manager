"""What the owner does to a task in the agent core: launch, steer, pause, resume, close; stop and
resume all.

Every method returns the short line the owner sees (a button's toast or a reply to a command),
in Russian, addressed to him as «ты». What the agent is to hear goes to it as an event; what it
must not do any more is held back by the task's status, which every tool call, every send and
every wake reads afresh (`agent/tools.ToolBox`, `agent/outgoing.Outgoing.send`, `Runner.wake`).
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections import defaultdict

from tg_agent.agent import chats, memory, people
from tg_agent.agent.people import task_targets
from tg_agent.agent.runner import Runner
from tg_agent.brain.context import hide_amount
from tg_agent.clock import Clock
from tg_agent.config import Settings
from tg_agent.control.cards import result_wake_key, result_waits_key
from tg_agent.db import NotFound, Repo
from tg_agent.gateway.base import Gateway
from tg_agent.models import (
    ApprovalStatus,
    ConvState,
    EventKind,
    Task,
    TaskStatus,
)
from tg_agent.policy.tg_safety import human_delay, next_working_time
from tg_agent.search.contacts import show
from tg_agent.tasks.recovery import replay_missed, replay_start
from tg_agent.tasks.templates import DECLINE, pick
from tg_agent.tasks.texts import who

S = ConvState

_log = logging.getLogger(__name__)

# For the owner.
NOBODY_TO_WRITE = (
    "Никому не написал: ни один контакт не нашёлся. Поправь контакты реплаем на карточку"
    " задачи."
)
NO_TASK = "Задачи #{id} нет"
NOT_A_DRAFT = "Задача уже запущена или закрыта"
EMPTY_INSTRUCTION = "Пустое указание"
PASSED_ON = "Передал агенту"
PASSED_ON_PAUSED = (
    "Передал. Задача на паузе: агент увидит это, когда ты её продолжишь"
    " — напиши «продолжи задачу {id}»"
)
TAKEN_UP = "Задача #{id} снова в работе: передал агенту"
NOT_IN_WORK = "Эта задача не в работе"
PAUSED = "Пауза в задаче #{id}. Продолжить — напиши «продолжи задачу {id}»"
PAUSED_ALREADY = "Задача #{id} уже на паузе"
NOT_WORKING = "Задача #{id} не в работе"
RESUMED = "Задача #{id} продолжена"
NOT_PAUSED = "Задача #{id} не на паузе"
DRAFT_CLOSED = "Задача закрыта"
CLOSED = "Задача #{id} закрыта"
SAID_GOODBYE = ". Попрощался: {who}"
CLOSED_ALREADY = "Задача #{id} уже закрыта"
ACKNOWLEDGED = "Задача #{id} завершена"
NOT_DONE = "Задача #{id} не завершена агентом — завершать нечего"
# Why a person of a task whose result the owner accepted was closed («Завершить», R56).
TASK_DONE = "задача завершена"

# Why a person was closed: the owner sees it in the pinned status, the model in «Люди».
TASK_CLOSED = "задача закрыта"
BUSY = "занят в задаче #{id}"

# For the owner: a target the gateway failed on in a way `people.attach` does not know.
CANNOT_OPEN = "не удалось открыть ({error})"
# For the model (a `send_refused` event): a reply the pause cancelled.
UNSENT = "задача была на паузе: сообщение не ушло"


class TaskOps:
    """What the owner does to a task: launch, steer, pause, resume, close; stop and resume all."""

    def __init__(
        self,
        repo: Repo,
        gateway: Gateway,
        clock: Clock,
        settings: Settings,
        runner: Runner,
        rng: random.Random,
    ) -> None:
        self._repo = repo
        self._gateway = gateway
        self._clock = clock
        self._settings = settings
        self._runner = runner
        self._rng = rng
        # One owner's action on a task at a time: two quick presses of a button run one
        # after the other, and the second finds the first one's result. The status moves
        # themselves are conditional too (`Repo.move_task_if`): a wake changes it as well.
        self._task_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def _task(self, task_id: int) -> Task | None:
        try:
            return await self._repo.get_task(task_id)
        except NotFound:
            return None

    async def _status(self, task_id: int) -> TaskStatus:
        return (await self._repo.get_task(task_id)).status

    # -- launch --

    async def launch(self, task_id: int) -> tuple[bool, list[str]]:
        """Attach the people the owner named and hand the task to the agent. Returns whether
        it was launched and the notes for the owner (who could not be attached, and why).

        People attached to the draft before count, without a note: a launch that attaches
        nobody new but finds people in the task still starts it. A draft whose owner named
        nobody starts with nobody: its agent finds people itself (ruling R26). One that named
        people of whom nobody could be attached stays a DRAFT for the owner to fix."""
        async with self._task_locks[task_id]:
            return await self._launch(task_id)

    async def _launch(self, task_id: int) -> tuple[bool, list[str]]:
        repo = self._repo
        task = await self._task(task_id)
        if task is None:
            return False, [NO_TASK.format(id=task_id)]
        if task.status is not TaskStatus.DRAFT:
            return False, [NOT_A_DRAFT]
        targets = await task_targets(repo, task_id)
        notes: list[str] = []
        for target in targets:
            try:
                result = await people.attach(repo, self._gateway, task_id, target)
            except Exception as exc:  # noqa: BLE001 - one target must not stop the others
                # `attach` turns Telegram's refusals into reasons; a dropped link or an RPC
                # error is not one of them.
                _log.exception("could not attach %s to task %s", target, task_id)
                result = people.NotAttached(
                    CANNOT_OPEN.format(error=type(exc).__name__)
                )
            if isinstance(result, people.NotAttached):
                if result.reason == people.IN_TASK_ALREADY:
                    continue  # he is in this draft: that is no failure
                notes.append(f"{show(target)}: {result.reason}")
        # A draft whose owner named nobody starts empty: its agent finds people (ruling
        # R26). One that named people of whom nobody could be attached does not: the owner
        # meant those people, and the notes say why each failed.
        if targets and not any(
            conv.state in people.OPEN_STATES
            for conv in await repo.task_conversations(task_id)
        ):
            notes.append(NOBODY_TO_WRITE)
            await repo.log_event("task_not_launched", None, task_id=task_id)
            return False, notes
        if not await self._runner.start(task_id):
            return False, [NOT_A_DRAFT]
        # Its people are live only now that the task is in work (`people.listened`):
        # without this the gateway would drop their replies.
        await people.listen(repo, self._gateway)
        await repo.log_event("task_launched", None, task_id=task_id)
        return True, notes

    # -- the owner's word --

    async def instruct(
        self,
        task_id: int,
        text: str,
        *,
        conv_id: int | None = None,
    ) -> str:
        """The owner's instruction for the task's agent (`conv_id`: about one person of this
        task; another conversation is ignored). It reaches the agent as an
        `owner_instruction` event, the price ceiling masked: the model never sees it,
        whoever wrote it. A finished task is taken up again by it."""
        async with self._task_locks[task_id]:
            return await self._instruct(task_id, text, conv_id)

    async def _instruct(self, task_id: int, text: str, conv_id: int | None) -> str:
        repo = self._repo
        task = await self._task(task_id)
        if task is None:
            return NO_TASK.format(id=task_id)
        if task.status not in (TaskStatus.ACTIVE, TaskStatus.PAUSED, TaskStatus.DONE):
            return NOT_IN_WORK
        text = text.strip()
        if not text:
            return EMPTY_INSTRUCTION
        if conv_id is not None and not any(
            conv.id == conv_id for conv in await repo.task_conversations(task_id)
        ):
            conv_id = None  # not a person of this task: the words are about the task
        status = task.status
        line = ""
        if await repo.get_kv(result_waits_key(task_id)):
            # His words on a result continue the task: nothing waits for «Завершить» (R109).
            await repo.set_kv(result_waits_key(task_id), "")
            await repo.set_kv(result_wake_key(task_id), "")
            if status is TaskStatus.ACTIVE:
                line = TAKEN_UP.format(id=task_id)
        if status is TaskStatus.DONE:
            if await self._take_up(task_id, status):
                line = TAKEN_UP.format(id=task_id)
            else:  # the agent or the owner moved it meanwhile: as it is now
                status = await self._status(task_id)
        if not line:
            if status is TaskStatus.ACTIVE:
                line = PASSED_ON
            elif status is TaskStatus.PAUSED:
                line = PASSED_ON_PAUSED.format(id=task_id)
            else:
                return NOT_IN_WORK
        # The event goes last: after a finished task's busy people are let go, so that the
        # wake it starts cannot write to one of them.
        await repo.add_event(
            task_id,
            EventKind.OWNER_INSTRUCTION,
            conv_id=conv_id,
            payload={"text": hide_amount(text, task.ceiling)},
            due_at=self._clock.now(),
        )
        await repo.log_event("task_instructed", None, task_id=task_id)
        return line

    async def _take_up(self, task_id: int, was: TaskStatus) -> bool:
        """A DONE (or CANCELLED) task goes back to work; False if it is not `was` any
        more. Its people kept their states; one who is in work in another task by now (his
        replies are read there) is closed here first, while the task is over and no wake of
        it can write to him. Once it is ACTIVE, what the others wrote meanwhile is read
        (`_catch_up`)."""
        repo = self._repo
        for conv in await repo.task_conversations(task_id):
            if conv.state not in people.OPEN_STATES:
                continue
            busy = await self._busy_elsewhere(task_id, conv.contact_id)
            if busy is None:
                continue
            async with self._runner.locks.of(conv.id):
                await people.close(repo, self._gateway, conv.id, BUSY.format(id=busy))
        moved = await repo.move_task_if(task_id, was, TaskStatus.ACTIVE)
        await people.listen(repo, self._gateway)
        if moved:
            await self._catch_up(task_id)
        return moved

    async def _catch_up(self, task_id: int) -> None:
        """What the task's open people wrote while it was finished was not stored (nobody
        listened to them): each chat is read from its newest stored message, as at start-up
        (`tasks/recovery`), so that the next wake sees it as new. A chat that cannot be read
        is logged, and the others are read all the same."""
        repo = self._repo
        for conv in await repo.task_conversations(task_id):
            if conv.state not in people.OPEN_STATES:
                continue
            try:
                start = await replay_start(repo, conv)
                if start is not None:
                    await replay_missed(repo, self._gateway, [start])
            except Exception:  # noqa: BLE001 - one chat must not keep the others unread
                _log.exception("task %s: chat %s not read on take-up", task_id, conv.id)

    async def _busy_elsewhere(self, task_id: int, contact_id: int) -> int | None:
        """The other task the person is in work in by now (his live conversation is there),
        or None."""
        contact = await self._repo.get_contact(contact_id)
        busy = await self._repo.blocking_conversation(contact.peer_id)
        if busy is None or busy.task_id == task_id:
            return None
        return busy.task_id

    # -- pause and resume --

    async def pause(self, task_id: int) -> str:
        """ACTIVE → PAUSED. What is queued for its people is cancelled, and the agent hears
        of each such message (a `send_refused` event that waits for the resume); its other
        events wait too, and messages that come meanwhile are only stored (the Inbox's rule):
        the agent sees them as new after the resume."""
        async with self._task_locks[task_id]:
            return await self._pause(task_id)

    async def _pause(self, task_id: int) -> str:
        repo = self._repo
        task = await self._task(task_id)
        if task is None:
            return NO_TASK.format(id=task_id)
        # The status first: a send judged meanwhile reads it under the conversation's lock
        # and queues nothing; what was queued before is cancelled below, under the same lock.
        if task.status is not TaskStatus.ACTIVE or not await repo.move_task_if(
            task_id, TaskStatus.ACTIVE, TaskStatus.PAUSED
        ):
            if await self._status(task_id) is TaskStatus.PAUSED:
                return PAUSED_ALREADY.format(id=task_id)
            return NOT_WORKING.format(id=task_id)
        for conv in await repo.task_conversations(task_id):
            async with self._runner.locks.of(conv.id):
                if await repo.cancel_pending(conv.id):
                    # The reply is gone: the agent decides after the resume what to write.
                    await repo.add_event(
                        task_id,
                        EventKind.SEND_REFUSED,
                        conv_id=conv.id,
                        payload={"reason": UNSENT, "kind": "paused", "closed": False},
                        due_at=self._clock.now(),
                    )
        await repo.log_event("task_paused", None, task_id=task_id)
        return PAUSED.format(id=task_id)

    async def resume(self, task_id: int) -> str:
        """PAUSED → ACTIVE, with no failed wakes counted (a task the Runner paused after
        three failures starts clean), and one `resumed` event that wakes the agent."""
        async with self._task_locks[task_id]:
            return await self._resume(task_id)

    async def _resume(self, task_id: int) -> str:
        repo = self._repo
        task = await self._task(task_id)
        if task is None:
            return NO_TASK.format(id=task_id)
        if not await repo.move_task_if(task_id, TaskStatus.PAUSED, TaskStatus.ACTIVE):
            return NOT_PAUSED.format(id=task_id)
        await self._runner.forget_failures(task_id)
        await repo.add_event(task_id, EventKind.RESUMED, due_at=self._clock.now())
        await repo.log_event("task_resumed", None, task_id=task_id)
        return RESUMED.format(id=task_id)

    # -- close --

    async def close(self, task_id: int) -> str:
        """The owner closes the task: CANCELLED, for good. Each person still open is closed.
        Where a talk is cut off — a person in TALKING in an ACTIVE or PAUSED task — he first
        gets the fixed goodbye (`templates.DECLINE`) in his language, queued straight into
        the outbox: the code's own words, no model and no gates. Nobody gets one in a DONE
        task or one whose result waits (its result may be a deal with him), in a chat the
        owner writes in himself right now, or when the person is in work in another task by
        now. The task's pending cards expire and its waiting events are dropped."""
        async with self._task_locks[task_id]:
            return await self._close(task_id)

    async def _close(self, task_id: int) -> str:
        repo = self._repo
        task = await self._task(task_id)
        if task is None:
            return NO_TASK.format(id=task_id)
        # The status first: a tool call, a send or a card of a wake running meanwhile reads
        # it and does nothing more (`ToolBox`, `Outgoing.send`'s second look under the lock).
        if task.status is TaskStatus.CANCELLED or not await repo.move_task_if(
            task_id, task.status, TaskStatus.CANCELLED
        ):
            if await self._status(task_id) is TaskStatus.CANCELLED:
                return CLOSED_ALREADY.format(id=task_id)
            return await self._close(
                task_id
            )  # it moved meanwhile: close it as it is now
        # A task whose result waits is as good as done: its result may be a deal (R109).
        cut_off = task.status in (TaskStatus.ACTIVE, TaskStatus.PAUSED) and not (
            await repo.get_kv(result_waits_key(task_id))
        )
        told: list[str] = []
        for conv in await repo.task_conversations(task_id):
            async with self._runner.locks.of(conv.id):
                if await self._close_person(task_id, conv.id, goodbye=cut_off):
                    told.append(who(await repo.get_contact(conv.contact_id)))
        for approval in await repo.pending_approvals():
            if approval.task_id == task_id:
                await repo.resolve_approval(
                    approval.id, approval.nonce, ApprovalStatus.EXPIRED
                )
        await repo.drop_task_events(task_id)
        await repo.log_event("task_closed", None, task_id=task_id)
        await people.listen(repo, self._gateway)  # nobody of it is heard now (R104)
        await chats.task_ended(repo, self._gateway, task_id)  # sound back (R105)
        if task.status is TaskStatus.DRAFT:
            return DRAFT_CLOSED
        line = CLOSED.format(id=task_id)
        if told:
            line += SAID_GOODBYE.format(who=", ".join(told))
        return line

    async def _close_person(self, task_id: int, conv_id: int, *, goodbye: bool) -> bool:
        """One person of a task being closed, under his conversation's lock. True if he got
        the goodbye (only with `goodbye`, and only where his talk is cut off)."""
        repo = self._repo
        conv = await repo.get_conversation(conv_id)  # afresh, under the lock
        if conv.state not in people.OPEN_STATES:
            return False
        await repo.cancel_pending(conv.id)
        now = self._clock.now()
        owner_writes = conv.paused_until is not None and conv.paused_until > now
        goodbye = goodbye and conv.state is S.TALKING and not owner_writes
        if goodbye and not await repo.sent_texts(conv.id):
            goodbye = False  # nothing of ours reached him: it would be his first message
        if goodbye and await self._busy_elsewhere(task_id, conv.contact_id) is not None:
            goodbye = False  # it would land in the middle of his talk in the other task
        if goodbye:
            contact = await repo.get_contact(conv.contact_id)
            text = pick(DECLINE, conv.language or contact.language)
            send_at = next_working_time(
                now + human_delay(self._rng, self._settings.limits),
                self._settings.schedule,
            )
            await repo.enqueue(conv.id, text, send_at)
        await people.close(repo, self._gateway, conv.id, TASK_CLOSED)
        return goodbye

    async def exclude(self, contact: str) -> int:
        """The owner said not to work with `contact` any more (ruling R92): every open talk
        with him, in any task, is closed with no goodbye, and what waits in the queue of
        every talk with him, closed ones too, is cancelled (R97), and the gateway hears him no
        more (R104). Returns how many were closed."""
        repo, key, closed = self._repo, contact.casefold(), 0
        for task in await repo.all_tasks():
            for conv in await repo.task_conversations(task.id):
                person = await repo.get_contact(conv.contact_id)
                if key not in memory.identifiers(person):
                    continue
                async with self._runner.locks.of(conv.id):
                    await repo.cancel_pending(conv.id)
                    fresh = await repo.get_conversation(conv.id)
                    if fresh.state not in people.OPEN_STATES:
                        continue
                    closed += await people.close(
                        repo, self._gateway, conv.id, memory.EXCLUDED
                    )
        await people.listen(repo, self._gateway)
        return closed

    async def listen(self) -> None:
        """The owner's memory changed («забудь»): who the gateway hears may have too."""
        await people.listen(self._repo, self._gateway)

    # -- the owner accepts the result --

    async def acknowledge(self, task_id: int) -> str:
        """«Завершить» under a result (rulings R56, R109): only the owner ends a task. A
        task whose result waits becomes DONE; its people still open are closed without a
        goodbye (R25: its result may be a deal with one of them, and what the last wake
        queued still goes), its pending cards expire, its result waits no more, nobody of it
        is heard and the chats the program muted get their sound back. A task with no result
        waiting is left as it is; a DONE one is ended again (a second press)."""
        async with self._task_locks[task_id]:
            repo = self._repo
            task = await self._task(task_id)
            if task is None:
                return NO_TASK.format(id=task_id)
            waits = await repo.get_kv(result_waits_key(task_id)) == "1"
            if task.status is not TaskStatus.DONE and not (
                waits
                and task.status in (TaskStatus.ACTIVE, TaskStatus.PAUSED)
                and await repo.move_task_if(task_id, task.status, TaskStatus.DONE)
            ):
                return NOT_DONE.format(id=task_id)
            result_wake = await repo.get_kv(result_wake_key(task_id))
            for conv in await repo.task_conversations(task_id):
                async with self._runner.locks.of(conv.id):
                    # afresh, under the lock
                    conv = await repo.get_conversation(conv.id)
                    if conv.state in people.OPEN_STATES:
                        # What a wake after the result queued is cancelled: nobody would
                        # read the answer. What the result's own wake queued still goes.
                        await repo.cancel_pending_after(
                            conv.id, int(result_wake) if result_wake else None
                        )
                        await people.close(repo, self._gateway, conv.id, TASK_DONE)
            for approval in await repo.pending_approvals():
                if approval.task_id == task_id:
                    await repo.resolve_approval(
                        approval.id, approval.nonce, ApprovalStatus.EXPIRED
                    )
            await repo.set_kv(result_waits_key(task_id), "")
            await repo.set_kv(result_wake_key(task_id), "")
            await repo.drop_task_events(task_id)  # nobody is woken for it now
            await repo.log_event("task_acknowledged", None, task_id=task_id)
            await people.listen(repo, self._gateway)  # nobody of it is heard now (R104)
            await chats.task_ended(repo, self._gateway, task_id)  # sound back (R105)
            return ACKNOWLEDGED.format(id=task_id)

    # -- stop and resume all --

    async def stop_all(self) -> None:
        """/stop: no wake starts and nothing is sent until /resume_all; events wait."""
        await self._repo.set_stopped(True)
        await self._repo.log_event("stopped", None)

    async def resume_all(self) -> None:
        """/resume_all: the agent works again; the events that waited fire now."""
        await self._repo.set_stopped(False)
        await self._repo.log_event("resumed", None)
