"""The test world of the agent core: the new core on fakes, driven as the program would drive it.

A real `Repo` in the test's temporary folder, `FakeClock`, `FakeLLM`, `FakeGateway` (dated by the
clock), `FakeUI`, `Runner(inline=True)`, its `Inbox` and a `Sender` that reports to it. The model
is scripted through `llm.steps["task"]`; tool calls are plain dicts (`call`, `send`, `ask`).
The agent's search is a `FakeSearch`: prepare its outcomes in `world.search.outcomes`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from tg_agent.agent import people
from tg_agent.agent.inbox import Inbox
from tg_agent.agent.runner import Runner
from tg_agent.brain.llm import FakeLLM
from tg_agent.clock import FakeClock
from tg_agent.config import Settings
from tg_agent.control.base import FakeUI
from tg_agent.db import Repo
from tg_agent.gateway.base import Incoming, PeerInfo
from tg_agent.gateway.fake import FakeGateway
from tg_agent.models import AgentEvent, Approval, ApprovalStatus, EventKind, Wake
from tg_agent.search.fake import FakeSearch
from tg_agent.tasks.first_contacts import GAP_MINUTES
from tg_agent.tasks.sender import Sender

CANARY = "ZX-CANARY-4242"
END = [{"type": "text", "text": "Жду ответа."}]  # a step that ends the turn
# Longer than the longest human pause before a send (`send_delay_max_seconds`).
DELIVERY = timedelta(seconds=181)
# Longer than the burst window of incoming messages (`batch_window_seconds`).
BURST = timedelta(seconds=46)

_ids = iter(range(1, 10_000))


def call(name: str, **input: Any) -> dict[str, Any]:
    """One tool call of a scripted model step."""
    return {"type": "tool_use", "id": f"t{next(_ids)}", "name": name, "input": input}


def send(
    person: int,
    *texts: str,
    commits: bool = False,
    attach: bool = False,
    language: str = "ru",
) -> dict[str, Any]:
    return call(
        "send_message",
        person_id=person,
        messages=list(texts),
        language=language,
        attach_task_files=attach,
        commits_owner=commits,
    )


def ask(
    kind: str,
    question: str,
    person: int | None,
    *,
    suggested: list[str] | None = None,
    amount: float | None = None,
    terms: str = "",
) -> dict[str, Any]:
    return call(
        "ask_owner",
        kind=kind,
        question=question,
        person_id=person,
        suggested_reply=suggested or [],
        amount=amount,
        terms=terms,
    )


@dataclass
class Person:
    conv_id: int
    peer_id: int
    username: str


class AgentWorld:
    """The new core on fakes, with the moves the flows need."""

    def __init__(
        self,
        repo: Repo,
        clock: FakeClock,
        settings: Settings,
        llm: FakeLLM | None = None,
        search: FakeSearch | None = None,
    ) -> None:
        self.repo = repo
        self.clock = clock
        self.settings = settings
        self.llm = llm or FakeLLM()
        self.search = search if search is not None else FakeSearch(repo=repo)
        self.gateway = FakeGateway(clock)
        self.ui = FakeUI()
        self._peers = 900
        self.build()

    def build(self) -> None:
        """The core's objects (again, after a "restart": the database is the state)."""
        self.runner = Runner(
            self.repo,
            self.llm,
            self.gateway,
            self.ui,
            self.clock,
            self.settings,
            random.Random(7),
            canary=CANARY,
            owner_secrets=[],
            inline=True,
            searcher=self.search,
        )
        self.inbox = Inbox(
            self.repo,
            self.clock,
            self.settings,
            outcomes=self.runner,
            locks=self.runner.locks,
        )
        self.sender = Sender(
            self.repo,
            self.gateway,
            self.ui,
            self.clock,
            self.settings,
            self.runner,
        )

    # -- a task and its people --

    async def task(
        self,
        *usernames: str,
        attachments: list[str] | None = None,
        start: bool = True,
    ) -> tuple[int, list[Person]]:
        """A task with these people attached (PLANNED), launched unless `start` is False."""
        task_id = await self.repo.create_task(
            "Дрель",
            "Купить дрель Bosch не дороже 600 000",
            {"what": "дрель"},
            600_000,
            "UZS",
            ["name"],
            attachments or [],
            source_text="Найди дрель Bosch",
        )
        found = []
        for username in usernames:
            self._peers += 1
            self.gateway.peers[username] = PeerInfo(
                self._peers, username, username.title(), False
            )
            attached = await people.attach(self.repo, self.gateway, task_id, username)
            assert isinstance(attached, people.Attached), attached
            found.append(Person(attached.conv_id, self._peers, username))
        if start:
            await self.runner.start(task_id)
        return task_id, found

    # -- the model --

    def script(self, *steps: list[dict[str, Any]] | Any) -> None:
        """Queue model steps: each a list of blocks (a step), or an exception to raise."""
        self.llm.steps["task"].extend(steps)

    def requests(self) -> list[dict[str, Any]]:
        return [req for purpose, req in self.llm.requests if purpose == "task"]

    def context(self, n: int = -1) -> str:
        """The data text of the n-th wake request (its first user message)."""
        [*_, text_block] = self.requests()[n]["messages"][0]["content"]
        return text_block["text"]

    def trusted(self, n: int = -1) -> str:
        return self.requests()[n]["system"][1]

    def result(self, block: dict[str, Any]) -> dict[str, Any]:
        """The tool result the model got back for this scripted tool call."""
        for request in self.requests():
            for message in request["messages"]:
                if message["role"] != "user" or isinstance(message["content"], str):
                    continue
                for item in message["content"]:
                    if item.get("tool_use_id") == block["id"]:
                        return item
        raise AssertionError(f"no result for {block['name']} ({block['id']})")

    # -- the world moves --

    async def write(self, person: Person, text: str) -> None:
        """The person writes: into the chat log, then reported to the inbox."""
        message = self.gateway.add_incoming(person.peer_id, text)
        await self.inbox.on_incoming(
            Incoming(person.peer_id, message.tg_msg_id, text, "text", message.date)
        )

    async def round(self, advance: timedelta | None = None) -> None:
        """Move the clock, then one round of the Scheduler: the core, then the Sender."""
        if advance is not None:
            self.clock.advance(advance)
        await self.runner.tick()
        await self.sender.tick()

    async def deliver(self) -> None:
        """Let what is queued go out (past the human pause)."""
        await self.round(DELIVERY)

    async def press(self, choice: str, text: str | None = None) -> str:
        """The owner presses a button on the newest pending card."""
        approval = (await self.repo.pending_approvals())[-1]
        return await self.runner.approvals.owner_decision(
            approval.id, approval.nonce, choice, text
        )

    # -- reading the world --

    def sent_to(self, person: Person) -> list[str]:
        return [text for peer, text in self.gateway.sent if peer == person.peer_id]

    async def state(self, person: Person) -> Any:
        return (await self.repo.get_conversation(person.conv_id)).state

    async def open_events(self, task_id: int) -> list[AgentEvent]:
        return await self.repo.open_events(task_id)

    async def kinds(self, task_id: int) -> list[EventKind]:
        return [event.kind for event in await self.repo.open_events(task_id)]

    async def approvals(self) -> list[Approval]:
        return [
            a
            for a in await self.repo.pending_approvals()
            if a.status is ApprovalStatus.PENDING
        ]

    async def wakes(self) -> list[Wake]:
        rows = await self.repo._all("SELECT * FROM wakes ORDER BY id")
        return [
            Wake(
                r["id"],
                r["task_id"],
                r["started_at"],
                r["finished_at"],
                r["outcome"],
                r["steps"],
            )
            for r in rows
        ]

    async def assert_clean(self) -> None:
        """Every scripted step was used, and no wake asked the model for a step that was
        not scripted (the fake raises AssertionError; the runner would release the wake)."""
        assert not self.llm.steps["task"], "scripted steps left unused"
        failed = await self.repo.events_of("wake_failed")
        assert all(e["error"] != "AssertionError" for e in failed), failed

    def owner_messages(self) -> int:
        """Everything the owner got from the bot: cards and notices."""
        return len(self.ui.sent) + len(self.ui.notes)

    # -- the common openings --

    async def contacted(self, *usernames: str, **kw: Any) -> tuple[int, list[Person]]:
        """A launched task whose first wake greeted every person, and the greetings went out."""
        task_id, found = await self.task(*usernames or ("drill_shop",), **kw)
        self.script(
            [send(p.conv_id, "Здравствуйте! Есть дрель Bosch?") for p in found], END
        )
        await self.round()
        await self.deliver()
        # First messages to new people are queued 1-2 minutes apart (Task 3.1, R70).
        for _ in found[1:]:
            if all(self.sent_to(p) for p in found):
                break
            await self.round(timedelta(minutes=GAP_MINUTES[1]))
        assert all(self.sent_to(p) for p in found)
        return task_id, found

    async def replied(self, text: str, **kw: Any) -> tuple[int, Person]:
        """A contacted person who answered; the wake for his answer is not run yet."""
        task_id, [person] = await self.contacted(**kw)
        self.clock.advance(timedelta(minutes=10))
        await self.write(person, text)
        return task_id, person
