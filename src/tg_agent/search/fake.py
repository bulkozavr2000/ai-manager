"""A scripted search for tests and evals: no model, no network, no browser.

`FakeSearch.run_query` has the signature of `Searcher.run_query`. It hands back the prepared
outcomes in order (an exception in the list is raised instead), records every call, and awaits
`still_wanted` when one is given: a search no longer wanted ends as the real one does, with what
it has and `stopped = "stopped"`. `affordable` is asked first, as the real search asks it once
its turn comes: a search that no longer fits the money ends at once with `stopped = "money"`.
A prepared outcome carries its `walls` (the sign-in walls it met) as the real one does.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace

from tg_agent.brain.schemas import Usage
from tg_agent.db import Repo
from tg_agent.search.agent import SearchOutcome


@dataclass
class SearchCall:
    """One call of the scripted search, as it was made."""

    what: str
    skip: list[str]
    task_id: int | None
    wake_id: int | None
    # What `still_wanted` answered (None: the caller gave none).
    wanted: bool | None = None
    sites: list[str] = field(default_factory=list)  # the owner's sites it was given
    # What `affordable` answered (None: the caller gave none).
    affordable: bool | None = None


@dataclass
class FakeSearch:
    """Prepared outcomes, handed back in order; `calls` holds what each call asked for.

    With a `repo`, an outcome that cost something is logged as one model call of the search
    for the call's task and wake, as the real search logs its calls (the cost counts toward the
    task and the day). `during` is awaited first in every call: a test does there what happens
    while a search runs (the owner pauses the task, a model call is paid).
    """

    outcomes: list[SearchOutcome | BaseException] = field(default_factory=list)
    repo: Repo | None = None
    during: Callable[[], Awaitable[None]] | None = None
    calls: list[SearchCall] = field(default_factory=list)

    async def run_query(
        self,
        what: str,
        skip: Sequence[str],
        *,
        task_id: int | None = None,
        wake_id: int | None = None,
        still_wanted: Callable[[], Awaitable[bool]] | None = None,
        sites: Sequence[str] = (),
        affordable: Callable[[], Awaitable[bool]] | None = None,
    ) -> SearchOutcome:
        call = SearchCall(what, list(skip), task_id, wake_id, sites=list(sites))
        self.calls.append(call)
        if affordable is not None:
            call.affordable = await affordable()
            if not call.affordable:
                return SearchOutcome(stopped="money")
        if self.during is not None:
            await self.during()
        if still_wanted is not None:
            call.wanted = await still_wanted()
        if not self.outcomes:
            raise AssertionError("FakeSearch: no outcome prepared")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if self.repo is not None and outcome.cost > 0:
            await self.repo.log_llm_call(
                None,
                "search",
                "fake-search",
                Usage(0, 0),
                outcome.cost,
                "end_turn",
                task_id=task_id,
                wake_id=wake_id,
            )
        if call.wanted is False:
            return replace(outcome, stopped="stopped")
        return outcome
