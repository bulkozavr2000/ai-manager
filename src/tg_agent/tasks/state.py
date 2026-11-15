"""The conversation state machine: which moves of a person's conversation are legal.

Whatever changes the state of a conversation goes through `transition`, so an impossible move
(reviving a finished conversation, say) fails loudly instead of quietly corrupting the data.
"""

from __future__ import annotations

from tg_agent.models import ConvState

S = ConvState  # short alias, so the tables below read at a glance and compare easily with the spec


class InvalidTransition(ValueError):
    """A conversation cannot move from the one state to the other."""


TRANSITIONS: dict[ConvState, frozenset[ConvState]] = {
    S.PLANNED: frozenset({S.CONTACTED, S.WAITING_OWNER, S.HANDED_TO_OWNER, S.CLOSED}),
    S.CONTACTED: frozenset({S.TALKING, S.WAITING_OWNER, S.HANDED_TO_OWNER, S.CLOSED}),
    S.TALKING: frozenset({S.WAITING_OWNER, S.HANDED_TO_OWNER, S.CLOSED}),
    # A card about a person who has not answered, or was not written to yet, returns him to
    # where he stood.
    S.WAITING_OWNER: frozenset(
        {S.TALKING, S.PLANNED, S.CONTACTED, S.HANDED_TO_OWNER, S.CLOSED}
    ),
    # The owner hands the chat back to the agent, or is done with the person.
    S.HANDED_TO_OWNER: frozenset({S.TALKING, S.PLANNED, S.CONTACTED, S.CLOSED}),
    # Done with this person. Only the owner's instruction opens the talk again, back to where
    # it stood (ruling R72, `agent/people.reopen_closed`).
    S.CLOSED: frozenset({S.TALKING, S.CONTACTED}),
}

TERMINAL: frozenset[ConvState] = frozenset({S.CLOSED})


def transition(cur: ConvState, new: ConvState) -> ConvState:
    """Check the move from `cur` to `new` and return `new`; raise InvalidTransition if illegal."""
    if new not in TRANSITIONS[cur]:
        raise InvalidTransition(f"a conversation cannot move from {cur} to {new}")
    return new
