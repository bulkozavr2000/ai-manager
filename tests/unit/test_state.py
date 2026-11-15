"""The conversation state machine: the six states of the agent core and their legal moves."""

from __future__ import annotations

import pytest

from tg_agent.models import ConvState, TaskStatus
from tg_agent.tasks.state import TERMINAL, TRANSITIONS, InvalidTransition, transition

S = ConvState

# The legal moves, with plain strings on purpose: this is the independent copy that the real
# table and `transition` are compared against.
SPEC_TRANSITIONS: dict[str, set[str]] = {
    # Attached to a task, nothing sent yet.
    "planned": {"contacted", "waiting_owner", "handed_to_owner", "closed"},
    "contacted": {"talking", "waiting_owner", "handed_to_owner", "closed"},
    "talking": {"waiting_owner", "handed_to_owner", "closed"},
    # A card about a person who has not answered, or was not written to yet, returns him to
    # where he stood.
    "waiting_owner": {"talking", "planned", "contacted", "handed_to_owner", "closed"},
    "handed_to_owner": {"talking", "planned", "contacted", "closed"},
    # Done with this person; only the owner's instruction reopens the talk (ruling R72).
    "closed": {"talking", "contacted"},
}


def accepts(cur: ConvState, new: ConvState) -> bool:
    try:
        transition(cur, new)
    except InvalidTransition:
        return False
    return True


def test_the_six_states_of_the_agent_core():
    assert [state.value for state in ConvState] == [
        "planned",
        "contacted",
        "talking",
        "waiting_owner",
        "handed_to_owner",
        "closed",
    ]


def test_a_person_s_way_from_planned_to_closed():
    path = [S.PLANNED, S.CONTACTED, S.TALKING, S.WAITING_OWNER, S.TALKING, S.CLOSED]

    state = path[0]
    for step in path[1:]:
        state = transition(state, step)
        assert state is step

    assert state in TERMINAL


def test_closed_is_the_only_final_state():
    assert TERMINAL == {S.CLOSED}
    # Ruling R72: only the owner's instruction reopens a closed conversation, back to where
    # the talk stood; nothing else leaves CLOSED.
    assert TRANSITIONS[S.CLOSED] == frozenset({S.TALKING, S.CONTACTED})
    with pytest.raises(InvalidTransition, match="closed.*waiting_owner"):
        transition(S.CLOSED, S.WAITING_OWNER)
    for target in ConvState:
        if target not in (S.TALKING, S.CONTACTED):
            assert not accepts(S.CLOSED, target), f"closed -> {target} must be rejected"

    # Every state has somewhere to go.
    assert all(TRANSITIONS[state] for state in ConvState)


def test_transition_table_matches_spec():
    table = {
        state.value: {move.value for move in moves}
        for state, moves in TRANSITIONS.items()
    }

    assert table == SPEC_TRANSITIONS


def test_transition_accepts_exactly_the_specified_moves():
    accepted = {
        (cur.value, new.value)
        for cur in ConvState
        for new in ConvState
        if accepts(cur, new)
    }
    specified = {(cur, new) for cur, moves in SPEC_TRANSITIONS.items() for new in moves}

    assert accepted == specified


async def _closed_conversation(repo, peer_id: int, *path: ConvState) -> int:
    task_id = await repo.create_task("Дрель", "цена", {}, None, "UZS", [], [])
    await repo.move_task_if(
        task_id, TaskStatus.DRAFT, TaskStatus.ACTIVE
    )  # a task in work
    contact_id = await repo.upsert_contact(peer_id, None, "Продавец", False)
    conv_id = await repo.create_conversation(task_id, contact_id)
    for state in path:
        await repo.set_state(conv_id, state)
    return conv_id


@pytest.mark.parametrize(
    "path",
    [(S.CLOSED,), (S.CONTACTED, S.CLOSED), (S.CONTACTED, S.WAITING_OWNER, S.CLOSED)],
)
async def test_a_closed_conversation_is_over_for_the_repo(repo, path) -> None:
    await _closed_conversation(repo, 1001, *path)

    assert await repo.conversation_by_peer(1001) is None
    assert await repo.blocking_conversation(1001) is None
    assert await repo.active_peer_ids() == set()


async def test_a_planned_conversation_is_open_for_the_repo(repo) -> None:
    conv_id = await _closed_conversation(repo, 1001)

    assert (await repo.conversation_by_peer(1001)).id == conv_id
    assert (await repo.blocking_conversation(1001)).id == conv_id
    assert await repo.active_peer_ids() == {1001}
