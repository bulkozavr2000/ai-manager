"""After a restart: the messages missed while the program was off.

Nothing here talks to Telegram or to the model: the gateway is a recording fake.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from tg_agent.clock import FakeClock
from tg_agent.db import Repo
from tg_agent.gateway.base import TgActionError
from tg_agent.gateway.fake import FakeGateway
from tg_agent.models import ConvState, Direction, MsgSource, TaskStatus
from tg_agent.policy.tg_safety import TgErrorKind
from tg_agent.tasks.recovery import replay_missed

S = ConvState
PEER = 501
_PATHS = {
    S.PLANNED: [],
    S.TALKING: [S.CONTACTED, S.TALKING],
}


async def allowed(repo: Repo) -> FakeGateway:
    """A gateway that listens to the open chats and finds one missed message in each."""
    gateway = FakeGateway()
    await gateway.set_allowed(await repo.active_peer_ids())
    gateway.replay_count = 1
    return gateway


async def open_chat(repo: Repo, state: ConvState = S.TALKING, peer: int = PEER) -> int:
    """A launched task with one conversation, moved along the state machine to `state`."""
    task_id = await repo.create_task(
        "3D-печать",
        "Узнать цену печати 5 корпусов",
        {"what": "корпус", "quantity": 5},
        None,
        "UZS",
        ["name"],
        [],
    )
    await repo.set_task_status(task_id, TaskStatus.ACTIVE)
    contact_id = await repo.upsert_contact(peer, f"shop_{peer}", "Магазин", False)
    conv_id = await repo.create_conversation(task_id, contact_id)
    for step in _PATHS[state]:
        await repo.set_state(conv_id, step)
    return conv_id


async def say(
    repo: Repo,
    conv_id: int,
    source: MsgSource,
    tg_msg_id: int | None,
    at: datetime,
) -> None:
    direction = Direction.IN if source is MsgSource.COUNTERPART else Direction.OUT
    await repo.add_message(conv_id, direction, source, "text", "…", tg_msg_id, at)


# --- missed messages -------------------------------------------------------------------------


async def test_open_chats_are_read_from_their_newest_known_message(
    repo: Repo, fake_clock: FakeClock
) -> None:
    conv_id = await open_chat(repo)
    now = fake_clock.now()
    await say(repo, conv_id, MsgSource.AGENT, 900, now - timedelta(hours=3))
    await say(repo, conv_id, MsgSource.COUNTERPART, 901, now - timedelta(hours=2))
    await say(repo, conv_id, MsgSource.OWNER_MANUAL, 902, now - timedelta(hours=1))
    # Written before the owner's own messages kept their ids.
    await say(repo, conv_id, MsgSource.OWNER_MANUAL, None, now - timedelta(minutes=5))
    gateway = await allowed(repo)

    assert await replay_missed(repo, gateway) == 1

    assert gateway.replays == [(PEER, 902, {900, 902})]


async def test_a_chat_with_nothing_sent_yet_is_not_read(repo: Repo) -> None:
    await open_chat(repo, S.PLANNED)
    gateway = await allowed(repo)

    assert await replay_missed(repo, gateway) == 0

    assert "replay" not in gateway.calls


async def test_a_chat_that_cannot_be_read_does_not_stop_the_others(
    repo: Repo, fake_clock: FakeClock
) -> None:
    for peer, msg_id in [(501, 900), (502, 950)]:
        conv_id = await open_chat(repo, peer=peer)
        await say(repo, conv_id, MsgSource.AGENT, msg_id, fake_clock.now())
    gateway = await allowed(repo)
    gateway.fail_next = TgActionError(TgErrorKind.OTHER, "TimeoutError")  # chat 501

    assert await replay_missed(repo, gateway) == 1

    assert gateway.calls.count("replay") == 2
    assert gateway.replays == [(502, 950, {950})]
