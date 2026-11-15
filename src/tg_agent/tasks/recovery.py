"""After a restart: catch up on what reached the open chats while the program was off.

`replay_missed` hands the inbox the messages that reached the open chats while nothing was
listening. It works in two halves: `replay_starts` reads where each chat's replay must start,
before the userbot connects, and `replay_missed` reads the chats from there once it is live. A
message that arrives while the userbot connects is stored at once, and read in the other order
it would move the starting point past the ones that were missed; the replay delivers it again,
and the inbox takes it for the duplicate it is.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from tg_agent.agent import people
from tg_agent.db import Repo
from tg_agent.gateway.base import Gateway, TgActionError
from tg_agent.models import Conversation

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReplayStart:
    """Where one open chat is read from: after its newest stored message, with the ids of
    our own stored messages, so that they are not taken for the owner's."""

    conv_id: int
    peer_id: int
    after_msg_id: int
    known_out_ids: frozenset[int]


async def replay_start(repo: Repo, conv: Conversation) -> ReplayStart | None:
    """Where one chat's replay starts: after its newest stored message. None for a chat with
    no stored message: nothing was sent there, so nothing can be missed, and its older history
    is the owner's own."""
    newest = await repo.last_tg_msg_id(conv.id)
    if newest is None:
        return None
    contact = await repo.get_contact(conv.contact_id)
    known = frozenset(await repo.out_tg_msg_ids(conv.id))
    return ReplayStart(conv.id, contact.peer_id, newest, known)


async def replay_starts(repo: Repo) -> list[ReplayStart]:
    """Where each listened chat's replay starts (`replay_start`): every person of a task in
    work, a closed one too (ruling R104 b: the inbox reopens his talk); not one the owner
    said not to work with (`people.listened` leaves him out)."""
    starts = []
    for peer_id in sorted(await people.listened(repo)):
        conv = await repo.conversation_by_peer(
            peer_id
        ) or await repo.closed_conversation_by_peer(peer_id)
        if conv is None:
            continue
        start = await replay_start(repo, conv)
        if start is not None:
            starts.append(start)
    return starts


async def replay_missed(
    repo: Repo, gateway: Gateway, starts: Sequence[ReplayStart] | None = None
) -> int:
    """Hand over what the open chats got while the program was off; return how many messages.

    Each chat is read from its start (`replay_starts`, read now when none are given). A chat
    that cannot be read is logged, and the others are read all the same.
    """
    if starts is None:
        starts = await replay_starts(repo)
    total = 0
    for start in starts:
        try:
            total += await gateway.replay(
                start.peer_id, start.after_msg_id, set(start.known_out_ids)
            )
        except TgActionError as error:
            _log.warning(
                "conversation %s: missed messages not read (%s: %s)",
                start.conv_id,
                error.kind.value,
                error.detail,
            )
    return total
