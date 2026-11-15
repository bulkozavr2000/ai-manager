"""The three protocols between the decision-making core and the parts around it.

The Telegram gateway reports what happens in the chats (`ChatEvents`), the Sender reports what
became of each queued message (`SendOutcomes`), and the Scheduler wakes the core up every round
(`Wakeups`). Those parts depend on these protocols only, never on the core itself, so that the
core can be replaced without touching them. The agent core implements them: its inbox
(`agent/inbox.Inbox`) the first, its runner (`agent/runner.Runner`) the other two.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from tg_agent.gateway.base import Incoming
from tg_agent.models import Conversation, OutboxItem
from tg_agent.policy.tg_safety import TgErrorKind


class ChatEvents(Protocol):
    """What the Telegram gateway reports about the chats the agent listens to."""

    async def on_incoming(self, msg: Incoming) -> None: ...
    async def on_edit(self, peer_id: int, tg_msg_id: int, text: str) -> None: ...
    async def on_owner_manual(
        self, peer_id: int, text: str, tg_msg_id: int | None = None
    ) -> None: ...
    async def on_delete(self, tg_msg_ids: Sequence[int]) -> None: ...


class SendOutcomes(Protocol):
    """What the Sender tells the core about a queued message."""

    def can_reanswer(self, conv: Conversation) -> bool:
        """Would the core write a new reply if the counterpart wrote again? Then a queued
        reply that a newer message made stale is dropped (`on_stale`); otherwise it goes out
        as it is: a goodbye, a handoff."""
        ...

    async def excluded(self, conv: Conversation) -> bool:
        """Did the owner say not to work with this person (ruling R92)? Then nothing queued
        for him goes out (R97)."""
        ...

    async def on_sent(
        self, item: OutboxItem, conv: Conversation, at: datetime, *, first: bool
    ) -> None:
        """The message went out and is recorded (`tasks.outbox.mark_sent`). `first`: it was
        the first message of the conversation."""
        ...

    async def on_stale(self, conv_id: int) -> None:
        """The chat's queue was dropped: the counterpart wrote after the reply was queued."""
        ...

    async def on_unclear(self, item: OutboxItem) -> None:
        """A send broke off: the message is UNKNOWN, it may or may not be in the chat."""
        ...

    async def on_refused(
        self,
        item: OutboxItem,
        conv: Conversation,
        kind: TgErrorKind,
        reason: str,
        cancelled: int,
    ) -> None:
        """Telegram refused this chat for good (privacy, a block, Premium, a fee, anything
        else): the message is FAILED and the chat's `cancelled` other queued messages are
        dropped. `reason` is the owner-facing reason, in Russian."""
        ...

    async def on_failed(
        self, item: OutboxItem, conv: Conversation, code: str, reason: str
    ) -> bool:
        """The Sender failed this message by itself, without Telegram's refusal (a missing
        file, a placeholder it cannot fill, "typing…" that kept failing, a message it could
        not prepare). `code` is what it wrote to the outbox row, `reason` the Russian phrase
        it would show the owner. True: the core took care of it and the Sender tells the
        owner nothing; False (or an exception): the Sender tells the owner itself."""
        ...


class Wakeups(Protocol):
    """What the Scheduler calls every round so that the core does whatever is due."""

    async def tick(self) -> None: ...
