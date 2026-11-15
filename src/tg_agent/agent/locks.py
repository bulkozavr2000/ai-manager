"""One lock per conversation for the agent core.

The inbox, the last section of the send pipeline and the cards each read a conversation's
state and then write for it. Two of them interleaving on one conversation (a reply queued
while the owner takes the chat, a message stored while a reply is queued) would break the
core's invariants, so they hold the same lock. The lock is not re-entrant: code that holds it
never calls into another holder.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict


class ConvLocks:
    """One asyncio lock per conversation, shared by everything that reads a conversation's
    state and then writes for it: the inbox, the last section of the send pipeline, the cards."""

    def __init__(self) -> None:
        self._locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    def of(self, conv_id: int) -> asyncio.Lock:
        return self._locks[conv_id]
