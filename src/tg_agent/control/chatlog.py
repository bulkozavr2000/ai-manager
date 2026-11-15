"""The owner's chat with the control bot, written down in both directions.

An agent that talks with the owner in the bot needs the recent chat and must know what a
message replies to; Telegram does not let a bot read its chat back, so the program keeps it.
"""

from __future__ import annotations

import logging

from tg_agent.control.base import UI, Card
from tg_agent.control.cards import MOVED
from tg_agent.db import Repo
from tg_agent.models import Direction

_log = logging.getLogger(__name__)

# The pinned status: edited at every change, it would flood the log with copies of itself.
_UNLOGGED = "board"


class ChatLog:
    """The owner's chat with the bot, both directions, in `owner_chat`. Never a secret: a
    future flow that takes a password from the owner must bypass it. A failure to write is
    logged and swallowed: the log must never keep a message from going out or being handled."""

    def __init__(self, repo: Repo) -> None:
        self._repo = repo

    async def incoming(
        self,
        kind: str,
        text: str,
        *,
        tg_msg_id: int | None = None,
        reply_to_msg_id: int | None = None,
    ) -> None:
        try:
            await self._repo.add_owner_chat(
                Direction.IN,
                kind,
                text,
                tg_msg_id=tg_msg_id,
                reply_to_msg_id=reply_to_msg_id,
            )
        except Exception:  # noqa: BLE001 - the log must never stop a message
            _log.warning(
                "could not log the owner's %s in the owner chat", kind, exc_info=True
            )

    async def outgoing(
        self,
        kind: str,
        text: str,
        *,
        tg_msg_id: int,
        task_id: int | None = None,
        approval_id: int | None = None,
    ) -> None:
        try:
            await self._repo.add_owner_chat(
                Direction.OUT,
                kind,
                text,
                tg_msg_id=tg_msg_id,
                task_id=task_id,
                approval_id=approval_id,
            )
        except Exception:  # noqa: BLE001 - the log must never stop a message
            _log.warning(
                "could not log the bot's %s in the owner chat", kind, exc_info=True
            )

    async def edited(self, tg_msg_id: int, text: str) -> None:
        try:
            await self._repo.set_owner_chat_text(tg_msg_id, text)
        except Exception:  # noqa: BLE001 - the log must never stop a message
            _log.warning("could not log an edit in the owner chat", exc_info=True)


class LoggedUI:
    """A `UI` that writes down what the inner one shows."""

    def __init__(self, inner: UI, chatlog: ChatLog) -> None:
        self._inner = inner
        self._chatlog = chatlog

    async def send(self, card: Card, *, silent: bool = False) -> int:
        msg_id = await self._inner.send(card, silent=silent)
        if card.kind != _UNLOGGED:
            await self._chatlog.outgoing(
                card.kind or "card",
                card.text,
                tg_msg_id=msg_id,
                task_id=card.task_id,
                approval_id=card.approval_id,
            )
        return msg_id

    async def edit(self, msg_id: int, card: Card) -> None:
        await self._inner.edit(msg_id, card)
        if card.kind != _UNLOGGED:
            await self._chatlog.edited(msg_id, card.text)

    async def delete(self, msg_id: int) -> None:
        await self._inner.delete(msg_id)
        # A deleted card was sent again lower down (stage 5, R52): its old row says so
        # instead of repeating the card.
        await self._chatlog.edited(msg_id, MOVED)

    async def pin(self, msg_id: int) -> None:
        await self._inner.pin(msg_id)

    async def unpin(self, msg_id: int) -> None:
        await self._inner.unpin(msg_id)

    async def notify(self, text: str) -> int:
        msg_id = await self._inner.notify(text)
        await self._chatlog.outgoing("notice", text, tg_msg_id=msg_id)
        return msg_id

    async def flush(self) -> int:
        return await self._inner.flush()

    async def typing(self) -> None:
        await self._inner.typing()
