"""The bookkeeping of a queued message that went out, and the texts about queued messages.

The Sender and the agent core's inbox and runner share them.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from tg_agent.db import FIRST_SENT, Repo
from tg_agent.models import (
    Contact,
    Conversation,
    Direction,
    MsgSource,
    OutboxItem,
    OutboxStatus,
)
from tg_agent.policy.dlp import ADDRESS_PLACEHOLDER, PHONE_PLACEHOLDER
from tg_agent.tasks.first_contacts import is_first_contact
from tg_agent.tasks.texts import quote, who

# The owner's data in a queued text, filled in only when it is sent.
_PLACEHOLDERS = re.compile(
    "|".join(re.escape(p) for p in (PHONE_PLACEHOLDER, ADDRESS_PLACEHOLDER))
)


async def mark_sent(
    repo: Repo,
    item: OutboxItem,
    conv: Conversation,
    msg_id: int | None,
    at: datetime,
    *,
    is_file: bool,
) -> bool:
    """Write down that a queued message went out as Telegram message `msg_id` at `at`
    (None: the owner says it went, and its Telegram id is not known). True when it was the
    conversation's first sent message.

    SENT first: it is what keeps a restart from sending it again. The transcript keeps the text
    as the model wrote it: placeholders, never the number. A first message to a person who had
    not written to us (`conv.last_in_at` unset when it went), and whom the account never wrote
    to in another conversation (`is_first_contact`), is logged as FIRST_SENT, which the day's
    quota of new people counts and the gap before the next stranger starts from; a reply to
    someone who wrote first, or a message to a person known from an earlier task, is no new
    contact. What follows a sent message is the caller's business.
    """
    # Read before this conversation's own `first_sent` is written below.
    new_contact = await is_first_contact(repo, conv)
    await repo.mark_outbox(item.id, OutboxStatus.SENT, tg_msg_id=msg_id, at=at)
    await repo.add_message(
        conv.id,
        Direction.OUT,
        MsgSource.AGENT,
        "document" if is_file else "text",
        item.text,
        msg_id,
        at,
    )
    first = not conv.first_sent
    fields: dict[str, object] = {"last_out_at": at}
    if first:
        fields["first_sent"] = True
    await repo.update_conversation(conv.id, **fields)
    if new_contact:
        await repo.log_event(FIRST_SENT, conv.id)
    return first


def same_text(queued: str, seen: str) -> bool:
    """Is `seen` what the queued text became once sent (its placeholders filled in)?

    Whitespace is compared loosely; an empty text matches nothing.
    """
    queued, seen = " ".join(queued.split()), " ".join(seen.split())
    if not queued:
        return False
    pattern = ".+?".join(re.escape(part) for part in _PLACEHOLDERS.split(queued))
    return re.fullmatch(pattern, seen) is not None


def unclear_text(item: OutboxItem, contact: Contact) -> str:
    """The question to the owner about a message whose send broke off."""
    if item.attach_path and not item.text.strip():
        head = f"Не знаю, ушёл ли файл «{Path(item.attach_path).name}»"
    else:
        head = f"Не знаю, ушло ли сообщение «{quote(item.text)}»"
    return (
        f"{head} в чат с {who(contact)}: отправка оборвалась. Проверь чат и ответь —"
        " до тех пор агент в этот чат ничего не отправляет"
    )
