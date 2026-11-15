"""Is this chat message our queued message whose send broke off (UNKNOWN)?

One rule for the two places that ask: the Runner, when it reads the chat a minute after the
attempt, and the Inbox, when Telegram (or the replay at start-up) reports an outgoing message
that the database does not know. Both must agree, or one of them takes our message for the
owner's own (and pauses the chat) or takes an old message for the one that broke off.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime, timedelta

from tg_agent.models import OutboxItem
from tg_agent.tasks.outbox import same_text

# Telegram dates are whole seconds: a message sent right after the item was queued may show a
# date up to a second before it.
DATE_SLACK = timedelta(seconds=1)


def is_file_item(item: OutboxItem) -> bool:
    """A queued file: it carries a path and no text (a file goes without a caption)."""
    return item.attach_path is not None and not item.text.strip()


def is_unclear_item(
    item: OutboxItem,
    *,
    out: bool,
    text: str,
    date: datetime,
    tg_msg_id: int | None,
    stored_out_ids: Collection[int],
) -> bool:
    """Does this chat message match the unclear item?

    It must be outgoing and not stored yet as a message of the conversation; not older than
    the item (an identical greeting from an earlier task is not this one); and a text item's
    words, or for a file item a message without text.
    """
    if not out:
        return False
    if tg_msg_id is not None and tg_msg_id in stored_out_ids:
        return False
    if date < item.created_at - DATE_SLACK:
        return False
    if is_file_item(item):
        return not text.strip()
    return same_text(item.text, text)
