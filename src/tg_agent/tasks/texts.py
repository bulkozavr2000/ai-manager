"""Texts the code shows the owner (and one the inbox keeps in a transcript), and how a notice
names a counterpart or quotes his message.

In Russian. The fixed texts a counterpart receives are in `templates`.
"""

from __future__ import annotations

from tg_agent.models import Contact

# How a message the counterpart deleted stays in the transcript (2.24).
DELETED_MARK = "(удалено собеседником)"
# The card that hands a chat to the owner after he pressed «Напишу сам».
HANDOFF_SELF = "Чат за тобой: напиши ему сам. Когда закончишь, нажми «Вернуть агенту»"
# A queued message did not go out (the Sender tells the owner): the notice, then the advice.
NOT_SENT = "Не отправил сообщение {who}: {reason}"
ADVICE = ". Напиши сам или реши, как быть"

# How an attachment the model cannot see is named for it.
KIND_LABELS = {
    "photo": "фото",
    "voice": "голосовое",
    "video": "видео",
    "document": "файл",
    "sticker": "стикер",
    "location": "геопозиция",
    "contact": "контакт",
    "poll": "опрос",
}


def who(contact: Contact) -> str:
    """How a notice to the owner names the counterpart: @username, else the display name."""
    return f"@{contact.username}" if contact.username else contact.display_name


QUOTE_CHARS = 80  # how much of a message the owner is shown when a notice is about it


def quote(text: str) -> str:
    """A message on one line, cut short: how a notice to the owner names it."""
    line = " ".join(text.split())
    return line if len(line) <= QUOTE_CHARS else line[: QUOTE_CHARS - 1] + "…"
