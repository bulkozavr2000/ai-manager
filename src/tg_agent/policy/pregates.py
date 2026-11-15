"""Checks that run before any model call: a canned auto-reply is not worth waking the agent.

Plain code, so it never spends a token.
"""

from __future__ import annotations

# Repeated text longer than this is treated as a canned auto-reply (greeting / away message).
_CANNED_MIN_CHARS = 40
# A repeated text this soon after our own message is an instant auto-reply.
_INSTANT_SECONDS = 3.0


def is_auto_reply(
    *,
    text: str,
    offline_flag: bool,
    seconds_since_our_last: float | None,
    previous_incoming: list[str],
) -> bool:
    """True for business greeting/away messages and other canned replies that need no answer."""
    if offline_flag:
        return True
    text = text.strip()
    seen = text in (previous.strip() for previous in previous_incoming)
    if not seen:
        return False
    instant = (
        seconds_since_our_last is not None
        and seconds_since_our_last <= _INSTANT_SECONDS
    )
    return instant or len(text) > _CANNED_MIN_CHARS
