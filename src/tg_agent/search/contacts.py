"""Contacts the vendor search may hand in, and the check that one was really seen.

A contact is a Telegram username (kept without "@") or a phone (kept as "+<digits>"). The
search model may only save a contact that appears in what the tools showed it — a page, a
listing, a Telegram profile — so a number it misremembers or makes up never becomes a target.
"""

from __future__ import annotations

import re

# Uzbek mobile operator codes: a bare 9-digit number counts as a phone only with one of these,
# so a price like "300 000 000" is not taken for one.
_UZ_MOBILE = frozenset(
    {"20", "33", "50", "55", "77", "88", "90", "91", "93", "94", "95", "97", "98", "99"}
)
# The same names the owner's text may hold (agent/people._TARGET): 5-32 letters, digits and "_", with
# at least one letter, so a number is never taken for a name.
_USERNAME = re.compile(
    r"^(?:(?:https?://)?(?:t|telegram)\.me/|@)?(?=[A-Za-z0-9_]*[A-Za-z])([A-Za-z0-9_]{5,32})$"
)
# Something that may be a phone inside a text: digits with spaces, dashes, dots, brackets.
_PHONE_IN_TEXT = re.compile(r"\+?\d[\d\s\-().]{6,22}\d")


def normalize_phone(raw: str) -> str | None:
    """The phone as "+<country><number>", or None if `raw` is not one."""
    text = raw.strip()
    digits = re.sub(r"\D", "", text)
    if text.startswith("+"):
        return f"+{digits}" if 10 <= len(digits) <= 15 else None
    if len(digits) == 12 and digits.startswith("998"):
        return f"+{digits}"
    if len(digits) == 9 and digits[:2] in _UZ_MOBILE:
        return f"+998{digits}"
    return None


def normalize_contact(raw: str) -> str | None:
    """A Telegram username (without "@") or a phone ("+..."); None for anything else."""
    text = raw.strip()
    match = _USERNAME.match(text)
    if match:
        return match.group(1)
    return normalize_phone(text)


def is_phone(contact: str) -> bool:
    return contact.startswith("+")


def find_phones(text: str) -> list[tuple[int, str]]:
    """The phones written in `text`: (where each starts, the phone in one form)."""
    found = []
    for match in _PHONE_IN_TEXT.finditer(text):
        phone = normalize_phone(match.group(0))
        if phone is not None:
            found.append((match.start(), phone))
    return found


def seen_in(contact: str, text: str) -> bool:
    """Does `text` show this contact — the same phone however it is written, or the username
    as a whole name (not a piece of a longer one)?"""
    if is_phone(contact):
        return any(phone == contact for _, phone in find_phones(text))
    name = re.escape(contact)
    return (
        re.search(rf"(?<![A-Za-z0-9_]){name}(?![A-Za-z0-9_])", text, re.IGNORECASE)
        is not None
    )


def show(contact: str) -> str:
    """The contact the way people write it: "@name", "+998 90 123 45 67"."""
    if not is_phone(contact):
        return f"@{contact}"
    digits = contact[1:]
    if len(digits) == 12 and digits.startswith("998"):
        d = digits[3:]
        return f"+998 {d[:2]} {d[2:5]} {d[5:7]} {d[7:]}"
    return contact
