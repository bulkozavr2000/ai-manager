"""Who gave a contact, decided by the code (stage 4; rulings R28b, R69).

* A counterpart of the task gave a contact when one of his messages shows it: a username as a
  whole word in any case, with or without «@», or in a t.me link (`search/contacts.seen_in`);
  a phone by its last nine digits in the digits of one message. Such a person the agent may
  add on its own (ruling R69), within the task's cap of referrals.
* The owner gives a contact, excludes it, or only names it: «не пиши @x», «@x не пиши, он
  мошенник», «кроме @x, @y и @z» exclude; «всё-таки напиши @x» tells to write; «я же сказал,
  @x — мошенник» only names him (`owner_words`).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from tg_agent.agent.people import targets_with_places
from tg_agent.search.contacts import is_phone, seen_in

# Words that exclude every contact of their clause, before or after them: «не пиши @x»,
# «@x не пиши, он мошенник», «@x в чёрный список».
_EXCLUDING = re.compile(
    r"(?<!\w)(?:не\s+(?:пиши|писать|пишите|добавляй|добавлять|связывайся|связываться"
    r"|трогай|беспокой|надо|нужно|стоит|звони|звонить|звоните)"
    r"|исключи\w*|убери\w*|убрать|вычеркни\w*|вычеркнуть"
    r"|(?<!не\s)забуд\w*|в\s+ч[её]рный\s+список|yozma\w*)(?!\w)",
    re.IGNORECASE,
)
# Words that exclude only the contacts after them in the clause: «напиши @y вместо @x»,
# «напиши @y, но не @x»; a bare «не» only right before a contact.
_EXCLUDING_AFTER = re.compile(
    r"(?<!\w)(?:кроме|без|вместо|только\s+не|а\s+не|но\s+не|не(?=\s*(?:@|\+|https?://"
    r"|t\.me|telegram\.me)))(?!\w)",
    re.IGNORECASE,
)
# A clause that tells to write to its contacts (when it holds no negation): only such a
# clause gives an excluded contact back (fix pass F3, I-A (c)).
_TELLING = re.compile(
    r"(?<!\w)(?:напиши\w*|написать|пиши|пишите|добавь\w*|добавить|свяжись|свяжитесь"
    r"|связаться|спроси\w*|позвони\w*|всё-таки|все-таки)(?!\w)",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"(?<!\w)(?:не|ни|нельзя)(?!\w)", re.IGNORECASE)
# Where a sentence splits into clauses: a comma or a dash between words.
_CLAUSE_BREAK = re.compile(r",|\s[—–-]\s|—")
# What may stand around contacts in a piece that only lists more of them: «и @y», «ни @y».
_JOINERS = re.compile(r"(?<!\w)(?:и|или|ни|а\s+также)(?!\w)", re.IGNORECASE)
# A line of a list: an optional bullet or number, then a contact.
_LIST_LINE = re.compile(
    r"^\s*(?:[-•*·]|\d+[.)])?\s*(?:@|\+|https?://|t\.me|telegram\.me)"
)
EXCLUDED, TOLD, PLAIN = "excluded", "told", "plain"
# A sentence ends at «.», «!», «?» or «;» before a space or the end, or at a line break:
# a dot inside «t.me/x» or «prom.uz» does not end it.
_SENTENCE = re.compile(r"(?:[^\n.!?;]|[.!?;](?!\s|$))+")
# The part of a phone that is the same however it is written: «8 90 …», «+998 90 …», «90 …».
_TAIL = 9


def _digits(text: str) -> str:
    """Every digit of a text, in order, as plain digits (full-width ones too)."""
    plain = unicodedata.normalize("NFKC", text)
    return "".join(str(unicodedata.digit(ch)) for ch in plain if ch.isdigit())


def given_by_a_counterpart(contact: str, texts: Iterable[str]) -> bool:
    """Does `contact` (a username without «@» or a phone «+…») stand in one of `texts`, the
    messages the task's counterparts wrote?"""
    if is_phone(contact):
        tail = contact[1:][-_TAIL:]
        return any(tail in _digits(text) for text in texts)
    return any(seen_in(contact, text) for text in texts)


def contacts_in_order(text: str) -> list[tuple[str, bool]]:
    """The contacts the owner's text names, in order, each with whether the text excludes
    him («не пиши @x»)."""
    return [(contact, verdict == EXCLUDED) for contact, verdict in owner_words(text)]


def owner_words(text: str) -> list[tuple[str, str]]:
    """The contacts the owner's text names, in order, each with what the text makes of him:
    `EXCLUDED` («не пиши @x», «@x не пиши», «кроме @x, @y и @z»), `TOLD` (a clause that
    tells to write to him and holds no negation: «всё-таки напиши @x»), or `PLAIN` (only
    named: «я же сказал, @x — мошенник»).

    A sentence ends at «.», «!», «?», «;» or a line break and splits into clauses at commas
    and dashes; a piece that only lists more contacts («и @y», «ни @y») belongs to the clause
    before it, and the lines of a list belong to the line ending with «:» above them."""
    found: list[tuple[str, str]] = []
    carried: str | None = None  # what a line ending with «:» says of the list below it
    for line in text.split("\n"):
        if carried is not None and _LIST_LINE.match(line):
            for contact, verdict in _read_sentence(line)[0]:
                if verdict != EXCLUDED and carried != PLAIN:
                    verdict = carried
                found.append((contact, verdict))
            continue
        carried = None
        header = PLAIN
        for sentence in _SENTENCE.finditer(line):
            verdicts, header = _read_sentence(sentence.group(0))
            found += verdicts
        if line.rstrip().endswith(":"):
            carried = header
    return found


# The written form of a contact, to tell a piece that only lists contacts.
_CONTACT_TEXT = re.compile(
    r"(?:@|https?://|(?:t|telegram)\.me/)[A-Za-z0-9_/.?=]+|\+?\d[\d\s\-().]{6,22}\d"
)


def _only_contacts(piece: str) -> bool:
    rest = _JOINERS.sub(" ", _CONTACT_TEXT.sub(" ", piece))
    return not re.search(r"\w", rest)


def _read_sentence(part: str) -> tuple[list[tuple[str, str]], str]:
    """The verdicts of one sentence's contacts, and what its last clause says by its words
    alone (for the list a line ending with «:» introduces)."""
    places = targets_with_places(part)
    pieces: list[tuple[int, int]] = []
    previous = 0
    for match in _CLAUSE_BREAK.finditer(part):
        pieces.append((previous, match.start()))
        previous = match.end()
    pieces.append((previous, len(part)))
    clauses: list[list[int]] = []  # [start, end]
    for start, end in pieces:
        listed = any(start <= at < end for at, _ in places)
        if clauses and listed and _only_contacts(part[start:end]):
            clauses[-1][1] = end
        else:
            clauses.append([start, end])
    found: list[tuple[str, str]] = []
    header = PLAIN
    for start, end in clauses:
        words = part[start:end]
        wide = _EXCLUDING.search(words) is not None
        after = [start + m.start() for m in _EXCLUDING_AFTER.finditer(words)]
        told = _TELLING.search(words) is not None and not _NEGATION.search(words)
        for at, contact in places:
            if not start <= at < end:
                continue
            if wide or any(a < at for a in after):
                found.append((contact, EXCLUDED))
            else:
                found.append((contact, TOLD if told else PLAIN))
        header = EXCLUDED if wide or after else TOLD if told else PLAIN
    return found, header
