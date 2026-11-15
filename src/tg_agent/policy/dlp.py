"""Data-loss checks for a text that is about to be sent to a counterpart.

`scan` finds what must never leave in the agent's own words: personal data of anyone (phone, card,
e-mail and passport numbers), links, the owner's real phone and address, the canary tokens planted
in the prompt, and the `{PHONE}` / `{ADDRESS}` placeholders that stand for the owner's data once the
owner has approved sharing it. It only reports; `postgates.check_outgoing` turns the findings into
a verdict. The sender fills the placeholders in with `substitute_placeholders`, which reads a text
exactly as `scan` does, so it fills in what the gate saw and nothing else.

Texts are made spelling-neutral before they are matched (`prepare`, `normalize`): one kind of
apostrophe, and Unicode compatibility forms folded (NFKC: a no-break space becomes a space).

Every pattern is meant to run in time linear in the length of the text, whatever the text is: a
repeat is bounded, and a pattern that reads on from its start only starts at the edge of a run of
letters. `tests/unit/test_dlp.py` feeds `scan` long hostile texts to keep it so.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

# What the model writes where the owner's real phone or address belongs; the sender fills them in
# only after the owner has approved sharing.
PHONE_PLACEHOLDER = "{PHONE}"
ADDRESS_PLACEHOLDER = "{ADDRESS}"

Kind = Literal["phone", "card", "email", "url", "passport", "canary", "owner_secret", "placeholder"]


@dataclass(frozen=True)
class Finding:
    """One thing `scan` found in a text.

    `match` is the piece of the text that matched, as `prepare` left it. For a `canary` or an
    `owner_secret` it is the value from the caller's list that was found, spelled as the caller
    gave it: never write the `match` of an `owner_secret` to a log.
    """

    kind: Kind
    match: str


# Uzbek writes the apostrophe of o', g', so'm, sun'iy in several ways. U+00B4 has to be mapped
# before NFKC, which would turn it into a space and an accent.
_APOSTROPHES = str.maketrans(
    dict.fromkeys(
        "\N{MODIFIER LETTER TURNED COMMA}"  # ʻ
        "\N{MODIFIER LETTER APOSTROPHE}"  # ʼ
        "\N{RIGHT SINGLE QUOTATION MARK}"  # ’
        "\N{LEFT SINGLE QUOTATION MARK}"  # ‘
        "`"
        "\N{ACUTE ACCENT}",  # ´
        "'",
    )
)

# Word edges that, unlike \b, do not count "_" as a letter: "_90 123 45 67_" (Markdown emphasis)
# is still a number.
_START = r"(?<![^\W_])"
_END = r"(?![^\W_])"

# A long digit group is a phone number unless it is a price: one of these words follows it.
_SUM_WORDS = (
    "сум",
    "сўм",
    "so'm",
    "som",
    "sum",
    "руб",
    "₽",
    "$",
    "usd",
    "доллар",
    "тыс",
    "ming",
    "минг",
    "млн",
    "mln",
)
_SYMBOL_SUM_WORDS = ("₽", "$")
# What Russian and Uzbek hang on a currency word: «сумов», «рублей», «долларов», «so'mga»,
# «mingta». Anything else is another word: «сумка», «сумасшедший», «somsa».
_SUM_ENDINGS = (
    "а",
    "ы",
    "ов",
    "ам",
    "ах",
    "ами",
    "ль",
    "ли",
    "ля",
    "лю",
    "лей",
    "лям",
    "лях",
    "лями",
    "яч",
    "яча",
    "ячи",
    "ячу",
    "ячей",
    "ni",
    "ga",
    "da",
    "dan",
    "ning",
    "lik",
    "lar",
    "ta",
    "ни",
    "га",
    "да",
    "дан",
    "нинг",
    "лик",
    "лар",
    "та",
)
_MIN_PHONE_DIGITS = 9
_CARD_DIGITS = range(13, 20)
# A number written without its country code is the same number: a secret phone is also looked for
# by the digits an Uzbek number has after +998.
_NATIONAL_NUMBER_DIGITS = 9

# Digits with phone punctuation between them: up to three separator characters in a row, so ") ",
# " (" and " - " stay inside a number; a hyphen, figure dash, en dash, em dash or minus sign all
# count as a dash. No dots: "12.11.2026".
_SEPARATOR = r"(?:[ \t]|[-\N{HYPHEN}\N{FIGURE DASH}\N{EN DASH}\N{EM DASH}\N{MINUS SIGN}()])"
_DIGIT_GROUP = re.compile(rf"{_START}\+?\(?\d(?:{_SEPARATOR}{{0,3}}\d)*")
# The currency word has to begin within three characters of the number, with no sentence
# punctuation and no line break between them, and has to end where a word ends (after at most one
# of the endings above): "90 123 45 67. Сумму уточню" is a phone, "300 000 000 сумов" is a price.
_SUM_LETTERS = "|".join(re.escape(word) for word in _SUM_WORDS if word not in _SYMBOL_SUM_WORDS)
_SUM_ENDING = "|".join(_SUM_ENDINGS)
_SUM_SYMBOLS = "".join(re.escape(symbol) for symbol in _SYMBOL_SUM_WORDS)
_SUM_WORD_AFTER = re.compile(
    rf"[^\w.,;:!?\r\n]{{0,3}}?(?:(?:{_SUM_LETTERS})(?:{_SUM_ENDING})?{_END}|[{_SUM_SYMBOLS}])",
    re.IGNORECASE,
)

# The lookbehind makes the pattern start only at the beginning of a run of address characters, so
# that a long word without an "@" is read once and not once per letter.
_EMAIL = re.compile(r"(?<![\w.+\-])[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")
# Domains that are links even without a path; with a path, any ending is a link.
_DOMAINS = (
    "com|uz|ru|org|net|io|me|app"
    "|kz|kg|tj|co|info|shop|store|online|site|xyz|dev|cc|ly|gl|to|tv|link|ee"
    "|ua|by|su|biz|рф"
)
_LABEL = r"[\w\-]{1,63}"
_PORT = r"(?::\d{1,5})?"
# In this order: any scheme://, a Telegram deep link, www., t.me (also "t . me/x"), an IPv4 address
# with a path, any domain with a path (the ending in Latin letters, so that "тыс.руб/мес" and
# "г.Ташкент/ул" are not links), a domain with a common ending. Every repeat is bounded.
_URL = re.compile(
    rf"""{_START}(?:
        [a-z][a-z0-9+.\-]{{1,15}}://\S+
      | tg:\S+
      | www\.\S+
      | (?:t|telegram)[ \t]*\.[ \t]*me{_END}(?:[ \t]*/[ \t]*\S*)?
      | (?:\d{{1,3}}\.){{3}}\d{{1,3}}{_PORT}/\S*
      | (?:{_LABEL}\.){{1,6}}[a-z]{{2,24}}{_PORT}/\S*
      | (?:{_LABEL}\.){{1,6}}(?:{_DOMAINS}){_END}{_PORT}(?:/\S*)?
    )""",
    re.IGNORECASE | re.VERBOSE,
)
_URL_TAIL = ".,;:!?)]}>»\"' \t"  # sentence punctuation that sticks to a link

# Two capital letters and seven digits, as one token: not the tail of a longer code.
_PASSPORT = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}[^\S\n]?\d{7}(?!\d)")
_PLACEHOLDER = re.compile(
    "|".join(re.escape(placeholder) for placeholder in (PHONE_PLACEHOLDER, ADDRESS_PLACEHOLDER)),
    re.IGNORECASE,
)
# A secret that is only digits and phone punctuation is a number; anything else is text.
_NUMBER_SECRET = re.compile(r"[+\d\s\-().]+")


def prepare(text: str) -> str:
    """The text with one kind of apostrophe and Unicode compatibility forms folded."""
    return unicodedata.normalize("NFKC", text.translate(_APOSTROPHES))


def normalize(text: str) -> str:
    """The form lexicons and text secrets are compared in: prepared, case-folded, single spaces."""
    return _fold(prepare(text))


def _fold(prepared: str) -> str:
    return " ".join(prepared.casefold().split())


# Case, spaces and punctuation do not hide the owner's address: only letters and digits are
# compared («Чиланзар 5, дом 12» and «ЧИЛАНЗАР-5 дом 12» are one text; rulings R91, R103).
_NOT_LETTER = re.compile(r"[\W_]+")


def plain(text: str) -> str:
    """Only the letters and digits of `text`, prepared and case-folded."""
    return _NOT_LETTER.sub("", prepare(text).casefold())


def _digits(text: str) -> str:
    """The decimal digits of `text`, as ASCII."""
    return "".join(str(int(char)) for char in text if char.isdecimal())


def _luhn_valid(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        digit = int(char)
        if position % 2:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


def _number_findings(prepared: str) -> list[Finding]:
    """Cards and phones: groups of digits, the card test first."""
    found = []
    for group in _DIGIT_GROUP.finditer(prepared):
        digits = _digits(group.group())
        if len(digits) in _CARD_DIGITS and _luhn_valid(digits):
            found.append(Finding("card", group.group()))
        elif len(digits) >= _MIN_PHONE_DIGITS and not _SUM_WORD_AFTER.match(prepared, group.end()):
            found.append(Finding("phone", group.group()))
    return found


def _url_findings(prepared: str, email_spans: list[tuple[int, int]]) -> list[Finding]:
    starts = [low for low, _ in email_spans]
    found = []
    for match in _URL.finditer(prepared):
        start = match.start()
        end = start + len(match.group().rstrip(_URL_TAIL))
        # the domain of an e-mail address belongs to the address, it is not a second finding; the
        # e-mails do not overlap, so only the last one that begins at or before the link can hold it
        inside = bisect_right(starts, start) - 1
        if inside < 0 or end > email_spans[inside][1]:
            found.append(Finding("url", prepared[start:end]))
    return found


def _contains(folded: str, needle: str) -> bool:
    """Is `needle` in the already normalized text? A blank needle matches nothing."""
    wanted = normalize(needle)
    return bool(wanted) and wanted in folded


def _holds_secret(secret: str, letters: str, digits: str) -> bool:
    cleaned = prepare(secret).strip()
    if _NUMBER_SECRET.fullmatch(cleaned) and any(char.isdecimal() for char in cleaned):
        number = _digits(cleaned)
        return number in digits or number[-_NATIONAL_NUMBER_DIGITS:] in digits
    wanted = plain(secret)
    return bool(wanted) and wanted in letters


def scan(
    text: str, *, canaries: Iterable[str] = (), owner_secrets: Iterable[str] = ()
) -> list[Finding]:
    """Everything in `text` that must not go out in the agent's own words.

    Kinds: `phone` (a group of digits, possibly with spaces, hyphens, dashes, parentheses and a
    leading plus, of at least nine digits and no currency word right after it), `card` (13-19
    digits that pass the Luhn check; tried before `phone`), `email`, `url` (any `scheme://`,
    `tg:`, `www.`, `t.me` also written `t . me`, an IPv4 address with a path, any `host.tld/path`,
    or a domain in one of the common endings), `passport` (two capital letters and seven digits),
    `canary` and `owner_secret` (one of the given values occurs in the text), `placeholder`
    (`{PHONE}`, `{ADDRESS}`).

    A secret that is only digits and phone punctuation is compared as digits, so it is found
    however the text formats it, and with or without its country code. Any other secret is
    compared by its letters and digits only (`plain`), a canary case-folded with whitespace
    collapsed. Blank values match nothing.

    The findings come in a fixed order: the kinds above, in the order they are listed, and by
    position within a kind.
    """
    prepared = prepare(text)
    folded = _fold(prepared)
    digits = _digits(prepared)
    emails = list(_EMAIL.finditer(prepared))

    findings = _number_findings(prepared)
    findings += [Finding("email", email.group()) for email in emails]
    findings += _url_findings(prepared, [email.span() for email in emails])
    findings += [Finding("passport", match.group()) for match in _PASSPORT.finditer(prepared)]
    findings += [Finding("canary", canary) for canary in canaries if _contains(folded, canary)]
    findings += [
        Finding("owner_secret", secret)
        for secret in owner_secrets
        if _holds_secret(secret, plain(prepared), digits)
    ]
    findings += [Finding("placeholder", match.group()) for match in _PLACEHOLDER.finditer(prepared)]
    return findings


# --- the placeholders, filled in by the sender ---------------------------------------------------

_PLACEHOLDER_NAMES = {
    placeholder.casefold(): placeholder for placeholder in (PHONE_PLACEHOLDER, ADDRESS_PLACEHOLDER)
}


def placeholders_in(text: str) -> frozenset[str]:
    """Which placeholders the text holds: `PHONE_PLACEHOLDER` and / or `ADDRESS_PLACEHOLDER`.

    The text is read the way `scan` reads it, so a placeholder in any case, or written with
    fullwidth letters and braces, counts: this is the one definition of "a placeholder".
    """
    return frozenset(
        _PLACEHOLDER_NAMES[match.group().casefold()]
        for match in _PLACEHOLDER.finditer(prepare(text))
    )


def _aligned(text: str) -> tuple[str, list[int]]:
    """`prepare(text)` and, for each of its characters, the index in `text` it came from.

    The text is folded one character at a time. That is the same as folding it whole, except for
    characters that combine with their neighbours, and these cannot make or break a placeholder,
    which is plain letters and braces: the letters and braces of a placeholder come out of single
    characters, and a combining mark between them keeps it from matching either way.
    """
    folded: list[str] = []
    origins: list[int] = []
    for index, char in enumerate(text):
        piece = prepare(char)
        folded.append(piece)
        origins.extend([index] * len(piece))
    return "".join(folded), origins


def substitute_placeholders(text: str, *, phone: str | None, address: str | None) -> str:
    """The text with each placeholder `scan` finds in it replaced by the matching value.

    Every other character of the text stays as it was written: `scan` reads a folded copy, but
    what is sent is the text itself. A value is inserted as it is (it is not a template, and
    nothing in it is filled in again). A placeholder whose value is None is left in the text; the
    caller checks first (`placeholders_in`) that every placeholder in it is allowed and has a
    value, and checks the result again: no placeholder may be left in what goes out.
    """
    values = {PHONE_PLACEHOLDER: phone, ADDRESS_PLACEHOLDER: address}
    folded, origins = _aligned(text)
    pieces: list[str] = []
    kept_from = 0
    for match in _PLACEHOLDER.finditer(folded):
        value = values[_PLACEHOLDER_NAMES[match.group().casefold()]]
        if value is None:
            continue
        start, end = origins[match.start()], origins[match.end() - 1] + 1
        pieces += [text[kept_from:start], value]
        kept_from = end
    pieces.append(text[kept_from:])
    return "".join(pieces)
