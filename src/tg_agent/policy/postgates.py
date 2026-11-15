"""The checks run on the model's draft before it is queued for sending.

The lexicons below are plain word lists, compared after `dlp.prepare` and a case-fold. An entry
matches from the start of a word on, so it also covers the endings and particles Russian and Uzbek
hang on a word; an entry that ends in a name or in «бот» ends where that word ends, and the short
words «ии» and «ai» have to stand alone. In the denial list the words of an entry may be parted by
punctuation, so a dash or a comma in the middle of a phrase does not hide it ("Я — человек"). A
«не» right before an entry (nothing but a space between) takes it back only for the three entries
that can be negated ("а не реальный человек", "Вам не пишет Саид"): "Не, я человек" is a denial.

An AI word with a negation next to it ("я не нейросеть", "ИИ тут нет", "sun'iy intellekt emasman")
is no admission but a denial.

The lists are a coarse net, not a reading of the text: the system prompt and the judge model are
the other layers.

`check_outgoing` runs every rule, keeps every reason, and takes `escalate` and `alert` from the
first rule that fired, in the order of the rules. Like the patterns of `dlp`, everything here runs
in time linear in the length of the text.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal

from tg_agent.models import ApprovalKind
from tg_agent.policy.dlp import ADDRESS_PLACEHOLDER, PHONE_PLACEHOLDER, normalize, prepare, scan

MAX_MESSAGES = 3
MAX_MESSAGE_LENGTH = 400
_UZBEK_LETTERS = re.compile("[ўқғҳЎҚҒҲ]")  # Uzbek Cyrillic letters Russian does not have
# Uzbek words in Latin that slip into a Russian reply («Rahmat!»).
_UZBEK_WORDS = re.compile(
    r"(?<![^\W_])(?:rahmat|raxmat|yaxshi|xo['‘’ʻ]?p|tushundim|aniqlashtirib|keyinroq)"
    r"(?![^\W_])",
    re.IGNORECASE,
)
_LATIN_LETTERS = re.compile(r"[A-Za-z]")
_CYRILLIC_LETTERS = re.compile(r"[Ѐ-ӿ]")


def looks_uzbek(text: str) -> bool:
    """Is there Uzbek in the text: a letter Russian lacks, or a common Uzbek word?"""
    return bool(_UZBEK_LETTERS.search(text) or _UZBEK_WORDS.search(text))

# Word edges that, unlike \b, do not count "_" as a letter.
_START = r"(?<![^\W_])"
_END = r"(?![^\W_])"

# Claiming to be a person or the owner, or denying being a bot. These end where a word ends:
# «это саиду нужно», «не ботинки», «я не ботаник» and «обычный человеческий» are not denials.
_DENIAL_WHOLE = (
    "я не бот",
    "я не робот",
    "я не ии",
    "я саид",
    # beyond the plan's list:
    "не бот",
    "это саид",
    "с вами саид",
    "говорит саид",
    "меня зовут саид",
)
# Whole words too, but a «не» right before them says the opposite: "Вам не пишет Саид", "я ИИ, а не
# реальный человек". Only these three: a «не» before any other entry ("Не, я человек") changes
# nothing.
_DENIAL_NEGATABLE = (
    "пишет саид",
    # beyond the plan's list:
    "обычный человек",
    "реальный человек",
)
# These may have an ending or a particle after them ("men odammanku"). The Uzbek phrases without
# «men» / «мен» are the same sentence with the pronoun dropped, which Uzbek does all the time.
_DENIAL_STEMS = (
    "я человек",
    "я живой",
    "не искусственный интеллект",
    "это саид пишет",
    "men bot emasman",
    "men odamman",
    "мен бот эмасман",
    "мен одамман",
    # beyond the plan's list:
    "men insonman",
    "мен инсонман",
    "robot emasman",
    "робот эмасман",
    "bot emasman",
    "odamman",
    "insonman",
    "бот эмасман",
    "одамман",
    "инсонман",
)
# «Бот не я», «Бот? Не-а»: a bot that is none. Only spaces and a question mark may stand between
# «бот» and «не»: after a comma, a dash or a full stop it is usually honest ("Я бот, не человек",
# "Я бот. Не волнуйтесь"), and «бот не человек» never is a denial.
_BOT_NE_RE = re.compile(rf"{_START}бот[ \t]*(?:\?[ \t]*)?не{_END}(?![ \t]+человек)")
# A reply that opens with a bare «нет» and then calls itself a helper ("Нет, я помощник
# Саида"): the «нет» answers «вы бот?» (ruling R93). Read on `_squash`ed text: the «нет» is a
# clause of its own. Saying what it is ("Нет, я ИИ-помощник") is no denial.
_BARE_NO_HELPER_RE = re.compile(
    r"^(?:\| )?(?:нет|yo'q|йўқ) \|.*?(?<![^\W_])(?:помощник|yordamchi|ёрдамчи)"
)
# Saying what the agent is. The short words are whole words ("линии", "Мии" and "Said" are not
# admissions); the rest are the beginnings of words ("нейросеть", "искусственного").
_ADMISSION_WORDS = ("ии", "ai")
_ADMISSION_STEMS = ("искусственн", "нейросет", "sun'iy intellekt", "сунъий интеллект")
# Firm first-person promises to buy or pay. Soft forms ("возьмём", "хочу заказать") are left out
# on purpose: they turn every price negotiation into an order.
_COMMITMENT = (
    "оформляйте",
    "оформляем",
    "подтверждаю",
    "заказываю",
    "оплачу",
    "оплачиваю",
    "переведу",
    "перевожу",
    "внесу предоплату",
    "договорились",
    "kelishdik",
    "tasdiqlayman",
    "to'layman",
    "o'tkazaman",
    "buyurtma beraman",
    "келишдик",
    "тасдиқлайман",
    "тўлайман",
    "ўтказаман",
    "буюртма бераман",
)
# "If" in Russian and in Uzbek (Latin, Cyrillic).
_CONDITIONAL = ("если", "agar", "агар")


def _lexicon(entries: Iterable[str], *, whole_words: bool = False) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(normalize(entry)) for entry in entries)
    return re.compile(f"{_START}(?:{alternatives}){_END if whole_words else ''}")


def _denial_lexicon(entries: Iterable[str], *, whole_words: bool = False) -> re.Pattern[str]:
    """Like `_lexicon`, for text from `_squash`: the words of an entry may be parted by a bar."""
    alternatives = "|".join(
        r" (?:\| )?".join(re.escape(word) for word in normalize(entry).split()) for entry in entries
    )
    return re.compile(f"{_START}(?:{alternatives}){_END if whole_words else ''}")


_DENIAL_RES = (_denial_lexicon(_DENIAL_WHOLE, whole_words=True), _denial_lexicon(_DENIAL_STEMS))
_DENIAL_NEGATABLE_RE = _denial_lexicon(_DENIAL_NEGATABLE, whole_words=True)
_ADMISSION_RES = (_lexicon(_ADMISSION_WORDS, whole_words=True), _lexicon(_ADMISSION_STEMS))
_COMMITMENT_RE = _lexicon(_COMMITMENT)
_CONDITIONAL_RE = _lexicon(_CONDITIONAL, whole_words=True)

# A word of a clause: letters and digits, with the Uzbek apostrophe inside it.
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
# A negation within two words of an AI word, in the same clause, turns an admission into a denial:
# before it «не» ("я не нейросеть"), after it «нет» ("ИИ тут нет") or Uzbek «emas» / «yo'q».
_NEGATION_REACH = 2
_NEGATION_BEFORE = frozenset({"не", "emas", "эмас"})
_NEGATION_AFTER = re.compile(r"нет|нету|emas\w*|эмас\w*|yo'q\w*|йўқ\w*")
# "Не просто ИИ", "не только нейросеть" say the writer is an AI and more.
_NOT_A_NEGATION = frozenset({"просто", "только"})

# Where one clause ends and the next begins: sentence punctuation, a line break, a comma, a dash.
_CLAUSE_END = re.compile(r"([.!?;:,\n\N{EM DASH}\N{EN DASH}]+|\s-\s)")

_THOUSAND = "тыс|ming|минг"
_MILLION = "млн|mln|миллион|million"
# A number, then optionally a scale. Groups of three digits after a space, dot or comma are
# thousands ("240 000", "1.200.000", "1,200,000"); any other dot or comma is a decimal one
# ("1,2"). "к" and "k" count as thousands only when they stick to the number ("240к").
_AMOUNT = re.compile(
    rf"""{_START}(?<![.,])
    (?:
        (?P<grouped>[1-9]\d{{0,2}}(?:[ .,]\d{{3}})+)(?:[.,](?P<cents>\d{{1,2}}))?(?!\d)
      | (?P<plain>\d+(?:[.,]\d+)?)
    )
    (?:
        [ ]*(?:(?P<thousand>{_THOUSAND})|(?P<million>{_MILLION}))
      | (?P<kilo>[kк]){_END}
    )?""",
    re.IGNORECASE | re.VERBOSE,
)


def _clauses(text: str) -> list[tuple[str, str]]:
    """The clauses of the case-folded text, each with the punctuation that ended it."""
    pieces = _CLAUSE_END.split(prepare(text).casefold())
    # re.split with a group gives clause, its end, clause, its end, ..., clause
    return [
        (" ".join(body.split()), end)
        for body, end in zip(pieces[::2], [*pieces[1::2], ""], strict=True)
    ]


def _gap(gap: re.Match[str]) -> str:
    # Only spaces and tabs keep a clause together; a line break (the next message too) or any mark
    # at all parts it.
    return " | " if gap.group().strip(" \t") else " "


def _squash(text: str) -> str:
    """The words of the text, one space between them, a bar where there was punctuation.

    A gap of spaces is a space; a gap with a comma, a dash, a bracket, a full stop, a symbol or a
    line break in it is " | ". So "Я — человек" becomes "я | человек": the denial entries read
    across the bar, and `_after_ne` can tell a «не» in the same clause from one before a pause.
    """
    return " ".join(re.sub(r"[^\w']+|_+", _gap, prepare(text).casefold()).split())


def _after_ne(squashed: str, start: int) -> bool:
    """Is «не» the word right before position `start`, with only a space between (one clause)?"""
    return squashed.endswith("не ", 0, start) and (start == 3 or not squashed[start - 4].isalnum())


def _negates(before: list[str], after: list[str]) -> bool:
    """Do the words around an AI word negate it?"""
    for position, word in enumerate(before):
        following = before[position + 1] if position + 1 < len(before) else ""
        if word in _NEGATION_BEFORE and following not in _NOT_A_NEGATION:
            return True
    return any(_NEGATION_AFTER.fullmatch(word) for word in after)


def _ai_mentions(text: str) -> tuple[bool, bool]:
    """(does an AI word stand without a negation, does one stand with a negation)"""
    stated = negated = False
    for clause, _ in _clauses(text):
        words = list(_WORD.finditer(clause))
        starts = [word.start() for word in words]
        for pattern in _ADMISSION_RES:
            for hit in pattern.finditer(clause):
                # the words that hold the hit (a stem sits inside its word, a phrase spans two)
                first = bisect_right(starts, hit.start()) - 1
                last = bisect_right(starts, hit.end() - 1) - 1
                before = [word.group() for word in words[max(0, first - _NEGATION_REACH) : first]]
                after = [word.group() for word in words[last + 1 : last + 1 + _NEGATION_REACH]]
                if _negates(before, after):
                    negated = True
                else:
                    stated = True
    return stated, negated


def contains_ai_denial(text: str) -> bool:
    """Does the text claim the writer is a person, or the owner, or deny being a bot or an AI?

    Punctuation does not hide a phrase ("Я — человек", "Я, человек, не бот"), and neither does a
    «не» before it ("Не, я человек"). An AI word that the writer negates ("я не нейросеть", "ИИ
    тут нет") counts as well.
    """
    squashed = _squash(text)
    if any(pattern.search(squashed) for pattern in _DENIAL_RES):
        return True
    for hit in _DENIAL_NEGATABLE_RE.finditer(squashed):
        # "а не реальный человек" says the opposite, but only a «не» in the same clause does
        if not _after_ne(squashed, hit.start()):
            return True
    if _BARE_NO_HELPER_RE.search(squashed) and not contains_ai_admission(text):
        return True
    return bool(_BOT_NE_RE.search(prepare(text).casefold())) or _ai_mentions(text)[1]


def contains_ai_admission(text: str) -> bool:
    """Does the text say the writer is an AI? An AI word the writer negates does not count."""
    return _ai_mentions(text)[0]


def contains_commitment(text: str) -> bool:
    """Does the text promise to order, pay or confirm?

    Only the listed firm forms count, and only in a clause that states them: a clause under an
    "if" ("Если оплачу сегодня, ...") or one that asks ("Оформляем?") proposes, it does not
    promise. A promise next to such a clause still counts ("Оплачу завтра, подойдёт?", "Если вас
    всё устраивает, оформляйте").
    """
    for clause, end in _clauses(text):
        hit = _COMMITMENT_RE.search(clause)
        if hit and "?" not in end and not _CONDITIONAL_RE.search(clause, 0, hit.start()):
            return True
    return False


def extract_amounts(text: str) -> list[float]:
    """The numbers in the text, in order, with "тыс", "ming", "млн", "mln" and the like applied.

    Every number counts, not only those next to a currency: a price is often written bare, and
    missing one is worse than counting a quantity. Amounts written in words are not found.
    """
    amounts = []
    for match in _AMOUNT.finditer(prepare(text)):
        if match["grouped"]:
            whole = re.sub(r"[ .,]", "", match["grouped"])
            number = Decimal(f"{whole}.{match['cents']}" if match["cents"] else whole)
        else:
            number = Decimal(match["plain"].replace(",", "."))
        if match["thousand"] or match["kilo"]:
            number *= 1_000
        elif match["million"]:
            number *= 1_000_000
        amounts.append(float(number))
    return amounts


@dataclass
class GateResult:
    """The verdict on a draft.

    `reasons` names every rule that fired. `escalate` is the question to put to the owner, if the
    first rule that fired has one (a commitment or a placeholder); `alert` is set when that rule
    is a leak (a canary or one of the owner's secrets).
    """

    ok: bool
    reasons: list[str]
    escalate: ApprovalKind | None = None
    alert: bool = False


@dataclass(frozen=True)
class _Fired:
    reason: str
    escalate: ApprovalKind | None = None
    alert: bool = False


def check_outgoing(
    messages: Sequence[str],
    *,
    commitment: bool,
    granted: Collection[ApprovalKind],
    canaries: Iterable[str],
    owner_secrets: Iterable[str],
    ceiling: float | None,
    language: str | None = None,
) -> GateResult:
    """Check `messages` before they are queued.

    `commitment` is what the writer declared: it counts even when the text sounds innocent.
    `granted` are the approvals the owner has given in this conversation. `ceiling` is the price
    limit of the task (None: no limit); it is compared with every number in a message that makes a
    commitment, whatever the currency. `language` is the counterpart's, as the code tells it (None: not
    known yet).
    """
    text = "\n".join(messages)
    findings = scan(text, canaries=canaries, owner_secrets=owner_secrets)
    kinds = {finding.kind for finding in findings}
    placeholders = {f.match.casefold() for f in findings if f.kind == "placeholder"}
    committed = commitment or contains_commitment(text)

    fired: list[_Fired] = []
    # 1-2. Something that must never leave: a canary from the prompt, the owner's real data.
    if "canary" in kinds:
        fired.append(_Fired("canary", alert=True))
    if "owner_secret" in kinds:
        fired.append(_Fired("owner_secret", alert=True))
    # 3-4. Anybody's personal data; links.
    if kinds & {"phone", "card", "email", "passport"}:
        fired.append(_Fired("pii"))
    if "url" in kinds:
        fired.append(_Fired("link"))
    # 5. A placeholder is filled in only once the owner has approved sharing it.
    if PHONE_PLACEHOLDER.casefold() in placeholders and ApprovalKind.SHARE_PHONE not in granted:
        fired.append(_Fired("placeholder_phone", escalate=ApprovalKind.SHARE_PHONE))
    if ADDRESS_PLACEHOLDER.casefold() in placeholders and ApprovalKind.SHARE_ADDRESS not in granted:
        fired.append(_Fired("placeholder_address", escalate=ApprovalKind.SHARE_ADDRESS))
    # 6. The agent never says it is a person or the owner (ruling R82: what it is, the
    # account's name says).
    if contains_ai_denial(text):
        fired.append(_Fired("ai_denial"))
    # 7-8. A commitment needs the owner's approval of the order, and then the price ceiling holds.
    if committed and ApprovalKind.CONFIRM_ORDER not in granted:
        fired.append(_Fired("commitment", escalate=ApprovalKind.CONFIRM_ORDER))
    elif committed and ceiling is not None:
        amounts = extract_amounts(text)
        if amounts and max(amounts) > ceiling:
            fired.append(_Fired("over_ceiling", escalate=ApprovalKind.CONFIRM_ORDER))
    # 9. Short, and in few messages.
    if len(messages) > MAX_MESSAGES or any(len(m) > MAX_MESSAGE_LENGTH for m in messages):
        fired.append(_Fired("too_long"))
    # 10-11. The counterpart's language: Russian without Uzbek words («Rahmat!» slips
    # in), and Uzbek always in Latin, whatever script the counterpart writes in.
    if language == "ru" and looks_uzbek(text):
        fired.append(_Fired("not_russian"))
    cyrillic = len(_CYRILLIC_LETTERS.findall(text))
    latin = len(_LATIN_LETTERS.findall(text))
    if language in ("uz_latn", "uz_cyrl") and cyrillic > latin:
        fired.append(_Fired("not_uzbek_latin"))

    escalate, alert = (fired[0].escalate, fired[0].alert) if fired else (None, False)
    return GateResult(
        ok=not fired, reasons=[rule.reason for rule in fired], escalate=escalate, alert=alert
    )
