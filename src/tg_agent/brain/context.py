"""Helpers for what the models see: the prompts, the transcript, the counterpart's language.

The price ceiling never reaches a model (`hide_amount` masks it wherever the owner wrote it), and
text that came from counterparts is escaped so it cannot close or open our tags.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable, Sequence
from importlib.resources import files
from zoneinfo import ZoneInfo

from tg_agent.models import Contact, Conversation, Message, MsgSource

MESSAGE_MAX_CHARS = 1000  # one counterpart message cannot flood the context

# Who wrote a line of a transcript, as the models read it.
SPEAKERS = {
    MsgSource.COUNTERPART: "Собеседник",
    MsgSource.AGENT: "Ассистент",
    MsgSource.OWNER_MANUAL: "Саид (сам)",
}
_KIND_LABELS = {
    "photo": "фото",
    "voice": "голосовое",
    "video": "видео",
    "document": "файл",
    "sticker": "стикер",
    "location": "геопозиция",
    "contact": "контакт",
    "poll": "опрос",
}
_UZBEK_CYRILLIC = frozenset("ўқғҳЎҚҒҲ")  # letters Russian does not have
_RUSSIAN_ONLY = frozenset("ыщЫЩ")  # letters the Uzbek alphabet does not have


def escape_untrusted(text: str) -> str:
    """Make angle brackets inert so untrusted text cannot fake our tags."""
    return text.replace("<", "‹").replace(">", "›")


# Every prompt the program uses; `main.start_up` loads them all before the loops run.
PROMPTS = ("task", "persona", "judge", "search", "secretary")


@functools.cache
def load_prompt(name: str) -> str:
    """The prompt `name`, read from the disk once per process. The live bot runs from the
    checkout the next stage is developed in: an edit of a prompt file must not reach it in the
    middle of a task, only with its next start."""
    return (files("tg_agent.brain") / "prompts" / f"{name}.md").read_text(
        encoding="utf-8"
    )


def load_prompts() -> None:
    """Read every prompt the program uses now (a missing one fails the start, not a wake)."""
    for name in PROMPTS:
        load_prompt(name)


def _clip(text: str) -> str:
    if len(text) <= MESSAGE_MAX_CHARS:
        return text
    return text[:MESSAGE_MAX_CHARS] + "…[обрезано]"


def _line(message: Message, tz: ZoneInfo) -> str:
    when = message.created_at.astimezone(tz).strftime("%d.%m %H:%M")
    body = escape_untrusted(_clip(message.text))
    if message.kind != "text":
        body = f"[{_KIND_LABELS.get(message.kind, 'вложение')}] {body}".rstrip()
    return f"[{when}] {SPEAKERS[message.source]}: {body}"


def render_transcript(messages: Iterable[Message], tz: ZoneInfo) -> str:
    return "\n".join(_line(message, tz) for message in messages)


def counterpart_language(
    conv: Conversation, contact: Contact, history: Sequence[Message]
) -> str | None:
    """The counterpart's language code, as far as the code can tell. Their latest message
    settles it, whatever was stored: Uzbek Cyrillic if it has ў, қ, ғ or ҳ, Russian if it has
    ы or щ. Otherwise what is known stands (a short Cyrillic line may be either)."""
    latest = next(
        (
            message.text
            for message in reversed(history)
            if message.source is MsgSource.COUNTERPART and message.text.strip()
        ),
        "",
    )
    if _UZBEK_CYRILLIC.intersection(latest):
        return "uz_cyrl"
    if _RUSSIAN_ONLY.intersection(latest):
        return "ru"
    return conv.language or contact.language


HIDDEN_AMOUNT = "[скрыто]"
# Digits grouped by thousands: a space (also a narrow or non-breaking one), a dot, a comma or
# an apostrophe between the groups («300 000», «300.000», «300'000», «300’000»).
_GROUP_MARK = "[ \u00a0\u202f.,'\u2019]"
# A number as people write money: 300 000 / 300'000 / 300000 / 0,3; then, maybe, a word that
# multiplies it: тыс / тыщ / т.р. / тр / k / к, млн / миллион / лям, the Uzbek минг / ming and
# million (a "к" or "тр" that starts a word, as in "5 корпусов" or "300 трубок", is not one).
_AMOUNT = re.compile(
    rf"(?<![\d.,'\u2019])(\d{{1,3}}(?:{_GROUP_MARK}\d{{3}})+(?!\d)|\d+(?:[.,]\d+)?)"
    r"(?:\s*(минг|ming|million\w*|тыс\w*|тыщ\w*|т\.\s?р\.?|тр|млн\w*|миллион\w*|лям\w*"
    r"|mln|k|к)(?![a-zа-яё]))?",
    re.IGNORECASE,
)
_GROUPED = re.compile(rf"\d{{1,3}}(?:{_GROUP_MARK}\d{{3}})+")
_MULTIPLIERS = (
    ("минг", 1_000),  # before "м": the Uzbek thousand is no million
    ("ming", 1_000),
    ("million", 1_000_000),
    ("т", 1_000),  # тыс, тыщ, т.р., тр
    ("м", 1_000_000),  # млн, миллион
    ("л", 1_000_000),  # лям
    ("mln", 1_000_000),
    ("k", 1_000),
    ("к", 1_000),
)
# A bare number is the ceiling in thousands or millions («до 300» for 300 000, «максимум 2»
# for 2 000 000) only right after one of these words (within `_BUDGET_REACH` characters), and
# only when no other word follows it but a currency or a conjunction («до 300 мм» is a size).
_BUDGET_WORD = re.compile(
    r"(?<![^\W\d_])(?:до|максимум|не дороже|не больше|бюджет\w*|потолок\w*|в пределах"
    r"|уложиться в)(?![^\W\d_])",
    re.IGNORECASE,
)
_BUDGET_REACH = 20
# Right after a bare number these make it something else than money.
_NOT_MONEY_NEXT = frozenset(":%/-\u2013\u2014")
_NEXT_WORD = re.compile(r"\s*([^\W\d_]+)")
_MAY_FOLLOW = re.compile(
    r"(?:сум\w*|сўм\w*|sum|so'?m|uzs|usd|руб\w*|долл\w*|и|или|а|но)", re.IGNORECASE
)

# A number written in words, then «тысяч…» or «миллион…»: «триста тысяч», «полтора
# миллиона», «один миллион двести тысяч» (nominative and genitive forms).
_NUMBER_WORDS: dict[str, float] = {
    **dict.fromkeys(("один", "одна", "одну", "одного", "одной"), 1),
    **dict.fromkeys(("два", "две", "двух"), 2),
    **dict.fromkeys(("три", "трёх", "трех"), 3),
    **dict.fromkeys(("четыре", "четырёх", "четырех"), 4),
    **dict.fromkeys(("пять", "пяти"), 5),
    **dict.fromkeys(("шесть", "шести"), 6),
    **dict.fromkeys(("семь", "семи"), 7),
    **dict.fromkeys(("восемь", "восьми"), 8),
    **dict.fromkeys(("девять", "девяти"), 9),
}
_TEENS: dict[str, float] = {
    **dict.fromkeys(("десять", "десяти"), 10),
    **dict.fromkeys(("одиннадцать", "одиннадцати"), 11),
    **dict.fromkeys(("двенадцать", "двенадцати"), 12),
    **dict.fromkeys(("тринадцать", "тринадцати"), 13),
    **dict.fromkeys(("четырнадцать", "четырнадцати"), 14),
    **dict.fromkeys(("пятнадцать", "пятнадцати"), 15),
    **dict.fromkeys(("шестнадцать", "шестнадцати"), 16),
    **dict.fromkeys(("семнадцать", "семнадцати"), 17),
    **dict.fromkeys(("восемнадцать", "восемнадцати"), 18),
    **dict.fromkeys(("девятнадцать", "девятнадцати"), 19),
}
_TENS: dict[str, float] = {
    **dict.fromkeys(("двадцать", "двадцати"), 20),
    **dict.fromkeys(("тридцать", "тридцати"), 30),
    **dict.fromkeys(("сорок", "сорока"), 40),
    **dict.fromkeys(("пятьдесят", "пятидесяти"), 50),
    **dict.fromkeys(("шестьдесят", "шестидесяти"), 60),
    **dict.fromkeys(("семьдесят", "семидесяти"), 70),
    **dict.fromkeys(("восемьдесят", "восьмидесяти"), 80),
    **dict.fromkeys(("девяносто", "девяноста"), 90),
}
_HUNDREDS: dict[str, float] = {
    **dict.fromkeys(("сто", "ста"), 100),
    **dict.fromkeys(("двести", "двухсот"), 200),
    **dict.fromkeys(("триста", "трёхсот", "трехсот"), 300),
    **dict.fromkeys(("четыреста", "четырёхсот", "четырехсот"), 400),
    **dict.fromkeys(("пятьсот", "пятисот"), 500),
    **dict.fromkeys(("шестьсот", "шестисот"), 600),
    **dict.fromkeys(("семьсот", "семисот"), 700),
    **dict.fromkeys(("восемьсот", "восьмисот"), 800),
    **dict.fromkeys(("девятьсот", "девятисот"), 900),
}
_HALF = frozenset({"полтора", "полторы", "полутора"})
# «тысяча» and «миллион» alone are one of them («миллион двести тысяч»).
_ALONE = frozenset({"тысяча", "миллион"})
_WORD = re.compile(r"[а-яё]+", re.IGNORECASE)


def _multiplier(word: str) -> float | None:
    if word.startswith(("тыс", "тыщ")):  # тысяч, тысячи, «тыс.»
        return 1_000
    if word.startswith(("миллион", "лям")):
        return 1_000_000
    return None


def _small(words: list[str], i: int) -> tuple[float | None, int]:
    """The number of up to three words at `i` («двести пятьдесят», «полтора»), and where it
    ends; None when no number word stands there."""
    if i < len(words) and words[i] in _HALF:
        return 1.5, i + 1
    value, j = 0.0, i
    for table in (_HUNDREDS, _TEENS, _TENS, _NUMBER_WORDS):
        if j < len(words) and words[j] in table:
            if table is _NUMBER_WORDS and j > i and words[j - 1] in _TEENS:
                break
            value += table[words[j]]
            j += 1
    return (value, j) if j > i else (None, i)


def _hide_words(text: str, amount: float) -> str:
    """`text` with the ceiling hidden where it is written in words."""
    found = list(_WORD.finditer(text))
    words = [m.group().casefold() for m in found]
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(found):
        total, j, end = 0.0, i, None
        while j < len(found):
            # only words one after another, nothing but spaces between them
            if j > i and not text[found[j - 1].end() : found[j].start()].isspace():
                break
            value, k = _small(words, j)
            if value is None and words[j] in _ALONE:
                value, k = 1.0, j
            if value is None or k >= len(found):
                break
            if k > j and not text[found[k - 1].end() : found[k].start()].isspace():
                break
            factor = _multiplier(words[k])
            if factor is None:
                break
            total += value * factor
            end, j = found[k].end(), k + 1
        if end is None:
            i += 1
            continue
        if abs(total - amount) < 0.5:
            spans.append((found[i].start(), end))
        i = j  # a part of a longer number is never taken for a number of its own
    for start, stop in reversed(spans):
        text = text[:start] + HIDDEN_AMOUNT + text[stop:]
    return text


def _after_a_budget_word(text: str, start: int, end: int) -> bool:
    """Does a budget word stand just before the bare number at `start`…`end`, and nothing
    but a currency or a conjunction after it?"""
    if not any(
        word.end() <= start and start - word.end() <= _BUDGET_REACH
        for word in _BUDGET_WORD.finditer(text, 0, start)
    ):
        return False
    # «до 10:00», «до 10%», «до 5-7 дней», «до 2/3»: a time, a share, a range, a fraction.
    if text[end : end + 1] in _NOT_MONEY_NEXT:
        return False
    following = _NEXT_WORD.match(text, end)
    return following is None or bool(_MAY_FOLLOW.fullmatch(following.group(1)))


def hide_amount(text: str, amount: float | None) -> str:
    """`text` with every spelling of `amount` hidden: 300 000, 300000, 300'000, 300 тыс,
    300 тыщ, 300 т.р., 300к, 0,3 млн, 0.3 миллиона, 2 ляма; «триста тысяч», «полтора
    миллиона»; and a bare 300 or 2 (the ceiling in thousands or millions) right after «до»,
    «максимум», «бюджет» and the like. Any other number stays as written.

    The price ceiling must never reach the model; the owner may have written it into the goal.
    """
    if not amount:
        return text

    def swap(match: re.Match[str]) -> str:
        number, suffix = match.group(1), (match.group(2) or "").lower()
        if _GROUPED.fullmatch(number):
            value = float(re.sub(r"\D", "", number))
        else:
            value = float(number.replace(",", "."))
        if not suffix:
            if abs(value - amount) < 0.5:
                return HIDDEN_AMOUNT
            bare = not _GROUPED.fullmatch(number)
            if (
                bare
                and any(abs(value * f - amount) < 0.5 for f in (1_000, 1_000_000))
                and _after_a_budget_word(match.string, match.start(), match.end())
            ):
                return HIDDEN_AMOUNT
            return match.group(0)
        for prefix, factor in _MULTIPLIERS:
            if suffix.startswith(prefix):
                value *= factor
                break
        return HIDDEN_AMOUNT if abs(value - amount) < 0.5 else match.group(0)

    return _AMOUNT.sub(swap, _hide_words(text, amount))


def money_amounts(text: str) -> list[float]:
    """The sums of money a text names (a thousand or more): «400 000», «400 тыс», «0,4 млн».
    A text holding several of the owner's messages may name a budget he took back: every such
    sum is hidden from the agent (`hide_amount` of each)."""
    sums: list[float] = []
    for match in _AMOUNT.finditer(text):
        number, suffix = match.group(1), (match.group(2) or "").lower()
        if _GROUPED.fullmatch(number):
            value = float(re.sub(r"\D", "", number))
        else:
            value = float(number.replace(",", "."))
        for prefix, factor in _MULTIPLIERS:
            if suffix.startswith(prefix):
                value *= factor
                break
        if value >= 1_000 and value not in sums:
            sums.append(value)
    return sums


def hide_money(text: str) -> str:
    """`text` with every sum of money it names hidden (`money_amounts`)."""
    for value in money_amounts(text):
        text = hide_amount(text, value)
    return text


def media_type(data: bytes) -> str:
    """The MIME type of an image the model is shown, from its first bytes."""
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"GIF8"):
        return "image/gif"
    return "image/jpeg"
