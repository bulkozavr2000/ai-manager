"""Fixed texts that the code itself sends to a counterpart, in the counterpart's language.

Closing an off-topic chat, handing a chat over to the owner and turning an offer down are
decided by code, so their words are fixed too; the model does not write them. Uzbek is always
written in Latin, also to someone who writes in Cyrillic. Russian stands in for a language
without its own text ("other", or not known yet).
"""

from __future__ import annotations

from collections.abc import Mapping

# The counterpart's language and script, as stored.
LANGUAGES = ("ru", "uz_latn", "uz_cyrl")

# How the agent says it is an AI in its first reply, when neither its own words nor the
# account's name do.
INTRO: Mapping[str, str] = {
    "ru": "Я ИИ-помощник Саида",
    "uz_latn": "Men Saidning sun'iy intellekt yordamchisiman",
}
# The counterpart does not want to talk to an AI: the owner will write himself.
HANDOFF: Mapping[str, str] = {
    "ru": "Понял, передам Саиду, он напишет сам",
    "uz_latn": "Tushundim, Saidga yetkazaman, o'zi yozadi",
}
# The owner turned the offer down.
DECLINE: Mapping[str, str] = {
    "ru": "Спасибо, пока не нужно",
    "uz_latn": "Rahmat, hozircha kerak emas",
}

_WRITTEN_IN = {"uz_cyrl": "uz_latn"}  # Uzbek goes out in Latin, whatever came in
# «Ассаляму алейкум!», «Добрый день!»: a short opening up to its "!".
_GREETING_CHARS = 30


def pick(texts: Mapping[str, str], language: str | None) -> str:
    """The text in this language (Uzbek in Latin), or the Russian one."""
    language = language or "ru"
    return texts.get(_WRITTEN_IN.get(language, language), texts["ru"])


def introduce(text: str, intro: str) -> str:
    """The message with `intro` right after the greeting it opens with («Ассаляму алейкум! Я
    ИИ-помощник Саида. Подскажите…»), or in front of it when it opens with none."""
    greeting, bang, rest = text.partition("!")
    if bang and len(greeting) <= _GREETING_CHARS:
        rest = rest.strip()
        return f"{greeting}! {intro}" + (f". {rest}" if rest else "")
    return f"{intro}. {text}"
