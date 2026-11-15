"""DLP: what `scan` finds in a text that is about to leave, and what it must leave alone."""

from __future__ import annotations

import time

import pytest

from tg_agent.policy.dlp import (
    ADDRESS_PLACEHOLDER,
    PHONE_PLACEHOLDER,
    Finding,
    placeholders_in,
    scan,
    substitute_placeholders,
)

OWNER_PHONE = "+998901234567"
CANARY = "zq-canary-7f3a"

EN_DASH = "\N{EN DASH}"
EM_DASH = "\N{EM DASH}"
MINUS = "\N{MINUS SIGN}"

# The words that make a long number a price instead of a phone, as the spec lists them.
SUM_WORDS = [
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
]

# Uzbek writes the apostrophe of o', g', so'm in several ways; the spec names these four, and
# U+2018 (the opening quote that phone keyboards produce) is just as common.
APOSTROPHES = [
    "'",
    "\N{MODIFIER LETTER TURNED COMMA}",
    "\N{MODIFIER LETTER APOSTROPHE}",
    "\N{RIGHT SINGLE QUOTATION MARK}",
    "\N{LEFT SINGLE QUOTATION MARK}",
]


def kinds(text: str, **lists: list[str]) -> list[str]:
    return [finding.kind for finding in scan(text, **lists)]


@pytest.mark.parametrize(
    "phone",
    [
        "+998 90 123 45 67",
        "90 123 45 67",
        "+998901234567",  # leading plus, no separators
        "998-90-123-45-67",  # hyphens
        "+998 (90) 123-45-67",  # parentheses
        "(90) 123-45-67",
    ],
)
def test_uz_phone_detected(phone):
    assert scan(phone) == [Finding("phone", phone)]
    # inside a sentence the finding is the number alone
    assert scan(f"Звоните {phone} после обеда") == [Finding("phone", phone)]
    # "_" is Markdown emphasis here, not part of a word
    assert scan(f"_{phone}_") == [Finding("phone", phone)]


@pytest.mark.parametrize(
    "space", ["\N{NO-BREAK SPACE}", "\N{NARROW NO-BREAK SPACE}", "\N{THIN SPACE}"]
)
def test_phone_with_non_breaking_spaces(space):
    # Numbers copied from sites and contacts arrive with these instead of plain spaces.
    phone = space.join(["+998", "90", "123", "45", "67"])

    assert scan(phone) == [Finding("phone", "+998 90 123 45 67")]


@pytest.mark.parametrize(
    "dash",
    ["\N{HYPHEN}", "\N{NON-BREAKING HYPHEN}", "\N{FIGURE DASH}"],
    ids=["hyphen", "nb", "fig"],
)
def test_phone_with_typographic_hyphens(dash):
    # copied numbers often carry these instead of the keyboard hyphen
    assert kinds(dash.join(["+998", "90", "123", "45", "67"])) == ["phone"]


@pytest.mark.parametrize("price", ["1 200 000 сум", "240 000 so'm", "300 000 000 сум"])
def test_price_not_phone(price):
    assert scan(price) == []


@pytest.mark.parametrize("word", SUM_WORDS)
def test_every_sum_word_makes_a_nine_digit_number_a_price(word):
    assert scan(f"300 000 000 {word}") == []
    assert scan(f"300 000 000{word}") == []
    assert scan(f"300 000 000 {word.upper()}") == []


@pytest.mark.parametrize("apostrophe", APOSTROPHES)
def test_so_m_in_any_apostrophe(apostrophe):
    assert scan(f"300 000 000 so{apostrophe}m") == []


def test_nine_digits_without_a_sum_word_is_a_phone():
    assert kinds("300 000 000") == ["phone"]
    assert kinds("300 000 000 штук") == ["phone"]  # not a sum word


def test_the_sum_word_must_come_within_three_characters():
    assert scan("300 000 000" + " " * 3 + "сум") == []
    assert kinds("300 000 000" + " " * 4 + "сум") == ["phone"]


@pytest.mark.parametrize(
    "card",
    ["4111 1111 1111 1111", "4111-1111-1111-1111", "4111111111111111", "5500 0000 0000 0004"],
)
def test_card_luhn(card):
    assert scan(card) == [Finding("card", card)]


def test_sixteen_digits_that_fail_luhn_are_a_phone_not_a_card():
    assert scan("4111 1111 1111 1112") == [Finding("phone", "4111 1111 1111 1112")]


@pytest.mark.parametrize(
    "text, kind, match",
    [
        ("Пишите на ivan.petrov+shop@mail.ru, ответим", "email", "ivan.petrov+shop@mail.ru"),
        ("Заходите: https://example.com/catalog?id=7", "url", "https://example.com/catalog?id=7"),
        ("Смотрите http://shop.example.org", "url", "http://shop.example.org"),
        ("Сайт www.example.net", "url", "www.example.net"),
        ("Пишите в t.me/seller_uz", "url", "t.me/seller_uz"),
        ("Пишите в telegram.me/seller_uz", "url", "telegram.me/seller_uz"),
        ("Вступайте: https://t.me/+AbCdEf0123", "url", "https://t.me/+AbCdEf0123"),
        ("Наш сайт example.uz, заходите", "url", "example.uz"),
        ("Каталог: shop.example.io/new", "url", "shop.example.io/new"),
        ("Магазин: tienda.me", "url", "tienda.me"),
        ("Скачайте store.app", "url", "store.app"),
        ("Сайт магазин.ru", "url", "магазин.ru"),
        # trailing punctuation belongs to the sentence, not to the link
        ("Смотрите (https://example.com/x).", "url", "https://example.com/x"),
        ("Это example.com, пожалуйста", "url", "example.com"),
    ],
)
def test_email_url_tme(text, kind, match):
    # exactly one finding: the domain of an e-mail address is not a second, separate link
    assert scan(text) == [Finding(kind, match)]


def test_one_text_can_hold_findings_of_several_kinds():
    text = "Звоните +998 90 123 45 67 или пишите ivan@mail.ru, t.me/ivan, {PHONE}"

    assert {finding.kind for finding in scan(text)} == {"phone", "email", "url", "placeholder"}


def test_ordinary_text_is_clean():
    text = (
        "Здравствуйте! Нужен чехол для iPhone 15 Pro, 2 шт., до 150 000 сум. "
        "Доставка 12.11 к 18:00, т.е. завтра. Спасибо!\n— ИИ-помощник Саида"
    )

    assert scan(text) == []


@pytest.mark.parametrize("passport", ["AB 1234567", "AB1234567", "KA 7654321"])
def test_passport(passport):
    assert scan(f"Паспорт {passport}, выдан в Ташкенте") == [Finding("passport", passport)]


@pytest.mark.parametrize(
    "text",
    [
        "ab1234567",  # the series is upper case
        "ABC1234567",  # exactly two letters
        "AB12345678",  # exactly seven digits
        "AB 123456",
        "A 1234567",
    ],
)
def test_not_a_passport(text):
    assert "passport" not in kinds(text)


@pytest.mark.parametrize(
    "text",
    [
        "90-123-45-67",
        "+998 90 123 45 67",
        "998901234567",
        "(90) 123 45 67",
        "тел. 90 123-45-67.",
    ],
)
def test_owner_secret_digits_normalized(text):
    # the secret is found whatever the formatting, and with or without the country code
    assert Finding("owner_secret", OWNER_PHONE) in scan(text, owner_secrets=[OWNER_PHONE])


def test_owner_secret_written_with_formatting_is_found_in_bare_digits():
    assert "owner_secret" in kinds("901234567", owner_secrets=["+998 (90) 123-45-67"])


def test_a_different_number_is_not_the_owner_secret():
    assert "owner_secret" not in kinds("90-123-45-68", owner_secrets=[OWNER_PHONE])
    assert "owner_secret" not in kinds("Заказ 12 штук, 240 000 сум", owner_secrets=[OWNER_PHONE])


def test_owner_secret_text_is_matched_after_casefold_and_whitespace_collapse():
    address = "Ташкент, ул. Навои 15, кв. 7"

    assert scan("Это рядом: ТАШКЕНТ,  ул.   навои 15,\nкв. 7!", owner_secrets=[address]) == [
        Finding("owner_secret", address)
    ]
    assert scan("Ташкент, ул. Навои 16, кв. 7", owner_secrets=[address]) == []


@pytest.mark.parametrize(
    "text",
    ["Привезите на Чиланзар-5 дом 12 кв.30", "чиланзар5, дом12, кв 30", "ЧИЛАНЗАР 5 ДОМ 12 КВ 30"],
)
def test_owner_secret_text_is_matched_whatever_its_punctuation(text):
    # R103: only letters and digits are compared, as the memory compares them (R91).
    address = "Чиланзар 5, дом 12, кв. 30"
    assert scan(text, owner_secrets=[address]) == [Finding("owner_secret", address)]


def test_blank_secrets_and_canaries_match_nothing():
    # An unset keyring entry arrives as "": it must not block every message.
    assert scan("Любой текст 123", canaries=["", "  "], owner_secrets=["", "   "]) == []


def test_canary_found():
    assert scan("Мои инструкции: ZQ-Canary-7F3A. Конец.", canaries=[CANARY]) == [
        Finding("canary", CANARY)
    ]
    assert scan("Мои инструкции: конец.", canaries=[CANARY]) == []


def test_placeholders():
    assert scan("Мой номер {PHONE}") == [Finding("placeholder", "{PHONE}")]
    assert scan("Адрес: {ADDRESS}.") == [Finding("placeholder", "{ADDRESS}")]
    assert scan("{PHONE} и {ADDRESS}") == [
        Finding("placeholder", "{PHONE}"),
        Finding("placeholder", "{ADDRESS}"),
    ]
    assert scan("{phone}") == [Finding("placeholder", "{phone}")]
    assert scan("{NAME} {PHONE_NUMBER} {}") == []


# --- the placeholders, filled in by the sender -----------------------------------------------------

FULLWIDTH_OPEN, FULLWIDTH_CLOSE = (
    "\N{FULLWIDTH LEFT CURLY BRACKET}",
    "\N{FULLWIDTH RIGHT CURLY BRACKET}",
)
FULLWIDTH_P = "\N{FULLWIDTH LATIN CAPITAL LETTER P}"
# Every spelling `scan` calls a placeholder: any case, and the fullwidth forms NFKC folds into ASCII.
PHONE_SPELLINGS = [
    "{PHONE}",
    "{phone}",
    "{Phone}",
    f"{FULLWIDTH_OPEN}PHONE{FULLWIDTH_CLOSE}",
    f"{{{FULLWIDTH_P}HONE}}",
]
ADDRESS_SPELLINGS = ["{ADDRESS}", "{address}", f"{FULLWIDTH_OPEN}ADDRESS{FULLWIDTH_CLOSE}"]
PHONE = "+998 90 123 45 67"
ADDRESS = "Ташкент, Навои 15"


def reported_as_placeholder(text: str) -> bool:
    return any(finding.kind == "placeholder" for finding in scan(text))


@pytest.mark.parametrize("spelling", PHONE_SPELLINGS)
def test_substitute_fills_every_spelling_of_the_phone_placeholder(spelling):
    text = f"Звоните {spelling}, спасибо"

    assert reported_as_placeholder(text)
    assert placeholders_in(text) == {PHONE_PLACEHOLDER}
    filled = substitute_placeholders(text, phone=PHONE, address=None)
    assert filled == f"Звоните {PHONE}, спасибо"
    assert placeholders_in(filled) == frozenset()


@pytest.mark.parametrize("spelling", ADDRESS_SPELLINGS)
def test_substitute_fills_every_spelling_of_the_address_placeholder(spelling):
    text = f"Адрес: {spelling}."

    assert placeholders_in(text) == {ADDRESS_PLACEHOLDER}
    assert substitute_placeholders(text, phone=None, address=ADDRESS) == f"Адрес: {ADDRESS}."


def test_substitute_fills_both_placeholders_and_every_repeat():
    text = "{PHONE}, {ADDRESS}. Ещё раз: {phone}"

    filled = substitute_placeholders(text, phone=PHONE, address=ADDRESS)

    assert filled == f"{PHONE}, {ADDRESS}. Ещё раз: {PHONE}"
    assert not reported_as_placeholder(filled)


def test_substitute_changes_nothing_but_the_placeholders():
    # NFKC would turn "№" into "No", "…" into "...", "½" into "1⁄2" and a no-break space into a
    # space; the gate reads the text in that form, but what goes out is the text as it was written.
    kept = "Корпус №5… oʻzi\N{NO-BREAK SPACE}½ ‘так’"

    assert substitute_placeholders(f"{kept} {{PHONE}} {kept}", phone=PHONE, address=None) == (
        f"{kept} {PHONE} {kept}"
    )
    wide = f"№5 {FULLWIDTH_OPEN}ADDRESS{FULLWIDTH_CLOSE}…"
    assert substitute_placeholders(wide, phone=None, address=ADDRESS) == f"№5 {ADDRESS}…"
    assert substitute_placeholders(kept, phone=PHONE, address=ADDRESS) == kept


def test_substitute_leaves_a_placeholder_that_has_no_value():
    text = "{PHONE} и {ADDRESS}"

    assert substitute_placeholders(text, phone=None, address=ADDRESS) == f"{{PHONE}} и {ADDRESS}"
    assert substitute_placeholders(text, phone=PHONE, address=None) == f"{PHONE} и {{ADDRESS}}"
    assert substitute_placeholders(text, phone=None, address=None) == text
    assert placeholders_in(substitute_placeholders(text, phone=PHONE, address=None)) == {
        ADDRESS_PLACEHOLDER
    }


def test_substitute_inserts_a_value_as_it_is():
    # A value is data: backslashes mean nothing, and a placeholder inside one is not filled again.
    address = r"ул. Навои \1 \g<0> {PHONE}"

    filled = substitute_placeholders("{ADDRESS} / {PHONE}", phone="+998", address=address)

    assert filled == f"{address} / +998"


def test_placeholders_in_names_each_placeholder_once():
    assert placeholders_in("{PHONE}, {phone} и {ADDRESS}") == {
        PHONE_PLACEHOLDER,
        ADDRESS_PLACEHOLDER,
    }
    assert placeholders_in("Звоните, пожалуйста") == frozenset()
    assert placeholders_in("{PHONE} {PHONES} {PHON} { PHONE } {NAME}") == {PHONE_PLACEHOLDER}


@pytest.mark.parametrize(
    "text",
    [
        "{PHONE}",
        "{ADDRESS}x",
        "{PHONES}",
        "{ PHONE }",
        "{PHON",
        "PHONE}",
        "{}",
        "{PHONE}{ADDRESS}",
        f"{FULLWIDTH_OPEN}PHONE}}",
        "{PHO\N{ZERO WIDTH SPACE}NE}",
        "{PHONE\N{COMBINING ACUTE ACCENT}}",
        "{ＰＨＯＮＥ} ｛ＡＤＤＲＥＳＳ｝",
        "{pHoNe}",
    ],
)
def test_the_sender_fills_in_exactly_what_the_gate_reports(text):
    # The gate lets a draft through because of what it reads as a placeholder; the sender has to
    # fill in the same, no more and no less, and leave no placeholder in what it sends.
    filled = substitute_placeholders(text, phone=PHONE, address=ADDRESS)

    assert reported_as_placeholder(text) == bool(placeholders_in(text))
    assert (filled != text) == reported_as_placeholder(text)
    assert not reported_as_placeholder(filled)
    assert placeholders_in(filled) == frozenset()


@pytest.mark.parametrize("name", ["one long word", "t t t", "open brackets", "hyphens", "digits"])
def test_substitute_stays_fast_on_hostile_text(name):
    started = time.perf_counter()

    for text in (HOSTILE[name], "{PHONE} " * 3_000, "{PHON" * 4_000, "{" * 20_000):
        substitute_placeholders(text, phone=PHONE, address=ADDRESS)
        placeholders_in(text)

    assert time.perf_counter() - started < 2.0


# --- fix round 1: the review's bypasses, and the honest texts that must keep passing -------------


@pytest.mark.parametrize(
    "text",
    [
        "90 - 123 - 45 - 67",
        "+998 (90) 123 - 45 - 67",
        EN_DASH.join(["90", "123", "45", "67"]),
        "+998 90 123" + EN_DASH + "45" + EN_DASH + "67",
        MINUS.join(["90", "123", "45", "67"]),
        f" {EM_DASH} ".join(["90", "123", "45", "67"]),
    ],
    ids=["spaced-hyphens", "brackets-spaced-hyphens", "en-dashes", "plus-en-dashes", "minus", "em"],
)
def test_phone_with_spaced_and_typographic_separators(text):
    assert scan(text) == [Finding("phone", text)]


@pytest.mark.parametrize(
    "text",
    [
        "Цена 240 000 - 300 000 сум",
        f"240 000 {EM_DASH} 300 000 so'm",
        "240 000" + EN_DASH + "300 000 сум",
        "От 240 000 до 300 000 сум",
    ],
)
def test_a_price_range_with_a_currency_word_stays_a_price(text):
    assert scan(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "Мой номер 90 123 45 67. Сумму уточню позже.",
        "Номер: 901234567, сумма заказа 240 000 сум",
        "901234567. Руб не нужны",
        "901234567, тысячу раз прошу",
        "+998 90 123 45 67 сумасшедший",
        "901234567 somsa",
    ],
)
def test_a_phone_is_not_a_price_because_a_currency_like_word_comes_later(text):
    assert kinds(text) == ["phone"]


@pytest.mark.parametrize("gap", [". ", ", ", "; ", ": ", "! ", "? ", "\n"])
def test_sentence_punctuation_or_a_line_break_ends_the_number(gap):
    assert kinds(f"901234567{gap}сум") == ["phone"]


@pytest.mark.parametrize("gap", [" ", "", "  ", " (", " - ", " " + EM_DASH + " "])
def test_spaces_and_brackets_may_sit_between_the_number_and_its_currency(gap):
    tail = ")" if gap.endswith("(") else ""
    assert scan(f"901234567{gap}сум{tail}") == []


@pytest.mark.parametrize(
    "text",
    [
        "300 000 000 сумов",
        "300 000 000 рублей",
        "300 000 000 рублях",
        "300 000 000 долларов",
        "300 000 000 тысяч",
        "300 000 000 so'mga",
        "300 000 000 so'mlik",
        "300 000 000 mingta",
        "300 000 000 сўмга",
        "300 000 000 сум.",
        "300 000 000 сум, оплата завтра",
        "300 000 000 USD.",
    ],
)
def test_an_inflected_currency_word_still_makes_a_price(text):
    assert scan(text) == []


@pytest.mark.parametrize(
    "link",
    [
        "bit.ly/abc",
        "goo.gl/abc",
        "t.co/abc",
        "tg://resolve?domain=seller",
        "tg:resolve?domain=seller",
        "my.shop.kz/p",
        "kaspi.kz",
        "example.info",
        "ozon.shop/x",
        "linktr.ee/x",
        "185.12.34.56/pay",
        "185.12.34.56:8080/pay",
        "t . me/x",
        "t. me/x",
        "telegram . me/x",
        "ftp://files.example.com/a.zip",
        "example.xyz/pay",
        "shop.example.online",
        "магазин.рф",
        "example.com:8080/pay",
    ],
)
def test_links_in_any_form_are_found(link):
    assert scan(f"Смотрите {link} сегодня") == [Finding("url", link)]


# The review's list, and a few more of the region's.
DOMAIN_ENDINGS = (
    "kz kg tj co info shop store online site xyz dev cc ly gl to tv link ee ua by su biz рф"
)


@pytest.mark.parametrize("tld", DOMAIN_ENDINGS.split())
def test_common_domain_endings_are_links(tld):
    assert scan(f"Заходите на magazin.{tld}.") == [Finding("url", f"magazin.{tld}")]


@pytest.mark.parametrize(
    "text",
    [
        "Спасибо. Хорошо",
        "Спасибо.Хорошо",
        "т.е. 5 шт",
        "и т.д.",
        "и т.п. и т.ч.",
        "Цена 1.5 млн сум, доставка 10.11.2026 к 9.00",
        "г. Ташкент, ул. Навои 15, кв. 7",
        "Rahmat. Yaxshi, ertaga yozaman.",
        "e.g. 5 pcs, i.e. two",
        "Ok.Thanks",
        "в 2.5 раза дешевле, 1.2 кг/день",
        "руб./шт., тыс.руб/мес",
        "Версия 2.0.1 и 3.14",
    ],
)
def test_honest_text_has_no_links(text):
    assert scan(text) == []


# About 20 000 characters each: a regex that is quadratic on one of them needs seconds, a linear one
# milliseconds, and a regression fails the test instead of hanging the run for minutes.
HOSTILE = {
    "dotted labels": "a." * 10_000,
    "spaced digits": "1 " * 10_000,
    "spaced hyphens": "1 - " * 5_000,
    "t and spaces": "t" + " " * 20_000,
    "t t t": "t " * 10_000,
    "one long word": "a" * 20_000,
    "scheme and a long tail": "http://" + "a" * 20_000,
    "digits": "9" * 20_000,
    "an e-mail with a long domain": "a@" + "b." * 10_000,
    "e-mail local parts": "a." * 10_000 + "b",
    "hyphens": "-" * 20_000,
    "open brackets": "(" * 10_000 + "1",
    "addresses": "1.1.1.1/" * 2_500,
    "labels with hyphens": "a-" * 10_000 + ".",
}


@pytest.mark.parametrize("name", list(HOSTILE))
def test_scan_stays_fast_on_hostile_text(name):
    started = time.perf_counter()

    scan(HOSTILE[name], canaries=[CANARY], owner_secrets=[OWNER_PHONE, "Ташкент, Навои 15"])

    assert time.perf_counter() - started < 1.0
