"""The checks run on the model's draft before it is queued: the lexicons and `check_outgoing`."""

from __future__ import annotations

import time
from collections.abc import Iterable

import pytest

from tg_agent.models import ApprovalKind
from tg_agent.policy.postgates import (
    GateResult,
    check_outgoing,
    contains_ai_admission,
    contains_ai_denial,
    contains_commitment,
    extract_amounts,
)

SHARE_PHONE = ApprovalKind.SHARE_PHONE
SHARE_ADDRESS = ApprovalKind.SHARE_ADDRESS
CONFIRM_ORDER = ApprovalKind.CONFIRM_ORDER

SIGNATURE = "— ИИ-помощник Саида"
CANARY = "zq-canary-7f3a"
EM_DASH = "\N{EM DASH}"

# The lexicons as the spec writes them.
DENIALS = [
    "я не бот",
    "я не робот",
    "я человек",
    "я живой",
    "я не ии",
    "не искусственный интеллект",
    "это саид пишет",
    "пишет саид",
    "я саид",
    "men bot emasman",
    "men odamman",
    "мен бот эмасман",
    "мен одамман",
]
COMMITMENTS = [
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
]
APOSTROPHES = [
    "'",
    "\N{MODIFIER LETTER TURNED COMMA}",
    "\N{MODIFIER LETTER APOSTROPHE}",
    "\N{RIGHT SINGLE QUOTATION MARK}",
    "\N{LEFT SINGLE QUOTATION MARK}",
    "`",
]


def check(
    *messages: str,
    commitment: bool = False,
    granted: Iterable[ApprovalKind] = (),
    canaries: Iterable[str] = (),
    owner_secrets: Iterable[str] = (),
    ceiling: float | None = None,
    language: str | None = None,
) -> GateResult:
    return check_outgoing(
        list(messages),
        commitment=commitment,
        granted=set(granted),
        canaries=canaries,
        owner_secrets=owner_secrets,
        ceiling=ceiling,
        language=language,
    )


def test_uzbek_letters_in_a_reply_to_a_russian_writer_are_stopped() -> None:
    # Russian has no ў, қ, ғ, ҳ: «Раҳмат!» to someone who writes Russian is the wrong language.
    for reply in ("Раҳмат! Уточню, позже напишу", "Rahmat! Уточню, позже напишу"):
        assert check(reply, language="ru").reasons == ["not_russian"]
        assert check(reply).ok  # the language is not known yet
    assert check("Спасибо! Уточню, позже напишу, PETG 5 шт", language="ru").ok


def test_a_reply_to_an_uzbek_writer_is_uzbek_in_latin() -> None:
    # Саид's rule: Uzbek always in Latin, also to someone who writes in Cyrillic.
    for language in ("uz_latn", "uz_cyrl"):
        assert check("Спасибо! Уточню", language=language).reasons == [
            "not_uzbek_latin"
        ]
        assert check("Раҳмат, тушундим", language=language).reasons == [
            "not_uzbek_latin"
        ]
        assert check("Rahmat! Aniqlashtirib, keyinroq yozaman", language=language).ok
        assert check("5 ta korpus, 250 000 so'm", language=language).ok


# --- the lexicons ---------------------------------------------------------------------------


@pytest.mark.parametrize("phrase", DENIALS)
def test_blocks_ai_denial_ru_uz(phrase):
    # casefolded, in capitals, in the middle of a sentence, and in Markdown emphasis
    for text in (
        phrase,
        phrase.upper(),
        f"Нет-нет, {phrase.capitalize()}, не переживайте!",
        f"__{phrase}__",
    ):
        assert contains_ai_denial(text), text
        result = check(text)
        assert result == GateResult(ok=False, reasons=["ai_denial"]), text


@pytest.mark.parametrize(
    "text",
    [
        "Я ИИ-помощник Саида",
        SIGNATURE,
        "Здравствуйте, Лидия Саидовна!",  # "я саид" inside a name is not "я Саид"
        "Я не знаю точной цены",
        "Он человек надёжный",
    ],
)
def test_honest_text_is_not_an_ai_denial(text):
    assert not contains_ai_denial(text)


@pytest.mark.parametrize(
    "text",
    [
        "Я ИИ",
        "я ИИ-помощник Саида",
        SIGNATURE,
        "ИИ.",
        "(ИИ)",
        "_ИИ_",
        "Я искусственный интеллект",
        "Искусственного интеллекта",
        "Я нейросеть",
        "Я работаю на нейросети",
        "Men sun'iy intellektman",
        "Men sun'iy intellekt asosidagi yordamchiman",
        "Мен сунъий интеллектман",
        "I am an AI",
        "AI-assistant",
    ],
)
def test_ai_admission_is_recognised(text):
    assert contains_ai_admission(text)


@pytest.mark.parametrize(
    "text",
    [
        "Проверим линии связи",
        "Мии",
        "Привет, Мии!",
        "Said and Aisha",  # "ai" is a whole word or nothing
        "main street, again",
        "Я помощник Саида. Чем могу помочь?",
        "неискусственный мех",
    ],
)
def test_ai_admission_needs_whole_words(text):
    assert not contains_ai_admission(text)


@pytest.mark.parametrize("apostrophe", APOSTROPHES)
def test_uzbek_apostrophes_are_one_spelling(apostrophe):
    assert contains_commitment(f"Ha, to{apostrophe}layman")
    assert contains_commitment(f"O{apostrophe}tkazaman ertaga")
    assert contains_ai_admission(f"Men sun{apostrophe}iy intellektman")


@pytest.mark.parametrize("phrase", COMMITMENTS)
def test_every_listed_form_is_a_commitment(phrase):
    assert contains_commitment(f"Хорошо, {phrase}.")
    assert contains_commitment(phrase.upper())


def test_conditional_question_not_commitment():
    assert not contains_commitment("Если возьмём 10 штук, будет скидка?")


@pytest.mark.parametrize(
    "text",
    [
        "Если возьмём 10 штук, будет скидка?",
        "Если оплачу сегодня, будет скидка?",
        "Если оплачу сегодня — будет скидка?",  # a dash ends a clause like a comma
        "А если переведу предоплату сейчас, отправите сегодня?",
        "Если оформляем заказ сейчас, доставка будет завтра?",
        "Если мы договорились, оформляем?",
        "Agar bugun to'layman, chegirma bo'ladimi?",
        "Агар бугун тўлайман, чегирма бўладими?",
        "Договорились?",
        "Оформляем?",
        "",
    ],
)
def test_conditional_and_question_forms_are_not_commitments(text):
    assert not contains_commitment(text)


@pytest.mark.parametrize(
    "text",
    [
        "Договорились, оплачу завтра.",
        "Оплачу завтра, подойдёт?",  # a tag question does not undo the promise
        "Если вас всё устраивает, оформляйте.",  # the condition is the seller's; the order is firm
        "Если вас всё устраивает — оформляйте.",
        "Если вас всё устраивает - оформляйте.",
        "Подтверждаю заказ. Если что, пишите.",
        "Хорошо!\nОформляем.",
        "Можно ли скинуть цену? Оплачу сразу.",
    ],
)
def test_firm_forms_stay_commitments_next_to_conditions_and_questions(text):
    assert contains_commitment(text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("240 000 so'm", [240_000]),
        ("1,2 млн", [1_200_000]),
        ("300 тыс", [300_000]),
        ("300 ming", [300_000]),
        ("300 минг", [300_000]),
        ("300 тыс.", [300_000]),
        ("2 mln", [2_000_000]),
        ("1.5 млн", [1_500_000]),
        ("2,5 тыс", [2_500]),
        ("240к", [240_000]),
        ("240000", [240_000]),
        ("1 200 000 сум", [1_200_000]),
        ("240.000 сум", [240_000]),
        ("1,200,000", [1_200_000]),
        ("$50", [50]),
        ("2 шт. по 120 000 сум", [2, 120_000]),
        ("240 кг", [240]),
        ("без цифр", []),
        ("", []),
    ],
)
def test_extract_amounts(text, expected):
    amounts = extract_amounts(text)

    assert amounts == expected
    assert all(isinstance(amount, float) for amount in amounts)


# --- check_outgoing -------------------------------------------------------------------------


def test_a_clean_message_passes():
    text = f"Здравствуйте! Подскажите, пожалуйста, цену на чехол для iPhone 15?\n{SIGNATURE}"

    assert check(text) == GateResult(ok=True, reasons=[])


def test_the_helper_s_answer_passes_without_an_admission():
    # Ruling R82: to «вы бот?» the agent answers «помощник Саида»; no gate demands more.
    assert check("Я помощник Саида. Чем могу помочь?") == GateResult(ok=True, reasons=[])
    assert check("Да, я ИИ-помощник Саида. Чем могу помочь?").ok


def test_commitment_without_approval_escalates_confirm_order():
    expected = GateResult(ok=False, reasons=["commitment"], escalate=CONFIRM_ORDER)

    assert check("Хорошо, оформляйте заказ.") == expected
    # the model's own flag counts even when the text sounds innocent
    assert check("Отлично, ждём.", commitment=True) == expected
    # an approval of another kind does not cover an order
    assert check("Хорошо, оформляйте заказ.", granted=[SHARE_PHONE]) == expected


def test_the_gates_take_plain_texts_and_the_declared_commitment():
    # The new core has no decision form: it hands over its texts and what the writer declared.
    texts = ["Отлично, ждём.", f"До связи!\n{SIGNATURE}"]
    assert not any(contains_commitment(text) for text in texts)

    def gate(commitment: bool) -> GateResult:
        return check_outgoing(
            texts,
            commitment=commitment,
            granted=set(),
            canaries=(),
            owner_secrets=(),
            ceiling=None,
        )

    assert gate(True) == GateResult(
        ok=False, reasons=["commitment"], escalate=CONFIRM_ORDER
    )
    assert gate(False) == GateResult(ok=True, reasons=[])


def test_commitment_allowed_after_approval_under_ceiling():
    granted = [CONFIRM_ORDER]
    ok = GateResult(ok=True, reasons=[])

    assert (
        check("Подтверждаю, оплачу 240 000 сум.", granted=granted, ceiling=300_000)
        == ok
    )
    # exactly at the ceiling is still allowed
    assert (
        check("Подтверждаю, оплачу 300 000 сум.", granted=granted, ceiling=300_000)
        == ok
    )
    # no ceiling, no limit
    assert (
        check("Подтверждаю, оплачу 90 000 000 сум.", granted=granted, ceiling=None)
        == ok
    )
    # the flag alone, with the approval and no amount in the text
    assert (
        check("Отлично, ждём.", commitment=True, granted=granted, ceiling=300_000) == ok
    )


def test_over_ceiling_escalates():
    granted = [CONFIRM_ORDER]
    expected = GateResult(ok=False, reasons=["over_ceiling"], escalate=CONFIRM_ORDER)

    assert (
        check("Подтверждаю, оплачу 240 000 сум.", granted=granted, ceiling=200_000)
        == expected
    )
    # scale words count: 1,2 млн is 1 200 000
    assert (
        check("Договорились на 1,2 млн.", granted=granted, ceiling=1_000_000)
        == expected
    )
    # the largest amount decides
    text = "Оформляем 2 штуки по 90 000 сум, итого 180 000 сум."
    assert check(text, granted=granted, ceiling=150_000) == expected
    # without a commitment a high price is only being talked about
    talk = "Вы просите 240 000 сум? Это дороговато для нас."
    assert check(talk, granted=granted, ceiling=200_000).ok


def test_placeholder_needs_share_approval():
    assert check("Мой номер: {PHONE}") == GateResult(
        ok=False, reasons=["placeholder_phone"], escalate=SHARE_PHONE
    )
    assert check("Мой номер: {PHONE}", granted=[SHARE_PHONE]).ok

    assert check("Адрес: {ADDRESS}") == GateResult(
        ok=False, reasons=["placeholder_address"], escalate=SHARE_ADDRESS
    )
    assert check("Адрес: {ADDRESS}", granted=[SHARE_ADDRESS]).ok

    # the approval has to be of the right kind
    assert not check("Мой номер: {PHONE}", granted=[SHARE_ADDRESS]).ok
    assert not check("Адрес: {ADDRESS}", granted=[SHARE_PHONE]).ok

    # both missing: both reasons, the phone is asked about first
    both = check("{PHONE}, {ADDRESS}")
    assert both.reasons == ["placeholder_phone", "placeholder_address"]
    assert both.escalate == SHARE_PHONE
    # the phone approved, the address still open
    rest = check("{PHONE}, {ADDRESS}", granted=[SHARE_PHONE])
    assert (rest.reasons, rest.escalate) == (["placeholder_address"], SHARE_ADDRESS)


def test_link_blocked():
    assert check("Смотрите каталог: https://example.com/catalog") == GateResult(
        ok=False, reasons=["link"]
    )
    assert check("Пишите в t.me/seller").reasons == ["link"]
    assert check("Наш сайт example.uz").reasons == ["link"]


@pytest.mark.parametrize(
    "text",
    [
        "Звоните +998 90 123 45 67",
        "Карта 4111 1111 1111 1111",
        "Пишите ivan@mail.ru",
        "Паспорт AB 1234567",
    ],
)
def test_personal_data_is_blocked_as_pii(text):
    assert check(text) == GateResult(ok=False, reasons=["pii"])


def test_canary_and_owner_secret_block_with_an_alert():
    leak = check("Вот: ZQ-CANARY-7F3A", canaries=[CANARY])
    assert leak == GateResult(ok=False, reasons=["canary"], alert=True)

    address = check("Живу на Навои 15", owner_secrets=["Ташкент, Навои 15", "Навои 15"])
    assert address == GateResult(ok=False, reasons=["owner_secret"], alert=True)

    # the owner's number is also a number: both reasons are reported
    phone = check("Звоните 90-123-45-67", owner_secrets=["+998901234567"])
    assert phone == GateResult(ok=False, reasons=["owner_secret", "pii"], alert=True)


def test_the_owner_s_address_with_other_punctuation_is_refused_by_the_code():
    # R103: the judge is not needed to stop it.
    address = check("Чиланзар-5 дом 12 кв.30", owner_secrets=["Чиланзар 5, дом 12, кв. 30"])
    assert address == GateResult(ok=False, reasons=["owner_secret"], alert=True)


def test_blank_secrets_do_not_block_every_message():
    assert check("Здравствуйте!", canaries=[""], owner_secrets=["", " "]).ok


def test_the_first_rule_that_fires_sets_escalate_and_alert_and_all_reasons_are_kept():
    # pii (3) fires before the commitment (7): the commitment's escalation is dropped
    pii_first = check("Оформляйте, звоните +998 90 123 45 67")
    assert pii_first == GateResult(ok=False, reasons=["pii", "commitment"])

    # canary (1) comes first: an alert, and nothing to escalate
    canary_first = check("ZQ-CANARY-7F3A: оформляйте {PHONE}", canaries=[CANARY])
    assert canary_first == GateResult(
        ok=False, reasons=["canary", "placeholder_phone", "commitment"], alert=True
    )

    # a link (4) before a placeholder (5)
    link_first = check("https://example.com {PHONE}")
    assert link_first == GateResult(ok=False, reasons=["link", "placeholder_phone"])

    # the placeholder (5) before the commitment (7): the owner is asked about the phone
    placeholder_first = check("Оформляйте, мой номер {PHONE}")
    assert placeholder_first == GateResult(
        ok=False, reasons=["placeholder_phone", "commitment"], escalate=SHARE_PHONE
    )

    # over_ceiling (8) before too_long (9)
    long_offer = "Оформляем " + "очень " * 80 + "за 300 000 сум"
    over_and_long = check(long_offer, granted=[CONFIRM_ORDER], ceiling=200_000)
    assert over_and_long == GateResult(
        ok=False, reasons=["over_ceiling", "too_long"], escalate=CONFIRM_ORDER
    )

    # a denial (6) alone: no rule asks for an admission any more (ruling R82)
    denial = check("Я человек, не бот")
    assert denial == GateResult(ok=False, reasons=["ai_denial"])


def test_too_long():
    assert check("а", "б", "в").ok  # three messages are fine
    assert check("а", "б", "в", "г") == GateResult(ok=False, reasons=["too_long"])
    assert check("а" * 400).ok
    assert check("а" * 401) == GateResult(ok=False, reasons=["too_long"])
    assert check("ок", "б" * 401).reasons == ["too_long"]


# --- fix round 1: the review's bypasses, and the honest texts that must keep passing -------------

# Important 1: an AI word with a negation next to it says the opposite of an admission.
NEGATED_ADMISSIONS = [
    "Нет, я не нейросеть",
    "Нет, это не ИИ",
    "Это не ИИ, это живой человек пишет",
    "Не нейросеть",
    "Нет, ИИ тут нет",
    "Я не искусственного интеллекта",
    "Men sun'iy intellekt emasman",
    "Мен сунъий интеллект эмасман",
]
HONEST_ADMISSIONS = [
    "Нет, я ИИ-помощник Саида",
    "Я не человек, я ИИ",
    "Я не Саид, а его ИИ-помощник",
    "Я не просто ИИ, а помощник Саида",
    "Я не только нейросеть, но и помощник",
    "Да, я ИИ-помощник Саида, а не сам Саид",
    "Я ИИ, а не человек",
    "Не волнуйтесь, я ИИ-помощник Саида",
    "Нет проблем: я ИИ-помощник",
    "Я ИИ-помощник Саида нет проблем",  # «нет» is the third word after: too far to negate
    "Не знаю как сказать но я ИИ",  # «не» is the third word before: too far to negate
    "Men sun'iy intellektman, odam emasman",
    "Мен сунъий интеллектман, одам эмасман",
]

# Important 2: a pause inside the phrase, and the owner impersonated in other words.
REVIEWED_DENIALS = [
    f"Я {EM_DASH} человек",
    "Я - человек",
    "Я, человек, не бот",
    f"Я {EM_DASH} живой человек",
    f"Я {EM_DASH} не бот",
    f"Я {EM_DASH} не робот",
    "Это Саид",
    "Здравствуйте, это Саид",
    "С вами Саид",
    "Говорит Саид",
    "Меня зовут Саид",
    f"Я {EM_DASH} Саид",
    "Да я обычный человек",
    "Я реальный человек",
    "Нет, не бот",
    "Не бот я",
    "Men insonman",
    "Men robot emasman",
    "Мен инсонман",
]
NEW_DENIALS = [
    "это саид",
    "с вами саид",
    "говорит саид",
    "меня зовут саид",
    "обычный человек",
    "реальный человек",
    "не бот",
    "men insonman",
    "мен инсонман",
    "robot emasman",
    "робот эмасман",
    # the same without the pronoun, which Uzbek drops
    "insonman",
    "инсонман",
    "odamman",
    "одамман",
    "bot emasman",
    "бот эмасман",
]


@pytest.mark.parametrize("text", NEGATED_ADMISSIONS)
def test_a_negated_admission_is_a_denial_not_an_admission(text):
    assert not contains_ai_admission(text)
    assert contains_ai_denial(text)
    assert check(text) == GateResult(ok=False, reasons=["ai_denial"])


@pytest.mark.parametrize("text", HONEST_ADMISSIONS)
def test_honest_admissions_survive_the_negation_rule(text):
    assert contains_ai_admission(text)
    assert not contains_ai_denial(text)
    assert check(text) == GateResult(ok=True, reasons=[])


@pytest.mark.parametrize("text", REVIEWED_DENIALS)
def test_a_pause_inside_the_phrase_does_not_hide_a_denial(text):
    assert contains_ai_denial(text)
    assert check(text) == GateResult(ok=False, reasons=["ai_denial"])


@pytest.mark.parametrize("separator", [", ", " - ", f" {EM_DASH} ", "... ", ": ", "; "])
def test_every_listed_phrase_is_still_a_denial_with_a_pause_in_it(separator):
    for phrase in DENIALS:
        assert contains_ai_denial(separator.join(phrase.split(" "))), phrase


@pytest.mark.parametrize("phrase", NEW_DENIALS)
def test_the_new_denial_phrases(phrase):
    # casefolded, in capitals, in the middle of a sentence, and in Markdown emphasis
    for text in (
        phrase,
        phrase.upper(),
        f"Нет-нет, {phrase.capitalize()}, не переживайте!",
        f"__{phrase}__",
    ):
        assert contains_ai_denial(text), text
        assert check(text) == GateResult(ok=False, reasons=["ai_denial"]), text


@pytest.mark.parametrize("text", ["Бот не я", "бот не", "Бот? Не-а"])
def test_bot_followed_by_ne_is_a_denial(text):
    assert contains_ai_denial(text)


@pytest.mark.parametrize(
    "text",
    [
        "Я бот, не человек",
        f"Я бот {EM_DASH} не человек",
        "Я бот - не человек",
        "Я бот не человек",
        "Да, я бот. Не волнуйтесь!",
        "Я бот; не переживайте",
    ],
)
def test_an_honest_bot_then_ne_is_not_a_denial(text):
    assert not contains_ai_denial(text)


@pytest.mark.parametrize(
    "text",
    [
        "Это Саиду нужно",
        "Это Саида просили уточнить цену",
        "С вами Саидов",
        "Вам пишет Саидова",
        "Говорит Саидова",
        "Меня зовут Саидов",
    ],
)
def test_an_entry_that_ends_with_a_name_matches_the_whole_word(text):
    assert not contains_ai_denial(text)


@pytest.mark.parametrize(
    "text",
    [
        "Нужны не ботинки, а кроссовки",
        "Это не ботинки",
        "Я не ботаник",
        "Я не роботизированная система",
        "Обычный человеческий фактор",
        "Реальный человечек из мультфильма",
    ],
)
def test_whole_word_entries_do_not_fire_inside_longer_words(text):
    assert not contains_ai_denial(text)


@pytest.mark.parametrize(
    "text",
    [
        "Я ИИ, а не реальный человек",
        "Я не реальный человек",
        "Я не обычный человек, я ИИ-помощник",
        "Вам не пишет Саид, а ИИ-помощник",
        "Я ИИ, не реальный человек",
        "Нет, не пишет Саид",  # an earlier «Нет,» is not the «не» that counts
    ],
)
def test_an_entry_the_writer_negates_is_the_opposite_of_a_denial(text):
    assert not contains_ai_denial(text)


# --- fix round 2: «не» before a phrase must not hide a denial ------------------------------------
# The «не» exemption belongs to the three entries that can be negated («реальный человек»,
# «обычный человек», «пишет саид»), and only when nothing but a space stands between «не» and the
# entry. A «Не,» that opens the sentence, or a «не» from an earlier clause, says nothing about it.
NE_FAMILY = [
    "Не, я человек",
    "Да не, я человек",
    "Не-не, я человек",
    "Не, я живой",
    "Не, я не бот",
    "Не, я не робот",
    "Не, я Саид",
    "Не, это Саид",
    "Не, пишет Саид",
    "Не, men odamman",
    "Не, men bot emasman",
    "цена) не Я живой",
    "не. Мен бот эмасман",
    # the entries that may be negated, after a pause
    "Не, реальный человек",
    "Не, обычный человек",
    "Не, я обычный человек",
    "Не; пишет Саид",
    # «не» right before an entry that cannot be negated
    "Не я человек",
    "Не я Саид",
    "Не это Саид",
    "Не men odamman",
]


@pytest.mark.parametrize("text", NE_FAMILY)
def test_a_ne_before_the_phrase_does_not_hide_a_denial(text):
    assert contains_ai_denial(text)
    assert check(text) == GateResult(ok=False, reasons=["ai_denial"])


@pytest.mark.parametrize("text", NE_FAMILY)
def test_a_denial_after_ne_is_not_hidden_by_the_signature_line(text):
    # The signature counts as an admission, so with it in the second message only the denial
    # rule can stop the reply.
    assert contains_ai_admission(SIGNATURE)
    result = check(text, SIGNATURE)
    assert result == GateResult(ok=False, reasons=["ai_denial"])


def test_a_ne_on_another_line_or_in_the_previous_message_does_not_count():
    # «не» keeps its meaning inside one clause: a line break parts clauses like a comma does
    denial = GateResult(ok=False, reasons=["ai_denial"])
    assert check("Я ИИ, а не\nреальный человек") == denial
    assert check("Я ИИ, а не", "реальный человек") == denial
    # while spaces, however many, do not
    assert check("Я ИИ, а не   реальный человек") == GateResult(ok=True, reasons=[])


@pytest.mark.parametrize(
    "prefix",
    [
        "Не, ",
        "Да не, ",
        "Не-не, ",
        "Не. ",
        f"Не {EM_DASH} ",
        "Не: ",
        "(не) ",
        "Не... ",
        "Нет, не, ",
        "не\n",
    ],
)
def test_every_phrase_is_a_denial_behind_a_ne_and_a_pause(prefix):
    for phrase in [*DENIALS, *NEW_DENIALS]:
        assert contains_ai_denial(prefix + phrase), prefix + phrase


@pytest.mark.parametrize("prefix", ["не ", "Да не ", "цена) не "])
def test_a_bare_ne_exempts_only_the_three_entries_that_can_be_negated(prefix):
    # nothing but a space between «не» and the phrase; whatever comes before the «не» is no matter
    can_be_negated = {"пишет саид", "обычный человек", "реальный человек"}

    for phrase in [*DENIALS, *NEW_DENIALS]:
        assert contains_ai_denial(prefix + phrase) == (phrase not in can_be_negated), (
            phrase
        )


# About 20 000 characters each (see the same table in test_dlp.py).
HOSTILE = {
    "ai words": "ии " * 7_000,
    "negations": "не " * 7_000,
    "pauses": "я - " * 5_000,
    "commitments": "оплачу, " * 2_500,
    "digits": "1 " * 10_000,
    "amounts": "1,2 млн " * 2_500,
    "one long word": "a" * 20_000,
    "punctuation": ", " * 10_000,
    "dashes": "-" * 20_000,
    "bot and spaces": "бот" + " " * 40_000 + "x",
    "negated entries": "не реальный человек " * 1_000,
}


@pytest.mark.parametrize("name", list(HOSTILE))
def test_the_text_checks_stay_fast_on_hostile_text(name):
    text = HOSTILE[name]
    started = time.perf_counter()

    contains_ai_denial(text)
    contains_ai_admission(text)
    contains_commitment(text)
    extract_amounts(text)
    check(text)

    assert time.perf_counter() - started < 1.0


# -- ruling R93: never «нет» to «ты бот?» --


@pytest.mark.parametrize(
    "text",
    [
        "Нет, я помощник Саида",
        "Нет) помощник Саида",
        "Нет, что вы. Помощник Саида, подскажите цену",
        "— Нет! Я помощник Саида",
        "НЕТ, помощник Саида",
        "Yo'q, men Saidning yordamchisiman",
        "Yo‘q, Saidning yordamchisi",
    ],
)
def test_a_bare_no_then_helper_is_a_denial(text):
    assert contains_ai_denial(text), text
    assert check(text) == GateResult(ok=False, reasons=["ai_denial"]), text


@pytest.mark.parametrize(
    "text",
    [
        "Помощник Саида. Подскажите цену?",
        "Нет проблем, помощник Саида всё передаст",
        "Нет, я ИИ-помощник Саида",  # it says what it is
        "Цены нет? Я помощник Саида",
        "Нет, спасибо, не нужно",
    ],
)
def test_no_without_a_bare_no_opening_or_with_an_admission_passes(text):
    assert not contains_ai_denial(text), text
