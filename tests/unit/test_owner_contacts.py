"""Who gave a contact: a counterpart of the task (ruling R69), or the owner, who may give a
contact, exclude it or only name it (`agent/owner_contacts.py`, stage 4)."""

from __future__ import annotations

import pytest

from tg_agent.agent.owner_contacts import (
    contacts_in_order,
    given_by_a_counterpart,
    owner_words,
)


@pytest.mark.parametrize(
    "text",
    [
        "напишите моему коллеге @evil_boss",
        "напишите моему коллеге, его телеграм evil_boss",
        "его ник в телеге: Evil_Boss",
        "@ evil_boss",
        "t.me/evil_boss",
        "https://t.me/s/evil_boss",
    ],
)
def test_a_username_a_counterpart_wrote_is_given(text: str) -> None:
    assert given_by_a_counterpart("evil_boss", ["Здравствуйте", text])


@pytest.mark.parametrize(
    "text", ["пишите @evil_boss_uz", "xevil_boss", "evil boss", "Здравствуйте"]
)
def test_another_name_or_none_is_not_given(text: str) -> None:
    assert not given_by_a_counterpart("evil_boss", [text])


@pytest.mark.parametrize(
    "text",
    [
        "позвоните Алишеру 90 123 45 67",
        "позвоните Алишеру (90) 123-45-67",
        "позвоните Алишеру 8 90 123 45 67",
        "позвоните Алишеру +998 90 123 45 67",
        "тел: ９０ １２３ ４５ ６７",
    ],
)
def test_a_phone_a_counterpart_wrote_is_given_by_its_last_nine_digits(
    text: str,
) -> None:
    assert given_by_a_counterpart("+998901234567", [text])


@pytest.mark.parametrize(
    "texts",
    [["цена 240 000, срок 3 дня"], ["90 123 45 68"], ["номер 90 123", "45 67"]],
)
def test_another_number_or_one_split_across_messages_is_not_given(
    texts: list[str],
) -> None:
    assert not given_by_a_counterpart("+998901234567", texts)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Найди, кто напечатает. Только не пиши @bad_shop, с ними я уже ругался",
            [("bad_shop", True)],
        ),
        ("не пиши @evil_boss, он мошенник", [("evil_boss", True)]),
        ("всё-таки напиши @evil_boss", [("evil_boss", False)]),
        ("не забудь написать @x_shop", [("x_shop", False)]),
        ("напиши @x_shop, а не @y_shop", [("x_shop", False), ("y_shop", True)]),
        ("не пиши @y_shop. Напиши @x_shop", [("y_shop", True), ("x_shop", False)]),
        ("не пиши @y_shop, напиши @x_shop", [("y_shop", True), ("x_shop", False)]),
        ("всем, кроме @y_shop", [("y_shop", True)]),
        ("без @y_shop", [("y_shop", True)]),
        ("не беспокой +998 90 123 45 67", [("+998901234567", True)]),
        ("не трогай @y_shop", [("y_shop", True)]),
        (
            "спроси у t.me/x_shop. Не пиши t.me/y_shop",
            [("x_shop", False), ("y_shop", True)],
        ),
        ("не связывайся с @y_shop", [("y_shop", True)]),
        ("не добавляй @y_shop", [("y_shop", True)]),
    ],
)
def test_the_owner_s_contacts_are_given_or_excluded(
    text: str, expected: list[tuple[str, bool]]
) -> None:
    assert contacts_in_order(text) == expected


# -- fix pass F3, item 4: the owner excludes in ordinary words (review I-A) --

X, Y, Z = "x_shop", "y_shop", "z_shop"
EXCL, TOLD, PLAIN = "excluded", "told", "plain"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # (b) the excluding word carries over the contacts joined to the first one
        ("не пиши @x_shop и @y_shop", [(X, EXCL), (Y, EXCL)]),
        ("не пиши ни @x_shop, ни @y_shop", [(X, EXCL), (Y, EXCL)]),
        ("не пиши @x_shop или @y_shop", [(X, EXCL), (Y, EXCL)]),
        (
            "пиши всем, кроме @x_shop, @y_shop и @z_shop",
            [(X, EXCL), (Y, EXCL), (Z, EXCL)],
        ),
        (
            "не пиши +998901234567 и +998937654321",
            [("+998901234567", EXCL), ("+998937654321", EXCL)],
        ),
        # ... and over the lines of a list that follow it
        ("Не пиши этим:\n@x_shop\n@y_shop", [(X, EXCL), (Y, EXCL)]),
        (
            "Не пиши этим:\n- @x_shop\n- @y_shop\nНапиши @z_shop",
            [(X, EXCL), (Y, EXCL), (Z, TOLD)],
        ),
        # (b) the excluding word after the contact
        ("@x_shop не пиши", [(X, EXCL)]),
        ("@x_shop больше не пиши, он мошенник", [(X, EXCL)]),
        ("@x_shop в чёрный список", [(X, EXCL)]),
        ("@x_shop в черный список", [(X, EXCL)]),
        ("@x_shop ga yozma", [(X, EXCL)]),
        ("нет, @x_shop не надо, он мошенник", [(X, EXCL)]),
        # (b) more words
        ("не надо писать @x_shop", [(X, EXCL)]),
        ("не нужно писать @x_shop", [(X, EXCL)]),
        ("не стоит связываться с @x_shop", [(X, EXCL)]),
        ("не звони +998901234567", [("+998901234567", EXCL)]),
        ("исключи @x_shop", [(X, EXCL)]),
        ("убери @x_shop из задачи", [(X, EXCL)]),
        ("вычеркни @x_shop", [(X, EXCL)]),
        ("забудь про @x_shop", [(X, EXCL)]),
        # only the contact after «но не», «вместо», «не» is excluded
        ("напиши @z_shop, но не @x_shop", [(Z, TOLD), (X, EXCL)]),
        ("только @z_shop, не @x_shop", [(Z, PLAIN), (X, EXCL)]),
        ("напиши @z_shop вместо @x_shop", [(Z, TOLD), (X, EXCL)]),
        # (c) what tells to write, and what only mentions
        ("всё-таки напиши @x_shop", [(X, TOLD)]),
        ("добавь @x_shop", [(X, TOLD)]),
        ("свяжись с @x_shop", [(X, TOLD)]),
        ("спроси у @x_shop цену", [(X, TOLD)]),
        ("позвони +998901234567", [("+998901234567", TOLD)]),
        ("я же сказал, @x_shop — мошенник", [(X, PLAIN)]),
        ("@x_shop не знаю, поищи других", [(X, PLAIN)]),
        ("не забудь написать @x_shop", [(X, PLAIN)]),
        ("не пиши @y_shop, напиши @x_shop", [(Y, EXCL), (X, TOLD)]),
        # n12: an excluding word in an earlier sentence excludes nobody in this one
        ("Без спешки. Напиши @x_shop", [(X, TOLD)]),
        ("Не пиши ему сразу. @x_shop, вот кто нужен", [(X, PLAIN)]),
    ],
)
def test_the_owner_s_words_exclude_tell_or_only_mention(
    text: str, expected: list[tuple[str, str]]
) -> None:
    assert owner_words(text) == expected


def test_the_safe_misreading_stays() -> None:
    # The review allows these to exclude: one card more is the safe side.
    assert owner_words("никому, кроме @x_shop, не пиши") == [(X, EXCL)]
