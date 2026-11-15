"""Contacts the search may hand in: a Telegram username or a phone, and proof it was seen."""

import pytest

from tg_agent.search.contacts import normalize_contact, normalize_phone, seen_in, show


@pytest.mark.parametrize(
    ("raw", "phone"),
    [
        ("+998 90 123-45-67", "+998901234567"),
        ("998901234567", "+998901234567"),
        ("90 123 45 67", "+998901234567"),
        ("(93) 555-12-34", "+998935551234"),
        ("+7 916 123-45-67", "+79161234567"),
    ],
)
def test_a_phone_is_kept_in_one_international_form(raw: str, phone: str) -> None:
    assert normalize_phone(raw) == phone


@pytest.mark.parametrize(
    "raw", ["123", "2026-11-09", "90 123", "+998 90 123 45 6789 00", ""]
)
def test_what_is_not_a_phone_is_refused(raw: str) -> None:
    assert normalize_phone(raw) is None


@pytest.mark.parametrize(
    ("raw", "contact"),
    [
        ("@print_lab", "print_lab"),
        ("print_lab", "print_lab"),
        ("https://t.me/print_lab", "print_lab"),
        ("t.me/print_lab", "print_lab"),
        ("@3d_tash", "3d_tash"),
        ("+998 90 123 45 67", "+998901234567"),
        ("998901234567", "+998901234567"),  # digits only: a phone, never a name
    ],
)
def test_a_contact_is_a_username_or_a_phone(raw: str, contact: str) -> None:
    assert normalize_contact(raw) == contact


@pytest.mark.parametrize(
    "raw", ["@ab", "print lab", "https://t.me/joinchat/abc", "olx.uz", ""]
)
def test_anything_else_is_no_contact(raw: str) -> None:
    assert normalize_contact(raw) is None


def test_a_phone_counts_as_seen_however_the_page_wrote_it() -> None:
    page = "Звоните: +998 (90) 123-45-67, Абдулхамид"
    assert seen_in("+998901234567", page)
    assert not seen_in("+998901234568", page)


def test_a_username_counts_as_seen_only_as_a_whole_name() -> None:
    page = "Пишите в Telegram @print_lab или t.me/printshop_uz"
    assert seen_in("print_lab", page)
    assert seen_in("printshop_uz", page)
    assert not seen_in("print", page)  # part of another name
    assert not seen_in("lab_print", page)


def test_contacts_are_shown_the_way_people_write_them() -> None:
    assert show("print_lab") == "@print_lab"
    assert show("+998901234567") == "+998 90 123 45 67"
    assert show("+79161234567") == "+79161234567"
