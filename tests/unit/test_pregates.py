from __future__ import annotations

from tg_agent.policy.pregates import is_auto_reply


def test_auto_reply_offline_flag() -> None:
    assert is_auto_reply(text="Здравствуйте!", offline_flag=True, seconds_since_our_last=None,
                         previous_incoming=[]) is True  # fmt: skip


def test_auto_reply_instant_repeat() -> None:
    greeting = "Ассалому алайкум"
    assert is_auto_reply(text=greeting, offline_flag=False, seconds_since_our_last=2.0,
                         previous_incoming=[greeting]) is True  # fmt: skip
    assert is_auto_reply(text=greeting, offline_flag=False, seconds_since_our_last=30.0,
                         previous_incoming=[greeting]) is False  # fmt: skip


def test_auto_reply_repeated_long_text() -> None:
    away = "Здравствуйте! Мы сейчас не на месте и ответим вам в рабочее время с 9 до 18"
    assert is_auto_reply(text=away, offline_flag=False, seconds_since_our_last=None,
                         previous_incoming=[away]) is True  # fmt: skip


def test_short_repeated_ok_is_not_auto_reply() -> None:
    assert is_auto_reply(text="ок", offline_flag=False, seconds_since_our_last=600.0,
                         previous_incoming=["ок"]) is False  # fmt: skip


def test_new_long_text_is_not_auto_reply() -> None:
    text = "Здравствуйте! Пять корпусов из PETG будут стоить 240 000 сум, срок три дня"
    assert is_auto_reply(text=text, offline_flag=False, seconds_since_our_last=1.0,
                         previous_incoming=[]) is False  # fmt: skip
