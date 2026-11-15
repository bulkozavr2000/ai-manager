"""The checks of stage 6 (ruling R80): the tools the agent used, its reminders, what the owner
saw. Hand-made contexts, as in `test_eval_checks.py`."""

from __future__ import annotations

from datetime import timedelta

import pytest
from test_eval_checks import FIRST, START, agent, make_ctx, vendor

from tg_agent.evals.checks import (
    CHECKS,
    TASK_LEVEL,
    Expect,
    Speaker,
    Turn,
    owner_saw_has,
    reminders_at_most,
    tool_not_used,
    tool_used,
)


def hours(n: float, text: str, speaker: Speaker = Speaker.AGENT) -> Turn:
    return Turn(speaker, text, START + timedelta(hours=n))


def calls(*names: str) -> list[dict[str, object]]:
    return [
        {"name": name, "input": {}, "answer": "ok", "error": False} for name in names
    ]


# --- reading the flags ------------------------------------------------------------------


def test_the_new_flags_are_read_from_the_scenario() -> None:
    expect = Expect.from_toml(
        {
            "tool_used": ["remind_me"],
            "tool_not_used": ["ask_owner", "finish"],
            "reminders_at_most": 2,
            "owner_saw_has": ["спрашивает по задаче|ответить"],
        }
    )
    assert expect.tool_used == ("remind_me",)
    assert expect.tool_not_used == ("ask_owner", "finish")
    assert expect.reminders_at_most == 2
    assert expect.owner_saw_has == ("спрашивает по задаче|ответить",)


def test_no_reminder_at_all_is_a_wish_too() -> None:
    expect = Expect.from_toml({"reminders_at_most": 0})
    assert expect.reminders_at_most == 0 and expect.wants("reminders_at_most")


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("tool_used", "remind_me"),
        ("tool_not_used", []),
        ("reminders_at_most", -1),
        ("reminders_at_most", True),
        ("owner_saw_has", [""]),
    ],
)
def test_a_wrong_value_of_a_new_flag_is_refused(flag: str, value: object) -> None:
    with pytest.raises(ValueError, match=flag):
        Expect.from_toml({flag: value})


def test_the_task_level_checks_run_once_and_the_chat_ones_in_each_chat() -> None:
    assert {"tool_used", "tool_not_used", "owner_saw_has"} <= TASK_LEVEL
    assert "reminders_at_most" not in TASK_LEVEL
    assert {"tool_used", "tool_not_used", "reminders_at_most"} <= set(CHECKS)


# --- tool_used / tool_not_used ----------------------------------------------------------


def test_tool_used_passes_when_every_named_tool_was_called() -> None:
    ctx = make_ctx(
        expect=Expect(tool_used=("remind_me", "finish")),
        tool_calls=calls("send_message", "remind_me", "finish"),
    )
    assert tool_used(ctx).passed


def test_tool_used_fails_on_a_tool_that_was_never_called() -> None:
    ctx = make_ctx(
        expect=Expect(tool_used=("remind_me",)), tool_calls=calls("send_message")
    )
    result = tool_used(ctx)
    assert not result.passed and "remind_me" in result.detail


def test_a_refused_call_still_counts_as_used() -> None:
    # The model reached for the tool: that is what the scenario asks about.
    refused = [{"name": "remind_me", "input": {}, "answer": "нет", "error": True}]
    ctx = make_ctx(expect=Expect(tool_used=("remind_me",)), tool_calls=refused)
    assert tool_used(ctx).passed


def test_tool_not_used_passes_when_none_was_called() -> None:
    ctx = make_ctx(
        expect=Expect(tool_not_used=("ask_owner",)),
        tool_calls=calls("send_message", "finish"),
    )
    assert tool_not_used(ctx).passed


def test_tool_not_used_fails_on_a_call_of_it() -> None:
    ctx = make_ctx(
        expect=Expect(tool_not_used=("ask_owner", "finish")),
        tool_calls=calls("send_message", "ask_owner"),
    )
    result = tool_not_used(ctx)
    assert not result.passed and "ask_owner" in result.detail


# --- reminders_at_most ------------------------------------------------------------------


def test_two_reminders_after_a_silence_pass_a_limit_of_two() -> None:
    chat = [hours(0, FIRST), hours(24, "Напоминаю"), hours(48, "Ещё раз напоминаю")]
    assert reminders_at_most(
        make_ctx(expect=Expect(reminders_at_most=2), chat=chat)
    ).passed


def test_a_third_reminder_fails_a_limit_of_two() -> None:
    chat = [
        hours(0, FIRST),
        hours(24, "Напоминаю"),
        hours(48, "Ещё раз"),
        hours(72, "И ещё раз"),
    ]
    result = reminders_at_most(make_ctx(expect=Expect(reminders_at_most=2), chat=chat))
    assert not result.passed
    assert "3" in result.detail and "И ещё раз" in result.detail


def test_a_message_after_his_answer_is_no_reminder() -> None:
    chat = [
        hours(0, FIRST),
        hours(1, "Отправлю в пятницу", Speaker.COUNTERPART),
        hours(1.1, "Хорошо, спасибо"),
        hours(73, "Отправили документы?"),  # he has not answered since «Хорошо»: one
    ]
    assert reminders_at_most(
        make_ctx(expect=Expect(reminders_at_most=1), chat=chat)
    ).passed
    assert not reminders_at_most(
        make_ctx(expect=Expect(reminders_at_most=0), chat=chat)
    ).passed


def test_messages_of_one_turn_are_no_reminders() -> None:
    # «Уточню и вернусь», then the phone the owner allowed minutes later: the same talk.
    chat = [
        agent(FIRST),
        vendor("Оставьте номер", 2),
        agent("Уточню и вернусь", 4),
        agent("Контактный номер: +998 99 111 22 33", 9),
    ]
    assert reminders_at_most(
        make_ctx(expect=Expect(reminders_at_most=0), chat=chat)
    ).passed


# --- owner_saw_has ----------------------------------------------------------------------


def test_owner_saw_has_finds_each_text_among_the_cards_and_notices() -> None:
    saw = [
        "❓ @ali_print спрашивает по задаче #1 «Печать»: «Кто это?»",
        "Ответить ему?",
    ]
    ctx = make_ctx(
        expect=Expect(owner_saw_has=("спрашивает по задаче", "ОТВЕТИТЬ|reply")),
        owner_saw=saw,
    )
    assert owner_saw_has(ctx).passed


def test_owner_saw_has_fails_on_a_text_he_never_saw() -> None:
    ctx = make_ctx(
        expect=Expect(owner_saw_has=("спрашивает по задаче|asks",)),
        owner_saw=["✅ Готово — Печать"],
    )
    result = owner_saw_has(ctx)
    assert not result.passed and "спрашивает по задаче|asks" in result.detail
