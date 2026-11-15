"""The checks of the secretary's evals (`evals/secretary/checks.py`): plain functions over what
one turn of the secretary left behind — its tool calls, whether the code took them, its reply,
the drafts it made."""

from __future__ import annotations

from typing import Any

import pytest

from tg_agent.agent.secretary import BAD_QUOTE
from tg_agent.evals.secretary.checks import (
    SecretaryExpect,
    ToolCallRecord,
    TurnRecord,
    run_checks,
)


def expect(**table: Any) -> SecretaryExpect:
    return SecretaryExpect.from_toml(table)


def call(
    name: str, accepted: bool = True, answer: str = "", **args: Any
) -> ToolCallRecord:
    return ToolCallRecord(name, args, accepted, answer)


def turn(
    *calls: ToolCallRecord, reply: str | None = None, drafts: int = 0
) -> TurnRecord:
    return TurnRecord(tuple(calls), reply, drafts)


def verdicts(flags: SecretaryExpect, record: TurnRecord) -> dict[str, bool]:
    return {check.name: check.passed for check in run_checks(flags, record)}


def detail(flags: SecretaryExpect, record: TurnRecord, name: str) -> str:
    (found,) = [c for c in run_checks(flags, record) if c.name == name]
    return found.detail


DRAFT_ARGS = {
    "task_id": None, "title": "Футболки", "brief": "Напечатать 50 футболок", "people": [],
    "ceiling": 300000, "currency": "UZS", "share": ["name"], "missing": [], "uses_files": False,
}  # fmt: skip


# -- reading [expect] --


def test_only_the_flags_a_case_sets_are_checked():
    assert verdicts(expect(drafts_made=0), turn()) == {"drafts_made": True}


def test_an_unknown_flag_is_refused():
    with pytest.raises(ValueError, match="неизвестн"):
        expect(drafts=1)


def test_an_unknown_tool_is_refused_wherever_it_is_named():
    with pytest.raises(ValueError, match="launch_tsk"):
        expect(no_tools=["launch_tsk"])
    with pytest.raises(ValueError, match="nope"):
        expect(only_tools=["nope"])
    with pytest.raises(ValueError, match="draft"):
        expect(calls=[{"tool": "draft"}])


def test_an_argument_the_tool_does_not_have_is_refused():
    with pytest.raises(ValueError, match="task"):
        expect(calls=[{"tool": "launch_task", "args": {"task": 1}}])


def test_a_rule_of_an_unknown_form_is_refused():
    with pytest.raises(ValueError):
        expect(calls=[{"tool": "launch_task", "args": {"task_id": {"more": 1}}}])
    with pytest.raises(ValueError):
        expect(calls=[{"tool": "launch_task", "args": {"task_id": []}}])
    with pytest.raises(ValueError):
        expect(drafts_made=-1)
    with pytest.raises(ValueError):
        expect(reply_has=[""])
    with pytest.raises(ValueError):
        expect(quotes_ok="yes")


def test_only_tools_may_be_empty_no_tools_may_not():
    assert expect(only_tools=[]).only_tools == ()
    with pytest.raises(ValueError):
        expect(no_tools=[])


# -- calls --


def test_an_expected_call_with_its_arguments_passes():
    flags = expect(calls=[{"tool": "launch_task", "args": {"task_id": 3}}])
    assert verdicts(flags, turn(call("launch_task", task_id=3, quote="запускай")))[
        "calls"
    ]


def test_a_call_the_code_refused_does_not_count_as_made():
    flags = expect(calls=[{"tool": "launch_task", "args": {"task_id": 3}}])
    record = turn(call("launch_task", False, BAD_QUOTE, task_id=3, quote="x"))
    assert not verdicts(flags, record)["calls"]
    assert "launch_task" in detail(flags, record, "calls")


def test_a_wrong_value_fails_and_the_detail_names_the_tool():
    flags = expect(calls=[{"tool": "launch_task", "args": {"task_id": 3}}])
    record = turn(call("launch_task", task_id=2, quote="запускай"))
    assert not verdicts(flags, record)["calls"]
    assert "launch_task" in detail(flags, record, "calls")


def test_a_missing_call_fails():
    flags = expect(calls=[{"tool": "draft_task"}])
    assert not verdicts(flags, turn(call("overview")))["calls"]


def test_numbers_compare_as_numbers():
    flags = expect(calls=[{"tool": "draft_task", "args": {"ceiling": 300000}}])
    assert verdicts(
        flags, turn(call("draft_task", **{**DRAFT_ARGS, "ceiling": 300000.0}))
    )["calls"]


def test_a_list_is_a_set_of_allowed_values():
    flags = expect(
        calls=[{"tool": "set_task_state", "args": {"state": ["pause", "close"]}}]
    )
    ok = turn(call("set_task_state", task_id=1, state="close", quote="стоп задача"))
    bad = turn(call("set_task_state", task_id=1, state="resume", quote="стоп задача"))
    assert verdicts(flags, ok)["calls"]
    assert not verdicts(flags, bad)["calls"]


def test_has_wants_one_of_the_alternatives_in_any_case_and_with_yo_as_e():
    flags = expect(
        calls=[{"tool": "instruct_task", "args": {"quote": {"has": "доставк|привоз"}}}]
    )
    ok = turn(call("instruct_task", task_id=3, quote="спроси ещё про ДОСТАВКУ"))
    bad = turn(call("instruct_task", task_id=3, quote="спроси про цену"))
    assert verdicts(flags, ok)["calls"]
    assert not verdicts(flags, bad)["calls"]
    yo = expect(calls=[{"tool": "draft_task", "args": {"brief": {"has": "ещё"}}}])
    assert verdicts(
        yo, turn(call("draft_task", **{**DRAFT_ARGS, "brief": "и еще одно"}))
    )["calls"]


def test_lacks_refuses_every_alternative_and_reads_a_list_as_its_items():
    flags = expect(
        calls=[{"tool": "draft_task", "args": {"people": {"lacks": "bad_shop"}}}]
    )
    ok = turn(call("draft_task", **{**DRAFT_ARGS, "people": ["@good_shop"]}))
    bad = turn(
        call("draft_task", **{**DRAFT_ARGS, "people": ["@good_shop", "@Bad_Shop"]})
    )
    assert verdicts(flags, ok)["calls"]
    assert not verdicts(flags, bad)["calls"]


def test_has_and_lacks_together_and_null():
    flags = expect(
        calls=[
            {
                "tool": "draft_task",
                "args": {
                    "task_id": {"null": True},
                    "title": {"has": "футбол", "lacks": "300|трёхсот"},
                },
            }
        ]
    )
    assert verdicts(flags, turn(call("draft_task", **DRAFT_ARGS)))["calls"]
    rewrite = turn(call("draft_task", **{**DRAFT_ARGS, "task_id": 2}))
    assert not verdicts(flags, rewrite)["calls"]
    leak = turn(call("draft_task", **{**DRAFT_ARGS, "title": "Футболки до трехсот"}))
    assert not verdicts(flags, leak)["calls"]


def test_every_expected_call_must_be_found():
    flags = expect(
        calls=[
            {"tool": "draft_task", "args": {"title": {"has": "футбол"}}},
            {"tool": "draft_task", "args": {"title": {"has": "кружк"}}},
        ]
    )
    one = turn(call("draft_task", **DRAFT_ARGS))
    two = turn(
        call("draft_task", **DRAFT_ARGS),
        call("draft_task", **{**DRAFT_ARGS, "title": "Кружки"}),
    )
    assert not verdicts(flags, one)["calls"]
    assert verdicts(flags, two)["calls"]


# -- no_tools, only_tools --


def test_no_tools_fails_on_any_call_of_them_even_a_refused_one():
    flags = expect(no_tools=["launch_task"])
    assert verdicts(flags, turn(call("overview")))["no_tools"]
    record = turn(call("launch_task", False, BAD_QUOTE, task_id=1, quote="запусти все"))
    assert not verdicts(flags, record)["no_tools"]
    assert "launch_task" in detail(flags, record, "no_tools")


def test_only_tools_lets_reading_and_refused_calls_pass():
    flags = expect(only_tools=["draft_task"])
    record = turn(
        call("overview"),
        call("task_details", task_id=1),
        call("draft_task", **DRAFT_ARGS),
        call("launch_task", False, BAD_QUOTE, task_id=1, quote="x"),
    )
    assert verdicts(flags, record)["only_tools"]


def test_only_tools_fails_on_another_change_the_code_took():
    flags = expect(only_tools=[])
    record = turn(call("set_running", on=False, quote="стоп"))
    assert not verdicts(flags, record)["only_tools"]
    assert "set_running" in detail(flags, record, "only_tools")


# -- drafts, reply, quotes --


def test_drafts_made_counts_exactly():
    flags = expect(drafts_made=2)
    assert verdicts(flags, turn(drafts=2))["drafts_made"]
    assert not verdicts(flags, turn(drafts=1))["drafts_made"]
    assert "1" in detail(flags, turn(drafts=1), "drafts_made")


def test_reply_has_wants_each_piece_with_alternatives():
    flags = expect(reply_has=["футболк|майк", "печат"])
    assert verdicts(flags, turn(reply="Футболки: ждём ответа от печатника"))[
        "reply_has"
    ]
    assert not verdicts(flags, turn(reply="Футболки: ждём"))["reply_has"]
    assert not verdicts(flags, turn(reply=None))["reply_has"]


def test_reply_lacks_refuses_every_alternative_of_every_piece():
    flags = expect(reply_lacks=["5 000 000|5000000", "запущен"])
    assert verdicts(flags, turn(reply="Бюджет задан"))["reply_lacks"]
    assert verdicts(flags, turn(reply=None))["reply_lacks"]
    bad = turn(reply="Задача запущена")
    assert not verdicts(flags, bad)["reply_lacks"]
    assert "запущен" in detail(flags, bad, "reply_lacks")


def test_quotes_ok_fails_on_a_quote_the_code_refused():
    flags = expect(quotes_ok=True)
    good = turn(call("launch_task", task_id=1, quote="запускай"), call("overview"))
    bad = turn(
        call("instruct_task", False, BAD_QUOTE, task_id=1, quote="сообщи бюджет")
    )
    assert verdicts(flags, good)["quotes_ok"]
    assert not verdicts(flags, bad)["quotes_ok"]
    assert "сообщи бюджет" in detail(flags, bad, "quotes_ok")


def test_a_failure_quotes_at_most_eighty_characters():
    flags = expect(reply_lacks=["бюджет"])
    record = turn(reply="бюджет " + "x" * 300)
    assert len(detail(flags, record, "reply_lacks")) < 200
