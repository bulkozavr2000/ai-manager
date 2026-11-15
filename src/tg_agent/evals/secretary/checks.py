"""The checks of a secretary eval: plain code over what one turn of the secretary left behind.

A turn leaves a `TurnRecord`: every tool call the model made, with its arguments and whether the
code took it (a call the code refused changed nothing, but the model still tried it), the bot's
reply to the owner, and how many new drafts the turn made. A case's `[expect]` names what must
hold; each flag is one check, run only when the case sets it:

- `calls`: each listed call was made and taken by the code, with arguments that match (a value;
  a list of allowed values; or a table `{has = "…", lacks = "…", null = true|false}`);
- `no_tools`: none of these tools was called at all, taken or refused;
- `only_tools`: no tool that changes something was taken except these (an empty list: nothing
  changed; the tools that only read are always allowed);
- `drafts_made`: exactly this many new drafts;
- `reply_has` / `reply_lacks`: pieces of the reply, alternatives split by «|»;
- `quotes_ok`: no call was refused for its quote (ruling R59).

Text is compared case-blind with «ё» read as «е»: the owner's words and the model's spelling
of them differ in nothing else that matters here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from tg_agent.agent.secretary import BAD_QUOTE, SECRETARY_TOOLS
from tg_agent.evals.checks import CheckResult
from tg_agent.tasks.texts import quote as cut

# Every tool of the secretary and the names of its arguments.
TOOLS: dict[str, tuple[str, ...]] = {
    str(tool["name"]): tuple(tool["input_schema"]["properties"])
    for tool in SECRETARY_TOOLS
}
# The tools that only read: always allowed by `only_tools`.
READING = frozenset({"overview", "task_details", "find_person"})
_RULE_KEYS = ("has", "lacks", "null")


@dataclass(frozen=True)
class ArgRule:
    """One condition on an argument: `equals`, `one_of`, `has`, `lacks` or `null`."""

    op: str
    value: Any


@dataclass(frozen=True)
class CallRule:
    """A call that must be made: the tool, and the conditions on its arguments."""

    tool: str
    args: tuple[tuple[str, tuple[ArgRule, ...]], ...] = ()


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool call of the turn: its arguments, whether the code took it, what it answered."""

    name: str
    input: Mapping[str, Any]
    accepted: bool
    answer: str = ""


@dataclass(frozen=True)
class TurnRecord:
    """What one turn of the secretary left behind."""

    calls: tuple[ToolCallRecord, ...]
    reply: str | None
    drafts_made: int


@dataclass(frozen=True)
class SecretaryExpect:
    """The `[expect]` table of a case. A flag left out is not checked."""

    calls: tuple[CallRule, ...] | None = None
    no_tools: tuple[str, ...] | None = None
    only_tools: tuple[str, ...] | None = None
    drafts_made: int | None = None
    reply_has: tuple[str, ...] | None = None
    reply_lacks: tuple[str, ...] | None = None
    quotes_ok: bool = False

    @classmethod
    def from_toml(cls, table: Mapping[str, object]) -> SecretaryExpect:
        """Read the table; an unknown flag, tool or argument, or a value of the wrong kind,
        is a ValueError: a typo must not quietly switch a check off."""
        unknown = sorted(set(table) - {flag.name for flag in fields(cls)})
        if unknown:
            raise ValueError(f"в [expect] неизвестные флаги: {', '.join(unknown)}")
        values: dict[str, Any] = {}
        for name, value in table.items():
            if name == "calls":
                values[name] = _calls(value)
            elif name in ("no_tools", "only_tools"):
                values[name] = _tools(name, value, empty=name == "only_tools")
            elif name == "drafts_made":
                if type(value) is not int or value < 0:
                    raise ValueError("[expect] drafts_made: нужно целое число от нуля")
                values[name] = value
            elif name in ("reply_has", "reply_lacks"):
                values[name] = _texts(name, value)
            elif not isinstance(value, bool):
                raise ValueError(f"[expect] {name}: нужно true или false")
            else:
                values[name] = value
        return cls(**values)

    def wanted(self) -> list[str]:
        """The checks this case asks for, in the order they run."""
        return [
            flag.name
            for flag in fields(self)
            if getattr(self, flag.name) is not None
            and getattr(self, flag.name) is not False
        ]


def _texts(name: str, value: object) -> tuple[str, ...]:
    ok = (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and _alternatives(item) for item in value)
    )
    if not ok:
        raise ValueError(f"[expect] {name}: нужен список непустых строк")
    return tuple(value)  # type: ignore[arg-type]


def _tools(name: str, value: object, *, empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not empty):
        raise ValueError(f"[expect] {name}: нужен список инструментов")
    for tool in value:
        if tool not in TOOLS:
            raise ValueError(f"[expect] {name}: нет инструмента «{tool}»")
    return tuple(value)


def _calls(value: object) -> tuple[CallRule, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("[[expect.calls]]: нужен хотя бы один вызов")
    rules = []
    for number, table in enumerate(value, 1):
        where = f"[[expect.calls]] №{number}"
        if not isinstance(table, dict) or set(table) - {"tool", "args"}:
            raise ValueError(f"{where}: нужны только tool и args")
        tool = table.get("tool")
        if tool not in TOOLS:
            raise ValueError(f"{where}: нет инструмента «{tool}»")
        args = table.get("args", {})
        if not isinstance(args, dict):
            raise ValueError(f"{where} args: нужна таблица")  # noqa: TRY004 - read as a case error
        conditions = []
        for arg, raw in args.items():
            if arg not in TOOLS[tool]:
                raise ValueError(f"{where}: у {tool} нет аргумента «{arg}»")
            conditions.append((arg, _arg_rules(f"{where} {arg}", raw)))
        rules.append(CallRule(tool, tuple(conditions)))
    return tuple(rules)


def _scalar(value: object) -> bool:
    return isinstance(value, str | int | float | bool)


def _arg_rules(where: str, raw: object) -> tuple[ArgRule, ...]:
    """A value; a non-empty list of allowed values; or `{has, lacks, null}`."""
    if _scalar(raw):
        return (ArgRule("equals", raw),)
    if isinstance(raw, list) and raw and all(_scalar(item) for item in raw):
        return (ArgRule("one_of", tuple(raw)),)
    if isinstance(raw, dict) and raw and not set(raw) - set(_RULE_KEYS):
        rules = []
        for key, value in raw.items():
            if key == "null":
                if not isinstance(value, bool):
                    raise ValueError(f"{where}: null — true или false")
            elif not isinstance(value, str) or not _alternatives(value):
                raise ValueError(f"{where}: {key} — непустая строка")
            rules.append(ArgRule(key, value))
        return tuple(rules)
    raise ValueError(
        f"{where}: нужно значение, список значений или таблица {{has, lacks, null}}"
    )


# -- matching --


def normal(text: str) -> str:
    """Text as the checks compare it: case-blind, «ё» as «е», spaces collapsed."""
    return " ".join(text.casefold().replace("ё", "е").split())


def _alternatives(piece: str) -> list[str]:
    return [normal(part) for part in piece.split("|") if normal(part)]


def _has(text: str, piece: str) -> bool:
    found = normal(text)
    return any(part in found for part in _alternatives(piece))


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list | tuple):
        return "\n".join(str(item) for item in value)
    return str(value)


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _equal(value: Any, wanted: Any) -> bool:
    if _number(value) and _number(wanted):
        return float(value) == float(wanted)
    if isinstance(value, str) and isinstance(wanted, str):
        return normal(value) == normal(wanted)
    return type(value) is type(wanted) and value == wanted


def arg_ok(value: Any, rule: ArgRule) -> bool:
    """Does the argument's value meet the condition?"""
    if rule.op == "equals":
        return _equal(value, rule.value)
    if rule.op == "one_of":
        return any(_equal(value, wanted) for wanted in rule.value)
    if rule.op == "null":
        return (value is None) is rule.value
    if rule.op == "has":
        return _has(_as_text(value), rule.value)
    return not _has(_as_text(value), rule.value)  # lacks


def matches(call: ToolCallRecord, rule: CallRule) -> bool:
    """The call is the rule's tool, the code took it, and every condition holds."""
    return (
        call.name == rule.tool
        and call.accepted
        and all(
            all(arg_ok(call.input.get(arg), cond) for cond in conds)
            for arg, conds in rule.args
        )
    )


def _show(call: ToolCallRecord) -> str:
    args = ", ".join(f"{name}={value!r}" for name, value in call.input.items())
    taken = "" if call.accepted else " (код отказал)"
    return cut(f"{call.name}({args}){taken}")


def _rule_text(rule: CallRule) -> str:
    conds = ", ".join(
        f"{arg} {cond.op} {cond.value!r}" for arg, conds in rule.args for cond in conds
    )
    return cut(f"{rule.tool}({conds})")


# -- the checks --


def check_calls(expect: SecretaryExpect, record: TurnRecord) -> CheckResult:
    for rule in expect.calls or ():
        if any(matches(call, rule) for call in record.calls):
            continue
        tried = [_show(call) for call in record.calls if call.name == rule.tool]
        made = "; ".join(tried) if tried else f"{rule.tool} не вызван"
        return CheckResult("calls", False, f"нет вызова {_rule_text(rule)}: {made}")
    return CheckResult("calls", True)


def check_no_tools(expect: SecretaryExpect, record: TurnRecord) -> CheckResult:
    banned = set(expect.no_tools or ())
    found = [call for call in record.calls if call.name in banned]
    if found:
        return CheckResult("no_tools", False, f"вызван {_show(found[0])}")
    return CheckResult("no_tools", True)


def check_only_tools(expect: SecretaryExpect, record: TurnRecord) -> CheckResult:
    allowed = set(expect.only_tools or ())
    found = [
        call
        for call in record.calls
        if call.accepted and call.name not in READING and call.name not in allowed
    ]
    if found:
        return CheckResult("only_tools", False, f"лишнее действие: {_show(found[0])}")
    return CheckResult("only_tools", True)


def check_drafts_made(expect: SecretaryExpect, record: TurnRecord) -> CheckResult:
    if record.drafts_made == expect.drafts_made:
        return CheckResult("drafts_made", True)
    return CheckResult(
        "drafts_made",
        False,
        f"новых черновиков {record.drafts_made}, а нужно {expect.drafts_made}",
    )


def check_reply_has(expect: SecretaryExpect, record: TurnRecord) -> CheckResult:
    reply = record.reply or ""
    for piece in expect.reply_has or ():
        if not _has(reply, piece):
            return CheckResult(
                "reply_has", False, f"в ответе нет «{piece}»: «{cut(reply)}»"
            )
    return CheckResult("reply_has", True)


def check_reply_lacks(expect: SecretaryExpect, record: TurnRecord) -> CheckResult:
    reply = record.reply or ""
    for piece in expect.reply_lacks or ():
        for part in piece.split("|"):
            if part.strip() and _has(reply, part):
                return CheckResult(
                    "reply_lacks",
                    False,
                    f"в ответе «{part.strip()}»: «{cut(_around(reply, part))}»",
                )
    return CheckResult("reply_lacks", True)


def _around(text: str, part: str) -> str:
    """The text from a little before the piece: the failure shows where it stands."""
    at = normal(text).find(normal(part))
    return " ".join(text.split())[max(0, at - 20) :]


def check_quotes_ok(expect: SecretaryExpect, record: TurnRecord) -> CheckResult:
    for call in record.calls:
        if not call.accepted and call.answer.strip() == BAD_QUOTE:
            said = call.input.get("quote")
            return CheckResult(
                "quotes_ok",
                False,
                f"{call.name}: цитата «{cut(str(said))}» не из слов владельца",
            )
    return CheckResult("quotes_ok", True)


CHECKS = {
    "calls": check_calls,
    "no_tools": check_no_tools,
    "only_tools": check_only_tools,
    "drafts_made": check_drafts_made,
    "reply_has": check_reply_has,
    "reply_lacks": check_reply_lacks,
    "quotes_ok": check_quotes_ok,
}


def run_checks(expect: SecretaryExpect, record: TurnRecord) -> list[CheckResult]:
    """Every check the case asks for, in a fixed order."""
    return [CHECKS[name](expect, record) for name in expect.wanted()]
