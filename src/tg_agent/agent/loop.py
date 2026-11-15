"""The generic agent loop: a model step, its tool calls, their results, again, within caps.

One run is one request built by the caller (system text, first user message, tools); inside the
run the message list only grows, so the model sees its own turns exactly as it wrote them. The
loop knows nothing about tasks: the handler does the tools, and the caller decides what an
outcome means for the wake.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from tg_agent.brain.llm import LLM, AgentStep, LLMError, ModelRefusal
from tg_agent.brain.pricing import WEB_SEARCH_USD, price


@dataclass(frozen=True)
class ToolResult:
    """What a tool handler answers. `is_error`: the call did not do what was asked, and `text`
    tells the model why. `stop`: nothing more may be done in this run (the task was finished)."""

    text: str
    is_error: bool = False
    stop: bool = False


Handler = Callable[
    [str, dict[str, Any]], Awaitable[ToolResult]
]  # (tool name, input) -> result


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    result: ToolResult


@dataclass
class LoopOutcome:
    steps: int = 0  # model calls made
    cost: float = 0.0  # USD, tokens plus web searches
    # None: the model ended its turn; else "steps", "budget", "refusal", "max_tokens",
    # "error", "finished"
    stopped: str | None = None
    calls: list[ToolCall] = field(default_factory=list)  # every tool call, in order
    error: str = ""  # the error's class name when stopped == "error"


# The model ended its turn by itself: nothing was cut off.
_TURN_ENDED = frozenset({"end_turn", "stop_sequence"})
NOT_RUN = "Этот вызов не выполнен: предыдущий вызов завершил работу"


def _get(block: Any, key: str) -> Any:
    """A field of a content block: an SDK object or, in tests, a dict."""
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


async def run_loop(
    llm: LLM,
    *,
    purpose: str,
    model: str,
    system: str | Sequence[str],  # several texts: stable first (cached)
    content: str | list[dict[str, Any]],
    tools: list[dict[str, Any]],
    handler: Handler,
    effort: str,
    max_tokens: int,
    max_steps: int,
    max_cost: float,
    on_step: Callable[[AgentStep, float], Awaitable[None]] | None = None,
) -> LoopOutcome:
    """Let the model act through `tools` until it ends its turn or a cap stops it.

    A handler that raises is a bug in a tool: the exception leaves this function and the caller
    decides what happens to the wake.
    """
    outcome = LoopOutcome()
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    # The model sometimes changes a tool name's letter case; it still means the declared tool.
    declared = {str(t["name"]).lower(): str(t["name"]) for t in tools if t.get("name")}
    while outcome.steps < max_steps:
        try:
            step = await llm.agent_step(
                purpose=purpose,
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                effort=effort,
                max_tokens=max_tokens,
            )
        except ModelRefusal:
            outcome.stopped = "refusal"
            return outcome
        except LLMError as error:
            outcome.stopped = "error"
            outcome.error = type(error).__name__
            return outcome
        outcome.steps += 1
        cost = price(step.model, step.usage) + step.web_searches * WEB_SEARCH_USD
        outcome.cost += cost
        if on_step is not None:
            await on_step(step, cost)
        messages.append({"role": "assistant", "content": step.content})
        if step.stop_reason == "refusal":
            # The real client returns a refused call marked so (it is paid for: logged above).
            outcome.stopped = "refusal"
            return outcome
        if step.stop_reason == "pause_turn":
            # The API resumes its server tool by itself; no user message goes in between.
            # A paused step costs money too, so the money cap still applies.
            if outcome.cost >= max_cost:
                outcome.stopped = "budget"
                return outcome
            continue
        if step.stop_reason in _TURN_ENDED:
            return outcome
        if step.stop_reason != "tool_use":
            # Cut off (max_tokens, the context window, a reason this code does not know yet):
            # a tool input that may be cut mid-way must never be executed.
            outcome.stopped = "max_tokens"
            return outcome
        uses = [b for b in step.content if _get(b, "type") == "tool_use"]
        if not uses:
            return outcome
        results: list[dict[str, Any]] = []
        finished = False
        for use in uses:
            name, args = str(_get(use, "name") or ""), _get(use, "input")
            known = declared.get(name.lower())
            if finished:
                # An earlier call of this step ended the run (the task was finished): what the
                # model asked for after it is answered, never done.
                name = known or name
                result = ToolResult(NOT_RUN, is_error=True)
            elif known is None:
                result = ToolResult(
                    f"Нет такого инструмента: {name}. Есть: "
                    + ", ".join(declared.values()),
                    is_error=True,
                )
            elif not isinstance(args, dict):
                name = known
                result = ToolResult(
                    f"Входные данные для {known} должны быть объектом с полями инструмента",
                    is_error=True,
                )
            else:
                name = known
                result = await handler(known, args)
            finished = finished or result.stop
            outcome.calls.append(
                ToolCall(name, args if isinstance(args, dict) else {}, result)
            )
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": _get(use, "id"),
                "content": result.text,
            }
            if result.is_error:
                block["is_error"] = True
            results.append(block)
        messages.append({"role": "user", "content": results})
        if finished:
            outcome.stopped = "finished"
            return outcome
        if outcome.cost >= max_cost:
            outcome.stopped = "budget"
            return outcome
    outcome.stopped = "steps"
    return outcome
