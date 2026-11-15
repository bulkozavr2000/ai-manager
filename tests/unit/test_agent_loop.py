"""The generic agent loop: model steps, tool calls, one user message of results, and the caps."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tg_agent.agent.loop import LoopOutcome, ToolResult, run_loop
from tg_agent.brain.llm import AgentStep, FakeLLM, LLMError, ModelRefusal
from tg_agent.brain.pricing import price
from tg_agent.brain.schemas import Usage

TOOLS: list[dict[str, Any]] = [
    {
        "name": "send_message",
        "description": "Отправить сообщение",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
    {
        "name": "update_notes",
        "description": "Записать заметки",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"notes": {"type": "string"}},
            "required": ["notes"],
            "additionalProperties": False,
        },
    },
]

STEP_COST = price("claude-sonnet-5-5", Usage(1000, 200))


class Recorder:
    """A handler that records its calls and answers from a script (default: "ok")."""

    def __init__(self, answers: dict[str, ToolResult] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.answers = answers or {}

    async def __call__(self, name: str, args: dict[str, Any]) -> ToolResult:
        self.calls.append((name, args))
        return self.answers.get(name, ToolResult("ok"))


def use(id_: str, name: str, args: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": id_, "name": name, "input": args}


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


async def run(
    llm: FakeLLM,
    handler: Recorder,
    *,
    content: str | list[dict[str, Any]] = "Задача: найти мастера",
    max_steps: int = 10,
    max_cost: float = 1.0,
    on_step: Any = None,
) -> LoopOutcome:
    return await run_loop(
        llm,
        purpose="task",
        model="claude-sonnet-5-5",
        system="Правила агента",
        content=content,
        tools=TOOLS,
        handler=handler,
        effort="medium",
        max_tokens=16000,
        max_steps=max_steps,
        max_cost=max_cost,
        on_step=on_step,
    )


def requests(llm: FakeLLM) -> list[dict[str, Any]]:
    return [req for purpose, req in llm.requests if purpose == "task"]


async def test_model_ends_its_turn_at_once() -> None:
    llm = FakeLLM()
    llm.steps["task"].append([text("Готово")])
    handler = Recorder()

    outcome = await run(llm, handler)

    assert outcome.steps == 1
    assert outcome.stopped is None
    assert outcome.calls == []
    assert handler.calls == []
    [req] = requests(llm)
    assert req["system"] == "Правила агента"
    assert req["messages"] == [{"role": "user", "content": "Задача: найти мастера"}]
    assert req["tools"] == TOOLS
    assert req["model"] == "claude-sonnet-5-5"
    assert req["effort"] == "medium"
    assert req["max_tokens"] == 16000


async def test_two_calls_in_one_step_get_one_user_message_of_results() -> None:
    llm = FakeLLM()
    first = [
        {"type": "thinking", "thinking": "думаю", "signature": "sig"},
        use("t1", "send_message", {"text": "Здравствуйте"}),
        use("t2", "update_notes", {"notes": "написал"}),
    ]
    llm.steps["task"].extend([first, [text("Жду ответа")]])
    handler = Recorder(
        {
            "send_message": ToolResult("отправлено"),
            "update_notes": ToolResult("записано"),
        }
    )

    outcome = await run(llm, handler)

    assert handler.calls == [
        ("send_message", {"text": "Здравствуйте"}),
        ("update_notes", {"notes": "написал"}),
    ]
    assert outcome.steps == 2
    assert outcome.stopped is None
    assert [(c.name, c.input, c.result.text) for c in outcome.calls] == [
        ("send_message", {"text": "Здравствуйте"}, "отправлено"),
        ("update_notes", {"notes": "написал"}, "записано"),
    ]
    second = requests(llm)[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert (
        second[1]["content"] == first
    )  # the blocks exactly as they came, thinking included
    assert second[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "отправлено"},
        {"type": "tool_result", "tool_use_id": "t2", "content": "записано"},
    ]


async def test_an_error_result_is_marked_and_the_loop_goes_on() -> None:
    llm = FakeLLM()
    llm.steps["task"].extend(
        [[use("t1", "send_message", {"text": "http://x"})], [text("Понял")]]
    )
    handler = Recorder({"send_message": ToolResult("ссылки нельзя", is_error=True)})

    outcome = await run(llm, handler)

    assert outcome.steps == 2
    assert outcome.stopped is None
    assert requests(llm)[1]["messages"][2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": "ссылки нельзя",
            "is_error": True,
        }
    ]


async def test_a_name_in_the_wrong_case_is_accepted_and_an_unknown_one_is_not() -> None:
    llm = FakeLLM()
    llm.steps["task"].extend(
        [
            [use("t1", "Send_Message", {"text": "Привет"}), use("t2", "fly", {})],
            [text("Всё")],
        ]
    )
    handler = Recorder()

    outcome = await run(llm, handler)

    assert handler.calls == [("send_message", {"text": "Привет"})]
    results = requests(llm)[1]["messages"][2]["content"]
    assert results[0] == {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
    assert results[1]["tool_use_id"] == "t2"
    assert results[1]["is_error"] is True
    assert "fly" in results[1]["content"]
    assert "send_message" in results[1]["content"]
    assert "update_notes" in results[1]["content"]
    assert [c.name for c in outcome.calls] == ["send_message", "fly"]
    assert outcome.calls[1].result.is_error


async def test_an_input_that_is_not_an_object_is_refused_without_the_handler() -> None:
    llm = FakeLLM()
    llm.steps["task"].extend([[use("t1", "send_message", "Привет")], [text("Всё")]])
    handler = Recorder()

    outcome = await run(llm, handler)

    assert handler.calls == []
    [result] = requests(llm)[1]["messages"][2]["content"]
    assert result["tool_use_id"] == "t1"
    assert result["is_error"] is True
    assert outcome.calls[0].result.is_error


async def test_a_stop_result_ends_the_loop_as_finished() -> None:
    llm = FakeLLM()
    llm.steps["task"].extend(
        [
            [use("t1", "send_message", {"text": "Итог"})],
            [text("этот шаг не должен быть запрошен")],
        ]
    )
    handler = Recorder({"send_message": ToolResult("задача завершена", stop=True)})

    outcome = await run(llm, handler)

    assert outcome.stopped == "finished"
    assert outcome.steps == 1
    assert len(requests(llm)) == 1
    assert len(llm.steps["task"]) == 1


async def test_max_steps_stops_a_model_that_keeps_calling_tools() -> None:
    llm = FakeLLM()
    for i in range(5):
        llm.steps["task"].append([use(f"t{i}", "update_notes", {"notes": str(i)})])
    handler = Recorder()

    outcome = await run(llm, handler, max_steps=3)

    assert outcome.stopped == "steps"
    assert outcome.steps == 3
    assert len(requests(llm)) == 3
    assert len(handler.calls) == 3


async def test_max_cost_runs_the_steps_tools_then_stops() -> None:
    llm = FakeLLM()
    llm.usage = Usage(100_000, 10_000)  # 0.3 USD a step
    llm.steps["task"].extend(
        [[use("t1", "update_notes", {"notes": "x"})], [text("не должен быть запрошен")]]
    )
    handler = Recorder()

    outcome = await run(llm, handler, max_cost=0.1)

    assert handler.calls == [("update_notes", {"notes": "x"})]
    assert outcome.stopped == "budget"
    assert outcome.steps == 1
    assert len(requests(llm)) == 1


async def test_a_cut_off_step_never_runs_its_tools() -> None:
    llm = FakeLLM()
    llm.steps["task"].append(
        AgentStep(
            [use("t1", "send_message", {"text": "Здравств"})],
            Usage(1000, 200),
            0,
            "claude-sonnet-5-5",
            "max_tokens",
        )
    )
    handler = Recorder()

    outcome = await run(llm, handler)

    assert handler.calls == []
    assert outcome.calls == []
    assert outcome.stopped == "max_tokens"
    assert outcome.steps == 1


async def test_pause_turn_continues_without_a_new_user_message() -> None:
    llm = FakeLLM()
    paused = [
        {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}}
    ]
    llm.steps["task"].extend(
        [
            AgentStep(paused, Usage(1000, 200), 1, "claude-sonnet-5-5", "pause_turn"),
            [text("Нашёл")],
        ]
    )
    handler = Recorder()

    outcome = await run(llm, handler)

    assert outcome.steps == 2
    assert outcome.stopped is None
    second = requests(llm)[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant"]
    assert second[1]["content"] == paused


async def test_a_refusal_stops_the_loop() -> None:
    llm = FakeLLM()
    llm.steps["task"].append(ModelRefusal("refusal"))

    outcome = await run(llm, Recorder())

    assert outcome.stopped == "refusal"
    assert outcome.steps == 0


async def test_another_model_error_stops_the_loop_and_names_the_class() -> None:
    class Overloaded(LLMError):
        pass

    llm = FakeLLM()
    llm.steps["task"].append(Overloaded("overloaded"))

    outcome = await run(llm, Recorder())

    assert outcome.stopped == "error"
    assert outcome.error == "Overloaded"
    assert outcome.steps == 0


async def test_an_exception_from_the_handler_propagates() -> None:
    llm = FakeLLM()
    llm.steps["task"].append([use("t1", "send_message", {"text": "x"})])

    async def broken(name: str, args: dict[str, Any]) -> ToolResult:
        raise RuntimeError("bug in a tool")

    with pytest.raises(RuntimeError, match="bug in a tool"):
        await run_loop(
            llm,
            purpose="task",
            model="claude-sonnet-5-5",
            system="Правила",
            content="Задача",
            tools=TOOLS,
            handler=broken,
            effort="medium",
            max_tokens=16000,
            max_steps=5,
            max_cost=1.0,
        )


async def test_on_step_is_awaited_per_step_with_its_cost() -> None:
    llm = FakeLLM()
    searched = AgentStep(
        [use("t1", "update_notes", {"notes": "x"})],
        Usage(1000, 200),
        2,
        "claude-sonnet-5-5",
        "tool_use",
    )
    llm.steps["task"].extend([searched, [text("Всё")]])
    seen: list[tuple[AgentStep, float]] = []

    async def on_step(step: AgentStep, cost: float) -> None:
        seen.append((step, cost))

    outcome = await run(llm, Recorder(), on_step=on_step)

    assert len(seen) == 2
    assert seen[0][0] is searched
    assert seen[0][1] == pytest.approx(STEP_COST + 2 * 0.01)
    assert seen[1][1] == pytest.approx(STEP_COST)
    assert outcome.cost == pytest.approx(seen[0][1] + seen[1][1])


async def test_the_first_message_may_be_content_blocks() -> None:
    llm = FakeLLM()
    llm.steps["task"].append([text("Вижу фото")])
    blocks = [
        {"type": "text", "text": "Вот фото"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"},
        },
    ]

    await run(llm, Recorder(), content=blocks)

    assert requests(llm)[0]["messages"] == [{"role": "user", "content": blocks}]


async def test_a_paused_step_over_the_money_cap_stops_the_loop() -> None:
    llm = FakeLLM()
    paused = [
        {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}}
    ]
    llm.steps["task"].extend(
        [
            AgentStep(
                paused, Usage(100_000, 10_000), 1, "claude-sonnet-5-5", "pause_turn"
            ),
            [text("не должен быть запрошен")],
        ]
    )

    outcome = await run(llm, Recorder(), max_cost=0.1)

    assert outcome.stopped == "budget"
    assert outcome.steps == 1
    assert len(requests(llm)) == 1


async def test_after_a_stop_result_the_other_calls_of_the_step_are_not_run() -> None:
    """`finish` and `send_message` in one step: the message must not go out after the end."""
    llm = FakeLLM()
    llm.steps["task"].append(
        [
            use("t1", "update_notes", {"notes": "итог"}),
            use("t2", "send_message", {"text": "И ещё вопрос"}),
        ]
    )
    handler = Recorder({"update_notes": ToolResult("Задача завершена", stop=True)})

    outcome = await run(llm, handler)

    assert handler.calls == [("update_notes", {"notes": "итог"})]
    assert outcome.stopped == "finished"
    assert [(call.name, call.result.is_error) for call in outcome.calls] == [
        ("update_notes", False),
        ("send_message", True),
    ]
    assert "не выполнен" in outcome.calls[1].result.text


async def test_a_step_that_arrives_as_a_refusal_runs_nothing() -> None:
    llm = FakeLLM()
    llm.steps["task"].append(
        AgentStep(
            [use("t1", "send_message", {"text": "Привет"})],
            Usage(1000, 200),
            0,
            "claude-sonnet-5-5",
            "refusal",
        )
    )
    handler = Recorder()

    outcome = await run(llm, handler)

    assert handler.calls == []
    assert outcome.stopped == "refusal"
    assert outcome.steps == 1


@pytest.mark.parametrize("reason", ["model_context_window_exceeded", "something_new"])
async def test_any_other_cut_off_never_runs_the_steps_tools(reason: str) -> None:
    llm = FakeLLM()
    llm.steps["task"].append(
        AgentStep(
            [use("t1", "send_message", {"text": "Здравств"})],
            Usage(1000, 200),
            0,
            "claude-sonnet-5-5",
            reason,
        )
    )
    handler = Recorder()

    outcome = await run(llm, handler)

    assert handler.calls == []
    assert outcome.stopped == "max_tokens"


async def test_blocks_of_the_real_client_are_read_like_dicts() -> None:
    """The real client returns objects with attributes, not dicts."""
    llm = FakeLLM()
    block = SimpleNamespace(
        type="tool_use", id="t1", name="send_message", input={"text": "Привет"}
    )
    llm.steps["task"].extend(
        [
            AgentStep([block], Usage(1000, 200), 0, "claude-sonnet-5-5", "tool_use"),
            [text("Готово")],
        ]
    )
    handler = Recorder()

    outcome = await run(llm, handler)

    assert handler.calls == [("send_message", {"text": "Привет"})]
    assert outcome.stopped is None
    assert requests(llm)[1]["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
    ]


async def test_several_system_texts_reach_the_model_as_given() -> None:
    llm = FakeLLM()
    llm.steps["task"].append([text("Готово")])

    await run_loop(
        llm,
        purpose="task",
        model="claude-sonnet-5-5",
        system=("Правила агента", "Задача этого пробуждения"),
        content="Задача: найти мастера",
        tools=TOOLS,
        handler=Recorder(),
        effort="medium",
        max_tokens=16000,
        max_steps=3,
        max_cost=1.0,
    )

    [req] = requests(llm)
    assert req["system"] == ("Правила агента", "Задача этого пробуждения")
    # Python does not check annotations at run time, and the repo has no type checker: the
    # signature itself must say it takes several texts.
    from collections.abc import Sequence
    from typing import get_type_hints

    assert get_type_hints(run_loop)["system"] == str | Sequence[str]
