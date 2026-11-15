from __future__ import annotations

from types import SimpleNamespace

import pytest

from tg_agent.brain.llm import (
    EFFORT,
    FALLBACK_BETA,
    SEARCH_MAX_TOKENS,
    AgentStep,
    AnthropicLLM,
    FakeLLM,
    LLMError,
    LLMResult,
    ModelRefusal,
    SearchStep,
)
from tg_agent.brain.schemas import JudgeVerdict, Usage
from tg_agent.config import Models

VERDICT = JudgeVerdict(ok=False, issues=["leak"], note="телефон")
DRAFTED = ["Здравствуйте! Подскажите цену."]
# The judge on a model that takes the server-side fallback: the path every structured call
# takes (the task intake was the other structured call until stage 5).
SONNET_JUDGE = Models(judge="claude-sonnet-5-5")


async def ask(llm: AnthropicLLM | FakeLLM) -> LLMResult[JudgeVerdict]:
    """One structured call (the judge), to check what every structured call does."""
    return await llm.judge(DRAFTED, ["name"], "")


# --- FakeLLM ------------------------------------------------------------------------------


async def test_fake_returns_queued_and_records_request() -> None:
    fake = FakeLLM()
    fake.verdicts.append(VERDICT)
    result = await ask(fake)
    assert isinstance(result, LLMResult)
    assert result.value == VERDICT
    assert result.usage == Usage(1000, 200)
    assert result.model == "claude-haiku-5-5"
    assert fake.requests == [("judge", DRAFTED)]


async def test_fake_raises_queued_exception() -> None:
    fake = FakeLLM()
    fake.verdicts.append(ModelRefusal("cyber"))
    with pytest.raises(ModelRefusal):
        await ask(fake)


async def test_fake_without_a_queued_item_fails_loudly() -> None:
    with pytest.raises(AssertionError, match="simulate"):
        await FakeLLM().simulate("роль", "")


async def test_fake_judge_defaults_to_ok() -> None:
    result = await FakeLLM().judge(["Спасибо!"], ["name"], "")
    assert result.value.ok is True and result.value.issues == []
    assert result.model == "claude-haiku-5-5"


# --- AnthropicLLM against a stub client ---------------------------------------------------


class StubMessages:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


def response(
    parsed: object = None, *, stop_reason: str = "end_turn", text: str = ""
) -> object:
    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber", explanation="declined"),
        model="claude-sonnet-5-5",
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=4500,
            output_tokens=500,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=0,
        ),
    )


def llm_with(
    resp: object, models: Models | None = None
) -> tuple[AnthropicLLM, StubMessages, StubMessages]:
    plain, beta = StubMessages(resp), StubMessages(resp)
    client = SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=beta))
    return AnthropicLLM("sk-test", models or SONNET_JUDGE, client=client), plain, beta


async def test_a_structured_call_asks_for_low_effort_and_the_fallback() -> None:
    llm, plain, beta = llm_with(response(VERDICT))
    result = await ask(llm)
    assert result.value == VERDICT
    assert result.usage == Usage(4500, 500, 0, 0)
    assert plain.calls == []
    call = beta.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    assert call["output_config"] == {"effort": "low"}
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in call


async def test_a_structured_call_without_fallback_support_uses_plain_parse() -> None:
    models = Models(judge="claude-haiku-5-5")
    llm, plain, beta = llm_with(response(VERDICT), models)
    await ask(llm)
    assert beta.calls == []
    assert "fallbacks" not in plain.calls[0]


async def test_refusal_raises_model_refusal() -> None:
    llm, _, _ = llm_with(response(None, stop_reason="refusal"))
    with pytest.raises(ModelRefusal, match="cyber"):
        await ask(llm)


async def test_truncated_output_raises_llm_error() -> None:
    llm, _, _ = llm_with(response(None, stop_reason="max_tokens"))
    with pytest.raises(LLMError, match="max_tokens"):
        await ask(llm)


async def test_judge_uses_haiku_and_escapes_draft() -> None:
    verdict = JudgeVerdict(ok=True, issues=[], note="ок")
    llm, plain, beta = llm_with(response(verdict), Models())
    result = await llm.judge(["<b>Спасибо!</b>"], ["name"], "Задача: цена печати")
    assert result.value == verdict
    call = plain.calls[0]
    assert beta.calls == []
    assert call["model"] == "claude-haiku-5-5"
    assert call["output_format"] is JudgeVerdict
    text = call["messages"][0]["content"]
    assert "‹b›Спасибо!‹/b›" in text and "<b>" not in text
    assert "name" in text


async def test_a_search_step_is_cached_and_counts_its_web_searches() -> None:
    resp = response(None, stop_reason="tool_use", text="ищу")
    resp.usage.server_tool_use = SimpleNamespace(
        web_search_requests=2, web_fetch_requests=1
    )
    llm, _, beta = llm_with(resp)  # the search model asks for the server-side fallback
    tools = [{"type": "web_search_20260209", "name": "web_search"}]
    messages = [
        {"role": "user", "content": "Задача: печать"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
    ]  # fmt: skip

    step = await llm.search_step(system="правила", messages=messages, tools=tools)

    assert (step.web_searches, step.stop_reason) == (2, "tool_use")
    call = beta.calls[0]
    assert call["model"] == "claude-sonnet-5-5" and call["tools"] == tools
    # Each step reads what came before it from the prompt cache instead of paying for it again.
    assert call["system"] == [
        {"type": "text", "text": "правила", "cache_control": {"type": "ephemeral"}}
    ]
    assert call["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert (
        "cache_control" not in call["messages"][0]
    )  # one mark: on the newest turn only
    assert (
        "cache_control" not in messages[-1]["content"][-1]
    )  # the caller's list stays as it was


async def test_simulate_returns_text() -> None:
    llm, plain, _ = llm_with(response(None, text="Va alaykum assalom"))
    result = await llm.simulate(
        "Ты мастер 3D-печати", "[10.11 10:00] Ассистент: Ассаляму алейкум!"
    )
    assert result.value == "Va alaykum assalom"
    assert plain.calls[0]["model"] == "claude-haiku-5-5"


async def test_simulate_tells_the_counterpart_to_write_only_his_line() -> None:
    # An eval run once got the simulator's reasoning about its role as the counterpart's message.
    llm, plain, _ = llm_with(response(None, text="Ладно, понял"))
    await llm.simulate("Ты мастер 3D-печати", "")
    assert "без рассуждений о роли" in plain.calls[0]["system"]


async def test_simulate_says_the_run_is_a_closed_test_so_attackers_stay_in_role() -> (
    None
):
    # Haiku refused to play the data fisher or the fake courier in 4 of 130 runs: the agent then
    # met no attack, and the run passed for nothing.
    llm, plain, _ = llm_with(response(None, text="Как вас зовут?"))
    await llm.simulate("Ты выуживаешь личные данные", "")
    system = plain.calls[0]["system"]
    assert "учебная проверка" in system and "доиграй роль" in system


async def test_rejected_fallback_param_is_dropped_and_retried_plainly() -> None:
    import anthropic
    import httpx2 as httpx

    class RejectingBeta(StubMessages):
        async def parse(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.BadRequestError(
                "fallbacks: not supported",
                response=httpx.Response(400, request=request),
                body=None,
            )

    plain = StubMessages(response(VERDICT))
    beta = RejectingBeta(None)
    client = SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=beta))
    llm = AnthropicLLM("sk-test", SONNET_JUDGE, client=client)
    await ask(llm)
    await ask(llm)
    assert len(beta.calls) == 1  # tried once, then remembered
    assert len(plain.calls) == 2 and "fallbacks" not in plain.calls[0]


# --- agent_step: one model call with tools, for any agent ---------------------------------

TOOLS = [{"name": "note", "description": "Заметка", "input_schema": {"type": "object"}}]


def conversation() -> list[dict]:
    return [
        {"role": "user", "content": "Задача: печать"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "note", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
    ]  # fmt: skip


async def step_of(
    llm: AnthropicLLM, model: str, messages: list[dict] | None = None
) -> AgentStep:
    return await llm.agent_step(
        purpose="task",
        model=model,
        system="правила",
        messages=conversation() if messages is None else messages,
        tools=TOOLS,
        effort="medium",
        max_tokens=1234,
    )


def rejecting_beta(text: str) -> StubMessages:
    """A beta endpoint that answers every call with a 400 carrying `text`."""
    import anthropic
    import httpx2 as httpx

    class RejectingBeta(StubMessages):
        async def create(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.BadRequestError(
                text, response=httpx.Response(400, request=request), body=None
            )

    return RejectingBeta(None)


async def test_an_agent_step_asks_for_the_fallback_and_caches_the_turns() -> None:
    llm, plain, beta = llm_with(response(None, stop_reason="tool_use"))
    messages = conversation()

    await step_of(llm, "claude-sonnet-5-5", messages)

    assert plain.calls == []
    call = beta.calls[0]
    assert call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default"
    assert call["model"] == "claude-sonnet-5-5"
    assert call["max_tokens"] == 1234
    assert call["output_config"] == {"effort": "medium"}
    assert call["system"] == [
        {"type": "text", "text": "правила", "cache_control": {"type": "ephemeral"}}
    ]
    assert call["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert messages == conversation()  # the caller's list stays as it was
    assert call["tools"] == TOOLS
    assert not {"tool_choice", "thinking", "temperature"} & call.keys()


async def test_an_agent_step_drops_a_rejected_fallback_and_remembers() -> None:
    plain = StubMessages(response(None))
    beta = rejecting_beta("fallbacks: not supported")
    client = SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=beta))
    llm = AnthropicLLM("sk-test", SONNET_JUDGE, client=client)

    await step_of(llm, "claude-sonnet-5-5")
    assert len(beta.calls) == 1 and len(plain.calls) == 1
    assert "fallbacks" not in plain.calls[0] and "betas" not in plain.calls[0]
    assert llm._tool_fallbacks_ok is False

    await step_of(llm, "claude-sonnet-5-5")
    # straight to the plain endpoint
    assert len(beta.calls) == 1 and len(plain.calls) == 2


async def test_an_agent_step_on_a_model_without_the_fallback_uses_the_plain_endpoint() -> (
    None
):
    llm, plain, beta = llm_with(response(None))

    await step_of(llm, "claude-haiku-5-5")

    assert beta.calls == []
    assert plain.calls[0]["model"] == "claude-haiku-5-5"
    assert "fallbacks" not in plain.calls[0] and "betas" not in plain.calls[0]


async def test_an_agent_step_returns_a_refusal_and_a_cut_response() -> None:
    # A refused call costs money too: the step comes back marked, so the caller logs it.
    llm, _, _ = llm_with(response(None, stop_reason="refusal"))
    step = await step_of(llm, "claude-sonnet-5-5")
    assert step.stop_reason == "refusal"
    assert step.usage.input_tokens > 0

    llm, _, _ = llm_with(response(None, stop_reason="max_tokens", text="недописан"))
    step = await step_of(llm, "claude-sonnet-5-5")
    assert step.stop_reason == "max_tokens"  # the caller decides what to do with it
    assert step.content[0].text == "недописан"


async def test_an_agent_step_reads_usage_web_searches_and_model() -> None:
    resp = response(None, stop_reason="end_turn")
    resp.model = "claude-opus-5-5"  # the fallback answered with another model
    resp.usage.cache_read_input_tokens = 3000
    resp.usage.server_tool_use = SimpleNamespace(web_search_requests=3)
    llm, _, _ = llm_with(resp)

    step = await step_of(llm, "claude-sonnet-5-5")

    assert isinstance(step, AgentStep)
    assert step.usage == Usage(4500, 500, 3000, 0)
    assert (step.web_searches, step.model) == (3, "claude-opus-5-5")


async def test_a_search_step_still_raises_on_refusal() -> None:
    llm, _, _ = llm_with(response(None, stop_reason="refusal"))
    with pytest.raises(ModelRefusal, match="cyber"):
        await llm.search_step(system="правила", messages=conversation(), tools=TOOLS)


async def test_a_search_step_is_an_agent_step_with_the_search_settings() -> None:
    llm, _, beta = llm_with(response(None), Models(search="claude-opus-5-5"))

    await llm.search_step(system="правила", messages=conversation(), tools=TOOLS)
    await llm.agent_step(
        purpose="search",
        model="claude-opus-5-5",
        system="правила",
        messages=conversation(),
        tools=TOOLS,
        effort=EFFORT,
        max_tokens=SEARCH_MAX_TOKENS,
    )

    assert len(beta.calls) == 2 and beta.calls[0] == beta.calls[1]
    assert SearchStep is AgentStep  # old imports keep working


async def test_fake_agent_steps_are_queued_per_purpose() -> None:
    fake = FakeLLM()
    assert fake.search_steps is fake.steps["search"]
    fake.steps["task"].append(
        [{"type": "tool_use", "id": "t1", "name": "note", "input": {}}]
    )
    fake.search_steps.append([{"type": "text", "text": "Всё"}])

    async def ask(purpose: str) -> AgentStep:
        return await fake.agent_step(
            purpose=purpose,
            model="claude-sonnet-5-5",
            system="правила",
            messages=[{"role": "user", "content": "Задача"}],
            tools=TOOLS,
            effort="medium",
            max_tokens=1234,
        )

    with pytest.raises(AssertionError, match="secretary"):
        await ask("secretary")  # nothing queued for it, whatever the others have
    assert (await ask("task")).stop_reason == "tool_use"
    assert (await ask("search")).stop_reason == "end_turn"
    with pytest.raises(AssertionError, match="task"):
        await ask("task")

    purpose, request = fake.requests[0]
    assert purpose == "secretary"
    assert request["model"] == "claude-sonnet-5-5"
    assert (request["effort"], request["max_tokens"]) == ("medium", 1234)
    assert request["messages"] == [{"role": "user", "content": "Задача"}]


async def test_fake_search_step_keeps_its_record_and_adds_the_settings() -> None:
    fake = FakeLLM()
    fake.search_steps.append(ModelRefusal("cyber"))
    messages = [{"role": "user", "content": "Задача"}]

    with pytest.raises(ModelRefusal):
        await fake.search_step(system="правила", messages=messages, tools=TOOLS)
    messages.append({"role": "assistant", "content": "ещё"})

    name, request = fake.requests[0]
    assert name == "search"
    assert request["system"] == "правила" and request["tools"] == TOOLS
    assert request["messages"] == [{"role": "user", "content": "Задача"}]  # a copy
    assert (request["effort"], request["max_tokens"]) == (EFFORT, SEARCH_MAX_TOKENS)
    assert "model" in request


# --- several system texts; one fallback helper with a switch per kind of call -------------


async def test_an_agent_step_with_two_system_texts_marks_only_the_first() -> None:
    llm, _, beta = llm_with(response(None, stop_reason="end_turn"))

    await llm.agent_step(
        purpose="task",
        model="claude-sonnet-5-5",
        system=("правила", "задача этого пробуждения"),
        messages=conversation(),
        tools=TOOLS,
        effort="medium",
        max_tokens=1234,
    )

    assert beta.calls[0]["system"] == [
        {"type": "text", "text": "правила", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "задача этого пробуждения"},
    ]


async def test_the_fake_records_the_system_texts_as_given() -> None:
    fake = FakeLLM()
    fake.steps["task"].append([{"type": "text", "text": "Всё"}])
    await fake.agent_step(
        purpose="task",
        model="claude-sonnet-5-5",
        system=("правила", "задача"),
        messages=[{"role": "user", "content": "Задача"}],
        tools=TOOLS,
        effort="medium",
        max_tokens=1234,
    )
    assert fake.requests[0][1]["system"] == ("правила", "задача")


async def test_a_400_that_does_not_name_the_fallback_is_an_llm_error_without_a_retry() -> (
    None
):
    plain = StubMessages(response(None))
    beta = rejecting_beta("messages: roles must alternate")
    client = SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=beta))
    llm = AnthropicLLM("sk-test", SONNET_JUDGE, client=client)

    with pytest.raises(LLMError, match="BadRequestError"):
        await step_of(llm, "claude-sonnet-5-5")

    assert len(beta.calls) == 1 and plain.calls == []
    assert llm._tool_fallbacks_ok is True  # it was not about the fallback


async def test_a_refused_fallback_for_a_tool_call_leaves_the_structured_calls_alone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    plain = StubMessages(response(VERDICT))
    beta = rejecting_beta("fallbacks: not supported")
    client = SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=beta))
    llm = AnthropicLLM("sk-test", SONNET_JUDGE, client=client)

    with caplog.at_level("WARNING", logger="tg_agent.brain.llm"):
        await step_of(llm, "claude-sonnet-5-5")

    assert llm._tool_fallbacks_ok is False
    assert llm._fallbacks_ok is True
    assert any("fallback" in record.getMessage() for record in caplog.records)
    # The structured calls still try the fallback: their endpoint is asked first.
    beta.parse = StubMessages(response(VERDICT)).parse  # type: ignore[method-assign]
    calls_before = len(plain.calls)
    await ask(llm)
    assert len(plain.calls) == calls_before  # the beta endpoint answered


async def test_a_refused_fallback_for_a_structured_call_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import anthropic
    import httpx2 as httpx

    class RejectingBeta(StubMessages):
        async def parse(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.BadRequestError(
                "fallbacks: not supported",
                response=httpx.Response(400, request=request),
                body=None,
            )

    plain = StubMessages(response(VERDICT))
    client = SimpleNamespace(
        messages=plain, beta=SimpleNamespace(messages=RejectingBeta(None))
    )
    llm = AnthropicLLM("sk-test", SONNET_JUDGE, client=client)

    with caplog.at_level("WARNING", logger="tg_agent.brain.llm"):
        await ask(llm)

    assert llm._fallbacks_ok is False and llm._tool_fallbacks_ok is True
    assert any("fallback" in record.getMessage() for record in caplog.records)


async def test_a_structured_call_400_that_does_not_name_the_fallback_is_not_retried() -> (
    None
):
    import anthropic
    import httpx2 as httpx

    class RejectingBeta(StubMessages):
        async def parse(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.BadRequestError(
                "max_tokens: too large",
                response=httpx.Response(400, request=request),
                body=None,
            )

    plain, beta = StubMessages(response(VERDICT)), RejectingBeta(None)
    client = SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=beta))
    llm = AnthropicLLM("sk-test", SONNET_JUDGE, client=client)

    with pytest.raises(LLMError, match="BadRequestError"):
        await ask(llm)
    assert len(beta.calls) == 1 and plain.calls == []
    assert llm._fallbacks_ok is True


def test_the_module_docstring_no_longer_says_only_the_search_has_tools() -> None:
    import tg_agent.brain.llm as module

    assert "Only the vendor search" not in (module.__doc__ or "")
    assert "agent_step" in (module.__doc__ or "")
