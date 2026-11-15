"""`agent_step` against the real API. Run explicitly:

    TG_AGENT_LIVE=1 uv run pytest tests/live/test_agent_step_live.py -q

The offline tests script the model, so they cannot tell whether the real endpoint accepts what
we send. These two calls do: strict tool schemas of the shapes the task agent uses, and the
search's server tools, both through the beta endpoint with the server-side fallback.
"""

from __future__ import annotations

from typing import Any

import pytest

from tg_agent.brain.llm import AnthropicLLM
from tg_agent.config import Models
from tg_agent.secrets import Key, require_secret

pytestmark = pytest.mark.live


def _strict(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


# The shapes of the task agent's tools: arrays, enums, nullable values, an array of objects.
STRICT_TOOLS = [
    _strict(
        "send_message",
        "Написать человеку из задачи в Telegram.",
        {
            "person_id": {"type": "integer"},
            "messages": {"type": "array", "items": {"type": "string"}},
            "language": {"type": "string", "enum": ["ru", "uz_latn", "other"]},
            "attach_task_files": {"type": "boolean"},
            "commits_owner": {"type": "boolean"},
        },
    ),
    _strict(
        "ask_owner",
        "Спросить владельца карточкой в боте.",
        {
            "kind": {
                "type": "string",
                "enum": ["question", "approval", "phone", "address", "handoff"],
            },
            "question": {"type": "string"},
            "person_id": {"type": ["integer", "null"]},
            "suggested_reply": {"type": "array", "items": {"type": "string"}},
            "amount": {"type": ["number", "null"]},
            "terms": {"type": "string"},
        },
    ),
    _strict(
        "update_notes",
        "Записать заметки и обновить статус задачи.",
        {
            "notes": {"type": "string"},
            "headline": {"type": "string"},
            "person_notes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "person_id": {"type": "integer"},
                        "note": {"type": "string"},
                    },
                    "required": ["person_id", "note"],
                    "additionalProperties": False,
                },
            },
        },
    ),
]


def _tool_calls(content: list[Any]) -> list[Any]:
    return [block for block in content if block.type == "tool_use"]


async def test_live_strict_tools_and_a_second_step() -> None:
    llm = AnthropicLLM(require_secret(Key.ANTHROPIC_API_KEY), Models())
    system = (
        "Ты ведёшь задачу владельца в Telegram. Действуй только вызовами инструментов."
        " Когда делать больше нечего, ответь одним словом: готово"
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": "Задача: узнать цену печати 5 корпусов из PETG.\n"
            "Люди: 1 — мастерская «Принт Лаб», ещё не писали.\n"
            "Напиши первому человеку одно короткое сообщение на русском.",
        }
    ]

    first = await llm.agent_step(
        purpose="task",
        model=Models().decide,
        system=system,
        messages=messages,
        tools=STRICT_TOOLS,
        effort="medium",
        max_tokens=16000,
    )

    assert llm._fallbacks_ok, "the beta endpoint refused the fallback parameter"
    assert first.stop_reason == "tool_use"
    calls = _tool_calls(first.content)
    assert [call.name for call in calls] == ["send_message"]
    sent = calls[0].input
    assert set(sent) == {
        "person_id",
        "messages",
        "language",
        "attach_task_files",
        "commits_owner",
    }
    assert sent["person_id"] == 1
    assert sent["language"] == "ru"
    assert sent["messages"] and all(isinstance(text, str) for text in sent["messages"])
    assert first.usage.input_tokens > 0

    # The second step sends the first one's blocks back unchanged, as the loop will.
    messages += [
        {"role": "assistant", "content": first.content},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": calls[0].id,
                    "content": "Сообщение поставлено в очередь. Больше делать нечего.",
                }
            ],
        },
    ]
    second = await llm.agent_step(
        purpose="task",
        model=Models().decide,
        system=system,
        messages=messages,
        tools=STRICT_TOOLS,
        effort="medium",
        max_tokens=16000,
    )

    assert second.stop_reason in {"end_turn", "tool_use"}
    assert all(call.name == "update_notes" for call in _tool_calls(second.content))


async def test_live_server_tools_through_the_fallback_endpoint() -> None:
    llm = AnthropicLLM(require_secret(Key.ANTHROPIC_API_KEY), Models())
    tools: list[dict[str, Any]] = [
        {
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": 1,
            "user_location": {"type": "approximate", "timezone": "Asia/Tashkent"},
        },
        {
            "type": "web_fetch_20260209",
            "name": "web_fetch",
            "max_uses": 1,
            "max_content_tokens": 2000,
            "allowed_callers": ["direct"],
        },
        {
            "name": "save_candidate",
            "description": "Сохранить найденного исполнителя.",
            "input_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    ]

    step = await llm.search_step(
        system="Ты ищешь исполнителей в интернете. Сделай ровно один поиск и ответь кратко.",
        messages=[
            {
                "role": "user",
                "content": "Найди одним поиском: 3D печать на заказ Ташкент. "
                "Назови одно найденное название, ничего не сохраняй.",
            }
        ],
        tools=tools,
    )

    assert llm._fallbacks_ok, "the beta endpoint refused the fallback parameter"
    assert step.stop_reason in {"end_turn", "tool_use", "pause_turn", "max_tokens"}
    assert step.web_searches >= 1
    assert step.usage.input_tokens > 0
