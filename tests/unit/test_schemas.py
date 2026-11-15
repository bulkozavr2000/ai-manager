"""The structured shapes the model fills in: strict, complete and fit for the API."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import fields, is_dataclass
from typing import Any

import pytest
from anthropic import transform_schema
from pydantic import BaseModel, ValidationError

from tg_agent.brain.schemas import JudgeVerdict, Usage

SCHEMAS = [JudgeVerdict]


def verdict_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "ok": False,
        "issues": ["leak", "commitment"],
        "note": "Назвал потолок цены.",
    }
    return {**payload, **overrides}


VALID_PAYLOADS: dict[type[BaseModel], dict[str, Any]] = {
    JudgeVerdict: verdict_payload(),
}


def parse(model: type[BaseModel], payload: dict[str, Any]) -> Any:
    """Validate the way the model's answer arrives: as JSON text."""
    return model.model_validate_json(json.dumps(payload, ensure_ascii=False))


def error_types(excinfo: pytest.ExceptionInfo[ValidationError]) -> list[str]:
    return [error["type"] for error in excinfo.value.errors()]


@pytest.mark.parametrize("model", SCHEMAS, ids=lambda m: m.__name__)
def test_the_schemas_reject_extra_fields(model):
    payload = VALID_PAYLOADS[model]
    parse(model, payload)  # the payload is fine as it is

    with pytest.raises(ValidationError) as excinfo:
        parse(model, {**payload, "surprise": "boo"})

    assert error_types(excinfo) == ["extra_forbidden"]


def test_judge_verdict_issue_tags():
    tags = ["leak", "invented_fact", "commitment", "claims_human", "off_policy"]

    assert parse(JudgeVerdict, verdict_payload(issues=tags)).issues == tags
    with pytest.raises(ValidationError):
        parse(JudgeVerdict, verdict_payload(issues=["rude"]))


@pytest.mark.parametrize("model", SCHEMAS, ids=lambda m: m.__name__)
def test_the_model_must_fill_in_every_field(model):
    # No silent defaults: a verdict without `ok` is an error, not a quiet "all fine".
    payload = VALID_PAYLOADS[model]

    for name in payload:
        incomplete = {key: value for key, value in payload.items() if key != name}
        with pytest.raises(ValidationError) as excinfo:
            parse(model, incomplete)
        assert error_types(excinfo) == ["missing"], name


def closed_objects(schema: Any) -> Iterator[dict[str, Any]]:
    """Every object description inside a JSON schema, at any depth (also under `$defs`)."""
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for value in schema.values():
            yield from closed_objects(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from closed_objects(value)


@pytest.mark.parametrize("model", SCHEMAS, ids=lambda m: m.__name__)
def test_schemas_fit_structured_output(model):
    """What `messages.parse` sends to the API: every field kept, every object closed.

    A free-form `dict` field would come out as an object that may contain nothing, so the model
    could never fill it in.
    """
    schema = transform_schema(model)

    assert set(schema["properties"]) == set(model.model_fields)
    nodes = list(closed_objects(schema))
    assert nodes
    for node in nodes:
        assert node.get("additionalProperties") is False, node
        assert node.get("properties"), f"free-form object in {model.__name__}: {node}"


def test_usage_is_a_plain_dataclass_counting_cache_traffic_from_zero():
    assert is_dataclass(Usage)
    assert not issubclass(Usage, BaseModel)
    assert [field.name for field in fields(Usage)] == [
        "input_tokens",
        "output_tokens",
        "cache_read",
        "cache_write",
    ]

    usage = Usage(1200, 85)
    assert (usage.cache_read, usage.cache_write) == (0, 0)
    assert Usage(1, 2, 3, 4) == Usage(
        input_tokens=1, output_tokens=2, cache_read=3, cache_write=4
    )
