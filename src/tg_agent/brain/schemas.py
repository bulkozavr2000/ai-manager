"""What the model fills in (structured output) and what one model call used up.

All schemas forbid unknown fields, so a key the model invents is a validation error instead of a
silently dropped value. They also have no defaults: the model has to fill in the whole form, and
an answer with a missing field (say the judge's `ok`) is rejected rather than read as a quiet
"nothing to see here".

The SDK sends these classes to the API as JSON schema and turns their docstrings into
descriptions, so a docstring on a schema class is read by the model: write it for the model.
Free-form `dict` fields do not work in structured output (the API wants closed objects).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JudgeVerdict(_Strict):
    """The checker's verdict on a draft of the agent's outgoing message."""

    ok: bool
    issues: list[
        Literal["leak", "invented_fact", "commitment", "claims_human", "off_policy"]
    ]
    note: str


@dataclass
class Usage:
    """Token counts of one model call, as the API reports them."""

    input_tokens: int
    output_tokens: int
    cache_read: int = 0
    cache_write: int = 0
