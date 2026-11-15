"""Domain model: enums are stored as their values, row dataclasses mirror the table columns."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import get_type_hints

import pytest

from tg_agent.models import (
    AgentEvent,
    Approval,
    ApprovalKind,
    ApprovalStatus,
    Contact,
    ConvState,
    Conversation,
    Direction,
    EventKind,
    Message,
    MsgSource,
    OutboxItem,
    OutboxStatus,
    Task,
    TaskStatus,
    Wake,
)

ENUM_MEMBERS: dict[type[StrEnum], list[str]] = {
    ConvState: [
        "PLANNED",
        "CONTACTED",
        "TALKING",
        "WAITING_OWNER",
        "HANDED_TO_OWNER",
        "CLOSED",
    ],
    TaskStatus: ["DRAFT", "ACTIVE", "PAUSED", "DONE", "CANCELLED"],
    EventKind: [
        "CREATED",
        "INCOMING",
        "EDITED",
        "OWNER_INSTRUCTION",
        "OWNER_ANSWER",
        "OWNER_WROTE",
        "SEND_REFUSED",
        "SILENCE",
        "REMINDER",
        "RESUMED",
        "RECOVERED",
    ],
    ApprovalKind: [
        "QUESTION",
        "CONFIRM_ORDER",
        "SHARE_PHONE",
        "SHARE_ADDRESS",
        "HANDOFF",
    ],
    ApprovalStatus: ["PENDING", "APPROVED", "REJECTED", "ANSWERED", "EXPIRED"],
    OutboxStatus: ["PENDING", "SENDING", "SENT", "FAILED", "CANCELLED", "UNKNOWN"],
    Direction: ["IN", "OUT"],
    MsgSource: ["COUNTERPART", "AGENT", "OWNER_MANUAL"],
}


@pytest.mark.parametrize("enum", ENUM_MEMBERS, ids=lambda e: e.__name__)
def test_enum_members_match_the_spec(enum):
    assert [member.name for member in enum] == ENUM_MEMBERS[enum]


@pytest.mark.parametrize("enum", ENUM_MEMBERS, ids=lambda e: e.__name__)
def test_enum_values_are_the_lowercase_member_names(enum):
    # The database stores the value, so the value is the contract.
    for member in enum:
        assert member.value == member.name.lower()
        assert str(member) == member.value
        assert enum(member.value) is member


# Table columns from the repository schema, as the dataclass fields that hold them: `*_json`
# columns become the parsed dict or list, flags are bool, times are timezone-aware datetimes.
ROW_FIELDS: dict[type, dict[str, object]] = {
    Task: {
        "id": int,
        "title": str,
        "goal": str,
        "item": dict,
        "ceiling": float | None,
        "currency": str,
        "allowed": list[str],
        "attachments": list[str],
        "owner_notes": str,
        "status": TaskStatus,
        "card_msg_id": int | None,
        "created_at": datetime,
        "updated_at": datetime,
        "notes": str,
        "headline": str,
        "source_text": str,
    },
    Contact: {
        "id": int,
        "peer_id": int,
        "username": str | None,
        "display_name": str,
        "is_bot": bool,
        "language": str | None,
        "card": dict,
        "do_not_contact": bool,
        "updated_at": datetime,
    },
    Conversation: {
        "id": int,
        "task_id": int,
        "contact_id": int,
        "state": ConvState,
        "prev_state": ConvState | None,
        "facts": dict,
        "summary": str,
        "language": str | None,
        "offtopic_strikes": int,
        "agent_messages": int,
        "followups_sent": int,
        "first_sent": bool,
        "paused_until": datetime | None,
        "last_in_at": datetime | None,
        "last_out_at": datetime | None,
        "created_at": datetime,
        "updated_at": datetime,
        "note": str,
        "close_reason": str | None,
    },
    Message: {
        "id": int,
        "conversation_id": int,
        "direction": Direction,
        "source": MsgSource,
        "tg_msg_id": int | None,
        "kind": str,
        "text": str,
        "created_at": datetime,
        "edited_at": datetime | None,
    },
    Approval: {
        "id": int,
        "task_id": int,
        "conversation_id": int | None,
        "kind": ApprovalKind,
        "payload": dict,
        "status": ApprovalStatus,
        "nonce": str,
        "control_msg_id": int | None,
        "answer": str | None,
        "created_at": datetime,
        "resolved_at": datetime | None,
    },
    OutboxItem: {
        "id": int,
        "conversation_id": int,
        "text": str,
        "attach_path": str | None,
        "status": OutboxStatus,
        "not_before": datetime,
        "created_at": datetime,
        "sent_at": datetime | None,
        "tg_msg_id": int | None,
        "error": str | None,
        "wake_id": int | None,
        "tried_at": datetime | None,
    },
    AgentEvent: {
        "id": int,
        "task_id": int,
        "conversation_id": int | None,
        "kind": EventKind,
        "payload": dict,
        "created_at": datetime,
        "due_at": datetime,
        "attempts": int,
        "wake_id": int | None,
        "handled_at": datetime | None,
    },
    Wake: {
        "id": int,
        "task_id": int | None,
        "started_at": datetime,
        "finished_at": datetime | None,
        "outcome": str | None,
        "steps": int,
    },
}


@pytest.mark.parametrize("row_class", ROW_FIELDS, ids=lambda c: c.__name__)
def test_row_dataclass_fields_follow_the_table_columns(row_class):
    hints = get_type_hints(row_class)
    expected = ROW_FIELDS[row_class]

    assert list(hints) == list(expected)  # same columns, same order
    assert hints == expected
