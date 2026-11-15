"""Domain model: the enums stored in the database and the dataclasses its rows are read into.

Every enum value is the lower-case member name, and the database stores the value, so renaming
one needs a data migration. Row dataclasses are named after the table columns and hold parsed
values: `*_json` columns as the dict or list they contain (without the suffix), flags as bool,
enum columns as enum members, times as timezone-aware datetimes (None for NULL).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class ConvState(StrEnum):
    """Where a conversation with one counterpart stands. The legal moves: tasks/state.py."""

    PLANNED = "planned"  # attached to a task, nothing sent yet
    CONTACTED = "contacted"  # our message is queued or sent, no reply yet
    TALKING = "talking"
    WAITING_OWNER = "waiting_owner"  # a card about this person waits for the owner
    HANDED_TO_OWNER = "handed_to_owner"  # the owner has the chat
    CLOSED = "closed"  # done with this person; `close_reason` says why


class TaskStatus(StrEnum):
    DRAFT = "draft"  # the card is shown, not launched
    ACTIVE = "active"  # the agent works on it
    PAUSED = "paused"  # the owner paused it, or its wakes kept failing
    DONE = "done"  # the agent finished it; an instruction can take it up again
    CANCELLED = "cancelled"  # the owner closed it: final


class EventKind(StrEnum):
    """What woke the agent core for a task: the `kind` of an `agent_events` row."""

    CREATED = "created"
    INCOMING = "incoming"
    EDITED = "edited"
    OWNER_INSTRUCTION = "owner_instruction"
    OWNER_ANSWER = "owner_answer"
    OWNER_WROTE = "owner_wrote"
    SEND_REFUSED = "send_refused"
    SILENCE = "silence"
    REMINDER = "reminder"
    RESUMED = "resumed"
    RECOVERED = "recovered"


class ApprovalKind(StrEnum):
    """What the owner is asked to approve or answer."""

    QUESTION = "question"
    CONFIRM_ORDER = "confirm_order"
    SHARE_PHONE = "share_phone"
    SHARE_ADDRESS = "share_address"
    HANDOFF = "handoff"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ANSWERED = "answered"
    EXPIRED = "expired"


class OutboxStatus(StrEnum):
    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class Direction(StrEnum):
    IN = "in"
    OUT = "out"


class MsgSource(StrEnum):
    """Who wrote a message."""

    COUNTERPART = "counterpart"
    AGENT = "agent"
    OWNER_MANUAL = "owner_manual"


@dataclass
class Task:
    """A row of `tasks`: what the owner wants done."""

    id: int
    title: str
    goal: str
    item: dict
    ceiling: float | None
    currency: str
    allowed: list[str]
    attachments: list[str]
    owner_notes: str
    status: TaskStatus
    card_msg_id: int | None
    created_at: datetime
    updated_at: datetime
    notes: str = ""
    headline: str = ""
    source_text: str = ""


@dataclass(frozen=True)
class TaskCandidate:
    """A row of `task_candidates`: someone the search found for a task, not yet one of its
    people. The search kept him only because his contact was seen in what a tool showed; his
    name, note and source are the search's words and so untrusted data."""

    id: int
    task_id: int
    contact: str  # a username without "@", or "+<digits>"
    name: str
    note: str
    source: str
    created_at: datetime


@dataclass
class Contact:
    """A row of `contacts`: a Telegram user the agent may talk to."""

    id: int
    peer_id: int
    # Not every Telegram account has a username.
    username: str | None
    display_name: str
    is_bot: bool
    language: str | None
    card: dict
    do_not_contact: bool
    updated_at: datetime


@dataclass
class Conversation:
    """A row of `conversations`: one task being worked with one contact."""

    id: int
    task_id: int
    contact_id: int
    state: ConvState
    prev_state: ConvState | None
    facts: dict
    summary: str
    language: str | None
    offtopic_strikes: int
    agent_messages: int
    followups_sent: int
    first_sent: bool
    paused_until: datetime | None
    last_in_at: datetime | None
    last_out_at: datetime | None
    created_at: datetime
    updated_at: datetime
    note: str = ""
    close_reason: str | None = None


@dataclass
class Message:
    """A row of `messages`: one message of a conversation."""

    id: int
    conversation_id: int
    direction: Direction
    source: MsgSource
    tg_msg_id: int | None
    kind: str
    text: str
    created_at: datetime
    edited_at: datetime | None


@dataclass
class Approval:
    """A row of `approvals`: a request to the owner and, once resolved, their answer."""

    id: int
    task_id: int
    conversation_id: int | None
    kind: ApprovalKind
    payload: dict
    status: ApprovalStatus
    nonce: str
    control_msg_id: int | None
    answer: str | None
    created_at: datetime
    resolved_at: datetime | None


@dataclass
class OutboxItem:
    """A row of `outbox`: a message queued for sending, never sent before `not_before`."""

    id: int
    conversation_id: int
    text: str
    attach_path: str | None
    status: OutboxStatus
    not_before: datetime
    created_at: datetime
    sent_at: datetime | None
    tg_msg_id: int | None
    error: str | None
    wake_id: int | None = None
    tried_at: datetime | None = None


@dataclass
class AgentEvent:
    """A row of `agent_events`: something that wakes the agent core for a task.

    Open while `handled_at` is None; claimed by a running wake while `wake_id` is set.
    """

    id: int
    task_id: int
    conversation_id: int | None
    kind: EventKind
    payload: dict
    created_at: datetime
    due_at: datetime
    attempts: int
    wake_id: int | None
    handled_at: datetime | None


@dataclass
class Wake:
    """A row of `wakes`: one run of the agent core over a task's events."""

    id: int
    task_id: int | None  # None: a wake of the owner-chat agent
    started_at: datetime
    finished_at: datetime | None
    outcome: str | None
    steps: int


@dataclass
class OwnerChatEntry:
    """A row of `owner_chat`: one message of the owner's chat with the control bot."""

    id: int
    direction: Direction
    kind: str
    tg_msg_id: int | None
    reply_to_msg_id: int | None
    task_id: int | None
    approval_id: int | None
    text: str
    handled_at: datetime | None
    created_at: datetime
