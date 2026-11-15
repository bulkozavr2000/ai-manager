"""SQLite storage: the schema and `Repo`, the only code that talks to the database.

What the rest of the program can rely on:

* Moments are stored as UTC ISO-8601 text (`2026-11-10T05:00:00+00:00`), which orders and
  compares chronologically in SQL, and come back as timezone-aware UTC datetimes. A naive
  datetime is refused with ValueError rather than guessed at.
* Rows come back as the dataclasses of `tg_agent.models`: enum columns as enum members (never
  raw strings), `*_json` columns as the parsed dict or list, flags as bool.
* There is one connection, in autocommit mode: a method that writes does it in one statement,
  so it is atomic and has committed when it returns. The one read-then-write, `set_state`,
  holds a lock.
* "Now" (created_at, updated_at, event times) comes from the `Clock` given to `Repo.open`.
* SQLite 3.35 or newer is needed (`RETURNING` in the two upserts).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from secrets import token_hex
from zoneinfo import ZoneInfo

import aiosqlite

from tg_agent.brain.schemas import Usage
from tg_agent.clock import Clock, SystemClock
from tg_agent.db_events import EventStore
from tg_agent.models import (
    Approval,
    ApprovalKind,
    ApprovalStatus,
    Contact,
    Conversation,
    ConvState,
    Direction,
    Message,
    MsgSource,
    OutboxItem,
    OutboxStatus,
    OwnerChatEntry,
    Task,
    TaskCandidate,
    TaskStatus,
)
from tg_agent.tasks.state import TERMINAL, transition

_log = logging.getLogger(__name__)

TASHKENT = ZoneInfo(
    "Asia/Tashkent"
)  # the owner's calendar: what "a day" means for the limits
# Event logged when the agent's first message goes to a person who had not written to us:
# a new contact (`tasks/outbox.mark_sent`).
FIRST_SENT = "first_sent"

SCHEMA_V0 = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    goal TEXT NOT NULL,
    item_json TEXT NOT NULL,
    ceiling REAL,
    currency TEXT NOT NULL DEFAULT 'UZS',
    allowed_json TEXT NOT NULL,
    attachments_json TEXT NOT NULL,
    owner_notes TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    card_msg_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS contacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    peer_id INTEGER NOT NULL UNIQUE,
    username TEXT,  -- not every Telegram account has one
    display_name TEXT NOT NULL,
    is_bot INTEGER NOT NULL,
    language TEXT,
    card_json TEXT NOT NULL DEFAULT '{}',
    do_not_contact INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL REFERENCES tasks (id),
    contact_id INTEGER NOT NULL REFERENCES contacts (id),
    state TEXT NOT NULL,
    prev_state TEXT,
    facts_json TEXT NOT NULL DEFAULT '{}',
    summary TEXT NOT NULL DEFAULT '',
    language TEXT,
    offtopic_strikes INTEGER NOT NULL DEFAULT 0,
    agent_messages INTEGER NOT NULL DEFAULT 0,
    followups_sent INTEGER NOT NULL DEFAULT 0,
    first_sent INTEGER NOT NULL DEFAULT 0,
    paused_until TEXT,
    last_in_at TEXT,
    last_out_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (task_id, contact_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations (id),
    direction TEXT NOT NULL,
    source TEXT NOT NULL,
    tg_msg_id INTEGER,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    edited_at TEXT
);
CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages (conversation_id, created_at);

CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL REFERENCES tasks (id),
    conversation_id INTEGER REFERENCES conversations (id),
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL,
    nonce TEXT NOT NULL,
    control_msg_id INTEGER,
    answer TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations (id),
    text TEXT NOT NULL,
    attach_path TEXT,
    status TEXT NOT NULL,
    not_before TEXT NOT NULL,
    created_at TEXT NOT NULL,
    sent_at TEXT,
    tg_msg_id INTEGER,
    error TEXT
);

CREATE TABLE IF NOT EXISTS timers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER REFERENCES conversations (id),
    task_id INTEGER REFERENCES tasks (id),
    kind TEXT NOT NULL,
    due_at TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending'  -- pending, done or cancelled
);
CREATE UNIQUE INDEX IF NOT EXISTS timers_one_pending_per_kind
    ON timers (conversation_id, kind) WHERE status = 'pending';

CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER REFERENCES conversations (id),
    purpose TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_read INTEGER NOT NULL,
    cache_write INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    stop_reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    type TEXT NOT NULL,
    conversation_id INTEGER REFERENCES conversations (id),
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# The schema version a database is at lives in `PRAGMA user_version`; every database created
# before versions existed is at 0 with the `SCHEMA_V0` tables. `MIGRATIONS[n]` takes a database
# from version n - 1 to n, one SQL statement per item (run one by one inside a transaction,
# never through `executescript`, which commits on its own). A fresh database is `SCHEMA_V0`
# plus every migration, so there is no second copy of the latest schema to drift.
LATEST_VERSION = 2
# "Now" inside a migration, in the form the program stores moments (UTC ISO-8601).
_SQL_NOW = "strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now')"
MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """CREATE TABLE wakes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER REFERENCES tasks (id),  -- NULL: a wake of the owner-chat agent
    started_at TEXT NOT NULL,
    finished_at TEXT,
    outcome TEXT,
    steps INTEGER NOT NULL DEFAULT 0
)""",
        """CREATE TABLE agent_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL REFERENCES tasks (id),
    conversation_id INTEGER REFERENCES conversations (id),
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    due_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    wake_id INTEGER REFERENCES wakes (id),
    handled_at TEXT
)""",
        (
            "CREATE INDEX agent_events_open ON agent_events (task_id, due_at)"
            " WHERE handled_at IS NULL"
        ),
        # At most one silence event waits per conversation. One that is handled (the person
        # answered, or its wake is over) or claimed by a running wake does not count.
        """CREATE UNIQUE INDEX agent_events_one_silence ON agent_events (conversation_id)
    WHERE kind = 'silence' AND wake_id IS NULL AND handled_at IS NULL""",
        """CREATE TABLE owner_chat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    direction TEXT NOT NULL,   -- 'in': the owner wrote; 'out': the bot wrote
    kind TEXT NOT NULL,        -- in: text, command, button, file; out: notice or the card's kind
    tg_msg_id INTEGER,
    reply_to_msg_id INTEGER,
    task_id INTEGER REFERENCES tasks (id),
    approval_id INTEGER REFERENCES approvals (id),
    text TEXT NOT NULL,
    handled_at TEXT,
    created_at TEXT NOT NULL
)""",
        "CREATE INDEX owner_chat_by_time ON owner_chat (created_at)",
        """CREATE TABLE task_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL REFERENCES tasks (id),
    contact TEXT NOT NULL,
    name TEXT NOT NULL,
    note TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (task_id, contact)
)""",
        "ALTER TABLE tasks ADD COLUMN notes TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE tasks ADD COLUMN headline TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE tasks ADD COLUMN source_text TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE conversations ADD COLUMN note TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE conversations ADD COLUMN close_reason TEXT",
        "ALTER TABLE outbox ADD COLUMN wake_id INTEGER REFERENCES wakes (id)",
        "ALTER TABLE outbox ADD COLUMN tried_at TEXT",
        "ALTER TABLE llm_calls ADD COLUMN task_id INTEGER REFERENCES tasks (id)",
        "ALTER TABLE llm_calls ADD COLUMN wake_id INTEGER REFERENCES wakes (id)",
    ),
    # A database at version 1 holds only the old purchase engine's data (the agent core never
    # ran on it), and that engine is gone: close what it left, so that nothing waits for code
    # that no longer exists and every row reads with the agent core's states and kinds.
    2: (
        # 1. Its unfinished tasks are cancelled.
        (
            f"UPDATE tasks SET status = 'cancelled', updated_at = {_SQL_NOW}"
            " WHERE status NOT IN ('done', 'cancelled')"
        ),
        # 2. Its people are closed, with the reason the owner reads on the board. The old
        # PAUSED remembered a state the agent core's machine does not know.
        (
            "UPDATE conversations SET prev_state = NULL WHERE prev_state IS NOT NULL"
            " AND prev_state NOT IN ('planned', 'contacted', 'talking', 'waiting_owner',"
            " 'handed_to_owner', 'closed')"
        ),
        """UPDATE conversations SET close_reason = CASE state
        WHEN 'done' THEN 'завершено'
        WHEN 'declined' THEN 'отказ'
        WHEN 'no_response' THEN 'не ответил'
        WHEN 'closed_offtopic' THEN 'разговор не по делу'
        ELSE 'закрыто при переходе на новую версию'
    END, state = 'closed'
    WHERE state != 'closed'""",
        # 3. Its open cards expire; the kinds the agent core does not have read as questions.
        (
            f"UPDATE approvals SET status = 'expired', resolved_at = {_SQL_NOW}"
            " WHERE status = 'pending'"
        ),
        (
            "UPDATE approvals SET kind = 'question' WHERE kind NOT IN"
            " ('question', 'confirm_order', 'share_phone', 'share_address', 'handoff')"
        ),
        # 4. Nothing it queued goes out.
        (
            "UPDATE outbox SET status = 'cancelled'"
            " WHERE status IN ('pending', 'sending', 'unknown')"
        ),
        # 5. Its timers never fire (the table stays until a later migration).
        "UPDATE timers SET status = 'cancelled' WHERE status = 'pending'",
        # 6. The pinned status redraws itself in place (its message id stays); the old cards
        # lose their codes, so their buttons answer «Уже решено». GLOB: `_` is literal there.
        (
            "DELETE FROM kv WHERE key = 'board_text' OR key GLOB 'task_nonce:*'"
            " OR key GLOB 'finish_nonce:*' OR key GLOB 'reply_terms:*' OR key GLOB 'found_msg:*'"
        ),
        # 7. Nothing of the agent core stays open over a cancelled task (none should be).
        f"UPDATE agent_events SET handled_at = {_SQL_NOW} WHERE handled_at IS NULL",
        (
            f"UPDATE wakes SET finished_at = {_SQL_NOW}, outcome = 'migrated'"
            " WHERE finished_at IS NULL"
        ),
        # 8. The first-contact guard reads the newest `first_sent` event on every send.
        "CREATE INDEX events_by_type ON events (type, ts)",
    ),
}


class MigrationError(RuntimeError):
    """The schema could not be brought to `LATEST_VERSION`; `version` is where it stopped."""

    def __init__(self, version: int, message: str) -> None:
        super().__init__(message)
        self.version = version


# A launched task that is not finished; a draft waits for the owner, DONE and CANCELLED are over.
_ACTIVE_TASK_STATUSES = (TaskStatus.ACTIVE, TaskStatus.PAUSED)

# A conversation is live while it is not over (by the state machine, so the two cannot
# disagree) and its task is in work: a person of a DONE or CANCELLED task is free for a new
# task and is not listened to. For queries that name the conversations table `c`; the
# parameters are `_LIVE_PARAMS`.
_FINISHED = tuple(sorted(state.value for state in TERMINAL))
_IN_WORK = (
    "c.task_id IN (SELECT id FROM tasks WHERE status IN"
    f" ({', '.join('?' for _ in _ACTIVE_TASK_STATUSES)}))"
)
_IN_WORK_PARAMS = tuple(status.value for status in _ACTIVE_TASK_STATUSES)
_STATES = ", ".join("?" for _ in _FINISHED)
_LIVE = f"c.state NOT IN ({_STATES}) AND {_IN_WORK}"
_LIVE_PARAMS = (*_FINISHED, *_IN_WORK_PARAMS)

# The conversation fields `update_conversation` may write, and the column each one lives in.
# The state is not among them: it changes only through `set_state`.
_CONVERSATION_COLUMNS = {
    "facts": "facts_json",
    "summary": "summary",
    "language": "language",
    "offtopic_strikes": "offtopic_strikes",
    "agent_messages": "agent_messages",
    "followups_sent": "followups_sent",
    "first_sent": "first_sent",
    "paused_until": "paused_until",
    "last_in_at": "last_in_at",
    "last_out_at": "last_out_at",
    "note": "note",
    "close_reason": "close_reason",
}

_STOPPED_KEY = "stopped"
_STOPPED_WHY_KEY = "stopped_why"


class NotFound(LookupError):
    """No row has the given id."""


# --- value conversion ------------------------------------------------------------------------


def _utc(moment: datetime) -> str:
    """A moment as the UTC text the database stores."""
    if moment.utcoffset() is None:
        raise ValueError(f"naive datetime {moment!r}: pass a timezone-aware one")
    return moment.astimezone(UTC).isoformat()


def _moment(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _maybe[T, R](convert: Callable[[T], R], value: T | None) -> R | None:
    """`convert(value)`, or None for a NULL."""
    return None if value is None else convert(value)


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _encode(value: object) -> object:
    """A Python value in the form the database stores: moments as UTC text, containers as JSON."""
    if isinstance(value, datetime):
        return _utc(value)
    if isinstance(value, dict | list):
        return _dumps(value)
    return value


def _task(r: aiosqlite.Row) -> Task:
    return Task(
        id=r["id"],
        title=r["title"],
        goal=r["goal"],
        item=json.loads(r["item_json"]),
        ceiling=r["ceiling"],
        currency=r["currency"],
        allowed=json.loads(r["allowed_json"]),
        attachments=json.loads(r["attachments_json"]),
        owner_notes=r["owner_notes"],
        status=TaskStatus(r["status"]),
        card_msg_id=r["card_msg_id"],
        created_at=_moment(r["created_at"]),
        updated_at=_moment(r["updated_at"]),
        notes=r["notes"],
        headline=r["headline"],
        source_text=r["source_text"],
    )


def _contact(r: aiosqlite.Row) -> Contact:
    return Contact(
        id=r["id"],
        peer_id=r["peer_id"],
        username=r["username"],
        display_name=r["display_name"],
        is_bot=bool(r["is_bot"]),
        language=r["language"],
        card=json.loads(r["card_json"]),
        do_not_contact=bool(r["do_not_contact"]),
        updated_at=_moment(r["updated_at"]),
    )


def _conversation(r: aiosqlite.Row) -> Conversation:
    return Conversation(
        id=r["id"],
        task_id=r["task_id"],
        contact_id=r["contact_id"],
        state=ConvState(r["state"]),
        prev_state=_maybe(ConvState, r["prev_state"]),
        facts=json.loads(r["facts_json"]),
        summary=r["summary"],
        language=r["language"],
        offtopic_strikes=r["offtopic_strikes"],
        agent_messages=r["agent_messages"],
        followups_sent=r["followups_sent"],
        first_sent=bool(r["first_sent"]),
        paused_until=_maybe(_moment, r["paused_until"]),
        last_in_at=_maybe(_moment, r["last_in_at"]),
        last_out_at=_maybe(_moment, r["last_out_at"]),
        created_at=_moment(r["created_at"]),
        updated_at=_moment(r["updated_at"]),
        note=r["note"],
        close_reason=r["close_reason"],
    )


def _message(r: aiosqlite.Row) -> Message:
    return Message(
        id=r["id"],
        conversation_id=r["conversation_id"],
        direction=Direction(r["direction"]),
        source=MsgSource(r["source"]),
        tg_msg_id=r["tg_msg_id"],
        kind=r["kind"],
        text=r["text"],
        created_at=_moment(r["created_at"]),
        edited_at=_maybe(_moment, r["edited_at"]),
    )


def _approval(r: aiosqlite.Row) -> Approval:
    return Approval(
        id=r["id"],
        task_id=r["task_id"],
        conversation_id=r["conversation_id"],
        kind=ApprovalKind(r["kind"]),
        payload=json.loads(r["payload_json"]),
        status=ApprovalStatus(r["status"]),
        nonce=r["nonce"],
        control_msg_id=r["control_msg_id"],
        answer=r["answer"],
        created_at=_moment(r["created_at"]),
        resolved_at=_maybe(_moment, r["resolved_at"]),
    )


def _outbox_item(r: aiosqlite.Row) -> OutboxItem:
    return OutboxItem(
        id=r["id"],
        conversation_id=r["conversation_id"],
        text=r["text"],
        attach_path=r["attach_path"],
        status=OutboxStatus(r["status"]),
        not_before=_moment(r["not_before"]),
        created_at=_moment(r["created_at"]),
        sent_at=_maybe(_moment, r["sent_at"]),
        tg_msg_id=r["tg_msg_id"],
        error=r["error"],
        wake_id=r["wake_id"],
        tried_at=_maybe(_moment, r["tried_at"]),
    )


def _candidate(r: aiosqlite.Row) -> TaskCandidate:
    return TaskCandidate(
        id=r["id"],
        task_id=r["task_id"],
        contact=r["contact"],
        name=r["name"],
        note=r["note"],
        source=r["source"],
        created_at=_moment(r["created_at"]),
    )


def _owner_chat_entry(r: aiosqlite.Row) -> OwnerChatEntry:
    return OwnerChatEntry(
        id=r["id"],
        direction=Direction(r["direction"]),
        kind=r["kind"],
        tg_msg_id=r["tg_msg_id"],
        reply_to_msg_id=r["reply_to_msg_id"],
        task_id=r["task_id"],
        approval_id=r["approval_id"],
        text=r["text"],
        handled_at=_maybe(_moment, r["handled_at"]),
        created_at=_moment(r["created_at"]),
    )


# --- schema versions -------------------------------------------------------------------------


async def _bring_schema_up_to_date(
    db: aiosqlite.Connection, path: str | Path, clock: Clock
) -> None:
    """Create a fresh database, or migrate an older one to `LATEST_VERSION`.

    The owner's live database is upgraded by this, so an upgrade first copies the database
    into `backup/` next to it, and each migration commits whole or not at all.
    """
    [(current,)] = await db.execute_fetchall("PRAGMA user_version")
    is_new = not await db.execute_fetchall(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'tasks'"
    )
    if is_new:  # nothing to lose, so no backup
        await db.executescript(SCHEMA_V0)
        for version in range(1, LATEST_VERSION + 1):
            await _migrate(db, version)
        return
    if current > LATEST_VERSION:
        raise MigrationError(
            current,
            f"the database is at schema version {current}, newer than this program"
            f" ({LATEST_VERSION}); it was left unchanged",
        )
    if current == LATEST_VERSION:
        return
    copied_to = None
    if str(path) != ":memory:":
        copied_to = await _back_up(db, Path(path), current, clock)
    for version in range(current + 1, LATEST_VERSION + 1):
        await _migrate(db, version)
    # The folder, not the file: the log masks long runs of digits, and the name has a date.
    _log.info(
        "the database was upgraded: schema version %s -> %s; the old file was copied to %s",
        current,
        LATEST_VERSION,
        copied_to.parent if copied_to is not None else "nowhere (in memory)",
    )


async def _back_up(
    db: aiosqlite.Connection, path: Path, version: int, clock: Clock
) -> Path:
    """Copy the database to `backup/agent-v{version}-{UTC time}.db`, never over an old copy;
    return the copy's path."""
    folder = path.parent / "backup"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"agent-v{version}-{clock.now().astimezone(UTC):%Y%m%d-%H%M%S}"
    target = folder / f"{stem}.db"
    n = 0
    while target.exists():
        n += 1
        target = folder / f"{stem}-{n}.db"
    # VACUUM INTO cannot run inside a transaction: the connection is in autocommit here.
    try:
        await db.execute_fetchall("VACUUM INTO ?", (str(target),))
    except Exception as exc:
        # No upgrade without a whole copy to go back to; a half-written one would look like
        # a good backup later.
        target.unlink(missing_ok=True)
        raise MigrationError(
            version, f"the database could not be backed up before the upgrade: {exc}"
        ) from exc
    return target


async def _migrate(db: aiosqlite.Connection, version: int) -> None:
    """Apply `MIGRATIONS[version]` and set the version, in one transaction."""
    try:
        await db.execute_fetchall("BEGIN IMMEDIATE")
        for statement in MIGRATIONS[version]:
            await db.execute_fetchall(statement)
        # DDL and user_version are transactional in SQLite, so a failure undoes both.
        await db.execute_fetchall(f"PRAGMA user_version = {int(version)}")
        await db.execute_fetchall("COMMIT")
    except Exception as exc:
        if db.in_transaction:
            await db.execute_fetchall("ROLLBACK")
        raise MigrationError(
            version, f"migration to schema version {version} failed: {exc}"
        ) from exc


# --- the repository --------------------------------------------------------------------------


class Repo(EventStore):
    """Everything the program stores, behind async methods. Create it with `Repo.open`.

    The agent core's events and wakes are in the `EventStore` mixin (`db_events.py`).
    """

    def __init__(
        self,
        db: aiosqlite.Connection,
        clock: Clock,
    ) -> None:
        self._db = db
        self._clock = clock
        # Held while a conversation's state is read, checked and written back.
        self._state_lock = asyncio.Lock()
        # Held while the agent core's events are read and then written (see `db_events`).
        self._event_lock = asyncio.Lock()

    @classmethod
    async def open(
        cls,
        path: str | Path,
        *,
        clock: Clock | None = None,
    ) -> Repo:
        """Open the database file at `path`, creating the tables the first time.

        An older database is upgraded to `LATEST_VERSION`, after a backup next to it (see
        `_bring_schema_up_to_date`); a newer one raises MigrationError.

        `clock` supplies the time for `created_at`, `updated_at` and event times; without one
        it is the system clock. Tests pass a FakeClock.
        """
        clock = clock if clock is not None else SystemClock(UTC)
        db = await aiosqlite.connect(path, isolation_level=None)  # None: autocommit
        try:
            db.row_factory = aiosqlite.Row
            await db.execute_fetchall("PRAGMA foreign_keys = ON")
            await db.execute_fetchall("PRAGMA journal_mode = WAL")
            await _bring_schema_up_to_date(db, path, clock)
            # «Добавить в задачу?» is gone (the simplification pass): an old card of its
            # kind is read as an expired question.
            await db.execute_fetchall(
                "UPDATE approvals SET kind = 'question', status = 'expired'"
                " WHERE kind = 'add_person'"
            )
        except BaseException:
            await db.close()  # else its thread outlives the failed open
            raise
        return cls(db, clock)

    async def close(self) -> None:
        await self._db.close()

    async def back_up_to(self, target: Path) -> None:
        """A whole copy of the database at `target`, by SQLite's own backup call (ruling R87).

        It is written next to its place and moved there only once whole: a copy that failed
        halfway never looks like a good one.
        """
        part = target.with_name(target.name + ".part")
        part.unlink(missing_ok=True)
        try:
            # The copy's connection is used from the database's own thread.
            copy = sqlite3.connect(part, check_same_thread=False)
            try:
                await self._db.backup(copy)
            finally:
                copy.close()
            part.replace(target)
        except BaseException:
            part.unlink(missing_ok=True)
            raise

    # -- plumbing --

    def _now(self) -> str:
        return _utc(self._clock.now())

    async def _all(
        self, sql: str, params: Sequence[object] = ()
    ) -> list[aiosqlite.Row]:
        return list(await self._db.execute_fetchall(sql, params))

    async def _one(
        self, sql: str, params: Sequence[object] = ()
    ) -> aiosqlite.Row | None:
        rows = await self._all(sql, params)
        return rows[0] if rows else None

    async def _run(self, sql: str, params: Sequence[object] = ()) -> int:
        """Run one statement that writes; return how many rows it changed."""
        async with self._db.execute(sql, params) as cursor:
            return cursor.rowcount

    async def _insert(self, sql: str, params: Sequence[object]) -> int:
        """Run one INSERT; return the id of the new row."""
        async with self._db.execute(sql, params) as cursor:
            return cursor.lastrowid

    async def _get(self, table: str, row_id: int) -> aiosqlite.Row:
        # `table` is always a literal of this module, never caller input.
        row = await self._one(f"SELECT * FROM {table} WHERE id = ?", (row_id,))
        if row is None:
            raise NotFound(f"{table} has no row with id {row_id}")
        return row

    async def _update(
        self, table: str, row_id: int, values: Mapping[str, object]
    ) -> None:
        """Set columns of one row; `table` and the column names are this module's own."""
        assignments = ", ".join(f"{column} = ?" for column in values)
        changed = await self._run(
            f"UPDATE {table} SET {assignments} WHERE id = ?", (*values.values(), row_id)
        )
        if not changed:
            raise NotFound(f"{table} has no row with id {row_id}")

    # -- tasks --

    async def create_task(
        self,
        title: str,
        goal: str,
        item: dict,
        ceiling: float | None,
        currency: str,
        allowed: list[str],
        attachments: list[str],
        *,
        source_text: str = "",
    ) -> int:
        """Store a new task. It starts as a DRAFT: the owner has not launched it yet.

        `source_text` is the owner's own words the task was made from.
        """
        now = self._now()
        return await self._insert(
            "INSERT INTO tasks (title, goal, item_json, ceiling, currency, allowed_json,"
            " attachments_json, status, source_text, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                title,
                goal,
                _dumps(item),
                ceiling,
                currency,
                _dumps(allowed),
                _dumps(attachments),
                TaskStatus.DRAFT.value,
                source_text,
                now,
                now,
            ),
        )

    async def rewrite_draft(
        self,
        task_id: int,
        *,
        title: str,
        goal: str,
        ceiling: float | None,
        currency: str,
        allowed: list[str],
        attachments: list[str],
        source_text: str,
    ) -> bool:
        """Replace a draft's fields in place (its number stays); True if it was rewritten.
        One statement that checks the status too: a draft launched or closed meanwhile is
        left as it is, so a correction can never change a task already in work."""
        changed = await self._run(
            "UPDATE tasks SET title = ?, goal = ?, ceiling = ?, currency = ?,"
            " allowed_json = ?, attachments_json = ?, source_text = ?, updated_at = ?"
            " WHERE id = ? AND status = ?",
            (
                title,
                goal,
                ceiling,
                currency,
                _dumps(allowed),
                _dumps(attachments),
                source_text,
                self._now(),
                task_id,
                TaskStatus.DRAFT.value,
            ),
        )
        return changed == 1

    async def get_task(self, task_id: int) -> Task:
        return _task(await self._get("tasks", task_id))

    async def set_task_status(self, task_id: int, status: TaskStatus) -> None:
        values = {"status": TaskStatus(status).value, "updated_at": self._now()}
        await self._update("tasks", task_id, values)

    async def move_task_if(
        self,
        task_id: int,
        expected: TaskStatus | Iterable[TaskStatus],
        new: TaskStatus,
    ) -> bool:
        """Move the task to `new` only if its status is still `expected` (one of them); True
        if it moved. One statement: of two callers that read the same status (two presses
        of a button, the owner's close and a wake's `finish`), only one moves it."""
        wanted = [expected] if isinstance(expected, TaskStatus) else list(expected)
        values = [TaskStatus(status).value for status in wanted]
        marks = ", ".join("?" for _ in values)
        changed = await self._run(
            f"UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?"
            f" AND status IN ({marks})",
            (TaskStatus(new).value, self._now(), task_id, *values),
        )
        return changed == 1

    async def set_task_notes(
        self, task_id: int, *, notes: str | None = None, headline: str | None = None
    ) -> None:
        """Write the agent's memory of the task and/or the one line of its pinned status;
        whichever is None stays as it is."""
        values: dict[str, object] = {}
        if notes is not None:
            values["notes"] = notes
        if headline is not None:
            values["headline"] = headline
        values["updated_at"] = self._now()
        await self._update("tasks", task_id, values)

    async def all_tasks(self) -> list[Task]:
        """Every task, whatever its status, oldest first."""
        rows = await self._all("SELECT * FROM tasks ORDER BY id", ())
        return [_task(row) for row in rows]

    # -- candidates the search found for a task --

    async def add_candidate(
        self, task_id: int, contact: str, name: str, note: str, source: str
    ) -> int:
        """Keep someone the search found for the task; return his row's id.

        A contact the task already has keeps its row and id (the agent may already know him
        by it); only what the newer search said about him replaces the old words.
        """
        await self._run(
            "INSERT INTO task_candidates (task_id, contact, name, note, source, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (task_id, contact) DO UPDATE SET"
            " name = excluded.name, note = excluded.note, source = excluded.source",
            (task_id, contact, name, note, source, self._now()),
        )
        row = await self._one(
            "SELECT id FROM task_candidates WHERE task_id = ? AND contact = ?",
            (task_id, contact),
        )
        assert row is not None  # written just above
        return row["id"]

    async def task_candidates(self, task_id: int) -> list[TaskCandidate]:
        """Everyone the searches found for the task, in the order they were first found."""
        rows = await self._all(
            "SELECT * FROM task_candidates WHERE task_id = ? ORDER BY id", (task_id,)
        )
        return [_candidate(row) for row in rows]

    async def get_candidate(self, candidate_id: int) -> TaskCandidate:
        return _candidate(await self._get("task_candidates", candidate_id))

    # -- contacts --

    async def upsert_contact(
        self, peer_id: int, username: str | None, display_name: str, is_bot: bool
    ) -> int:
        """Add the contact or refresh its profile; the card, language and do-not-contact stay."""
        [row] = await self._all(
            "INSERT INTO contacts (peer_id, username, display_name, is_bot, updated_at)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT (peer_id) DO UPDATE SET username = excluded.username,"
            " display_name = excluded.display_name, is_bot = excluded.is_bot,"
            " updated_at = excluded.updated_at"
            " RETURNING id",
            (peer_id, username, display_name, is_bot, self._now()),
        )
        return row["id"]

    async def get_contact(self, contact_id: int) -> Contact:
        return _contact(await self._get("contacts", contact_id))

    async def contact_by_peer(self, peer_id: int) -> Contact | None:
        row = await self._one("SELECT * FROM contacts WHERE peer_id = ?", (peer_id,))
        return _maybe(_contact, row)

    async def set_do_not_contact(self, contact_id: int, flag: bool) -> None:
        """Mark (or unmark) a person the agent must not write to in a new task."""
        await self._update(
            "contacts",
            contact_id,
            {"do_not_contact": int(flag), "updated_at": self._now()},
        )

    async def update_contact_card(
        self, contact_id: int, card: dict, language: str | None
    ) -> None:
        values = {
            "card_json": _dumps(card),
            "language": language,
            "updated_at": self._now(),
        }
        await self._update("contacts", contact_id, values)

    # -- conversations --

    async def create_conversation(
        self, task_id: int, contact_id: int, state: ConvState = ConvState.PLANNED
    ) -> int:
        """Start a conversation of a task with a contact, in PLANNED or in the state given."""
        now = self._now()
        return await self._insert(
            "INSERT INTO conversations (task_id, contact_id, state, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (task_id, contact_id, ConvState(state).value, now, now),
        )

    async def get_conversation(self, conv_id: int) -> Conversation:
        return _conversation(await self._get("conversations", conv_id))

    async def conversation_by_peer(self, peer_id: int) -> Conversation | None:
        """The newest live conversation with this Telegram user (see `_LIVE`)."""
        row = await self._one(
            "SELECT c.* FROM conversations c JOIN contacts k ON k.id = c.contact_id"
            f" WHERE k.peer_id = ? AND {_LIVE} ORDER BY c.id DESC LIMIT 1",
            (peer_id, *_LIVE_PARAMS),
        )
        return _maybe(_conversation, row)

    async def blocking_conversation(self, peer_id: int) -> Conversation | None:
        """A live conversation with this Telegram user, which keeps a new task from writing
        to him (one whose chat the owner has, HANDED_TO_OWNER, too); the newest if several."""
        row = await self._one(
            "SELECT c.* FROM conversations c JOIN contacts k ON k.id = c.contact_id"
            f" WHERE k.peer_id = ? AND {_LIVE} ORDER BY c.id DESC LIMIT 1",
            (peer_id, *_LIVE_PARAMS),
        )
        return _maybe(_conversation, row)

    async def task_conversations(self, task_id: int) -> list[Conversation]:
        rows = await self._all(
            "SELECT * FROM conversations WHERE task_id = ? ORDER BY id", (task_id,)
        )
        return [_conversation(row) for row in rows]

    async def update_conversation(self, conv_id: int, **fields: object) -> None:
        """Write the given fields (see `_CONVERSATION_COLUMNS`); the state is not among them."""
        unknown = sorted(fields.keys() - _CONVERSATION_COLUMNS.keys())
        if unknown:
            raise TypeError(
                f"cannot update conversation field(s) {unknown}; "
                "the state changes only through set_state"
            )
        values = {
            _CONVERSATION_COLUMNS[name]: _encode(value)
            for name, value in fields.items()
        }
        values["updated_at"] = self._now()
        await self._update("conversations", conv_id, values)

    async def set_state(self, conv_id: int, new: ConvState) -> ConvState:
        """Move the conversation to `new` if the state machine allows it, and return it.

        An impossible move raises InvalidTransition and changes nothing.
        """
        async with self._state_lock:
            conv = await self.get_conversation(conv_id)
            new = transition(conv.state, ConvState(new))
            values = {"state": new.value, "updated_at": self._now()}
            await self._update("conversations", conv_id, values)
        return new

    async def move_if(self, conv_id: int, expected: ConvState, new: ConvState) -> bool:
        """Move the conversation to `new` only if it is still in `expected`; True if it moved.

        For a move decided from a state read earlier: a wake may have moved the person since
        (to a card, to the owner, closed), and then nothing changes. The check and the move
        hold the state lock, as `set_state` does; a move the state machine forbids from
        `expected` raises InvalidTransition.
        """
        async with self._state_lock:
            conv = await self.get_conversation(conv_id)
            if conv.state is not ConvState(expected):
                return False
            new = transition(conv.state, ConvState(new))
            values = {"state": new.value, "updated_at": self._now()}
            await self._update("conversations", conv_id, values)
        return True

    async def closed_conversation_by_peer(self, peer_id: int) -> Conversation | None:
        """The newest closed conversation with this Telegram user in a task in work: he is
        still heard (ruling R104 b)."""
        row = await self._one(
            "SELECT c.* FROM conversations c JOIN contacts k ON k.id = c.contact_id"
            f" WHERE k.peer_id = ? AND c.state IN ({_STATES}) AND {_IN_WORK}"
            " ORDER BY c.id DESC LIMIT 1",
            (peer_id, *_FINISHED, *_IN_WORK_PARAMS),
        )
        return _maybe(_conversation, row)

    async def contacts_in_work(self) -> list[Contact]:
        """Every person of a task in work, closed ones too; nobody of a task that is over
        (ruling R104). The gateway listens to them (`people.listen`)."""
        rows = await self._all(
            "SELECT DISTINCT k.* FROM conversations c"
            f" JOIN contacts k ON k.id = c.contact_id WHERE {_IN_WORK}",
            _IN_WORK_PARAMS,
        )
        return [_contact(row) for row in rows]

    async def active_peer_ids(self) -> set[int]:
        """Telegram ids of the contacts with a live conversation (see `_LIVE`)."""
        rows = await self._all(
            "SELECT DISTINCT k.peer_id FROM conversations c JOIN contacts k ON k.id = c.contact_id"
            f" WHERE {_LIVE}",
            _LIVE_PARAMS,
        )
        return {row["peer_id"] for row in rows}

    # -- messages --

    async def add_message(
        self,
        conv_id: int,
        direction: Direction,
        source: MsgSource,
        kind: str,
        text: str,
        tg_msg_id: int | None,
        at: datetime,
    ) -> int:
        return await self._insert(
            "INSERT INTO messages (conversation_id, direction, source, tg_msg_id, kind, text,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                conv_id,
                Direction(direction).value,
                MsgSource(source).value,
                tg_msg_id,
                kind,
                text,
                _utc(at),
            ),
        )

    async def incoming_texts(self, task_id: int) -> list[str]:
        """The texts of every message the task's counterparts wrote to us, oldest first."""
        rows = await self._all(
            "SELECT m.text FROM messages m JOIN conversations c"
            " ON m.conversation_id = c.id"
            " WHERE c.task_id = ? AND m.direction = ? ORDER BY m.id",
            (task_id, Direction.IN.value),
        )
        return [row["text"] for row in rows]

    async def last_messages(self, conv_id: int, n: int) -> list[Message]:
        """The `n` newest messages of the conversation, oldest of them first."""
        if n <= 0:  # SQLite reads a negative LIMIT as "no limit"
            return []
        rows = await self._all(
            "SELECT * FROM (SELECT * FROM messages WHERE conversation_id = ?"
            " ORDER BY created_at DESC, id DESC LIMIT ?) ORDER BY created_at, id",
            (conv_id, n),
        )
        return [_message(row) for row in rows]

    async def max_message_id(self, task_id: int) -> int:
        """The highest `messages.id` among the task's conversations; 0 when there is none.

        The agent core keeps it as the task's "seen" watermark: messages above it are new.
        """
        row = await self._one(
            "SELECT COALESCE(MAX(m.id), 0) AS top FROM messages m"
            " JOIN conversations c ON c.id = m.conversation_id WHERE c.task_id = ?",
            (task_id,),
        )
        return row["top"]

    async def max_incoming_id(self, conv_id: int) -> int:
        """The highest `messages.id` the person of this conversation wrote; 0 when none.

        A reply drafted before a message above it came in is stale.
        """
        row = await self._one(
            "SELECT COALESCE(MAX(id), 0) AS top FROM messages"
            " WHERE conversation_id = ? AND direction = ?",
            (conv_id, Direction.IN.value),
        )
        return row["top"]

    async def has_message(self, conv_id: int, tg_msg_id: int) -> bool:
        """Is a message with this Telegram id already stored in the conversation?"""
        row = await self._one(
            "SELECT 1 FROM messages WHERE conversation_id = ? AND tg_msg_id = ? LIMIT 1",
            (conv_id, tg_msg_id),
        )
        return row is not None

    async def last_tg_msg_id(
        self, conv_id: int, direction: Direction | None = None
    ) -> int | None:
        """The Telegram id of the newest stored message that has one (ids only grow in a
        chat), of either direction or only of `direction`."""
        sql = "SELECT MAX(tg_msg_id) AS newest FROM messages WHERE conversation_id = ?"
        params: list[object] = [conv_id]
        if direction is not None:
            sql += " AND direction = ?"
            params.append(Direction(direction).value)
        row = await self._one(sql, params)
        return row["newest"] if row is not None else None

    async def out_tg_msg_ids(self, conv_id: int) -> set[int]:
        """The Telegram ids of the stored outgoing messages: the agent's and the owner's own."""
        rows = await self._all(
            "SELECT tg_msg_id FROM messages WHERE conversation_id = ? AND direction = ?"
            " AND tg_msg_id IS NOT NULL",
            (conv_id, Direction.OUT.value),
        )
        return {row["tg_msg_id"] for row in rows}

    async def find_incoming(self, tg_msg_id: int) -> tuple[int, Message] | None:
        """The counterpart's message with this Telegram id, and its conversation; the newest
        if several. (In private chats Telegram's ids are the account's own, one sequence for
        all chats, so the id tells the message.)"""
        rows = await self._all(
            "SELECT * FROM messages WHERE tg_msg_id = ? AND direction = ?"
            " ORDER BY id DESC LIMIT 1",
            (tg_msg_id, Direction.IN.value),
        )
        if not rows:
            return None
        message = _message(rows[0])
        return int(rows[0]["conversation_id"]), message

    async def edit_message(
        self, conv_id: int, tg_msg_id: int, text: str, at: datetime
    ) -> bool:
        """Replace the text of the message with this Telegram id; False if there is none."""
        changed = await self._run(
            "UPDATE messages SET text = ?, edited_at = ?"
            " WHERE conversation_id = ? AND tg_msg_id = ?",
            (text, _utc(at), conv_id, tg_msg_id),
        )
        return changed > 0

    async def purge_messages_before(self, moment: datetime) -> int:
        """Delete the messages older than `moment`; return how many went."""
        return await self._run(
            "DELETE FROM messages WHERE created_at < ?", (_utc(moment),)
        )

    # -- the owner's chat with the bot --

    async def add_owner_chat(
        self,
        direction: Direction | str,
        kind: str,
        text: str,
        *,
        tg_msg_id: int | None = None,
        reply_to_msg_id: int | None = None,
        task_id: int | None = None,
        approval_id: int | None = None,
    ) -> int:
        return await self._insert(
            "INSERT INTO owner_chat (direction, kind, tg_msg_id, reply_to_msg_id, task_id,"
            " approval_id, text, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                Direction(direction).value,
                kind,
                tg_msg_id,
                reply_to_msg_id,
                task_id,
                approval_id,
                text,
                self._now(),
            ),
        )

    async def set_owner_chat_text(self, tg_msg_id: int, text: str) -> bool:
        """Replace the text of the bot's newest logged message with this Telegram id (a card
        edited in place); False if there is none."""
        changed = await self._run(
            "UPDATE owner_chat SET text = ? WHERE id = (SELECT MAX(id) FROM owner_chat"
            " WHERE tg_msg_id = ? AND direction = ?)",
            (text, tg_msg_id, Direction.OUT.value),
        )
        return changed > 0

    async def owner_chat_tail(self, limit: int) -> list[OwnerChatEntry]:
        """The `limit` newest entries of the owner's chat, oldest of them first."""
        if limit <= 0:  # SQLite reads a negative LIMIT as "no limit"
            return []
        rows = await self._all(
            "SELECT * FROM (SELECT * FROM owner_chat ORDER BY created_at DESC, id DESC"
            " LIMIT ?) ORDER BY created_at, id",
            (limit,),
        )
        return [_owner_chat_entry(row) for row in rows]

    # -- approvals --

    async def create_approval(
        self, task_id: int, conv_id: int | None, kind: ApprovalKind, payload: dict
    ) -> Approval:
        """Ask the owner something. The nonce goes into the buttons and is good for one answer."""
        approval_id = await self._insert(
            "INSERT INTO approvals (task_id, conversation_id, kind, payload_json, status, nonce,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                conv_id,
                ApprovalKind(kind).value,
                _dumps(payload),
                ApprovalStatus.PENDING.value,
                token_hex(3),
                self._now(),
            ),
        )
        return await self.get_approval(approval_id)

    async def get_approval(self, approval_id: int) -> Approval:
        return _approval(await self._get("approvals", approval_id))

    async def resolve_approval(
        self,
        approval_id: int,
        nonce: str,
        status: ApprovalStatus,
        answer: str | None = None,
    ) -> bool:
        """Record the owner's answer. False if the nonce is not this approval's or it is settled."""
        status = ApprovalStatus(status)
        if status is ApprovalStatus.PENDING:
            raise ValueError("an approval cannot be resolved to 'pending'")
        changed = await self._run(
            "UPDATE approvals SET status = ?, answer = ?, resolved_at = ?"
            " WHERE id = ? AND nonce = ? AND status = ?",
            (
                status.value,
                answer,
                self._now(),
                approval_id,
                nonce,
                ApprovalStatus.PENDING.value,
            ),
        )
        return changed == 1

    async def approval_by_control_msg(self, msg_id: int) -> Approval | None:
        """The approval whose card is the control bot's message `msg_id`, settled or not."""
        row = await self._one(
            "SELECT * FROM approvals WHERE control_msg_id = ? ORDER BY id DESC LIMIT 1",
            (msg_id,),
        )
        return _approval(row) if row is not None else None

    async def pending_approvals(self, conv_id: int | None = None) -> list[Approval]:
        """The unanswered approvals, of one conversation or (without `conv_id`) of all."""
        sql = "SELECT * FROM approvals WHERE status = ?"
        params: list[object] = [ApprovalStatus.PENDING.value]
        if conv_id is not None:
            sql += " AND conversation_id = ?"
            params.append(conv_id)
        return [_approval(row) for row in await self._all(sql + " ORDER BY id", params)]

    async def granted_kinds(self, conv_id: int) -> set[ApprovalKind]:
        """What the owner has approved for this conversation (for example sharing the phone)."""
        rows = await self._all(
            "SELECT DISTINCT kind FROM approvals WHERE conversation_id = ? AND status = ?",
            (conv_id, ApprovalStatus.APPROVED.value),
        )
        return {ApprovalKind(row["kind"]) for row in rows}

    async def set_approval_msg(self, approval_id: int, msg_id: int) -> None:
        """Remember the control-chat message that carries the approval's buttons."""
        await self._update("approvals", approval_id, {"control_msg_id": msg_id})

    async def approvals_of(self, conv_id: int) -> list[Approval]:
        """Every approval of the conversation, whatever its status, oldest first."""
        rows = await self._all(
            "SELECT * FROM approvals WHERE conversation_id = ? ORDER BY id", (conv_id,)
        )
        return [_approval(row) for row in rows]

    async def approvals_of_task(self, task_id: int) -> list[Approval]:
        """Every approval of the task, about a person or about the task as a whole, whatever
        its status, oldest first."""
        rows = await self._all(
            "SELECT * FROM approvals WHERE task_id = ? ORDER BY id", (task_id,)
        )
        return [_approval(row) for row in rows]

    async def set_approval_payload(self, approval_id: int, payload: dict) -> None:
        """Replace the approval's payload (to record what the owner approved, for example)."""
        await self._update("approvals", approval_id, {"payload_json": _dumps(payload)})

    # -- outbox --

    async def enqueue(
        self,
        conv_id: int,
        text: str,
        not_before: datetime,
        attach_path: str | None = None,
        *,
        wake_id: int | None = None,
    ) -> int:
        """Queue a message; it is never sent before `not_before`. The agent core names the
        wake that queued it."""
        return await self._insert(
            "INSERT INTO outbox (conversation_id, text, attach_path, status, not_before,"
            " created_at, wake_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                conv_id,
                text,
                attach_path,
                OutboxStatus.PENDING.value,
                _utc(not_before),
                self._now(),
                wake_id,
            ),
        )

    async def due_outbox(self, now: datetime) -> list[OutboxItem]:
        """The waiting messages whose time has come, the earliest first."""
        rows = await self._all(
            "SELECT * FROM outbox WHERE status = ? AND not_before <= ? ORDER BY not_before, id",
            (OutboxStatus.PENDING.value, _utc(now)),
        )
        return [_outbox_item(row) for row in rows]

    async def mark_outbox(
        self,
        outbox_id: int,
        status: OutboxStatus,
        *,
        tg_msg_id: int | None = None,
        error: str | None = None,
        at: datetime | None = None,
    ) -> None:
        """Set the item's status along with its outcome: the Telegram id, the error, the send time.

        The outcome fields are written as given, so a value left out is cleared.
        """
        await self._update(
            "outbox",
            outbox_id,
            {
                "status": OutboxStatus(status).value,
                "tg_msg_id": tg_msg_id,
                "error": error,
                "sent_at": _maybe(_utc, at),
            },
        )

    async def get_outbox(self, outbox_id: int) -> OutboxItem:
        return _outbox_item(await self._get("outbox", outbox_id))

    async def claim_outbox(self, outbox_id: int) -> bool:
        """Take a waiting message for sending: PENDING to SENDING, in one step.

        False if the message is no longer waiting (cancelled meanwhile, or taken already). The
        sender sends only what it has claimed, so a message that was cancelled or is being sent
        by another caller is never sent twice, and a cancelled one never comes back to life.
        """
        changed = await self._run(
            "UPDATE outbox SET status = ? WHERE id = ? AND status = ?",
            (OutboxStatus.SENDING.value, outbox_id, OutboxStatus.PENDING.value),
        )
        return changed == 1

    async def reschedule_outbox(self, outbox_id: int, not_before: datetime) -> bool:
        """Move a waiting message to a later send time (out of hours, or a Telegram flood wait).

        False if the message is not waiting (any other status is left as it is).
        """
        changed = await self._run(
            "UPDATE outbox SET not_before = ? WHERE id = ? AND status = ?",
            (_utc(not_before), outbox_id, OutboxStatus.PENDING.value),
        )
        return changed == 1

    async def cancel_pending(self, conv_id: int) -> int:
        """Cancel the conversation's waiting messages (not those already being sent)."""
        return await self._run(
            "UPDATE outbox SET status = ? WHERE conversation_id = ? AND status = ?",
            (OutboxStatus.CANCELLED.value, conv_id, OutboxStatus.PENDING.value),
        )

    async def cancel_pending_after(self, conv_id: int, keep_wake_id: int | None) -> int:
        """Cancel the conversation's waiting messages queued by a wake other than
        `keep_wake_id`; those queued by no wake (a reply the owner approved) stay."""
        return await self._run(
            "UPDATE outbox SET status = ? WHERE conversation_id = ? AND status = ?"
            " AND wake_id IS NOT NULL AND wake_id IS NOT ?",
            (
                OutboxStatus.CANCELLED.value,
                conv_id,
                OutboxStatus.PENDING.value,
                keep_wake_id,
            ),
        )

    async def cancel_outbox(self, outbox_id: int) -> bool:
        """Cancel one waiting message; False if it is not waiting any more (the Sender may
        have claimed it meanwhile, and a message on its way is not called back)."""
        changed = await self._run(
            "UPDATE outbox SET status = ? WHERE id = ? AND status = ?",
            (OutboxStatus.CANCELLED.value, outbox_id, OutboxStatus.PENDING.value),
        )
        return changed == 1

    async def mark_outbox_tried(self, outbox_id: int, at: datetime) -> None:
        """Remember when a send of the item broke off (the agent core checks the chat for an
        UNKNOWN item a while after that)."""
        await self._update("outbox", outbox_id, {"tried_at": _utc(at)})

    async def outbox_in(self, status: OutboxStatus) -> list[OutboxItem]:
        rows = await self._all(
            "SELECT * FROM outbox WHERE status = ? ORDER BY id",
            (OutboxStatus(status).value,),
        )
        return [_outbox_item(row) for row in rows]

    async def outbox_of(self, conv_id: int) -> list[OutboxItem]:
        """Every item queued for the conversation, whatever its status, in the order queued."""
        rows = await self._all(
            "SELECT * FROM outbox WHERE conversation_id = ? ORDER BY id", (conv_id,)
        )
        return [_outbox_item(row) for row in rows]

    async def sent_texts(self, conv_id: int | None = None) -> list[str]:
        """The texts that went out, in the order they were sent; of one conversation or all."""
        sql = "SELECT text FROM outbox WHERE status = ?"
        params: list[object] = [OutboxStatus.SENT.value]
        if conv_id is not None:
            sql += " AND conversation_id = ?"
            params.append(conv_id)
        return [
            row["text"]
            for row in await self._all(sql + " ORDER BY sent_at, id", params)
        ]

    # -- spending --

    async def log_llm_call(
        self,
        conv_id: int | None,
        purpose: str,
        model: str,
        usage: Usage,
        cost: float,
        stop_reason: str,
        *,
        task_id: int | None = None,
        wake_id: int | None = None,
    ) -> None:
        """Record one model call and its cost in USD (`conv_id` is None outside a conversation).

        The agent core names the task and the wake the call was made for.
        """
        await self._run(
            "INSERT INTO llm_calls (conversation_id, purpose, model, input_tokens, output_tokens,"
            " cache_read, cache_write, cost_usd, stop_reason, created_at, task_id, wake_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                conv_id,
                purpose,
                model,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_read,
                usage.cache_write,
                cost,
                stop_reason,
                self._now(),
                task_id,
                wake_id,
            ),
        )

    async def task_cost(self, task_id: int) -> float:
        """USD spent on the model calls logged with this task (the agent core's)."""
        row = await self._one(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS total FROM llm_calls WHERE task_id = ?",
            (task_id,),
        )
        return row["total"]

    async def wake_cost(self, wake_id: int, purpose: str) -> float:
        """USD spent on the model calls of this purpose logged with this wake."""
        row = await self._one(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS total FROM llm_calls"
            " WHERE wake_id = ? AND purpose = ?",
            (wake_id, purpose),
        )
        return row["total"]

    async def cost_since(self, since: datetime) -> float:
        """USD spent on model calls made at or after `since`."""
        row = await self._one(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS total FROM llm_calls WHERE created_at >= ?",
            (_utc(since),),
        )
        return row["total"]

    # -- events, flags and statistics --

    async def log_event(
        self, type: str, conv_id: int | None = None, **payload: object
    ) -> None:
        """Append to the event log; `payload` must be JSON-serialisable."""
        await self._run(
            "INSERT INTO events (ts, type, conversation_id, payload_json) VALUES (?, ?, ?, ?)",
            (self._now(), type, conv_id, _dumps(payload)),
        )

    async def events_of(self, type: str, conv_id: int | None = None) -> list[dict]:
        """The payloads of the logged events of this type, of one conversation or of all.

        Oldest first.
        """
        sql = "SELECT payload_json FROM events WHERE type = ?"
        params: list[object] = [type]
        if conv_id is not None:
            sql += " AND conversation_id = ?"
            params.append(conv_id)
        rows = await self._all(sql + " ORDER BY id", params)
        return [json.loads(row["payload_json"]) for row in rows]

    async def searches_of(self, task_id: int, kind: str = "research") -> list[dict]:
        """The payloads of this task's events of `kind`, oldest first: `research` is written
        when a search of its agent starts (`agent/tools.ToolBox`), `research_done` when it
        returns, with what it found, its cost and whether it paid for a model call. The
        search caps count them (ruling R27)."""
        rows = await self._all(
            "SELECT payload_json FROM events WHERE type = ?"
            " AND json_extract(payload_json, '$.task_id') = ? ORDER BY id",
            (kind, task_id),
        )
        return [json.loads(row["payload_json"]) for row in rows]

    async def get_kv(self, key: str) -> str | None:
        row = await self._one("SELECT value FROM kv WHERE key = ?", (key,))
        return None if row is None else row["value"]

    async def kv_keys(self, prefix: str) -> list[str]:
        """The keys that start with `prefix` and hold a value (not ""), in order."""
        rows = await self._all(
            "SELECT key FROM kv WHERE substr(key, 1, ?) = ? AND value != '' ORDER BY key",
            (len(prefix), prefix),
        )
        return [row["key"] for row in rows]

    async def set_kv(self, key: str, value: str) -> None:
        await self._run(
            "INSERT INTO kv (key, value) VALUES (?, ?)"
            " ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    async def is_stopped(self) -> bool:
        """Has the owner stopped the agent? The flag survives a restart."""
        return await self.get_kv(_STOPPED_KEY) == "1"

    async def set_stopped(self, flag: bool, *, why: str = "") -> None:
        """Stop the agent or let it work again. `why` names what stopped it when that was
        not the owner (Telegram restricted the account): the board then warns him before he
        resumes. It is forgotten when the agent works again."""
        await self.set_kv(_STOPPED_KEY, "1" if flag else "0")
        await self.set_kv(_STOPPED_WHY_KEY, why if flag else "")

    async def stopped_why(self) -> str:
        """What stopped the agent, if it was not the owner's own /stop; "" otherwise."""
        return await self.get_kv(_STOPPED_WHY_KEY) or ""

    async def new_contacts_on(self, day: date) -> int:
        """How many new people were written to for the first time on this calendar day
        (Tashkent)."""
        start = datetime.combine(day, time.min, tzinfo=TASHKENT)
        end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=TASHKENT)
        row = await self._one(
            "SELECT COUNT(*) AS n FROM events WHERE type = ? AND ts >= ? AND ts < ?",
            (FIRST_SENT, _utc(start), _utc(end)),
        )
        return row["n"]

    async def last_first_contact_at(self) -> datetime | None:
        """When a new person was last written to for the first time (the newest
        `first_sent`)."""
        row = await self._one(
            "SELECT MAX(ts) AS ts FROM events WHERE type = ?", (FIRST_SENT,)
        )
        return _maybe(_moment, row["ts"])

    async def was_written_to(self, contact_id: int) -> bool:
        """Has the account written to this person in some conversation already (one of them
        has `first_sent`)?"""
        row = await self._one(
            "SELECT EXISTS (SELECT 1 FROM conversations WHERE contact_id = ?"
            " AND first_sent = 1) AS written",
            (contact_id,),
        )
        return bool(row["written"])

    async def waiting_first_contacts(
        self, exclude_conv_id: int | None = None
    ) -> list[datetime]:
        """The first messages to new people that wait in the outbox: for each conversation
        never written to, whose person never wrote to us either and was never written to in
        another conversation (`tasks/first_contacts.is_first_contact`), that has a PENDING
        item (but `exclude_conv_id`), the earliest `not_before` of its items; sorted."""
        rows = await self._all(
            "SELECT MIN(o.not_before) AS at FROM outbox o"
            " JOIN conversations c ON c.id = o.conversation_id"
            " WHERE o.status = ? AND c.first_sent = 0 AND c.last_in_at IS NULL"
            " AND NOT EXISTS (SELECT 1 FROM conversations w"
            " WHERE w.contact_id = c.contact_id AND w.first_sent = 1)"
            " AND c.id IS NOT ?"
            " GROUP BY o.conversation_id ORDER BY at",
            (OutboxStatus.PENDING.value, exclude_conv_id),
        )
        return [_moment(row["at"]) for row in rows]

    async def first_message_slots(self, conv_ids: Sequence[int]) -> dict[int, datetime]:
        """For each of these conversations that has PENDING outbox items, the earliest
        `not_before` of them: when its queued messages go (the board shows it for a person
        whose first message waits for its paced slot). One query for all of them."""
        if not conv_ids:
            return {}
        marks = ", ".join("?" for _ in conv_ids)
        rows = await self._all(
            "SELECT conversation_id, MIN(not_before) AS at FROM outbox"
            f" WHERE status = ? AND conversation_id IN ({marks})"
            " GROUP BY conversation_id",
            (OutboxStatus.PENDING.value, *conv_ids),
        )
        return {row["conversation_id"]: _moment(row["at"]) for row in rows}
