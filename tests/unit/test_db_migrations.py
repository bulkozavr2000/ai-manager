"""Schema versions: a fresh database, the upgrade of an old one, its backup and its rollback.

The live database of the owner is upgraded by this code, so these tests build an old database
the way the program before schema versions left it (`SCHEMA_V0`, `user_version = 0`) and check
that every row survives and that a backup exists before anything changes.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from contextlib import closing
from datetime import timedelta
from pathlib import Path

import aiosqlite
import pytest

from tg_agent import db as db_module
from tg_agent.db import LATEST_VERSION, SCHEMA_V0, MigrationError, Repo

V0_TABLES = (
    "tasks",
    "contacts",
    "conversations",
    "messages",
    "approvals",
    "outbox",
    "timers",
    "llm_calls",
    "events",
    "kv",
)
NEW_TABLES = ("wakes", "agent_events", "owner_chat", "task_candidates")
NEW_COLUMNS = {
    "tasks": ("notes", "headline", "source_text"),
    "conversations": ("note", "close_reason"),
    "outbox": ("wake_id", "tried_at"),
    "llm_calls": ("task_id", "wake_id"),
}
T = "2026-11-01T05:00:00+00:00"


def raw(db_path: Path, sql: str, params: tuple = ()) -> list[tuple]:
    """Read the file through a plain sqlite3 connection: what is really stored."""
    with closing(sqlite3.connect(db_path)) as db:
        return db.execute(sql, params).fetchall()


def user_version(db_path: Path) -> int:
    return raw(db_path, "PRAGMA user_version")[0][0]


def tables(db_path: Path) -> set[str]:
    return {
        name
        for (name,) in raw(
            db_path, "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def columns(db_path: Path, table: str) -> list[str]:
    return [name for _, name, *_ in raw(db_path, f"PRAGMA table_info({table})")]


def row_counts(db_path: Path) -> dict[str, int]:
    return {
        table: raw(db_path, f"SELECT COUNT(*) FROM {table}")[0][0]
        for table in V0_TABLES
    }


def backups(db_path: Path) -> list[Path]:
    folder = db_path.parent / "backup"
    return sorted(folder.iterdir()) if folder.exists() else []


def make_v0_database(db_path: Path) -> None:
    """A database as the program before schema versions left it: V0 tables, one row in each."""
    with closing(sqlite3.connect(db_path)) as db:
        db.executescript(SCHEMA_V0)
        db.executescript(
            f"""
            INSERT INTO tasks (id, title, goal, item_json, ceiling, currency, allowed_json,
                attachments_json, owner_notes, status, card_msg_id, created_at, updated_at)
            VALUES (1, 'Дрель', 'самое дешёвое', '{{"товар": "дрель"}}', 500000.0, 'UZS',
                '["name"]', '[]', 'заметка', 'active', 77, '{T}', '{T}');
            INSERT INTO contacts (id, peer_id, username, display_name, is_bot, language,
                card_json, do_not_contact, updated_at)
            VALUES (1, 1001, 'seller', 'Продавец', 0, 'ru', '{{}}', 0, '{T}');
            INSERT INTO conversations (id, task_id, contact_id, state, prev_state, facts_json,
                summary, language, offtopic_strikes, agent_messages, followups_sent,
                first_sent, paused_until, last_in_at, last_out_at, created_at, updated_at)
            VALUES (1, 1, 1, 'talking', NULL, '{{"цена": 450000}}', 'торгуемся', 'ru', 0, 2,
                1, 1, NULL, '{T}', '{T}', '{T}', '{T}');
            INSERT INTO messages (conversation_id, direction, source, tg_msg_id, kind, text,
                created_at, edited_at)
            VALUES (1, 'in', 'counterpart', 501, 'text', 'Здравствуйте', '{T}', NULL);
            INSERT INTO approvals (task_id, conversation_id, kind, payload_json, status, nonce,
                control_msg_id, answer, created_at, resolved_at)
            VALUES (1, 1, 'question', '{{}}', 'pending', 'abc123', NULL, NULL, '{T}', NULL);
            INSERT INTO outbox (id, conversation_id, text, attach_path, status, not_before,
                created_at, sent_at, tg_msg_id, error)
            VALUES (1, 1, 'Добрый день', NULL, 'sent', '{T}', '{T}', '{T}', 502, NULL);
            INSERT INTO timers (conversation_id, task_id, kind, due_at, payload_json, status)
            VALUES (1, 1, 'followup', '{T}', '{{}}', 'pending');
            INSERT INTO llm_calls (conversation_id, purpose, model, input_tokens,
                output_tokens, cache_read, cache_write, cost_usd, stop_reason, created_at)
            VALUES (1, 'decide', 'claude', 100, 20, 0, 0, 0.01, 'end_turn', '{T}');
            INSERT INTO events (ts, type, conversation_id, payload_json)
            VALUES ('{T}', 'first_sent', 1, '{{}}');
            INSERT INTO kv (key, value) VALUES ('stopped', '0');
            """
        )
        _add_old_engine_rows(db)
        db.commit()
    assert user_version(db_path) == 0


def _add_old_engine_rows(db: sqlite3.Connection) -> None:
    """What the old engine left besides the one row of each table: a task in each of its
    statuses, people in its states, open cards, queued messages, timers and its kv keys."""
    for task_id, status in enumerate(
        ("draft", "deciding", "ordered", "done", "cancelled", "paused"), start=2
    ):
        db.execute(
            "INSERT INTO tasks (id, title, goal, item_json, allowed_json, attachments_json,"
            " status, created_at, updated_at) VALUES (?, ?, '', '{}', '[]', '[]', ?, ?, ?)",
            (task_id, f"Задача {task_id}", status, T, T),
        )
    for contact_id in range(2, 10):
        db.execute(
            "INSERT INTO contacts (id, peer_id, display_name, is_bot, updated_at)"
            " VALUES (?, ?, ?, 0, ?)",
            (contact_id, 1000 + contact_id, f"Человек {contact_id}", T),
        )
    for conv_id, (task_id, contact_id, state, prev_state) in OLD_CONVERSATIONS.items():
        db.execute(
            "INSERT INTO conversations (id, task_id, contact_id, state, prev_state,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (conv_id, task_id, contact_id, state, prev_state, T, T),
        )
    db.executemany(
        "INSERT INTO approvals (id, task_id, conversation_id, kind, payload_json, status,"
        " nonce, answer, created_at, resolved_at) VALUES (?, 1, ?, ?, '{}', ?, ?, ?, ?, ?)",
        [
            (2, None, "compare", "pending", "n2", None, T, None),
            (3, 3, "share_phone", "approved", "n3", None, T, T),
            (4, 2, "alert", "rejected", "n4", None, T, T),
            (5, 6, "unclear", "pending", "n5", None, T, None),
            (6, 3, "question", "answered", "n6", "да", T, T),
        ],
    )
    db.executemany(
        "INSERT INTO outbox (id, conversation_id, text, status, not_before, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        [
            (2, 2, "ждёт", "pending", T, T),
            (3, 3, "уходит", "sending", T, T),
            (4, 3, "не знаю, ушло ли", "unknown", T, T),
            (5, 6, "не ушло", "failed", T, T),
            (6, 6, "отменено", "cancelled", T, T),
        ],
    )
    db.executemany(
        "INSERT INTO timers (conversation_id, task_id, kind, due_at, status)"
        " VALUES (?, ?, ?, ?, ?)",
        [(2, 1, "decide", T, "done"), (None, 1, "compare", T, "pending")],
    )
    db.executemany(
        "INSERT INTO kv (key, value) VALUES (?, ?)",
        [
            ("board_text", "Статус задач"),
            ("board_msg_id", "900"),
            ("task_nonce:1", "aa11"),
            ("finish_nonce:1", "bb22"),
            ("reply_terms:3", "{}"),
            ("found_msg:901", "1"),
            ("task_card:902", "1"),
            ("digest_sent_on", "2026-11-01"),
        ],
    )


# conv id: (task id, contact id, state, prev_state); conversation 1 is make_v0_database's own
# (task 1, `talking`).
OLD_CONVERSATIONS = {
    2: (1, 2, "ready", None),
    3: (1, 3, "waiting_owner", None),
    4: (4, 4, "ordered", None),
    5: (5, 5, "done", None),
    6: (1, 6, "no_response", None),
    7: (1, 7, "declined", None),
    8: (1, 8, "closed_offtopic", None),
    9: (1, 9, "paused", "ready"),
    10: (6, 2, "draft", None),
    11: (3, 3, "tracking", None),
    12: (3, 4, "handed_to_owner", None),
}
MIGRATED = "закрыто при переходе на новую версию"
CLOSE_REASONS = {
    1: MIGRATED,
    2: MIGRATED,
    3: MIGRATED,
    4: MIGRATED,
    5: "завершено",
    6: "не ответил",
    7: "отказ",
    8: "разговор не по делу",
    9: MIGRATED,
    10: MIGRATED,
    11: MIGRATED,
    12: MIGRATED,
}
REMAINING_KINDS = {
    "question",
    "confirm_order",
    "share_phone",
    "share_address",
    "handoff",
}
RETIRED_KV = (
    "board_text",
    "task_nonce:1",
    "finish_nonce:1",
    "reply_terms:3",
    "found_msg:901",
)
SQLITE_NOW = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00$")


def without_retired_kv(counts: dict[str, int]) -> dict[str, int]:
    """The row counts of a version-0 database after migration 2: the same in every table
    but kv, which loses only the old engine's keys."""
    return {**counts, "kv": counts["kv"] - len(RETIRED_KV)}


def make_v1_database(db_path: Path) -> None:
    """A version-0 database brought to version 1 by migration 1 alone, as the owner's is now,
    with an open agent event and an unfinished wake the old engine could not have left."""
    make_v0_database(db_path)
    with closing(sqlite3.connect(db_path, isolation_level=None)) as db:
        db.execute("BEGIN")
        for statement in db_module.MIGRATIONS[1]:
            db.execute(statement)
        db.execute("INSERT INTO wakes (id, task_id, started_at) VALUES (1, 1, ?)", (T,))
        db.execute(
            "INSERT INTO wakes (id, task_id, started_at, finished_at, outcome, steps)"
            " VALUES (2, 1, ?, ?, 'done', 3)",
            (T, T),
        )
        db.execute(
            "INSERT INTO agent_events (id, task_id, conversation_id, kind, created_at, due_at)"
            " VALUES (1, 1, 1, 'incoming', ?, ?)",
            (T, T),
        )
        db.execute(
            "INSERT INTO agent_events (id, task_id, kind, created_at, due_at, handled_at)"
            " VALUES (2, 1, 'created', ?, ?, ?)",
            (T, T, T),
        )
        db.execute("PRAGMA user_version = 1")
        db.execute("COMMIT")
    assert user_version(db_path) == 1


def dump(db_path: Path) -> dict[str, list[tuple]]:
    """Every row of every table, to compare a database before and after something."""
    return {
        table: raw(db_path, f"SELECT * FROM {table} ORDER BY rowid")
        for table in sorted(tables(db_path))
        if table != "sqlite_sequence"
    }


def assert_old_engine_data_closed(db_path: Path) -> None:
    """Each rule of migration 2 holds for the rows `make_v0_database` wrote."""
    tasks = dict(raw(db_path, "SELECT id, status FROM tasks"))
    assert tasks == {
        1: "cancelled",
        2: "cancelled",
        3: "cancelled",
        4: "cancelled",
        5: "done",
        6: "cancelled",
        7: "cancelled",
    }
    for task_id, updated_at in raw(db_path, "SELECT id, updated_at FROM tasks"):
        if task_id in (5, 6):  # done and cancelled already: untouched
            assert updated_at == T, task_id
        else:
            assert SQLITE_NOW.match(updated_at) and updated_at != T, task_id

    conversations = {
        conv_id: (state, prev_state, reason)
        for conv_id, state, prev_state, reason in raw(
            db_path, "SELECT id, state, prev_state, close_reason FROM conversations"
        )
    }
    assert {conv_id: state for conv_id, (state, _, _) in conversations.items()} == {
        conv_id: "closed" for conv_id in CLOSE_REASONS
    }
    assert {
        conv_id: reason for conv_id, (_, _, reason) in conversations.items()
    } == CLOSE_REASONS
    # The old PAUSED remembered a state the new machine does not know.
    assert conversations[9][1] is None

    approvals = {
        approval_id: (kind, status, answer, resolved_at)
        for approval_id, kind, status, answer, resolved_at in raw(
            db_path, "SELECT id, kind, status, answer, resolved_at FROM approvals"
        )
    }
    for approval_id in (1, 2, 5):  # were pending
        assert approvals[approval_id][1] == "expired", approval_id
        assert SQLITE_NOW.match(approvals[approval_id][3]), approval_id
    assert (approvals[2][0], approvals[5][0]) == ("question", "question")
    assert approvals[3] == ("share_phone", "approved", None, T)  # settled: unchanged
    assert approvals[4] == ("question", "rejected", None, T)  # was `alert`
    assert approvals[6] == ("question", "answered", "да", T)
    assert {kind for kind, *_ in approvals.values()} <= REMAINING_KINDS

    assert dict(raw(db_path, "SELECT id, status FROM outbox")) == {
        1: "sent",
        2: "cancelled",
        3: "cancelled",
        4: "cancelled",
        5: "failed",
        6: "cancelled",
    }
    assert raw(
        db_path, "SELECT id, text, tg_msg_id, sent_at FROM outbox WHERE id = 1"
    ) == [(1, "Добрый день", 502, T)]
    assert sorted(raw(db_path, "SELECT kind, status FROM timers")) == [
        ("compare", "cancelled"),
        ("decide", "done"),
        ("followup", "cancelled"),
    ]

    kv = dict(raw(db_path, "SELECT key, value FROM kv"))
    assert not set(RETIRED_KV) & kv.keys()
    assert kv == {
        "stopped": "0",
        "board_msg_id": "900",
        "task_card:902": "1",
        "digest_sent_on": "2026-11-01",
    }
    indexes = {
        name: sql
        for name, sql in raw(
            db_path, "SELECT name, sql FROM sqlite_master WHERE type = 'index'"
        )
    }
    assert any(sql and "ON events (type, ts)" in sql for sql in indexes.values()), (
        indexes
    )


async def test_a_fresh_file_database_is_created_at_the_latest_version(
    db_path, fake_clock
):
    repo = await Repo.open(db_path, clock=fake_clock)
    await repo.close()

    assert user_version(db_path) == LATEST_VERSION
    assert tables(db_path) >= set(V0_TABLES) | set(NEW_TABLES)
    for table, new_columns in NEW_COLUMNS.items():
        assert set(new_columns) <= set(columns(db_path, table)), table
    assert not (
        db_path.parent / "backup"
    ).exists()  # nothing to lose, nothing to back up


async def test_a_v0_database_upgrades_without_losing_a_row_after_a_backup(
    db_path, fake_clock
):
    make_v0_database(db_path)
    before = row_counts(db_path)

    repo = await Repo.open(db_path, clock=fake_clock)
    try:
        task = await repo.get_task(1)
        conversation = await repo.get_conversation(1)
        outbox = await repo.outbox_of(1)
    finally:
        await repo.close()

    assert user_version(db_path) == LATEST_VERSION
    assert row_counts(db_path) == without_retired_kv(before)
    assert (task.title, task.owner_notes, task.card_msg_id) == ("Дрель", "заметка", 77)
    assert (task.notes, task.headline, task.source_text) == ("", "", "")
    assert (conversation.summary, conversation.agent_messages) == ("торгуемся", 2)
    # Migration 2 closed the old engine's conversation.
    assert (conversation.note, conversation.close_reason) == ("", MIGRATED)
    assert [(item.text, item.tg_msg_id) for item in outbox] == [("Добрый день", 502)]
    assert (outbox[0].wake_id, outbox[0].tried_at) == (None, None)

    # One backup, named by the version it holds and the clock's moment in UTC
    # (10:00 in Tashkent is 05:00 UTC), and it is the old database untouched.
    [backup] = backups(db_path)
    assert backup.name == "agent-v0-20261110-050000.db"
    assert user_version(backup) == 0
    assert row_counts(backup) == before
    assert not tables(backup) & set(NEW_TABLES)


async def test_opening_a_migrated_database_again_changes_nothing(db_path, fake_clock):
    make_v0_database(db_path)
    repo = await Repo.open(db_path, clock=fake_clock)
    await repo.close()
    fake_clock.advance(timedelta(hours=1))  # a second backup would get its own name

    repo = await Repo.open(db_path, clock=fake_clock)
    await repo.close()

    assert user_version(db_path) == LATEST_VERSION
    assert len(backups(db_path)) == 1


async def test_an_existing_backup_is_never_overwritten(db_path, fake_clock):
    make_v0_database(db_path)
    folder = db_path.parent / "backup"
    folder.mkdir()
    taken = folder / "agent-v0-20261110-050000.db"
    taken.write_bytes(b"an earlier backup")

    repo = await Repo.open(db_path, clock=fake_clock)
    await repo.close()

    assert taken.read_bytes() == b"an earlier backup"
    assert [path.name for path in backups(db_path)] == [
        "agent-v0-20261110-050000-1.db",
        "agent-v0-20261110-050000.db",
    ]
    assert user_version(folder / "agent-v0-20261110-050000-1.db") == 0


async def test_a_failing_migration_rolls_back_entirely(
    db_path, fake_clock, monkeypatch
):
    # A database at the latest version.
    repo = await Repo.open(db_path, clock=fake_clock)
    await repo.close()
    next_version = LATEST_VERSION + 1
    monkeypatch.setitem(
        db_module.MIGRATIONS,
        next_version,
        ("CREATE TABLE half_done (id INTEGER)", "THIS IS NOT SQL"),
    )
    monkeypatch.setattr(db_module, "LATEST_VERSION", next_version)

    with pytest.raises(MigrationError) as failure:
        await Repo.open(db_path, clock=fake_clock)

    assert failure.value.version == next_version
    assert isinstance(failure.value.__cause__, sqlite3.Error)
    assert user_version(db_path) == LATEST_VERSION
    assert "half_done" not in tables(db_path)


async def test_a_database_newer_than_the_program_is_refused_untouched(
    db_path, fake_clock
):
    make_v0_database(db_path)
    with closing(sqlite3.connect(db_path)) as db:
        db.execute("PRAGMA user_version = 99")

    with pytest.raises(MigrationError) as failure:
        await Repo.open(db_path, clock=fake_clock)

    assert failure.value.version == 99
    assert user_version(db_path) == 99
    assert not tables(db_path) & set(NEW_TABLES)
    assert not (db_path.parent / "backup").exists()


async def test_an_in_memory_database_works_at_the_latest_version(fake_clock):
    repo = await Repo.open(":memory:", clock=fake_clock)
    try:
        [(version,)] = await repo._db.execute_fetchall("PRAGMA user_version")
        task_id = await repo.create_task("Дрель", "цель", {}, None, "UZS", [], [])
        contact_id = await repo.upsert_contact(1001, None, "Продавец", False)
        conv_id = await repo.create_conversation(task_id, contact_id)
        await repo.update_conversation(conv_id, note="ждёт фото", close_reason="купили")
        conversation = await repo.get_conversation(conv_id)
    finally:
        await repo.close()

    assert version == LATEST_VERSION
    assert (conversation.note, conversation.close_reason) == ("ждёт фото", "купили")


async def test_a_dropped_silence_event_does_not_block_the_next_one(db_path, fake_clock):
    """A person who answers ends the silence (the event is handled, never claimed); our next
    message arms a new one. Two open ones for one conversation are still refused."""
    make_v0_database(db_path)
    repo = await Repo.open(db_path, clock=fake_clock)
    await repo.close()
    insert = (
        "INSERT INTO agent_events (task_id, conversation_id, kind, created_at, due_at,"
        " handled_at) VALUES (1, 1, 'silence', ?, ?, ?)"
    )

    with closing(sqlite3.connect(db_path)) as db:
        db.execute(insert, (T, T, T))  # armed, then dropped
        db.execute(insert, (T, T, None))  # armed again after our next message
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(insert, (T, T, None))
        db.commit()

    assert raw(db_path, "SELECT COUNT(*) FROM agent_events") == [(2,)]


async def test_an_upgrade_is_written_to_the_program_log(db_path, fake_clock, caplog):
    make_v0_database(db_path)

    with caplog.at_level(logging.INFO, logger="tg_agent.db"):
        repo = await Repo.open(db_path, clock=fake_clock)
        await repo.close()

    [line] = [r.getMessage() for r in caplog.records if r.name == "tg_agent.db"]
    assert f"0 -> {LATEST_VERSION}" in line
    assert str(db_path.parent / "backup") in line


async def test_a_fresh_database_and_an_up_to_date_one_log_nothing(
    db_path, fake_clock, caplog
):
    with caplog.at_level(logging.INFO, logger="tg_agent.db"):
        for _ in range(2):
            repo = await Repo.open(db_path, clock=fake_clock)
            await repo.close()

    assert not [r for r in caplog.records if r.name == "tg_agent.db"]


# --- migration 2: what the old engine left is closed ---------------------------------------


def test_the_latest_version_is_2():
    assert LATEST_VERSION == 2


async def test_a_v0_database_with_the_old_engines_data_is_closed_by_migration_2(
    db_path, fake_clock
):
    make_v0_database(db_path)
    before = row_counts(db_path)

    repo = await Repo.open(db_path, clock=fake_clock)
    await repo.close()

    assert user_version(db_path) == 2
    assert row_counts(db_path) == without_retired_kv(before)  # no table lost a row
    assert_old_engine_data_closed(db_path)


async def test_after_migration_2_every_row_reads_through_the_repo(db_path, fake_clock):
    make_v0_database(db_path)

    repo = await Repo.open(db_path, clock=fake_clock)
    try:
        tasks = await repo.all_tasks()
        conversations = [
            conv for task in tasks for conv in await repo.task_conversations(task.id)
        ]
        approvals = [
            approval
            for task in tasks
            for approval in await repo.approvals_of_task(task.id)
        ]
        for conv in conversations:
            await repo.approvals_of(conv.id)
        outbox = [
            item for conv in conversations for item in await repo.outbox_of(conv.id)
        ]
    finally:
        await repo.close()

    assert len(tasks) == 7
    assert len(conversations) == 12
    assert len(approvals) == 6
    assert len(outbox) == 6


async def test_a_v1_database_is_closed_the_same_way_and_a_second_start_changes_nothing(
    db_path, fake_clock
):
    make_v1_database(db_path)
    before = row_counts(db_path)

    repo = await Repo.open(db_path, clock=fake_clock)
    await repo.close()

    assert user_version(db_path) == 2
    assert row_counts(db_path) == without_retired_kv(before)
    assert_old_engine_data_closed(db_path)
    [backup] = backups(db_path)
    assert backup.name == "agent-v1-20261110-050000.db"
    assert user_version(backup) == 1
    # An open event is handled and an unfinished wake finished; the settled ones stay.
    events = raw(db_path, "SELECT id, handled_at FROM agent_events ORDER BY id")
    assert SQLITE_NOW.match(events[0][1])
    assert events[1] == (2, T)
    wakes = raw(
        db_path, "SELECT id, finished_at, outcome, steps FROM wakes ORDER BY id"
    )
    assert SQLITE_NOW.match(wakes[0][1])
    assert wakes[0][2:] == ("migrated", 0)
    assert wakes[1] == (2, T, "done", 3)

    upgraded = dump(db_path)
    fake_clock.advance(timedelta(hours=1))
    repo = await Repo.open(db_path, clock=fake_clock)
    await repo.close()

    assert dump(db_path) == upgraded
    assert len(backups(db_path)) == 1


async def test_a_conversation_closed_before_migration_2_keeps_its_reason(
    db_path, fake_clock
):
    make_v1_database(db_path)
    with closing(sqlite3.connect(db_path)) as db:
        db.execute(
            "UPDATE conversations SET state = 'closed', close_reason = 'купили' WHERE id = 5"
        )
        db.commit()

    repo = await Repo.open(db_path, clock=fake_clock)
    await repo.close()

    assert raw(
        db_path, "SELECT state, close_reason FROM conversations WHERE id = 5"
    ) == [("closed", "купили")]


async def test_a_failed_backup_stops_the_upgrade_and_leaves_no_partial_file(
    db_path, fake_clock, monkeypatch
):
    make_v1_database(db_path)
    before = dump(db_path)
    real = aiosqlite.Connection.execute_fetchall

    async def vacuum_breaks_halfway(self, sql, parameters=None):
        if sql.startswith("VACUUM INTO"):
            Path(parameters[0]).write_bytes(b"half a copy")
            raise sqlite3.OperationalError("disk I/O error")
        return await real(self, sql, parameters)

    monkeypatch.setattr(aiosqlite.Connection, "execute_fetchall", vacuum_breaks_halfway)

    with pytest.raises(MigrationError) as failure:
        await Repo.open(db_path, clock=fake_clock)

    assert failure.value.version == 1
    assert isinstance(failure.value.__cause__, sqlite3.OperationalError)
    assert user_version(db_path) == 1
    assert dump(db_path) == before  # nothing was migrated
    assert backups(db_path) == []  # the partial copy is gone


async def test_a_fresh_database_is_at_version_2_and_empty(db_path, fake_clock):
    repo = await Repo.open(db_path, clock=fake_clock)
    await repo.close()

    assert user_version(db_path) == 2
    assert all(rows == [] for rows in dump(db_path).values())
