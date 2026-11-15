"""The SQLite repository, exercised against a real temporary database (no mocks)."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import threading
import types
from contextlib import closing
from dataclasses import fields
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import get_args, get_origin, get_type_hints
from zoneinfo import ZoneInfo

import pytest

from tg_agent.brain.schemas import Usage
from tg_agent.db import NotFound, Repo
from tg_agent.models import (
    Approval,
    ApprovalKind,
    ApprovalStatus,
    Contact,
    Conversation,
    ConvState,
    Direction,
    EventKind,
    Message,
    MsgSource,
    OutboxItem,
    OutboxStatus,
    Task,
    TaskCandidate,
    TaskStatus,
)
from tg_agent.tasks.state import TERMINAL, InvalidTransition

S = ConvState
TASHKENT = ZoneInfo("Asia/Tashkent")


# --- helpers ------------------------------------------------------------------------------


def raw(db_path: Path, sql: str, params: tuple = ()) -> list[tuple]:
    """Read the database file through a plain sqlite3 connection: what is really stored."""
    with closing(sqlite3.connect(db_path)) as db:
        return db.execute(sql, params).fetchall()


async def make_task(
    repo: Repo, title: str = "Дрель", status: TaskStatus = TaskStatus.ACTIVE
) -> int:
    """A task, launched unless `status` says otherwise: a conversation is live only while its
    task is in work."""
    task_id = await repo.create_task(
        title,
        "самое дешёвое предложение",
        {"товар": "дрель"},
        500_000.0,
        "UZS",
        ["name"],
        [],
    )
    if status is not TaskStatus.DRAFT:
        await repo.set_task_status(task_id, status)
    return task_id


async def make_conversation(
    repo: Repo, peer_id: int = 1001, task_id: int | None = None
) -> int:
    """A new conversation (state PLANNED) with the contact that has this Telegram id."""
    if task_id is None:
        task_id = await make_task(repo)
    contact_id = await repo.upsert_contact(
        peer_id, f"user{peer_id}", f"Продавец {peer_id}", False
    )
    return await repo.create_conversation(task_id, contact_id)


async def walk(repo: Repo, conv_id: int, *states: ConvState) -> None:
    """Move a conversation through the given states, one legal step at a time."""
    for state in states:
        await repo.set_state(conv_id, state)


async def say(
    repo: Repo,
    conv_id: int,
    text: str,
    at: datetime,
    *,
    source: MsgSource = MsgSource.COUNTERPART,
    tg_msg_id: int | None = None,
) -> int:
    """Add a text message; anyone but the counterpart is outgoing."""
    direction = Direction.IN if source is MsgSource.COUNTERPART else Direction.OUT
    return await repo.add_message(
        conv_id, direction, source, "text", text, tg_msg_id, at
    )


def assert_matches_annotations(row: object) -> None:
    """Every field holds what its annotation promises: enum members, parsed JSON, aware UTC."""
    name_of_row = type(row).__name__
    for name, hint in get_type_hints(type(row)).items():
        value = getattr(row, name)
        options = get_args(hint) if get_origin(hint) is types.UnionType else (hint,)
        if value is None:
            assert type(None) in options, (
                f"{name_of_row}.{name} is None but not Optional"
            )
            continue
        expected = next(option for option in options if option is not type(None))
        expected = get_origin(expected) or expected  # list[str] -> list
        assert type(value) is expected, (
            f"{name_of_row}.{name}: {value!r} is not a {expected}"
        )
        if expected is datetime:
            assert value.utcoffset() == timedelta(0), f"{name_of_row}.{name} is not UTC"


# --- the database itself --------------------------------------------------------------------

TABLE_ROWS: dict[str, type] = {
    "tasks": Task,
    "contacts": Contact,
    "conversations": Conversation,
    "messages": Message,
    "approvals": Approval,
    "outbox": OutboxItem,
    "task_candidates": TaskCandidate,
}


@pytest.mark.parametrize("table", TABLE_ROWS)
def test_table_columns_follow_the_row_dataclass(repo, db_path, table):
    # The dataclass fields are the columns without the `_json` suffix (they hold the parsed
    # value), and a field is Optional exactly when its column may be NULL: model and schema
    # cannot drift apart without this test failing.
    row_class = TABLE_ROWS[table]
    columns = raw(
        db_path, f"PRAGMA table_info({table})"
    )  # (cid, name, type, notnull, default, pk)

    assert [name.removesuffix("_json") for _, name, *_ in columns] == [
        field.name for field in fields(row_class)
    ]

    hints = get_type_hints(row_class)
    for _, name, _type, notnull, _default, pk in columns:
        column_is_nullable = not notnull and not pk
        field_is_optional = type(None) in get_args(hints[name.removesuffix("_json")])
        assert column_is_nullable == field_is_optional, f"{table}.{name}"


async def test_foreign_keys_are_enforced_and_wal_is_on(repo, db_path, fake_clock):
    assert raw(db_path, "PRAGMA journal_mode") == [("wal",)]

    with pytest.raises(sqlite3.IntegrityError):
        await say(
            repo,
            9999,
            "a message of a conversation that does not exist",
            fake_clock.now(),
        )


async def test_data_and_stop_flag_survive_a_restart(db_path, fake_clock):
    repo = await Repo.open(db_path, clock=fake_clock)
    try:
        task_id = await make_task(repo)
        await repo.set_kv("digest_sent_on", "2026-11-10")
        await repo.set_kv("digest_sent_on", "2026-11-11")  # overwrites
        assert not await repo.is_stopped()
        await repo.set_stopped(True)
    finally:
        await repo.close()

    repo = await Repo.open(db_path, clock=fake_clock)
    try:
        assert (await repo.get_task(task_id)).title == "Дрель"
        assert await repo.get_kv("digest_sent_on") == "2026-11-11"
        assert await repo.get_kv("never_set") is None
        assert await repo.is_stopped()
        await repo.set_stopped(False)
        assert not await repo.is_stopped()
    finally:
        await repo.close()


async def test_a_card_of_a_kind_that_is_gone_does_not_stop_the_start(
    db_path, fake_clock
):
    # The simplification pass (S1): «Добавить в задачу?» is no more.
    repo = await Repo.open(db_path, clock=fake_clock)
    try:
        task_id = await make_task(repo)
        await repo._db.execute(
            "INSERT INTO approvals (task_id, kind, payload_json, status, nonce,"
            " created_at) VALUES (?, 'add_person', '{}', 'pending', 'abcdef', ?)",
            (task_id, repo._now()),
        )
    finally:
        await repo.close()

    repo = await Repo.open(db_path, clock=fake_clock)
    try:
        assert await repo.pending_approvals() == []
        [card] = await repo.approvals_of_task(task_id)
        assert (card.kind, card.status) == (
            ApprovalKind.QUESTION,
            ApprovalStatus.EXPIRED,
        )
    finally:
        await repo.close()


async def test_open_works_in_a_folder_with_a_cyrillic_name(tmp_path, fake_clock):
    # The project itself lives under a Cyrillic path on the owner's machine.
    folder = tmp_path / "Заметки"
    folder.mkdir()
    repo = await Repo.open(folder / "agent.db", clock=fake_clock)
    try:
        task_id = await make_task(repo)
        assert (await repo.get_task(task_id)).title == "Дрель"
    finally:
        await repo.close()
    assert (folder / "agent.db").exists()


async def test_a_failed_open_leaves_no_connection_thread_behind(tmp_path):
    # A leaked worker thread is not a daemon: the process would hang on exit.
    not_a_database = tmp_path / "broken.db"
    not_a_database.write_bytes(b"this is not a database" * 100)
    threads_before = threading.active_count()

    with pytest.raises(sqlite3.DatabaseError):
        await Repo.open(not_a_database)

    for _ in range(100):  # the thread may need a moment to wind down
        if threading.active_count() <= threads_before:
            break
        await asyncio.sleep(0.02)
    assert threading.active_count() <= threads_before


async def test_open_without_a_clock_uses_the_real_time(db_path):
    repo = await Repo.open(db_path)
    try:
        created_at = (await repo.get_task(await make_task(repo))).created_at
    finally:
        await repo.close()

    assert abs(datetime.now(UTC) - created_at) < timedelta(seconds=30)


# --- time and types ----------------------------------------------------------------------------


async def test_times_are_stored_as_utc_text_and_read_back_aware(repo, db_path):
    conv_id = await make_conversation(repo)
    local = datetime(2026, 11, 10, 15, 30, 15, 250_000, tzinfo=TASHKENT)

    await say(repo, conv_id, "привет", local)

    # Sortable text in UTC, whatever zone the caller used ...
    stored = raw(db_path, "SELECT created_at FROM messages")
    assert stored == [("2026-11-10T10:30:15.250000+00:00",)]
    # ... and the same moment, aware and in UTC, on the way back.
    [message] = await repo.last_messages(conv_id, 1)
    assert message.created_at == local
    assert message.created_at.utcoffset() == timedelta(0)


NAIVE = datetime(2026, 11, 10, 10, 0)

NAIVE_CALLS = [
    pytest.param(lambda r, c, o: say(r, c, "hi", NAIVE), id="add_message"),
    pytest.param(lambda r, c, o: r.edit_message(c, 1, "x", NAIVE), id="edit_message"),
    pytest.param(
        lambda r, c, o: r.purge_messages_before(NAIVE), id="purge_messages_before"
    ),
    pytest.param(lambda r, c, o: r.enqueue(c, "hi", NAIVE), id="enqueue"),
    pytest.param(lambda r, c, o: r.due_outbox(NAIVE), id="due_outbox"),
    pytest.param(
        lambda r, c, o: r.mark_outbox(o, OutboxStatus.SENT, at=NAIVE), id="mark_outbox"
    ),
    pytest.param(lambda r, c, o: r.reschedule_outbox(o, NAIVE), id="reschedule_outbox"),
    pytest.param(
        lambda r, c, o: r.update_conversation(c, paused_until=NAIVE),
        id="update_conversation",
    ),
    pytest.param(lambda r, c, o: r.cost_since(NAIVE), id="cost_since"),
]


@pytest.mark.parametrize("call", NAIVE_CALLS)
async def test_naive_datetimes_are_rejected(repo, fake_clock, call):
    conv_id = await make_conversation(repo)
    outbox_id = await repo.enqueue(conv_id, "queued", fake_clock.now())

    with pytest.raises(ValueError, match="naive"):
        await call(repo, conv_id, outbox_id)


async def test_rows_come_back_as_model_types(repo, db_path, fake_clock):
    # Everything the repository returns matches the annotations of its row dataclass: enum
    # members (never raw strings), parsed JSON, bool flags, aware UTC datetimes.
    now = fake_clock.now()
    task_id = await repo.create_task(
        "Дрель",
        "дешевле",
        {"товар": "дрель", "шт": 2},
        450.5,
        "UZS",
        ["name", "city"],
        ["a.jpg"],
    )
    await repo.set_task_status(task_id, TaskStatus.ACTIVE)

    contact_id = await repo.upsert_contact(1001, "seller", "Продавец", True)
    nameless_id = await repo.upsert_contact(1002, None, "Без имени", False)
    await repo.update_contact_card(contact_id, {"role": "магазин"}, "ru")

    conv_id = await repo.create_conversation(task_id, contact_id)
    await walk(repo, conv_id, S.CONTACTED, S.TALKING)
    await repo.update_conversation(
        conv_id,
        facts={"price": "450"},
        summary="цена 450",
        language="ru",
        offtopic_strikes=1,
        agent_messages=2,
        followups_sent=1,
        first_sent=True,
        paused_until=now + timedelta(hours=12),
        last_in_at=now,
        last_out_at=now,
        note="ждёт фото",
        close_reason="купили",
    )
    await say(repo, conv_id, "до правки", now, tg_msg_id=5)
    await repo.edit_message(conv_id, 5, "после правки", now)

    approval = await repo.create_approval(
        task_id, conv_id, ApprovalKind.QUESTION, {"q": "?"}
    )
    await repo.set_approval_msg(approval.id, 9)
    await repo.resolve_approval(
        approval.id, approval.nonce, ApprovalStatus.ANSWERED, "да"
    )

    sent_id = await repo.enqueue(conv_id, "hello", now, attach_path="a.jpg")
    await repo.mark_outbox(sent_id, OutboxStatus.SENT, tg_msg_id=12, at=now)
    failed_id = await repo.enqueue(conv_id, "oops", now)
    await repo.mark_outbox(failed_id, OutboxStatus.FAILED, error="PRIVACY")
    # No Repo method writes the outbox's wake columns, the task's card message or the
    # conversation's previous state (columns of the old engine): fill them straight in the
    # file, so their conversion is still checked below.
    with closing(sqlite3.connect(db_path)) as db:
        db.execute("UPDATE tasks SET card_msg_id = 77 WHERE id = ?", (task_id,))
        db.execute(
            "UPDATE conversations SET prev_state = 'talking' WHERE id = ?", (conv_id,)
        )
        wake_id = db.execute(
            "INSERT INTO wakes (task_id, started_at) VALUES (?, ?)",
            (task_id, now.astimezone(UTC).isoformat()),
        ).lastrowid
        db.execute(
            "UPDATE outbox SET wake_id = ?, tried_at = ? WHERE id = ?",
            (wake_id, now.astimezone(UTC).isoformat(), sent_id),
        )
        db.commit()
    candidate_id = await repo.add_candidate(task_id, "seller", "Продавец", "—", "u")

    rows = [
        await repo.get_task(task_id),
        await repo.get_contact(contact_id),
        await repo.get_contact(nameless_id),
        await repo.contact_by_peer(1001),
        await repo.get_conversation(conv_id),
        await repo.conversation_by_peer(1001),
        *await repo.task_conversations(task_id),
        *await repo.last_messages(conv_id, 5),
        approval,
        await repo.get_approval(approval.id),
        *await repo.outbox_in(OutboxStatus.SENT),
        *await repo.outbox_in(OutboxStatus.FAILED),
        *await repo.due_outbox(now),
        await repo.get_candidate(candidate_id),
        *await repo.task_candidates(task_id),
    ]

    assert {type(row) for row in rows} == set(TABLE_ROWS.values())
    for row in rows:
        assert_matches_annotations(row)
    # No vacuous pass: every field of every row type was filled in at least once, so its
    # conversion (enum, JSON, datetime, flag) really ran.
    for row_class in TABLE_ROWS.values():
        seen = [row for row in rows if type(row) is row_class]
        for field in fields(row_class):
            assert any(getattr(row, field.name) is not None for row in seen), (
                f"{row_class.__name__}.{field.name} was never filled in"
            )


# --- tasks and contacts ------------------------------------------------------------------------


async def test_create_task_conversation_and_lookup_by_peer(repo, fake_clock):
    task_id = await repo.create_task(
        "Дрель",
        "самое дешёвое предложение",
        {"товар": "дрель", "шт": 2},
        450.5,
        "UZS",
        ["name", "city"],
        ["a.jpg"],
    )
    contact_id = await repo.upsert_contact(1001, "seller", "Продавец Анвар", False)
    conv_id = await repo.create_conversation(task_id, contact_id)

    now = fake_clock.now()
    assert await repo.get_task(task_id) == Task(
        id=task_id,
        title="Дрель",
        goal="самое дешёвое предложение",
        item={"товар": "дрель", "шт": 2},
        ceiling=450.5,
        currency="UZS",
        allowed=["name", "city"],
        attachments=["a.jpg"],
        owner_notes="",
        status=TaskStatus.DRAFT,  # the owner has not launched it yet
        card_msg_id=None,
        created_at=now,
        updated_at=now,
    )
    assert await repo.get_contact(contact_id) == Contact(
        id=contact_id,
        peer_id=1001,
        username="seller",
        display_name="Продавец Анвар",
        is_bot=False,
        language=None,
        card={},
        do_not_contact=False,
        updated_at=now,
    )
    await repo.set_task_status(
        task_id, TaskStatus.ACTIVE
    )  # launched: its people are live
    expected = Conversation(
        id=conv_id,
        task_id=task_id,
        contact_id=contact_id,
        state=S.PLANNED,
        prev_state=None,
        facts={},
        summary="",
        language=None,
        offtopic_strikes=0,
        agent_messages=0,
        followups_sent=0,
        first_sent=False,
        paused_until=None,
        last_in_at=None,
        last_out_at=None,
        created_at=now,
        updated_at=now,
    )
    assert await repo.conversation_by_peer(1001) == expected
    assert await repo.get_conversation(conv_id) == expected
    assert await repo.task_conversations(task_id) == [expected]
    assert await repo.conversation_by_peer(9999) is None
    assert await repo.task_conversations(task_id + 1) == []

    # One conversation per task and contact.
    with pytest.raises(sqlite3.IntegrityError):
        await repo.create_conversation(task_id, contact_id)


async def test_task_status_and_active_tasks(repo, fake_clock):
    ids = {
        status: await make_task(repo, status.value, TaskStatus.DRAFT)
        for status in TaskStatus
    }
    fake_clock.advance(timedelta(minutes=5))
    for status, task_id in ids.items():
        await repo.set_task_status(task_id, status)

    task = await repo.get_task(ids[TaskStatus.ACTIVE])
    assert task.status is TaskStatus.ACTIVE
    assert task.updated_at == fake_clock.now() > task.created_at


async def test_rewrite_draft_replaces_a_draft_in_place_and_nothing_else(
    repo, fake_clock
):
    task_id = await repo.create_task(
        "Дрель", "Купить дрель", {"what": "дрель"}, None, "UZS", ["name"], [],
        source_text="Найди дрель",
    )  # fmt: skip
    fake_clock.advance(timedelta(minutes=5))

    changed = await repo.rewrite_draft(
        task_id,
        title="Дрель Bosch",
        goal="Купить дрель Bosch, можно на OLX",
        ceiling=600_000,
        currency="USD",
        allowed=["name", "city"],
        attachments=["/files/a.pdf"],
        source_text="Найди дрель\nищи и на OLX",
    )

    assert changed is True
    task = await repo.get_task(task_id)  # the same id: the draft's number stays
    assert (task.title, task.goal, task.ceiling, task.currency) == (
        "Дрель Bosch",
        "Купить дрель Bosch, можно на OLX",
        600_000,
        "USD",
    )
    assert (task.allowed, task.attachments) == (["name", "city"], ["/files/a.pdf"])
    assert task.source_text == "Найди дрель\nищи и на OLX"
    assert task.item == {"what": "дрель"}  # not a field of the rewrite
    assert task.status is TaskStatus.DRAFT
    assert task.updated_at == fake_clock.now() > task.created_at

    # A task that is no longer a draft is left as it is.
    for status in (TaskStatus.ACTIVE, TaskStatus.CANCELLED):
        await repo.set_task_status(task_id, status)
        assert not await repo.rewrite_draft(
            task_id, title="X", goal="Y", ceiling=None, currency="UZS",
            allowed=[], attachments=[], source_text="",
        )  # fmt: skip
        assert (await repo.get_task(task_id)).title == "Дрель Bosch"
    assert not await repo.rewrite_draft(
        9999, title="X", goal="Y", ceiling=None, currency="UZS",
        allowed=[], attachments=[], source_text="",
    )  # fmt: skip


async def test_upsert_contact_refreshes_the_profile_and_keeps_the_rest(
    repo, fake_clock
):
    first = await repo.upsert_contact(1001, None, "Anvar", False)
    assert (await repo.get_contact(first)).username is None  # not every account has one
    await repo.update_contact_card(first, {"shop": "Tools"}, "uz_latn")

    fake_clock.advance(timedelta(hours=1))
    again = await repo.upsert_contact(1001, "anvar_tools", "Anvar T.", True)

    assert again == first
    contact = await repo.contact_by_peer(1001)
    assert (contact.username, contact.display_name) == ("anvar_tools", "Anvar T.")
    assert contact.is_bot is True
    assert (contact.card, contact.language) == (
        {"shop": "Tools"},
        "uz_latn",
    )  # not reset
    assert contact.do_not_contact is False
    assert contact.updated_at == fake_clock.now()
    assert await repo.contact_by_peer(404) is None

    await repo.upsert_contact(1001, None, "Anvar T.", True)  # the username was removed
    assert (await repo.get_contact(first)).username is None
    await repo.update_contact_card(first, {}, None)
    assert (await repo.get_contact(first)).language is None


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda r: r.get_task(99), id="get_task"),
        pytest.param(lambda r: r.get_contact(99), id="get_contact"),
        pytest.param(lambda r: r.get_conversation(99), id="get_conversation"),
        pytest.param(lambda r: r.get_approval(99), id="get_approval"),
        pytest.param(
            lambda r: r.set_task_status(99, TaskStatus.ACTIVE), id="set_task_status"
        ),
        pytest.param(
            lambda r: r.update_contact_card(99, {}, None), id="update_contact_card"
        ),
        pytest.param(
            lambda r: r.update_conversation(99, summary="x"), id="update_conversation"
        ),
        pytest.param(lambda r: r.set_state(99, S.CONTACTED), id="set_state"),
        pytest.param(lambda r: r.set_approval_msg(99, 1), id="set_approval_msg"),
        pytest.param(lambda r: r.mark_outbox(99, OutboxStatus.SENT), id="mark_outbox"),
    ],
)
async def test_missing_rows_raise_not_found(repo, call):
    with pytest.raises(NotFound):
        await call(repo)


# --- conversations -------------------------------------------------------------------------


async def test_set_state_moves_and_rejects_an_impossible_move(repo, fake_clock):
    conv_id = await make_conversation(repo)
    created_at = fake_clock.now()

    fake_clock.advance(timedelta(minutes=3))
    assert await repo.set_state(conv_id, S.CONTACTED) is S.CONTACTED
    await walk(repo, conv_id, S.TALKING, S.WAITING_OWNER)
    conv = await repo.get_conversation(conv_id)
    assert conv.state is S.WAITING_OWNER
    assert conv.prev_state is None  # nothing writes it any more
    assert conv.updated_at == fake_clock.now() > created_at

    # An impossible move fails loudly and changes nothing.
    await walk(repo, conv_id, S.CLOSED)
    before = await repo.get_conversation(conv_id)
    # CLOSED reopens only to TALKING or CONTACTED (ruling R72).
    with pytest.raises(InvalidTransition):
        await repo.set_state(conv_id, S.WAITING_OWNER)
    assert await repo.get_conversation(conv_id) == before
    assert before.state is S.CLOSED


async def test_set_state_serialises_concurrent_changes(repo):
    conv_id = await make_conversation(repo)
    await walk(repo, conv_id, S.CONTACTED)

    # The move is legal from CONTACTED, but once it has happened it is not (TALKING cannot
    # move to TALKING): the second must not be judged against the same stale state.
    results = await asyncio.gather(
        repo.set_state(conv_id, S.TALKING),
        repo.set_state(conv_id, S.TALKING),
        return_exceptions=True,
    )

    assert sum(isinstance(result, InvalidTransition) for result in results) == 1
    assert (await repo.get_conversation(conv_id)).state is S.TALKING


FINISHING_MOVES: dict[ConvState, tuple[ConvState, ...]] = {
    S.CLOSED: (S.CONTACTED, S.CLOSED),
}


@pytest.mark.parametrize("final", sorted(TERMINAL))
async def test_finished_conversations_are_not_found_by_peer(repo, final):
    assert set(FINISHING_MOVES) == TERMINAL
    contact_id = await repo.upsert_contact(1001, "seller", "Продавец", False)
    older = await repo.create_conversation(await make_task(repo, "older"), contact_id)
    newer = await repo.create_conversation(await make_task(repo, "newer"), contact_id)

    assert (await repo.conversation_by_peer(1001)).id == newer  # the latest one
    assert await repo.active_peer_ids() == {1001}

    await walk(repo, newer, *FINISHING_MOVES[final])
    assert (
        await repo.conversation_by_peer(1001)
    ).id == older  # falls back to the open one

    await walk(repo, older, *FINISHING_MOVES[final])
    assert await repo.conversation_by_peer(1001) is None
    assert await repo.active_peer_ids() == set()


async def test_active_peer_ids(repo):
    task_id = await make_task(repo)
    await make_conversation(
        repo, peer_id=1, task_id=task_id
    )  # planned: live all the same
    talking = await make_conversation(repo, peer_id=2, task_id=task_id)
    waiting = await make_conversation(repo, peer_id=3, task_id=task_id)
    closed = await make_conversation(repo, peer_id=4, task_id=task_id)
    await make_task(repo)  # a task with no conversations adds nobody
    await walk(repo, talking, S.CONTACTED, S.TALKING)
    await walk(repo, waiting, S.CONTACTED, S.WAITING_OWNER)
    await walk(repo, closed, S.CONTACTED, S.TALKING, S.CLOSED)

    assert await repo.active_peer_ids() == {1, 2, 3}


@pytest.mark.parametrize(
    ("status", "live"),
    [
        (TaskStatus.ACTIVE, True),
        (TaskStatus.PAUSED, True),  # the owner paused the task: its people stay his
        (TaskStatus.DONE, False),  # the agent finished: its people are free
        (TaskStatus.CANCELLED, False),
        (TaskStatus.DRAFT, False),
    ],
)
async def test_a_conversation_is_live_only_while_its_task_is_in_work(
    repo, status, live
):
    task_id = await make_task(repo, status=status)
    conv_id = await make_conversation(repo, task_id=task_id)
    await walk(repo, conv_id, S.CONTACTED, S.TALKING)

    by_peer = await repo.conversation_by_peer(1001)
    blocking = await repo.blocking_conversation(1001)
    assert (by_peer.id if by_peer else None) == (conv_id if live else None)
    assert (blocking.id if blocking else None) == (conv_id if live else None)
    assert await repo.active_peer_ids() == ({1001} if live else set())


async def test_a_person_of_a_done_task_can_be_attached_to_a_new_one(repo):
    done = await make_conversation(repo, task_id=await make_task(repo))
    await walk(repo, done, S.CONTACTED, S.TALKING)
    first_task = (await repo.get_conversation(done)).task_id
    await repo.set_task_status(first_task, TaskStatus.DONE)  # his state stays TALKING

    assert await repo.blocking_conversation(1001) is None
    newer = await make_conversation(repo, task_id=await make_task(repo, "Вторая"))

    assert (await repo.conversation_by_peer(1001)).id == newer
    assert (await repo.blocking_conversation(1001)).id == newer
    assert await repo.active_peer_ids() == {1001}
    assert (await repo.get_conversation(done)).state is S.TALKING


async def test_the_newest_live_conversation_is_the_one_found(repo):
    older = await make_conversation(repo, task_id=await make_task(repo, "Первая"))
    newer = await make_conversation(repo, task_id=await make_task(repo, "Вторая"))
    assert (await repo.conversation_by_peer(1001)).id == newer

    await repo.set_task_status(
        (await repo.get_conversation(newer)).task_id, TaskStatus.DONE
    )

    assert (await repo.conversation_by_peer(1001)).id == older


async def test_update_conversation_stores_typed_values(repo, fake_clock):
    conv_id = await make_conversation(repo)
    before = await repo.get_conversation(conv_id)
    fake_clock.advance(timedelta(minutes=5))
    now = fake_clock.now()

    await repo.update_conversation(
        conv_id,
        facts={"price": "120 000", "currency": "UZS"},
        summary="Цена 120 000",
        language="ru",
        offtopic_strikes=1,
        agent_messages=3,
        followups_sent=2,
        first_sent=True,
        paused_until=now + timedelta(hours=12),
        last_in_at=now,
        last_out_at=None,
    )

    conv = await repo.get_conversation(conv_id)
    assert conv.facts == {"price": "120 000", "currency": "UZS"}
    assert (conv.summary, conv.language) == ("Цена 120 000", "ru")
    assert (conv.offtopic_strikes, conv.agent_messages, conv.followups_sent) == (
        1,
        3,
        2,
    )
    assert conv.first_sent is True
    assert (conv.paused_until, conv.last_in_at, conv.last_out_at) == (
        now + timedelta(hours=12),
        now,
        None,
    )
    assert conv.updated_at == now != before.updated_at
    assert conv.state is S.PLANNED  # nothing else moved

    await repo.update_conversation(
        conv_id, paused_until=None, first_sent=False, language=None
    )
    conv = await repo.get_conversation(conv_id)
    assert (conv.paused_until, conv.first_sent, conv.language) == (None, False, None)


@pytest.mark.parametrize(
    "field",
    ["state", "prev_state", "id", "task_id", "contact_id", "created_at", "nonsense"],
)
async def test_update_conversation_refuses_fields_it_does_not_own(repo, field):
    # `state` and `prev_state` change only through `set_state`, so no move skips the state machine.
    conv_id = await make_conversation(repo)
    before = await repo.get_conversation(conv_id)

    with pytest.raises(TypeError, match=field):
        await repo.update_conversation(
            conv_id, summary="would be written", **{field: S.CLOSED}
        )

    assert await repo.get_conversation(conv_id) == before


# --- messages ----------------------------------------------------------------------------------


async def test_last_messages_newest_n_in_order(repo, fake_clock):
    conv_id = await make_conversation(repo)
    other = await make_conversation(repo, peer_id=2002)
    t0 = fake_clock.now()
    # Inserted out of order on purpose: "newest" is by time, not by insertion.
    for minute in (3, 0, 4, 1, 2):
        await say(repo, conv_id, f"m{minute}", t0 + timedelta(minutes=minute))
    await say(repo, other, "someone else's", t0 + timedelta(minutes=9))

    assert [m.text for m in await repo.last_messages(conv_id, 3)] == ["m2", "m3", "m4"]
    assert [m.text for m in await repo.last_messages(conv_id, 1)] == ["m4"]
    assert [m.text for m in await repo.last_messages(conv_id, 50)] == [
        f"m{i}" for i in range(5)
    ]
    assert await repo.last_messages(conv_id, 0) == []
    assert (
        await repo.last_messages(conv_id, -1) == []
    )  # (SQLite's LIMIT -1 would mean "all")
    assert await repo.last_messages(conv_id + 100, 5) == []


async def test_messages_sharing_a_time_keep_their_insertion_order(repo, fake_clock):
    # Telegram dates have a one-second resolution, so a burst of messages shares a timestamp.
    conv_id = await make_conversation(repo)
    for text in ("a", "b", "c"):
        await say(repo, conv_id, text, fake_clock.now())

    assert [m.text for m in await repo.last_messages(conv_id, 2)] == ["b", "c"]


async def test_message_round_trip(repo, fake_clock):
    conv_id = await make_conversation(repo)
    message_id = await repo.add_message(
        conv_id,
        Direction.OUT,
        MsgSource.OWNER_MANUAL,
        "voice",
        "Салом!",
        None,
        fake_clock.now(),
    )

    assert await repo.last_messages(conv_id, 1) == [
        Message(
            id=message_id,
            conversation_id=conv_id,
            direction=Direction.OUT,
            source=MsgSource.OWNER_MANUAL,
            tg_msg_id=None,
            kind="voice",
            text="Салом!",
            created_at=fake_clock.now(),
            edited_at=None,
        )
    ]


async def test_edit_message_updates_text(repo, fake_clock):
    conv_id = await make_conversation(repo)
    other = await make_conversation(repo, peer_id=2002)
    t0 = fake_clock.now()
    await say(repo, conv_id, "цена 100", t0, tg_msg_id=555)
    await say(
        repo, other, "не трогать", t0, tg_msg_id=555
    )  # same Telegram id, another chat
    edited_at = t0 + timedelta(minutes=2)

    assert await repo.edit_message(conv_id, 555, "цена 120", edited_at) is True

    [message] = await repo.last_messages(conv_id, 10)
    assert (message.text, message.edited_at, message.created_at) == (
        "цена 120",
        edited_at,
        t0,
    )
    [untouched] = await repo.last_messages(other, 10)
    assert (untouched.text, untouched.edited_at) == ("не трогать", None)
    assert await repo.edit_message(conv_id, 999, "нет такого", edited_at) is False


async def test_has_message_looks_for_a_telegram_id_in_one_conversation(
    repo, fake_clock
):
    conv_id = await make_conversation(repo)
    other = await make_conversation(repo, peer_id=2002)
    await say(repo, conv_id, "цена 100", fake_clock.now(), tg_msg_id=555)
    await say(repo, conv_id, "без id", fake_clock.now())

    assert await repo.has_message(conv_id, 555) is True
    assert await repo.has_message(conv_id, 556) is False
    assert await repo.has_message(other, 555) is False


async def test_purge_messages_before_keeps_recent(repo, fake_clock):
    conv_id = await make_conversation(repo)
    other = await make_conversation(repo, peer_id=2002)
    now = fake_clock.now()
    cutoff = now - timedelta(days=90)
    await say(repo, conv_id, "old", cutoff - timedelta(seconds=1))
    await say(repo, other, "old too", now - timedelta(days=200))
    await say(repo, conv_id, "on the cutoff", cutoff)
    await say(repo, conv_id, "recent", now - timedelta(days=89))
    await say(repo, conv_id, "new", now)

    assert await repo.purge_messages_before(cutoff) == 2
    assert [m.text for m in await repo.last_messages(conv_id, 10)] == [
        "on the cutoff",
        "recent",
        "new",
    ]
    assert await repo.last_messages(other, 10) == []
    assert await repo.purge_messages_before(cutoff) == 0
    assert (await repo.get_conversation(conv_id)).state is S.PLANNED  # only messages go


# --- approvals ---------------------------------------------------------------------------------


async def test_approval_nonce_single_use(repo):
    task_id = await make_task(repo)
    conv_id = await make_conversation(repo, task_id=task_id)
    kind = ApprovalKind.SHARE_PHONE
    approval = await repo.create_approval(task_id, conv_id, kind, {"why": "звонок"})
    assert re.fullmatch(r"[0-9a-f]{6}", approval.nonce)
    assert approval.status is ApprovalStatus.PENDING
    assert approval == await repo.get_approval(approval.id)
    foreign = "zzzzzz"  # can never be a real nonce

    assert (
        await repo.resolve_approval(approval.id, foreign, ApprovalStatus.APPROVED)
        is False
    )
    assert (await repo.get_approval(approval.id)).status is ApprovalStatus.PENDING

    assert await repo.resolve_approval(
        approval.id, approval.nonce, ApprovalStatus.APPROVED, "ок"
    )
    rejected = ApprovalStatus.REJECTED
    assert await repo.resolve_approval(approval.id, approval.nonce, rejected) is False
    assert (
        await repo.resolve_approval(approval.id, foreign, ApprovalStatus.REJECTED)
        is False
    )

    resolved = await repo.get_approval(approval.id)  # the replays changed nothing
    assert (resolved.status, resolved.answer) == (ApprovalStatus.APPROVED, "ок")


async def test_every_approval_gets_its_own_nonce(repo):
    # The nonce is what makes an old or forged button press harmless, so it must not repeat.
    task_id = await make_task(repo)
    approvals = [
        await repo.create_approval(task_id, None, ApprovalKind.QUESTION, {})
        for _ in range(5)
    ]

    assert len({approval.nonce for approval in approvals}) > 1


async def test_resolving_to_pending_is_refused(repo):
    # Otherwise the nonce could be "used" without being used up.
    task_id = await make_task(repo)
    approval = await repo.create_approval(task_id, None, ApprovalKind.QUESTION, {})

    with pytest.raises(ValueError, match="pending"):
        await repo.resolve_approval(approval.id, approval.nonce, ApprovalStatus.PENDING)


async def test_pending_and_granted_approvals(repo, fake_clock):
    task_id = await make_task(repo)
    conv_id = await make_conversation(repo, task_id=task_id)
    other_conv = await make_conversation(repo, peer_id=2002, task_id=task_id)
    phone = await repo.create_approval(
        task_id, conv_id, ApprovalKind.SHARE_PHONE, {"ask": "phone"}
    )
    address = await repo.create_approval(
        task_id, conv_id, ApprovalKind.SHARE_ADDRESS, {}
    )
    question = await repo.create_approval(
        task_id, conv_id, ApprovalKind.QUESTION, {"q": "Доставка?"}
    )
    elsewhere = await repo.create_approval(
        task_id, other_conv, ApprovalKind.SHARE_PHONE, {}
    )
    compare = await repo.create_approval(
        task_id, None, ApprovalKind.QUESTION, {"options": [1, 2]}
    )

    ours = [phone.id, address.id, question.id]
    assert [a.id for a in await repo.pending_approvals(conv_id)] == ours
    assert [a.id for a in await repo.pending_approvals()] == [
        phone.id,
        address.id,
        question.id,
        elsewhere.id,
        compare.id,
    ]
    assert await repo.granted_kinds(conv_id) == set()

    fake_clock.advance(timedelta(minutes=10))
    for approval, status, answer in (
        (phone, ApprovalStatus.APPROVED, None),
        (address, ApprovalStatus.REJECTED, None),
        (question, ApprovalStatus.ANSWERED, "Бесплатная"),
        (elsewhere, ApprovalStatus.APPROVED, None),
    ):
        assert await repo.resolve_approval(approval.id, approval.nonce, status, answer)

    # Only what was approved, only for this conversation.
    assert await repo.granted_kinds(conv_id) == {ApprovalKind.SHARE_PHONE}
    assert await repo.granted_kinds(other_conv) == {ApprovalKind.SHARE_PHONE}
    assert await repo.granted_kinds(other_conv + 100) == set()
    assert [a.id for a in await repo.pending_approvals()] == [compare.id]
    assert await repo.pending_approvals(conv_id) == []

    await repo.set_approval_msg(compare.id, 4242)
    task_level = await repo.get_approval(compare.id)
    assert (task_level.control_msg_id, task_level.conversation_id) == (4242, None)
    assert task_level.payload == {"options": [1, 2]}
    assert task_level.resolved_at is None

    answered = await repo.get_approval(question.id)
    assert (answered.status, answered.answer) == (ApprovalStatus.ANSWERED, "Бесплатная")
    assert answered.resolved_at == fake_clock.now()


async def test_approvals_of_lists_every_approval_of_the_conversation(repo):
    task_id = await make_task(repo)
    conv_id = await make_conversation(repo, task_id=task_id)
    other_conv = await make_conversation(repo, peer_id=2002, task_id=task_id)
    order = await repo.create_approval(task_id, conv_id, ApprovalKind.CONFIRM_ORDER, {})
    await repo.create_approval(task_id, other_conv, ApprovalKind.CONFIRM_ORDER, {})
    alert = await repo.create_approval(
        task_id, conv_id, ApprovalKind.QUESTION, {"text": "!"}
    )
    await repo.resolve_approval(order.id, order.nonce, ApprovalStatus.APPROVED)

    approvals = await repo.approvals_of(conv_id)

    assert [(a.id, a.status) for a in approvals] == [
        (order.id, ApprovalStatus.APPROVED),
        (alert.id, ApprovalStatus.PENDING),
    ]
    assert await repo.approvals_of(conv_id + 100) == []


async def test_approvals_of_task_lists_every_approval_of_the_task(repo):
    task_id = await make_task(repo)
    other_task = await make_task(repo)
    conv_id = await make_conversation(repo, task_id=task_id)
    other_conv = await make_conversation(repo, peer_id=2002, task_id=other_task)
    order = await repo.create_approval(task_id, conv_id, ApprovalKind.CONFIRM_ORDER, {})
    await repo.create_approval(other_task, other_conv, ApprovalKind.QUESTION, {})
    # a question about the task as a whole is about nobody: it is the task's all the same
    about_task = await repo.create_approval(task_id, None, ApprovalKind.QUESTION, {})
    await repo.resolve_approval(order.id, order.nonce, ApprovalStatus.REJECTED)

    approvals = await repo.approvals_of_task(task_id)

    assert [(a.id, a.status) for a in approvals] == [
        (order.id, ApprovalStatus.REJECTED),
        (about_task.id, ApprovalStatus.PENDING),
    ]
    assert await repo.approvals_of_task(other_task + 100) == []


async def test_set_approval_payload_replaces_it_and_keeps_the_rest(repo):
    task_id = await make_task(repo)
    conv_id = await make_conversation(repo, task_id=task_id)
    order = await repo.create_approval(
        task_id, conv_id, ApprovalKind.CONFIRM_ORDER, {"text": "Оформляем?"}
    )
    await repo.resolve_approval(order.id, order.nonce, ApprovalStatus.APPROVED)

    await repo.set_approval_payload(
        order.id, {"text": "Оформляем?", "approved_amount": 350000}
    )

    stored = await repo.get_approval(order.id)
    assert stored.payload == {"text": "Оформляем?", "approved_amount": 350000}
    assert (stored.status, stored.nonce) == (ApprovalStatus.APPROVED, order.nonce)
    with pytest.raises(NotFound):
        await repo.set_approval_payload(order.id + 100, {})


# --- outbox ------------------------------------------------------------------------------------


async def test_due_outbox_respects_not_before_and_status(repo, fake_clock):
    conv_id = await make_conversation(repo)
    other = await make_conversation(repo, peer_id=2002)
    now = fake_clock.now()

    five_minutes_ago = now - timedelta(minutes=5)
    overdue = await repo.enqueue(
        conv_id, "overdue", five_minutes_ago, attach_path="p.jpg"
    )
    on_time = await repo.enqueue(other, "on time", now)
    await repo.enqueue(conv_id, "not yet", now + timedelta(seconds=1))
    for status in OutboxStatus:  # whatever is not waiting is never due
        if status is not OutboxStatus.PENDING:
            item = await repo.enqueue(conv_id, f"is {status}", now - timedelta(hours=1))
            await repo.mark_outbox(item, status)

    due = await repo.due_outbox(now)
    assert [(item.id, item.text) for item in due] == [
        (overdue, "overdue"),
        (on_time, "on time"),
    ]
    assert (due[0].attach_path, due[1].attach_path) == ("p.jpg", None)
    assert all(item.status is OutboxStatus.PENDING for item in due)
    assert (due[0].not_before, due[0].created_at) == (five_minutes_ago, now)

    later = await repo.due_outbox(now + timedelta(seconds=1))
    assert [item.text for item in later] == ["overdue", "on time", "not yet"]


async def test_cancel_pending_only_touches_pending(repo, fake_clock):
    conv_id = await make_conversation(repo)
    other = await make_conversation(repo, peer_id=2002)
    now = fake_clock.now()
    await repo.enqueue(conv_id, "a", now)
    await repo.enqueue(conv_id, "b", now)
    sent = await repo.enqueue(conv_id, "sent", now)
    sending = await repo.enqueue(conv_id, "sending", now)
    await repo.enqueue(other, "elsewhere", now)
    await repo.mark_outbox(sent, OutboxStatus.SENT, tg_msg_id=1, at=now)
    await repo.mark_outbox(sending, OutboxStatus.SENDING)

    assert await repo.cancel_pending(conv_id) == 2

    assert [i.text for i in await repo.outbox_in(OutboxStatus.CANCELLED)] == ["a", "b"]
    assert [i.text for i in await repo.outbox_in(OutboxStatus.SENT)] == ["sent"]
    assert [i.text for i in await repo.outbox_in(OutboxStatus.SENDING)] == ["sending"]
    assert [i.text for i in await repo.outbox_in(OutboxStatus.PENDING)] == ["elsewhere"]
    assert await repo.cancel_pending(conv_id) == 0


async def test_mark_outbox_and_sent_texts(repo, fake_clock):
    conv_id = await make_conversation(repo)
    other = await make_conversation(repo, peer_id=2002)
    now = fake_clock.now()
    first = await repo.enqueue(conv_id, "first", now)
    second = await repo.enqueue(conv_id, "second", now)
    elsewhere = await repo.enqueue(other, "elsewhere", now)
    never = await repo.enqueue(conv_id, "never went out", now)

    await repo.mark_outbox(first, OutboxStatus.SENDING)
    # "second" happens to go out before "first" does.
    await repo.mark_outbox(
        second, OutboxStatus.SENT, tg_msg_id=502, at=now + timedelta(seconds=10)
    )
    await repo.mark_outbox(
        first, OutboxStatus.SENT, tg_msg_id=501, at=now + timedelta(seconds=20)
    )
    thirty = now + timedelta(seconds=30)
    await repo.mark_outbox(elsewhere, OutboxStatus.SENT, tg_msg_id=601, at=thirty)
    await repo.mark_outbox(never, OutboxStatus.FAILED, error="PRIVACY")

    sent = {item.text: item for item in await repo.outbox_in(OutboxStatus.SENT)}
    assert (sent["first"].tg_msg_id, sent["first"].sent_at) == (
        501,
        now + timedelta(seconds=20),
    )
    assert sent["first"].error is None
    [failed] = await repo.outbox_in(OutboxStatus.FAILED)
    assert (failed.error, failed.sent_at, failed.tg_msg_id) == ("PRIVACY", None, None)

    assert await repo.sent_texts(conv_id) == ["second", "first"]
    assert await repo.sent_texts() == ["second", "first", "elsewhere"]
    assert await repo.sent_texts(conv_id + 100) == []


async def test_outbox_of_lists_every_item_of_the_conversation_in_queue_order(
    repo, fake_clock
):
    conv_id = await make_conversation(repo)
    other = await make_conversation(repo, peer_id=2002)
    now = fake_clock.now()
    first = await repo.enqueue(conv_id, "first", now + timedelta(minutes=5))
    await repo.enqueue(other, "elsewhere", now)
    await repo.enqueue(conv_id, "", now, attach_path="files/1/a.stl")
    await repo.mark_outbox(first, OutboxStatus.SENT, tg_msg_id=501, at=now)

    items = await repo.outbox_of(conv_id)

    assert [(i.text, i.attach_path, i.status) for i in items] == [
        ("first", None, OutboxStatus.SENT),
        ("", "files/1/a.stl", OutboxStatus.PENDING),
    ]
    assert await repo.outbox_of(conv_id + 100) == []


async def test_claim_outbox_takes_a_waiting_item_once(repo, fake_clock):
    conv_id = await make_conversation(repo)
    now = fake_clock.now()
    cancelled = await repo.enqueue(conv_id, "cancelled", now)
    await repo.cancel_pending(conv_id)
    waiting = await repo.enqueue(conv_id, "waiting", now)
    sent = await repo.enqueue(conv_id, "sent", now)
    await repo.mark_outbox(sent, OutboxStatus.SENT, tg_msg_id=1, at=now)
    failed = await repo.enqueue(conv_id, "failed", now)
    await repo.mark_outbox(failed, OutboxStatus.FAILED, error="PRIVACY")
    await repo.enqueue(conv_id, "still waiting", now)

    assert await repo.claim_outbox(waiting) is True
    assert (
        await repo.claim_outbox(waiting) is False
    )  # taken already: whoever lost must not send
    # Only a waiting item can be claimed: a cancelled one is not brought back, a sent or failed one
    # is not sent again, and an id nobody has gives nothing.
    for item_id in (cancelled, sent, failed, 9999):
        assert await repo.claim_outbox(item_id) is False

    statuses = {item.text: item.status for item in await repo.outbox_of(conv_id)}
    assert statuses == {
        "waiting": OutboxStatus.SENDING,
        "cancelled": OutboxStatus.CANCELLED,
        "sent": OutboxStatus.SENT,
        "failed": OutboxStatus.FAILED,
        "still waiting": OutboxStatus.PENDING,
    }
    assert (
        await repo.cancel_pending(conv_id) == 1
    )  # the claimed item is out of reach of a cancel
    assert [item.id for item in await repo.outbox_in(OutboxStatus.SENDING)] == [waiting]


async def test_reschedule_outbox_moves_only_waiting_items(repo, fake_clock):
    conv_id = await make_conversation(repo)
    now = fake_clock.now()
    morning = now + timedelta(hours=14)
    moved = await repo.enqueue(conv_id, "moved", now, attach_path="files/1/a.stl")
    sending = await repo.enqueue(conv_id, "sending", now)
    await repo.claim_outbox(sending)
    sent = await repo.enqueue(conv_id, "sent", now)
    await repo.mark_outbox(sent, OutboxStatus.SENT, tg_msg_id=1, at=now)
    cancelled = await repo.enqueue(conv_id, "cancelled", now)
    await repo.mark_outbox(cancelled, OutboxStatus.CANCELLED)

    assert await repo.reschedule_outbox(moved, morning) is True
    for item_id in (sending, sent, cancelled, 9999):
        assert await repo.reschedule_outbox(item_id, morning) is False

    items = {item.text: item for item in await repo.outbox_of(conv_id)}
    assert items["moved"].not_before == morning
    assert (items["moved"].status, items["moved"].attach_path) == (
        OutboxStatus.PENDING,
        "files/1/a.stl",
    )
    for text in ("sending", "sent", "cancelled"):
        assert items[text].not_before == now  # nothing but the waiting one is touched
    assert [item.text for item in await repo.due_outbox(now)] == []
    assert [item.text for item in await repo.due_outbox(morning)] == ["moved"]

    # a time in another zone is stored as the same moment
    await repo.reschedule_outbox(moved, datetime(2026, 11, 11, 9, 0, tzinfo=TASHKENT))
    [item] = await repo.due_outbox(datetime(2026, 11, 11, 4, 0, tzinfo=UTC))
    assert item.not_before == datetime(2026, 11, 11, 4, 0, tzinfo=UTC)


# --- costs and events ----------------------------------------------------------------------------


async def test_cost_since_sums_llm_calls(repo, db_path, fake_clock):
    conv_id = await make_conversation(repo)
    other = await make_conversation(repo, peer_id=2002)
    start = fake_clock.now()

    decide_usage = Usage(1000, 200)
    await repo.log_llm_call(
        conv_id, "decide", "claude-sonnet-5-5", decide_usage, 0.004, "end_turn"
    )
    fake_clock.advance(timedelta(hours=2))
    judge_usage = Usage(500, 50, cache_read=300, cache_write=40)
    await repo.log_llm_call(
        conv_id, "judge", "claude-haiku-5-5", judge_usage, 0.001, "end_turn"
    )
    await repo.log_llm_call(
        None, "intake", "claude-sonnet-5-5", Usage(800, 300), 0.01, "end_turn"
    )
    await repo.log_llm_call(
        other, "decide", "claude-sonnet-5-5", Usage(1, 1), 0.02, "max_tokens"
    )

    assert await repo.cost_since(start) == pytest.approx(0.035)  # `since` is inclusive
    assert await repo.cost_since(start + timedelta(minutes=1)) == pytest.approx(0.031)
    assert await repo.cost_since(fake_clock.now() + timedelta(seconds=1)) == 0.0

    assert raw(
        db_path,
        "SELECT purpose, model, input_tokens, output_tokens, cache_read, cache_write, stop_reason "
        "FROM llm_calls WHERE conversation_id = ? ORDER BY id",
        (conv_id,),
    ) == [
        ("decide", "claude-sonnet-5-5", 1000, 200, 0, 0, "end_turn"),
        ("judge", "claude-haiku-5-5", 500, 50, 300, 40, "end_turn"),
    ]


async def test_log_event_stores_type_conversation_and_payload(repo, db_path):
    conv_id = await make_conversation(repo)

    await repo.log_event("first_sent", conv_id, via="bot", attempt=2)
    await repo.log_event("stopped")

    rows = raw(
        db_path,
        "SELECT ts, type, conversation_id, payload_json FROM events ORDER BY id",
    )
    assert [
        (ts, kind, conv, json.loads(payload)) for ts, kind, conv, payload in rows
    ] == [
        (
            "2026-11-10T05:00:00+00:00",
            "first_sent",
            conv_id,
            {"via": "bot", "attempt": 2},
        ),
        ("2026-11-10T05:00:00+00:00", "stopped", None, {}),
    ]


async def test_events_of_returns_the_payloads_of_one_type_oldest_first(repo):
    conv_id = await make_conversation(repo)
    other = await make_conversation(repo, peer_id=2002)
    await repo.log_event("owner_instruction", conv_id, text="first")
    await repo.log_event("auto_reply", conv_id, tg_msg_id=7)
    await repo.log_event("owner_instruction", other, text="elsewhere")
    await repo.log_event("owner_instruction", conv_id, text="second")
    await repo.log_event("stopped")

    assert await repo.events_of("owner_instruction", conv_id) == [
        {"text": "first"},
        {"text": "second"},
    ]
    assert await repo.events_of("owner_instruction") == [
        {"text": "first"},
        {"text": "elsewhere"},
        {"text": "second"},
    ]
    assert await repo.events_of("stopped") == [{}]
    assert await repo.events_of("auto_reply", other) == []


async def test_new_contacts_on_counts_first_sent_events_by_tashkent_day(
    repo, fake_clock
):
    conv_id = await make_conversation(repo)

    async def event(kind: str, local: datetime) -> None:
        fake_clock.set(local)
        await repo.log_event(kind, conv_id)

    await event("first_sent", datetime(2026, 11, 9, 23, 59, 59, tzinfo=TASHKENT))
    await event(
        "first_sent", datetime(2026, 11, 10, 0, 0, tzinfo=TASHKENT)
    )  # 19:00 UTC on the 9th
    await event(
        "first_sent", datetime(2026, 11, 10, 23, 59, tzinfo=TASHKENT)
    )  # 18:59 UTC the 10th
    await event(
        "reply", datetime(2026, 11, 10, 12, 0, tzinfo=TASHKENT)
    )  # another kind of event
    await event("first_sent", datetime(2026, 11, 11, 0, 0, tzinfo=TASHKENT))

    assert await repo.new_contacts_on(date(2026, 11, 9)) == 1
    assert await repo.new_contacts_on(date(2026, 11, 10)) == 2
    assert await repo.new_contacts_on(date(2026, 11, 11)) == 1
    assert await repo.new_contacts_on(date(2026, 11, 12)) == 0


# --- Task 10 additions --------------------------------------------------------------------


async def test_all_tasks_lists_every_task_oldest_first(repo: Repo) -> None:
    first = await make_task(repo, "Первая")
    second = await make_task(repo, "Вторая")
    await repo.set_task_status(first, TaskStatus.CANCELLED)
    assert [t.id for t in await repo.all_tasks()] == [first, second]


# --- after a restart ------------------------------------------------------------------------


async def test_newest_telegram_id_and_the_ids_of_our_own_messages(repo, fake_clock):
    conv_id = await make_conversation(repo)
    empty = await make_conversation(repo, peer_id=1002)
    now = fake_clock.now()
    await say(repo, conv_id, "Добрый день", now, source=MsgSource.AGENT, tg_msg_id=900)
    await say(repo, conv_id, "Здравствуйте", now, tg_msg_id=901)
    await say(repo, conv_id, "Сам", now, source=MsgSource.OWNER_MANUAL, tg_msg_id=905)
    await say(repo, conv_id, "Без id", now, source=MsgSource.OWNER_MANUAL)

    assert await repo.last_tg_msg_id(conv_id) == 905
    assert await repo.out_tg_msg_ids(conv_id) == {900, 905}
    assert await repo.last_tg_msg_id(empty) is None
    assert await repo.out_tg_msg_ids(empty) == set()


# --- additions for the agent core (stage 2) -------------------------------------------------


async def test_a_new_conversation_starts_in_planned(repo):
    task_id = await make_task(repo)
    contact_id = await repo.upsert_contact(1001, None, "Продавец", False)
    planned = await repo.create_conversation(task_id, contact_id)
    other = await repo.upsert_contact(1002, None, "Другой", False)
    given = await repo.create_conversation(task_id, other, state=S.CONTACTED)

    assert (await repo.get_conversation(planned)).state is S.PLANNED
    assert (await repo.get_conversation(given)).state is S.CONTACTED


async def test_create_task_keeps_the_owner_s_words(repo):
    words = "Найди, где напечатать деталь на 3D-принтере, до 200 тысяч"
    task_id = await repo.create_task(
        "3D-печать", "цена", {}, None, "UZS", [], [], source_text=words
    )

    assert (await repo.get_task(task_id)).source_text == words
    assert (await repo.get_task(await make_task(repo))).source_text == ""  # the default


async def test_llm_calls_of_a_task_and_its_cost(repo, db_path):
    task_id = await make_task(repo)
    other_task = await make_task(repo, "Другая")
    await repo.log_llm_call(
        None,
        "task",
        "claude-sonnet-5-5",
        Usage(10, 5),
        0.25,
        "end_turn",
        task_id=task_id,
        wake_id=None,
    )
    await repo.log_llm_call(
        None,
        "task",
        "claude-sonnet-5-5",
        Usage(10, 5),
        0.5,
        "tool_use",
        task_id=task_id,
    )
    await repo.log_llm_call(
        None,
        "task",
        "claude-sonnet-5-5",
        Usage(1, 1),
        4.0,
        "end_turn",
        task_id=other_task,
    )
    await repo.log_llm_call(
        None, "intake", "claude-sonnet-5-5", Usage(1, 1), 8.0, "end_turn"
    )

    assert await repo.task_cost(task_id) == pytest.approx(0.75)
    assert await repo.task_cost(other_task) == pytest.approx(4.0)
    assert await repo.task_cost(other_task + 100) == 0.0
    assert raw(db_path, "SELECT task_id, wake_id FROM llm_calls ORDER BY id") == [
        (task_id, None),
        (task_id, None),
        (other_task, None),
        (None, None),
    ]


async def test_llm_calls_record_the_wake(repo, db_path):
    task_id = await make_task(repo)
    async with repo._db.execute(  # a wake row for the foreign key
        "INSERT INTO wakes (task_id, started_at) VALUES (?, '2026-11-10T05:00:00+00:00')",
        (task_id,),
    ) as cursor:
        wake_id = cursor.lastrowid
    await repo.log_llm_call(
        None,
        "task",
        "m",
        Usage(1, 1),
        0.1,
        "end_turn",
        task_id=task_id,
        wake_id=wake_id,
    )

    assert raw(db_path, "SELECT task_id, wake_id FROM llm_calls") == [
        (task_id, wake_id)
    ]


async def test_set_task_notes_writes_what_is_given(repo, fake_clock):
    task_id = await make_task(repo)
    fake_clock.advance(timedelta(minutes=5))

    await repo.set_task_notes(task_id, notes="Али: 150 000, ждём фото")
    task = await repo.get_task(task_id)
    assert (task.notes, task.headline) == ("Али: 150 000, ждём фото", "")
    assert task.updated_at == fake_clock.now()

    fake_clock.advance(timedelta(minutes=5))
    await repo.set_task_notes(task_id, headline="Жду ответа от двоих")
    task = await repo.get_task(task_id)
    assert (task.notes, task.headline) == (
        "Али: 150 000, ждём фото",
        "Жду ответа от двоих",
    )
    assert task.updated_at == fake_clock.now()

    await repo.set_task_notes(task_id, notes="", headline="Готово")
    task = await repo.get_task(task_id)
    assert (task.notes, task.headline) == ("", "Готово")

    with pytest.raises(NotFound):
        await repo.set_task_notes(task_id + 100, notes="x")


async def test_a_candidate_of_a_task_is_added_and_read_back(repo, fake_clock):
    task_id = await make_task(repo)

    candidate_id = await repo.add_candidate(
        task_id, "print_lab", "Студия PrintLab", "печатает PETG", "https://printlab.uz"
    )

    candidate = await repo.get_candidate(candidate_id)
    assert candidate == TaskCandidate(
        id=candidate_id,
        task_id=task_id,
        contact="print_lab",
        name="Студия PrintLab",
        note="печатает PETG",
        source="https://printlab.uz",
        created_at=fake_clock.now(),
    )
    assert_matches_annotations(candidate)
    assert await repo.task_candidates(task_id) == [candidate]


async def test_the_same_contact_again_updates_its_row_and_keeps_its_id(repo):
    task_id = await make_task(repo)
    first = await repo.add_candidate(task_id, "print_lab", "PrintLab", "—", "u1")
    second = await repo.add_candidate(task_id, "+998901234567", "Али", "PLA", "u2")

    again = await repo.add_candidate(
        task_id, "print_lab", "Студия PrintLab", "PETG и PLA", "u3"
    )

    assert again == first
    candidates = await repo.task_candidates(task_id)
    assert [c.id for c in candidates] == [first, second]  # by id
    assert (candidates[0].name, candidates[0].note, candidates[0].source) == (
        "Студия PrintLab",
        "PETG и PLA",
        "u3",
    )


async def test_candidates_of_another_task_are_not_listed(repo):
    task_id = await make_task(repo)
    other_task = await make_task(repo, "Другая")
    await repo.add_candidate(other_task, "print_lab", "PrintLab", "—", "u")
    mine = await repo.add_candidate(task_id, "print_lab", "PrintLab", "—", "u")

    assert [c.id for c in await repo.task_candidates(task_id)] == [mine]
    assert len(await repo.task_candidates(other_task)) == 1
    assert await repo.task_candidates(task_id + 100) == []


async def test_an_unknown_candidate_is_not_found(repo):
    with pytest.raises(NotFound):
        await repo.get_candidate(999)


async def test_max_message_id_of_a_task(repo, fake_clock):
    task_id = await make_task(repo)
    assert await repo.max_message_id(task_id) == 0  # no conversation yet

    first = await make_conversation(repo, peer_id=1, task_id=task_id)
    second = await make_conversation(repo, peer_id=2, task_id=task_id)
    elsewhere = await make_conversation(repo, peer_id=3)
    assert await repo.max_message_id(task_id) == 0  # no message yet

    now = fake_clock.now()
    await say(repo, first, "Здравствуйте", now)
    newest = await say(repo, second, "Цена 100", now)
    await say(repo, elsewhere, "Другая задача", now)  # a higher id, of another task

    assert await repo.max_message_id(task_id) == newest


async def test_max_incoming_id_of_a_conversation(repo, fake_clock):
    conv = await make_conversation(repo, peer_id=1)
    other = await make_conversation(repo, peer_id=2)
    assert await repo.max_incoming_id(conv) == 0  # nothing came in yet

    now = fake_clock.now()
    first = await say(repo, conv, "Здравствуйте", now)
    await say(repo, conv, "Цена?", now, source=MsgSource.AGENT)  # ours: not incoming
    await say(repo, other, "Другая беседа", now)  # a higher id, of another conversation

    assert await repo.max_incoming_id(conv) == first


async def test_enqueue_keeps_the_wake_it_was_queued_in(repo, fake_clock):
    conv = await make_conversation(repo)
    task_id = (await repo.get_conversation(conv)).task_id
    await repo.add_event(task_id, EventKind.CREATED)
    wake_id, _ = await repo.claim_events(task_id, fake_clock.now())

    plain = await repo.enqueue(conv, "Здравствуйте", fake_clock.now())
    from_wake = await repo.enqueue(
        conv, "Есть дрель?", fake_clock.now(), wake_id=wake_id
    )

    assert (await repo.get_outbox(plain)).wake_id is None
    assert (await repo.get_outbox(from_wake)).wake_id == wake_id


async def test_move_if_moves_only_from_the_expected_state(repo):
    conv_id = await make_conversation(repo)
    await walk(repo, conv_id, S.CONTACTED)

    # Not in the expected state: nothing changes.
    assert await repo.move_if(conv_id, S.TALKING, S.WAITING_OWNER) is False
    assert (await repo.get_conversation(conv_id)).state is S.CONTACTED

    assert await repo.move_if(conv_id, S.CONTACTED, S.TALKING) is True
    assert (await repo.get_conversation(conv_id)).state is S.TALKING

    # In the expected state, but the state machine forbids the move: it still raises.
    await walk(repo, conv_id, S.CLOSED)
    with pytest.raises(InvalidTransition):
        await repo.move_if(conv_id, S.CLOSED, S.WAITING_OWNER)
    assert (await repo.get_conversation(conv_id)).state is S.CLOSED


async def test_first_message_slots_give_the_earliest_pending_slot_per_conversation(
    repo, fake_clock
):
    now = fake_clock.now()
    queued = await make_conversation(repo, 1001)
    await repo.enqueue(queued, "второе", now + timedelta(minutes=30))
    await repo.enqueue(queued, "первое", now + timedelta(minutes=15))
    cancelled = await make_conversation(repo, 1002)
    item = await repo.enqueue(cancelled, "отменено", now + timedelta(minutes=5))
    await repo.cancel_pending(cancelled)
    other = await make_conversation(repo, 1003)
    await repo.enqueue(other, "не спрошено", now + timedelta(minutes=1))
    nothing = await make_conversation(repo, 1004)

    slots = await repo.first_message_slots([queued, cancelled, nothing])

    assert slots == {queued: now + timedelta(minutes=15)}
    assert (await repo.get_outbox(item)).status is OutboxStatus.CANCELLED
    assert await repo.first_message_slots([]) == {}


async def test_a_stop_remembers_why_until_the_agent_works_again(db_path, fake_clock):
    repo = await Repo.open(db_path, clock=fake_clock)
    try:
        assert await repo.stopped_why() == ""
        await repo.set_stopped(
            True, why="peer_flood"
        )  # Telegram restricted the account
        assert await repo.is_stopped() and await repo.stopped_why() == "peer_flood"
        await repo.set_stopped(False)
        assert not await repo.is_stopped() and await repo.stopped_why() == ""
        await repo.set_stopped(True)  # the owner's /stop names no reason
        assert await repo.is_stopped() and await repo.stopped_why() == ""
    finally:
        await repo.close()
