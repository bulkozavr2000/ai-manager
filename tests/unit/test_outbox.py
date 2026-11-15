"""The neutral bookkeeping of a sent message (`mark_sent`) and the outbox text helpers."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import timedelta
from pathlib import Path

from tg_agent.clock import FakeClock
from tg_agent.db import FIRST_SENT, Repo
from tg_agent.models import Direction, MsgSource, OutboxStatus
from tg_agent.tasks.outbox import mark_sent, same_text


async def make_conversation(repo: Repo) -> int:
    task_id = await repo.create_task(
        "Дрель",
        "самое дешёвое предложение",
        {"товар": "дрель"},
        500_000.0,
        "UZS",
        [],
        [],
    )
    contact_id = await repo.upsert_contact(1001, "seller", "Продавец", False)
    return await repo.create_conversation(task_id, contact_id)


def timer_rows(db_path: Path) -> list[tuple]:
    with closing(sqlite3.connect(db_path)) as db:
        return db.execute("SELECT * FROM timers").fetchall()


async def test_the_first_message_is_marked_sent_and_counted_without_a_timer(
    repo: Repo, fake_clock: FakeClock, db_path: Path
) -> None:
    conv_id = await make_conversation(repo)
    at = fake_clock.now()
    item = await repo.get_outbox(
        await repo.enqueue(conv_id, "Здравствуйте! {PHONE}", at)
    )

    first = await mark_sent(
        repo, item, await repo.get_conversation(conv_id), 77, at, is_file=False
    )

    assert first is True
    sent = await repo.get_outbox(item.id)
    assert (sent.status, sent.tg_msg_id, sent.sent_at) == (OutboxStatus.SENT, 77, at)
    messages = await repo.last_messages(conv_id, 10)
    assert [(m.direction, m.source, m.kind, m.text, m.tg_msg_id) for m in messages] == [
        (Direction.OUT, MsgSource.AGENT, "text", "Здравствуйте! {PHONE}", 77)
    ]
    conv = await repo.get_conversation(conv_id)
    assert conv.first_sent is True
    assert conv.last_out_at == at
    assert len(await repo.events_of(FIRST_SENT, conv_id)) == 1
    assert timer_rows(db_path) == []


async def test_a_second_message_is_not_the_first(
    repo: Repo, fake_clock: FakeClock
) -> None:
    conv_id = await make_conversation(repo)
    at = fake_clock.now()
    one = await repo.get_outbox(await repo.enqueue(conv_id, "Первое", at))
    two = await repo.get_outbox(await repo.enqueue(conv_id, "Второе", at))
    await mark_sent(
        repo, one, await repo.get_conversation(conv_id), 1, at, is_file=False
    )

    later = at + timedelta(minutes=1)
    first = await mark_sent(
        repo, two, await repo.get_conversation(conv_id), 2, later, is_file=False
    )

    assert first is False
    assert len(await repo.events_of(FIRST_SENT, conv_id)) == 1
    assert (await repo.get_conversation(conv_id)).last_out_at == later


async def test_a_file_goes_into_the_transcript_as_a_document(
    repo: Repo, fake_clock: FakeClock
) -> None:
    conv_id = await make_conversation(repo)
    at = fake_clock.now()
    item = await repo.get_outbox(
        await repo.enqueue(conv_id, "", at, attach_path="model.stl")
    )

    await mark_sent(
        repo, item, await repo.get_conversation(conv_id), 5, at, is_file=True
    )

    [message] = await repo.last_messages(conv_id, 10)
    assert message.kind == "document"


def test_a_placeholder_matches_whatever_stands_in_its_place() -> None:
    assert same_text(
        "Мой номер {PHONE}, адрес {ADDRESS}", "Мой номер +998 90 123, адрес Ц-1"
    )
    assert not same_text("Мой номер {PHONE}", "Мой адрес +998 90 123")


def test_whitespace_is_compared_loosely() -> None:
    assert same_text("Добрый  день,\nесть дрель?", " Добрый день, есть   дрель? ")
    assert not same_text("Добрый день", "Добрый день!")


def test_an_empty_queued_text_matches_nothing() -> None:
    assert not same_text("", "")
    assert not same_text("   ", "что-то")
