from __future__ import annotations

import logging

import pytest

from tg_agent.control.base import Card, CardGone, FakeUI
from tg_agent.control.chatlog import ChatLog, LoggedUI
from tg_agent.db import Repo
from tg_agent.models import ApprovalKind, Direction, OwnerChatEntry


def logged(repo: Repo) -> tuple[LoggedUI, FakeUI]:
    inner = FakeUI()
    return LoggedUI(inner, ChatLog(repo)), inner


async def test_fake_ui_gives_unique_increasing_ids_across_send_and_notify() -> None:
    ui = FakeUI()
    ids = [
        await ui.send(Card("a")),
        await ui.notify("b"),
        await ui.send(Card("c")),
        await ui.notify("d"),
    ]
    assert ids == sorted(ids) and len(set(ids)) == 4
    assert ui.sent_ids == [ids[0], ids[2]] and ui.note_ids == [ids[1], ids[3]]
    assert [c.text for c in ui.sent] == ["a", "c"] and ui.notes == ["b", "d"]


async def test_send_returns_the_inner_id_and_logs_the_card(repo: Repo) -> None:
    ui, inner = logged(repo)
    await inner.notify("before")  # so the inner id differs from the row count
    task_id = await repo.create_task("3D-печать", "Цена", {}, None, "UZS", [], [])
    asked = await repo.create_approval(task_id, None, ApprovalKind.SHARE_PHONE, {})
    approval = Card("Дать телефон?", kind="approval", approval_id=asked.id)
    task = Card("Задача", task_id=task_id, kind="task")
    plain = Card("Просто")
    ids = [await ui.send(approval), await ui.send(task), await ui.send(plain)]
    assert ids == inner.sent_ids
    rows = await repo.owner_chat_tail(10)
    assert [(r.direction, r.kind, r.text, r.tg_msg_id) for r in rows] == [
        (Direction.OUT, "approval", "Дать телефон?", ids[0]),
        (Direction.OUT, "task", "Задача", ids[1]),
        (Direction.OUT, "card", "Просто", ids[2]),
    ]
    assert [(r.task_id, r.approval_id) for r in rows] == [
        (None, asked.id),
        (task_id, None),
        (None, None),
    ]
    assert rows[0].reply_to_msg_id is None and rows[0].handled_at is None


async def test_notify_returns_the_id_and_logs_a_notice(repo: Repo) -> None:
    ui, inner = logged(repo)
    msg_id = await ui.notify("Готово")
    assert inner.note_ids == [msg_id]
    (row,) = await repo.owner_chat_tail(5)
    assert (row.direction, row.kind, row.text, row.tg_msg_id) == (
        Direction.OUT,
        "notice",
        "Готово",
        msg_id,
    )


async def test_edit_replaces_the_logged_text_and_adds_no_row(repo: Repo) -> None:
    ui, inner = logged(repo)
    first = await ui.send(Card("Было", kind="task"))
    second = await ui.send(Card("Другая"))
    await ui.edit(first, Card("Стало", kind="task"))
    assert inner.edits[-1][0] == first and inner.edits[-1][1].text == "Стало"
    rows = await repo.owner_chat_tail(5)
    assert [(r.tg_msg_id, r.text) for r in rows] == [
        (first, "Стало"),
        (second, "Другая"),
    ]


async def test_the_board_is_never_logged(repo: Repo) -> None:
    ui, inner = logged(repo)
    msg_id = await ui.send(Card("Статус задач", kind="board"))
    await ui.edit(msg_id, Card("Статус задач\n#1", kind="board"))
    await ui.pin(msg_id)
    assert inner.sent_ids == [msg_id] and inner.pins == [msg_id]
    assert len(inner.edits) == 1
    assert await repo.owner_chat_tail(5) == []


class BrokenLogRepo:
    """Stands in for a Repo whose writes fail (a locked or broken database)."""

    async def add_owner_chat(self, *args: object, **kwargs: object) -> int:
        raise RuntimeError("database is locked")

    async def set_owner_chat_text(self, *args: object, **kwargs: object) -> bool:
        raise RuntimeError("database is locked")


async def test_a_failing_log_never_stops_a_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    inner = FakeUI()
    ui = LoggedUI(inner, ChatLog(BrokenLogRepo()))  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING):
        card_id = await ui.send(Card("Карточка"))
        note_id = await ui.notify("Заметка")
        await ui.edit(card_id, Card("Новая"))
        await ChatLog(BrokenLogRepo()).incoming("text", "привет", tg_msg_id=3)  # type: ignore[arg-type]
    assert inner.sent_ids == [card_id] and inner.note_ids == [note_id]
    assert inner.edits[-1][1].text == "Новая"
    assert "owner chat" in caplog.text


class GoneUI(FakeUI):
    async def edit(self, msg_id: int, card: Card) -> None:
        raise CardGone("message to edit not found")


async def test_an_error_of_the_inner_ui_propagates(repo: Repo) -> None:
    ui = LoggedUI(GoneUI(), ChatLog(repo))
    msg_id = await ui.send(Card("Было"))
    with pytest.raises(CardGone):
        await ui.edit(msg_id, Card("Стало"))
    (row,) = await repo.owner_chat_tail(5)
    assert row.text == "Было"


async def test_flush_passes_through(repo: Repo) -> None:
    ui, _ = logged(repo)
    assert await ui.flush() == 0


async def test_incoming_messages_are_logged_with_their_kind_and_ids(
    repo: Repo,
) -> None:
    chatlog = ChatLog(repo)
    await chatlog.incoming("text", "добавь @x", tg_msg_id=41, reply_to_msg_id=40)
    await chatlog.incoming("command", "/tasks")
    await chatlog.incoming("button", "tk:1:launch:abcdef", tg_msg_id=40)
    await chatlog.incoming("file", "макет.stl")
    rows = await repo.owner_chat_tail(10)
    assert [
        (r.direction, r.kind, r.text, r.tg_msg_id, r.reply_to_msg_id) for r in rows
    ] == [
        (Direction.IN, "text", "добавь @x", 41, 40),
        (Direction.IN, "command", "/tasks", None, None),
        (Direction.IN, "button", "tk:1:launch:abcdef", 40, None),
        (Direction.IN, "file", "макет.stl", None, None),
    ]


async def test_the_tail_is_the_latest_entries_oldest_first(repo: Repo) -> None:
    for n in range(1, 4):
        await repo.add_owner_chat("out", "notice", f"#{n}", tg_msg_id=n)
    tail = await repo.owner_chat_tail(2)
    assert all(isinstance(entry, OwnerChatEntry) for entry in tail)
    assert [entry.text for entry in tail] == ["#2", "#3"]
    assert tail[0].created_at.tzinfo is not None
    assert await repo.owner_chat_tail(0) == []


async def test_setting_the_text_of_an_unknown_message_is_false(repo: Repo) -> None:
    await repo.add_owner_chat("in", "text", "привет", tg_msg_id=5)
    assert await repo.set_owner_chat_text(5, "x") is False  # only bot messages change
    assert await repo.set_owner_chat_text(99, "x") is False
    await repo.add_owner_chat("out", "notice", "старое", tg_msg_id=6)
    await repo.add_owner_chat("out", "notice", "новое", tg_msg_id=6)
    assert await repo.set_owner_chat_text(6, "правка") is True
    assert [e.text for e in await repo.owner_chat_tail(5)] == [
        "привет",
        "старое",
        "правка",
    ]


async def test_a_deleted_card_is_logged_as_moved_and_a_silent_send_goes_through(
    repo: Repo,
) -> None:
    # Stage 5 (R52): a card moved to the bottom is deleted and sent again; its old row says
    # so instead of repeating the card, and the new one is sent without sound.
    ui, inner = logged(repo)
    first = await ui.send(Card("Вопрос", kind="approval"))
    await ui.delete(first)
    again = await ui.send(Card("Вопрос", kind="approval"), silent=True)
    assert inner.deleted == [first] and inner.silent == [again]
    rows = await repo.owner_chat_tail(5)
    assert [(r.text, r.tg_msg_id) for r in rows] == [
        ("⬇ перенесено ниже", first),
        ("Вопрос", again),
    ]
