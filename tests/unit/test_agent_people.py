"""People of a task: attaching a person, closing and reopening his conversation."""

from __future__ import annotations

from datetime import timedelta

import pytest

from tg_agent.agent.people import (
    OPEN_STATES,
    Attached,
    NotAttached,
    attach,
    close,
    extract_targets,
    reopen,
)
from tg_agent.db import Repo
from tg_agent.gateway.base import PeerInfo, TgActionError
from tg_agent.gateway.fake import FakeGateway
from tg_agent.models import ConvState, EventKind, OutboxStatus, TaskStatus
from tg_agent.policy.tg_safety import TgErrorKind

S = ConvState


async def new_task(repo: Repo, title: str = "Мастер") -> int:
    task_id = await repo.create_task(title, "найти мастера", {}, None, "UZS", [], [])
    await repo.set_task_status(task_id, TaskStatus.ACTIVE)
    return task_id


def peer(peer_id: int, username: str, *, is_bot: bool = False) -> PeerInfo:
    return PeerInfo(peer_id, username, username.title(), is_bot)


@pytest.fixture
def gateway() -> FakeGateway:
    gw = FakeGateway()
    gw.peers["ivan"] = peer(501, "ivan")
    return gw


async def conv_in(repo: Repo, task_id: int, peer_id: int, *path: ConvState) -> int:
    contact_id = await repo.upsert_contact(peer_id, f"u{peer_id}", f"U{peer_id}", False)
    conv_id = await repo.create_conversation(task_id, contact_id, state=S.PLANNED)
    for state in path:
        await repo.set_state(conv_id, state)
    return conv_id


def test_open_states() -> None:
    assert OPEN_STATES == {
        S.PLANNED,
        S.CONTACTED,
        S.TALKING,
        S.WAITING_OWNER,
        S.HANDED_TO_OWNER,
    }


# -- attach --


async def test_attach_creates_a_planned_conversation(
    repo: Repo, gateway: FakeGateway
) -> None:
    task_id = await new_task(repo)

    result = await attach(repo, gateway, task_id, "ivan", found="OLX: объявление")

    assert isinstance(result, Attached)
    conv = await repo.get_conversation(result.conv_id)
    assert conv.task_id == task_id
    assert conv.state is S.PLANNED
    assert result.contact.peer_id == 501
    contact = await repo.get_contact(conv.contact_id)
    assert contact.card["found"] == "OLX: объявление"
    assert gateway.allowed == {501}
    assert gateway.calls[-1] == "set_allowed"


async def test_attach_without_found_leaves_the_card(
    repo: Repo, gateway: FakeGateway
) -> None:
    task_id = await new_task(repo)

    result = await attach(repo, gateway, task_id, "@ivan")

    assert isinstance(result, Attached)
    assert "found" not in (await repo.get_contact(result.contact.id)).card


@pytest.mark.parametrize(
    ("target", "error", "reason"),
    [
        (
            "ivan",
            TgActionError(TgErrorKind.FLOOD_WAIT, "wait", 30),
            "не удалось открыть (flood_wait)",
        ),
        (
            "+998901234567",
            None,
            "нет в Telegram или он скрыт по номеру — позвони сам",
        ),
        ("nobody", None, "не найден — проверь имя"),
    ],
)
async def test_attach_reports_a_person_who_cannot_be_resolved(
    repo: Repo,
    gateway: FakeGateway,
    target: str,
    error: TgActionError | None,
    reason: str,
) -> None:
    task_id = await new_task(repo)
    gateway.fail_next = error

    result = await attach(repo, gateway, task_id, target)

    assert result == NotAttached(reason)
    assert await repo.task_conversations(task_id) == []
    assert "set_allowed" not in gateway.calls


async def test_attach_refuses_a_bot(repo: Repo, gateway: FakeGateway) -> None:
    task_id = await new_task(repo)
    gateway.peers["shopbot"] = peer(777, "shopbot", is_bot=True)

    result = await attach(repo, gateway, task_id, "shopbot")

    assert result == NotAttached("это бот: агент с ботами не переписывается")
    assert await repo.task_conversations(task_id) == []


async def test_attach_refuses_a_person_who_asked_not_to_be_written_to(
    repo: Repo, gateway: FakeGateway
) -> None:
    task_id = await new_task(repo)
    contact_id = await repo.upsert_contact(501, "ivan", "Ivan", False)
    await repo.set_do_not_contact(contact_id, True)

    result = await attach(repo, gateway, task_id, "ivan")

    assert result == NotAttached("просил не писать")
    assert await repo.task_conversations(task_id) == []


async def test_attach_refuses_a_person_already_in_the_task(
    repo: Repo, gateway: FakeGateway
) -> None:
    task_id = await new_task(repo)
    assert isinstance(await attach(repo, gateway, task_id, "ivan"), Attached)

    result = await attach(repo, gateway, task_id, "ivan")

    assert result == NotAttached("уже в задаче")
    assert len(await repo.task_conversations(task_id)) == 1


async def test_attach_refuses_a_person_busy_in_another_task(
    repo: Repo, gateway: FakeGateway
) -> None:
    other = await new_task(repo, "Другая")
    assert isinstance(await attach(repo, gateway, other, "ivan"), Attached)
    task_id = await new_task(repo)

    result = await attach(repo, gateway, task_id, "ivan")

    assert result == NotAttached(f"уже в задаче #{other} — пока она не закончится")
    assert await repo.task_conversations(task_id) == []


# -- close --


async def test_close_writes_the_reason_and_drops_the_silence(
    repo: Repo, gateway: FakeGateway, fake_clock
) -> None:
    task_id = await new_task(repo)
    attached = await attach(repo, gateway, task_id, "ivan")
    assert isinstance(attached, Attached)
    conv_id = attached.conv_id
    await repo.set_state(conv_id, S.CONTACTED)
    await repo.arm_silence(task_id, conv_id, fake_clock.now() + timedelta(hours=24))
    assert gateway.allowed == {501}

    assert await close(repo, gateway, conv_id, "отказался") is True

    conv = await repo.get_conversation(conv_id)
    assert conv.state is S.CLOSED
    assert conv.close_reason == "отказался"
    assert [
        e for e in await repo.open_events(task_id) if e.kind is EventKind.SILENCE
    ] == []
    assert gateway.allowed == {501}  # his task is in work: he is still heard (R104 b)

    assert await close(repo, gateway, conv_id, "ещё раз") is False
    assert (await repo.get_conversation(conv_id)).close_reason == "отказался"


async def test_close_refuses_a_move_the_state_machine_forbids(
    repo: Repo, gateway: FakeGateway
) -> None:
    task_id = await new_task(repo)
    contact_id = await repo.upsert_contact(501, "ivan", "Ivan", False)
    # Closed already: CLOSED cannot move to CLOSED again.
    conv_id = await repo.create_conversation(task_id, contact_id, state=S.CLOSED)
    await repo.update_conversation(conv_id, close_reason="отказ")

    assert await close(repo, gateway, conv_id, "закрыть") is False

    conv = await repo.get_conversation(conv_id)
    assert conv.state is S.CLOSED
    assert conv.close_reason == "отказ"


# -- reopen --


@pytest.mark.parametrize("waiting", [S.WAITING_OWNER, S.HANDED_TO_OWNER])
async def test_reopen_returns_to_talking_when_the_person_wrote(
    repo: Repo, fake_clock, waiting: ConvState
) -> None:
    task_id = await new_task(repo)
    conv_id = await conv_in(repo, task_id, 501, S.CONTACTED, S.TALKING, waiting)
    await repo.update_conversation(
        conv_id, last_in_at=fake_clock.now(), first_sent=True
    )

    assert await reopen(repo, conv_id) is S.TALKING
    assert (await repo.get_conversation(conv_id)).state is S.TALKING


@pytest.mark.parametrize("waiting", [S.WAITING_OWNER, S.HANDED_TO_OWNER])
async def test_reopen_returns_to_contacted_after_our_first_message(
    repo: Repo, waiting: ConvState
) -> None:
    task_id = await new_task(repo)
    conv_id = await conv_in(repo, task_id, 501, S.CONTACTED, waiting)
    await repo.update_conversation(conv_id, first_sent=True)

    assert await reopen(repo, conv_id) is S.CONTACTED
    assert (await repo.get_conversation(conv_id)).state is S.CONTACTED


@pytest.mark.parametrize(
    "status", [OutboxStatus.PENDING, OutboxStatus.SENDING, OutboxStatus.UNKNOWN]
)
async def test_reopen_returns_to_contacted_while_the_first_message_is_on_its_way(
    repo: Repo, fake_clock, status: OutboxStatus
) -> None:
    task_id = await new_task(repo)
    conv_id = await conv_in(repo, task_id, 501, S.WAITING_OWNER)
    item = await repo.enqueue(conv_id, "Здравствуйте", fake_clock.now())
    if status is not OutboxStatus.PENDING:
        await repo.mark_outbox(item, status)

    assert await reopen(repo, conv_id) is S.CONTACTED


@pytest.mark.parametrize("waiting", [S.WAITING_OWNER, S.HANDED_TO_OWNER])
async def test_reopen_returns_to_planned_when_nothing_went(
    repo: Repo, fake_clock, waiting: ConvState
) -> None:
    task_id = await new_task(repo)
    conv_id = await conv_in(repo, task_id, 501, waiting)
    # A message that will never go does not count.
    item = await repo.enqueue(conv_id, "Здравствуйте", fake_clock.now())
    await repo.mark_outbox(item, OutboxStatus.CANCELLED)

    assert await reopen(repo, conv_id) is S.PLANNED
    assert (await repo.get_conversation(conv_id)).state is S.PLANNED


@pytest.mark.parametrize(
    "path", [(S.CONTACTED, S.TALKING), (S.CLOSED,)], ids=["talking", "closed"]
)
async def test_reopen_leaves_other_states_alone(
    repo: Repo, fake_clock, path: tuple[ConvState, ...]
) -> None:
    task_id = await new_task(repo)
    conv_id = await conv_in(repo, task_id, 501, *path)
    await repo.update_conversation(conv_id, last_in_at=fake_clock.now())

    assert await reopen(repo, conv_id) is path[-1]
    assert (await repo.get_conversation(conv_id)).state is path[-1]


@pytest.mark.parametrize("meanwhile", [S.TALKING, S.CLOSED])
async def test_reopen_leaves_a_conversation_moved_meanwhile(
    repo: Repo, fake_clock, monkeypatch, meanwhile: ConvState
) -> None:
    task_id = await new_task(repo)
    conv_id = await conv_in(repo, task_id, 501, S.CONTACTED, S.WAITING_OWNER)
    outbox_of = repo.outbox_of

    async def moved_first(cid: int):
        # A wake moves the person between reopen's read and its move.
        await repo.set_state(conv_id, meanwhile)
        return await outbox_of(cid)

    monkeypatch.setattr(repo, "outbox_of", moved_first)

    assert await reopen(repo, conv_id) is meanwhile
    assert (await repo.get_conversation(conv_id)).state is meanwhile


# -- the contacts the owner names (moved here from the old intake, Task 5.3) --


def test_extract_targets() -> None:
    text = "у @print_lab и https://t.me/maker_uz, а ещё @3d"
    assert extract_targets(text) == ["print_lab", "maker_uz"]


def test_targets_keep_order_and_drop_duplicates() -> None:
    text = "@maker_uz, t.me/print_lab и снова @Maker_Uz и telegram.me/print_lab"
    assert extract_targets(text) == ["maker_uz", "print_lab"]


def test_email_addresses_are_not_targets() -> None:
    assert extract_targets("пиши на said@example.com или @print_lab") == [
        "print_lab"
    ]


def test_usernames_longer_than_32_are_ignored() -> None:
    assert extract_targets("@" + "a" * 33) == []


def test_phones_are_targets_too_in_one_form() -> None:
    text = "у @print_lab и +998 90 123-45-67, ещё раз +998901234567, бюджет 300 000 000"
    assert extract_targets(text) == ["print_lab", "+998901234567"]
