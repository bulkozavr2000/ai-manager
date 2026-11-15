"""The memory about the owner (`agent/memory.py`, ruling R77) and its one door, `people.attach`."""

from __future__ import annotations

import json

import pytest

from tg_agent.agent import memory
from tg_agent.agent.people import Attached, NotAttached, attach
from tg_agent.brain.context import HIDDEN_AMOUNT
from tg_agent.clock import FakeClock
from tg_agent.db import Repo
from tg_agent.gateway.base import PeerInfo
from tg_agent.gateway.fake import FakeGateway
from tg_agent.models import TaskStatus
from tg_agent.secrets import Key, set_secret


async def test_a_fact_is_kept_with_its_id_and_rendered_numbered(
    repo: Repo, fake_clock: FakeClock
) -> None:
    line, kept = await memory.remember(
        repo, "Я в Ташкенте,  встречи удобны после 15:00", None, fake_clock.now()
    )
    await memory.remember(repo, "Не работать с этим магазином", "bad_shop", fake_clock.now())

    assert kept and line == "Запомнил: Я в Ташкенте, встречи удобны после 15:00"
    stored = json.loads(await repo.get_kv(memory.KEY) or "")
    assert stored[0] == {
        "id": 1,
        "fact": "Я в Ташкенте, встречи удобны после 15:00",
        "contact": None,
        "at": fake_clock.now().isoformat(),
    }
    assert memory.render(await memory.load(repo)) == [
        "1. Я в Ташкенте, встречи удобны после 15:00",
        "2. Не работать с этим магазином (не работать: @bad_shop)",
    ]
    assert await memory.excluded(repo) == {"bad_shop"}


@pytest.mark.parametrize(
    "fact",
    [
        "мой номер +998 90 123 45 67",
        "карта 4111 1111 1111 1111",
        "почта said@example.com",
    ],
)
async def test_a_phone_a_card_or_an_e_mail_is_refused(
    repo: Repo, fake_clock: FakeClock, fact: str
) -> None:
    assert await memory.remember(repo, fact, None, fake_clock.now()) == (
        memory.PRIVATE,
        False,
    )
    assert await memory.load(repo) == []


async def test_a_sum_is_masked(repo: Repo, fake_clock: FakeClock) -> None:
    line, kept = await memory.remember(
        repo, "обычно трачу на подарки до 500 000 сум", None, fake_clock.now()
    )

    [fact] = await memory.load(repo)
    assert kept and "500" not in fact.fact + line
    assert HIDDEN_AMOUNT in fact.fact


async def test_the_201st_character_and_the_41st_fact_are_refused(
    repo: Repo, fake_clock: FakeClock
) -> None:
    now = fake_clock.now()
    assert await memory.remember(repo, "я" * 201, None, now) == (memory.TOO_LONG, False)
    assert (await memory.remember(repo, "я" * 200, None, now))[1]
    for n in range(39):
        assert (await memory.remember(repo, f"факт {n}", None, now))[1]

    assert await memory.remember(repo, "ещё один", None, now) == (memory.FULL, False)
    assert len(await memory.load(repo)) == 40


async def test_forget_drops_the_fact_by_its_id_and_ids_are_not_reused(
    repo: Repo, fake_clock: FakeClock
) -> None:
    now = fake_clock.now()
    await memory.remember(repo, "встречи после 15", None, now)
    await memory.remember(repo, "с ними не работать", "bad_shop", now)

    assert await memory.forget(repo, 2) == (
        "Забыл: с ними не работать (не работать: @bad_shop)",
        True,
    )
    assert await memory.forget(repo, 2) == ("В памяти нет факта №2", False)
    await memory.remember(repo, "я в Ташкенте", None, now)

    assert [f.id for f in await memory.load(repo)] == [1, 3]
    assert await memory.excluded(repo) == set()


# -- the one door: people.attach --


async def test_attach_refuses_whom_he_said_not_to_work_with_until_he_forgets(
    repo: Repo, fake_clock: FakeClock
) -> None:
    gateway = FakeGateway()
    gateway.peers["bad_shop"] = PeerInfo(501, "bad_shop", "Bad", False)
    gateway.peers["+998901112233"] = PeerInfo(501, "bad_shop", "Bad", False)
    task_id = await repo.create_task("Т", "найти", {}, None, "UZS", [], [])
    await repo.set_task_status(task_id, TaskStatus.ACTIVE)
    await memory.remember(repo, "с ними не работать", "Bad_Shop", fake_clock.now())

    by_name = await attach(repo, gateway, task_id, "bad_shop")
    by_phone = await attach(repo, gateway, task_id, "+998901112233")  # resolves to him

    assert by_name == NotAttached(memory.NOT_WITH)
    assert by_phone == NotAttached(memory.NOT_WITH)
    assert await repo.task_conversations(task_id) == []
    await memory.forget(repo, 1)
    assert isinstance(await attach(repo, gateway, task_id, "bad_shop"), Attached)


# -- ruling R91: the owner's address and phone never enter the memory --


@pytest.fixture
def owner_data() -> None:
    """The made-up phone and address the «Дать телефон?» / «Дать адрес?» cards give."""
    set_secret(Key.OWNER_PHONE, "+998 90 111 22 33")
    set_secret(Key.OWNER_ADDRESS, "Чиланзар 5, дом 12, кв. 30")


@pytest.mark.parametrize(
    "fact",
    [
        "Доставка всегда на Чиланзар 5, дом 12, кв. 30",
        "доставка: ЧИЛАНЗАР-5 дом 12 кв.30, после обеда",
        "мой номер +998 90 111 22 33",
        "звонить на 998901112233",
    ],
)
async def test_his_stored_address_or_phone_is_refused_whatever_the_spelling(
    repo: Repo, fake_clock: FakeClock, owner_data: None, fact: str
) -> None:
    assert await memory.remember(repo, fact, None, fake_clock.now()) == (
        memory.OWNER_DATA,
        False,
    )
    assert await memory.load(repo) == []
    assert "Дать адрес?" in memory.OWNER_DATA and "Дать телефон?" in memory.OWNER_DATA


async def test_a_fact_that_only_names_his_district_is_kept(
    repo: Repo, fake_clock: FakeClock, owner_data: None
) -> None:
    line, kept = await memory.remember(
        repo, "Доставка на Чиланзар, после 15:00", None, fake_clock.now()
    )

    assert kept and line == "Запомнил: Доставка на Чиланзар, после 15:00"


async def test_with_no_stored_address_nothing_is_compared(
    repo: Repo, fake_clock: FakeClock
) -> None:
    assert (
        await memory.remember(repo, "Доставка на Чиланзар 5, дом 12", None, fake_clock.now())
    )[1]


# -- fix round C of stage 6: R102 — «не работай с +998…» --


async def test_the_excluded_contact_s_own_phone_in_the_fact_is_kept_and_shown_masked(
    repo: Repo, fake_clock: FakeClock, owner_data: None
) -> None:
    fact = "не работать с +998 90 123 45 67, не отвечает на звонки"

    line, kept = await memory.remember(repo, fact, "+998901234567", fake_clock.now())

    assert kept
    assert "123 45" not in line and "+998 90 ••• 45 67" in line
    (stored,) = await memory.load(repo)
    assert stored.contact == "+998901234567"
    assert "123" not in stored.fact and "не отвечает на звонки" in stored.fact
    assert await memory.excluded(repo) == {"+998901234567"}


@pytest.mark.parametrize(
    ("fact", "contact"),
    [
        ("не работать с ним, звонить на +998 90 555 66 77", "+998901234567"),
        ("не работать с @bad_shop, его номер +998 90 555 66 77", "bad_shop"),
    ],
)
async def test_another_phone_beside_the_excluded_contact_is_still_refused(
    repo: Repo, fake_clock: FakeClock, fact: str, contact: str
) -> None:
    assert await memory.remember(repo, fact, contact, fake_clock.now()) == (
        memory.PRIVATE,
        False,
    )


async def test_his_own_phone_as_the_contact_is_still_refused(
    repo: Repo, fake_clock: FakeClock, owner_data: None
) -> None:
    line, kept = await memory.remember(
        repo, "не работать с +998 90 111 22 33", "+998901112233", fake_clock.now()
    )
    assert (line, kept) == (memory.OWNER_DATA, False)
