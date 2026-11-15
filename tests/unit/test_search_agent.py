"""The vendor search: Claude with tools (web, OLX, Telegram's public search), its candidates,
and what keeps it honest and cheap."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tg_agent.brain.llm import FakeLLM, LLMError, ModelRefusal, SearchStep
from tg_agent.brain.schemas import Usage
from tg_agent.clock import FakeClock
from tg_agent.config import Search, Settings
from tg_agent.db import Repo
from tg_agent.gateway.base import PublicPeer
from tg_agent.gateway.fake import FakeGateway
from tg_agent.models import Task
from tg_agent.search.agent import (
    NOT_LINKED,
    NOT_SEEN,
    PAGES_USED,
    Candidate,
    Searcher,
    Wall,
)
from tg_agent.search.olx import (
    BASE,
    CAPTCHA,
    LOGIN_URL,
    NEEDS_LOGIN,
    REVEALS_USED,
    Listing,
    ListingDetails,
    OlxError,
)

LISTING = "https://www.olx.uz/d/obyavlenie/3d-pechat-zapchastey-ID4wDKf.html"


class FakeOlx:
    def __init__(self) -> None:
        self.searched: list[str] = []
        self.opened: list[str] = []
        self.fail: OlxError | None = None
        self.phone_note = ""  # set: the listing shows no phone, and says why

    async def search(self, query: str) -> list[Listing]:
        self.searched.append(query)
        if self.fail:
            raise self.fail
        return [
            Listing(LISTING, "3D печать запчастей", "", "Ташкент, Яшнабадский район")
        ]

    async def listing(self, url: str) -> ListingDetails:
        self.opened.append(url)
        return ListingDetails(
            url=url,
            title="3D печать запчастей",
            price="от 25 000 сум",
            location="Ташкент, Яшнабадский район",
            seller="Абдулхамид",
            description="Печатаем PETG и PLA, модели по образцу",
            phones=[] if self.phone_note else ["+998901234567"],
            phone_note=self.phone_note,
        )


def call(tool: str, tool_id: str, /, **arguments: object) -> dict:
    return {"type": "tool_use", "id": tool_id, "name": tool, "input": arguments}


def text(words: str) -> dict:
    return {"type": "text", "text": words}


def fetched(url: str, page: str) -> list[dict]:
    """A web_fetch Anthropic ran during the turn: the call and its result."""
    return [
        {
            "type": "server_tool_use",
            "id": "srv1",
            "name": "web_fetch",
            "input": {"url": url},
        },
        {
            "type": "web_fetch_tool_result",
            "tool_use_id": "srv1",
            "content": {
                "type": "web_fetch_result",
                "url": url,
                "content": {
                    "type": "document",
                    "source": {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": page,
                    },
                },
            },
        },
    ]


async def a_task(repo: Repo, ceiling: float | None = 250_000) -> Task:
    task_id = await repo.create_task(
        "Печать корпусов",
        "Узнать цену печати 5 корпусов из PETG",
        {"what": "корпус для платы", "quantity": 5, "deadline": None},
        ceiling,
        "UZS",
        ["name"],
        [],
    )
    return await repo.get_task(task_id)


def asked(task: Task) -> str:
    """What the task's agent would ask the search for this task (`run_query`'s `what`)."""
    return f"{task.title}: {task.goal}"


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def olx() -> FakeOlx:
    return FakeOlx()


@pytest.fixture
def gateway() -> FakeGateway:
    return FakeGateway()


@pytest.fixture
def searcher(
    repo: Repo, settings: Settings, llm: FakeLLM, gateway: FakeGateway, olx: FakeOlx
):
    return Searcher(llm, repo, settings, gateway, olx)


def last_results(llm: FakeLLM) -> list[dict]:
    """The tool results the newest search request sent back."""
    _, request = llm.requests[-1]
    return request["messages"][-1]["content"]


async def test_a_vendor_found_on_olx_comes_back_with_where_he_was_found(
    searcher: Searcher, repo: Repo, llm: FakeLLM, olx: FakeOlx
) -> None:
    llm.search_steps.extend(
        [
            [call("olx_search", "t1", query="3д печать")],
            [call("olx_listing", "t2", url=LISTING)],
            [
                call(
                    "save_candidate",
                    "t3",
                    name="Абдулхамид, 3D-печать",
                    contact="+998 90 123-45-67",
                    source=LISTING,
                    note="печатает PETG, от 25 000 сум, Яшнабадский район",
                )
            ],
            [text("Нашёл одного на OLX")],
        ]
    )

    outcome = await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert outcome.candidates == [
        Candidate(
            name="Абдулхамид, 3D-печать",
            contact="+998901234567",
            source=LISTING,
            note="печатает PETG, от 25 000 сум, Яшнабадский район",
        )
    ]
    assert olx.searched == ["3д печать"] and olx.opened == [LISTING]
    assert outcome.summary == "Нашёл одного на OLX"
    assert outcome.stopped is None
    [saved] = last_results(llm)
    assert saved["tool_use_id"] == "t3" and "Сохранено" in saved["content"]


async def test_a_contact_no_tool_showed_is_not_saved(
    searcher: Searcher, repo: Repo, llm: FakeLLM
) -> None:
    llm.search_steps.extend(
        [
            [
                call(
                    "save_candidate",
                    "t1",
                    name="Мастер",
                    contact="+998 90 000 00 00",
                    source="https://example.uz",
                    note="—",
                )
            ],
            [text("Всё")],
        ]
    )

    outcome = await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert outcome.candidates == []
    [refused] = last_results(llm)
    assert refused["is_error"] is True
    assert "web_fetch" in refused["content"]  # it is told how to make it count


async def test_a_page_read_with_web_fetch_counts_as_seen(
    searcher: Searcher, repo: Repo, llm: FakeLLM
) -> None:
    llm.search_steps.extend(
        [
            [
                *fetched(
                    "https://printlab.uz", "Студия 3D-печати. Telegram: @print_lab"
                ),
                call(
                    "save_candidate",
                    "t1",
                    name="Студия PrintLab",
                    contact="@print_lab",
                    source="https://printlab.uz",
                    note="студия 3D-печати",
                ),
            ],
            [text("Готово")],
        ]
    )

    outcome = await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert [c.contact for c in outcome.candidates] == ["print_lab"]


async def test_telegram_search_finds_public_accounts(
    searcher: Searcher, repo: Repo, llm: FakeLLM, gateway: FakeGateway
) -> None:
    gateway.public = [
        PublicPeer("print3d_tash", "3D печать Ташкент", "channel", "Заказы: @ali_3d"),
    ]
    llm.search_steps.extend(
        [
            [call("telegram_search", "t1", query="3D печать Ташкент")],
            [
                call(
                    "save_candidate",
                    "t2",
                    name="Али, канал 3D печать",
                    contact="@ali_3d",
                    source="https://t.me/print3d_tash",
                    note="канал о 3D-печати",
                )
            ],
            [text("Готово")],
        ]
    )

    outcome = await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert gateway.searches == ["3D печать Ташкент"]
    assert [c.contact for c in outcome.candidates] == ["ali_3d"]


async def test_telegram_search_is_capped_per_search(
    repo: Repo, settings: Settings, llm: FakeLLM, gateway: FakeGateway, olx: FakeOlx
) -> None:
    settings = replace(settings, search=replace(settings.search, telegram_searches=1))
    searcher = Searcher(llm, repo, settings, gateway, olx)
    llm.search_steps.extend(
        [
            [call("telegram_search", "t1", query="3D печать")],
            [call("telegram_search", "t2", query="3D принтер")],
            [text("Всё")],
        ]
    )

    await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert gateway.searches == ["3D печать"]
    [refused] = last_results(llm)
    assert refused["is_error"] is True


async def test_olx_listing_opens_only_olx_listings(
    searcher: Searcher, repo: Repo, llm: FakeLLM, olx: FakeOlx
) -> None:
    llm.search_steps.extend(
        [
            [call("olx_listing", "t1", url="https://evil.example/olx.uz/x.html")],
            [text("Всё")],
        ]
    )

    await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert olx.opened == []
    [refused] = last_results(llm)
    assert refused["is_error"] is True


async def test_an_olx_captcha_is_reported_not_solved(
    searcher: Searcher, repo: Repo, llm: FakeLLM, olx: FakeOlx
) -> None:
    olx.fail = OlxError("OLX просит капчу: открой OLX в браузере и пройди её сам")
    llm.search_steps.extend(
        [[call("olx_search", "t1", query="3д печать")], [text("Всё")]]
    )

    outcome = await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert any("капчу" in note for note in outcome.notes)
    [refused] = last_results(llm)
    assert refused["is_error"] is True


async def test_an_olx_listing_that_wants_a_login_is_a_wall_at_olx_s_sign_in(
    searcher: Searcher, repo: Repo, llm: FakeLLM, olx: FakeOlx
) -> None:
    # Task 4б.2: the note stays, and the wall leads the sign-in card to OLX's own sign-in.
    olx.phone_note = NEEDS_LOGIN
    llm.search_steps.extend(
        [
            [call("olx_listing", "t1", url=LISTING)],
            [call("olx_listing", "t2", url=LISTING + "?again")],
            [text("Всё")],
        ]
    )

    outcome = await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert outcome.notes == [f"OLX: {NEEDS_LOGIN}"]
    assert outcome.walls == [Wall("olx.uz", LOGIN_URL, "login")]


async def test_over_the_day_s_cap_the_listing_says_so_to_the_model(
    searcher: Searcher, repo: Repo, llm: FakeLLM, olx: FakeOlx
) -> None:
    # Ruling R51: the listing is read without «Показать телефон», and the answer says why.
    olx.phone_note = REVEALS_USED
    llm.search_steps.extend([[call("olx_listing", "t1", url=LISTING)], [text("Всё")]])

    outcome = await searcher.run_query(asked(await a_task(repo)), skip=[])

    [shown] = last_results(llm)
    assert "Телефон: дневной предел показов номеров на OLX исчерпан" in shown["content"]
    assert outcome.walls == []


async def test_an_olx_captcha_is_a_wall_and_another_failure_is_not(
    searcher: Searcher, repo: Repo, llm: FakeLLM, olx: FakeOlx
) -> None:
    olx.fail = OlxError("страница не открылась: TimeoutError")
    llm.search_steps.extend([[call("olx_search", "t1", query="3д")], [text("Всё")]])
    outcome = await searcher.run_query(asked(await a_task(repo)), skip=[])
    assert outcome.walls == []

    olx.fail = OlxError(CAPTCHA)
    llm.search_steps.extend([[call("olx_search", "t1", query="3д")], [text("Всё")]])
    outcome = await searcher.run_query(asked(await a_task(repo)), skip=[])
    assert outcome.notes == [f"OLX: {CAPTCHA}"]
    assert outcome.walls == [Wall("olx.uz", f"{BASE}/", "captcha")]


async def test_known_contacts_and_doubles_are_not_saved_again(
    searcher: Searcher, repo: Repo, llm: FakeLLM
) -> None:
    page = "Мастерские: @print_lab, @maker_uz"
    llm.search_steps.extend(
        [
            [
                *fetched("https://catalog.uz", page),
                call(
                    "save_candidate",
                    "t1",
                    name="A",
                    contact="@print_lab",
                    source="u",
                    note="n",
                ),
                call(
                    "save_candidate",
                    "t2",
                    name="B",
                    contact="@maker_uz",
                    source="u",
                    note="n",
                ),
                call(
                    "save_candidate",
                    "t3",
                    name="B",
                    contact="@Maker_UZ",
                    source="u",
                    note="n",
                ),
            ],
            [text("Всё")],
        ]
    )

    outcome = await searcher.run_query(asked(await a_task(repo)), skip=["print_lab"])

    assert [c.contact for c in outcome.candidates] == ["maker_uz"]
    results = last_results(llm)
    assert [r.get("is_error", False) for r in results] == [True, False, True]


async def test_no_more_than_the_wanted_number_is_saved(
    repo: Repo, settings: Settings, llm: FakeLLM, gateway: FakeGateway, olx: FakeOlx
) -> None:
    settings = replace(settings, search=replace(settings.search, max_candidates=1))
    searcher = Searcher(llm, repo, settings, gateway, olx)
    llm.search_steps.extend(
        [
            [
                *fetched("https://catalog.uz", "@print_lab @maker_uz"),
                call(
                    "save_candidate",
                    "t1",
                    name="A",
                    contact="@print_lab",
                    source="u",
                    note="n",
                ),
                call(
                    "save_candidate",
                    "t2",
                    name="B",
                    contact="@maker_uz",
                    source="u",
                    note="n",
                ),
            ],
            [text("Всё")],
        ]
    )

    outcome = await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert [c.contact for c in outcome.candidates] == ["print_lab"]


async def test_the_budget_stops_the_search(
    repo: Repo, settings: Settings, llm: FakeLLM, gateway: FakeGateway, olx: FakeOlx
) -> None:
    settings = replace(settings, search=replace(settings.search, max_cost_usd=0.05))
    searcher = Searcher(llm, repo, settings, gateway, olx)
    expensive = Usage(input_tokens=20_000, output_tokens=2_000)  # $0.06 on Sonnet 5.5
    llm.search_steps.extend(
        [
            SearchStep(
                [call("olx_search", "t1", query="3д печать")],
                expensive,
                1,
                "claude-sonnet-5-5",
                "tool_use",
            ),
            [text("never asked")],
        ]
    )

    outcome = await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert outcome.stopped == "budget"
    assert len(llm.requests) == 1
    assert outcome.cost == pytest.approx(0.06 + 0.01)


async def test_every_step_is_paid_for_and_logged(
    searcher: Searcher, repo: Repo, llm: FakeLLM, fake_clock
) -> None:
    llm.search_steps.extend(
        [
            SearchStep(
                [text("Ничего не нашёл")],
                Usage(10_000, 1_000),
                3,
                "claude-sonnet-5-5",
                "end_turn",
            ),
        ]
    )

    outcome = await searcher.run_query(asked(await a_task(repo)), skip=[])

    # 10k in × $2/M + 1k out × $10/M + 3 searches × $0.01
    assert outcome.cost == pytest.approx(0.02 + 0.01 + 0.03)
    assert await repo.cost_since(fake_clock.now().replace(hour=0)) == pytest.approx(
        0.06
    )


async def test_a_paused_turn_goes_on(
    searcher: Searcher, repo: Repo, llm: FakeLLM
) -> None:
    paused = [
        {
            "type": "server_tool_use",
            "id": "srv1",
            "name": "web_search",
            "input": {"query": "3d печать ташкент"},
        }
    ]
    llm.search_steps.extend(
        [
            SearchStep(paused, Usage(100, 10), 1, "claude-sonnet-5-5", "pause_turn"),
            [text("Готово")],
        ]
    )

    await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert len(llm.requests) == 2
    _, second = llm.requests[1]
    assert second["messages"][-1] == {"role": "assistant", "content": paused}


async def test_the_model_never_sees_the_budget(
    searcher: Searcher, repo: Repo, llm: FakeLLM
) -> None:
    llm.search_steps.append([text("Всё")])

    await searcher.run_query(asked(await a_task(repo, ceiling=250_000)), skip=[])

    _, request = llm.requests[0]
    seen = request["system"] + str(request["messages"])
    assert "250" not in seen and "потол" not in seen.lower()
    assert "Ташкент" in seen  # the city it looks in


async def test_tools_on_offer(searcher: Searcher, repo: Repo, llm: FakeLLM) -> None:
    llm.search_steps.append([text("Всё")])

    await searcher.run_query(asked(await a_task(repo)), skip=[])

    _, request = llm.requests[0]
    names = [tool["name"] for tool in request["tools"]]
    assert names == [
        "web_search",
        "web_fetch",
        "telegram_search",
        "olx_search",
        "olx_listing",
        "save_candidate",
    ]
    web_fetch = request["tools"][1]
    assert web_fetch["type"] == "web_fetch_20260209"
    # Pages come back to us, not only to code the model writes: that is what "seen" checks.
    assert web_fetch["allowed_callers"] == ["direct"]


async def test_without_olx_the_olx_tools_are_not_offered(
    repo: Repo, settings: Settings, llm: FakeLLM, gateway: FakeGateway
) -> None:
    searcher = Searcher(llm, repo, settings, gateway, None)
    llm.search_steps.append([text("Всё")])

    await searcher.run_query(asked(await a_task(repo)), skip=[])

    _, request = llm.requests[0]
    assert "olx_search" not in [tool["name"] for tool in request["tools"]]


@pytest.mark.parametrize(
    ("failure", "stopped"),
    [(ModelRefusal("refused"), "refusal"), (LLMError("down"), "error")],
)
async def test_a_failed_model_call_ends_the_search_with_what_it_has(
    searcher: Searcher, repo: Repo, llm: FakeLLM, failure: Exception, stopped: str
) -> None:
    llm.search_steps.extend(
        [
            [
                *fetched("https://catalog.uz", "@print_lab"),
                call(
                    "save_candidate",
                    "t1",
                    name="A",
                    contact="@print_lab",
                    source="u",
                    note="n",
                ),
            ],
            failure,
        ]
    )

    outcome = await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert outcome.stopped == stopped
    assert [c.contact for c in outcome.candidates] == ["print_lab"]


async def test_steps_are_capped(
    repo: Repo, settings: Settings, llm: FakeLLM, gateway: FakeGateway, olx: FakeOlx
) -> None:
    settings = replace(settings, search=replace(settings.search, max_steps=2))
    searcher = Searcher(llm, repo, settings, gateway, olx)
    llm.search_steps.extend(
        [[call("olx_search", f"t{i}", query="3д")] for i in range(3)]
    )

    outcome = await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert outcome.stopped == "steps"
    assert len(llm.requests) == 2


def test_search_settings_are_checked() -> None:
    with pytest.raises(ValueError):
        Search(browser="firefox")
    with pytest.raises(ValueError):
        Search(max_cost_usd=0)
    with pytest.raises(ValueError):
        Search(max_minutes=0)
    assert Search().max_minutes == 8  # fix pass E2.4


# --- a search by a free-text request, for a task's agent -------------------------------------


async def a_wake(repo: Repo, task_id: int) -> int:
    """A wake row for the foreign key of the logged calls."""
    async with repo._db.execute(
        "INSERT INTO wakes (task_id, started_at) VALUES (?, '2026-11-10T05:00:00+00:00')",
        (task_id,),
    ) as cursor:
        return cursor.lastrowid


def logged_calls(db_path: Path) -> list[tuple]:
    with closing(sqlite3.connect(db_path)) as db:
        return db.execute(
            "SELECT purpose, task_id, wake_id FROM llm_calls ORDER BY id"
        ).fetchall()


async def test_a_query_is_briefed_in_its_own_words_and_nothing_of_a_task(
    repo: Repo, settings: Settings, llm: FakeLLM, gateway: FakeGateway, olx: FakeOlx
) -> None:
    settings = replace(
        settings,
        search=replace(settings.search, sites=("texnomart.uz",), max_candidates=5),
    )
    searcher = Searcher(llm, repo, settings, gateway, olx)
    await a_task(repo, ceiling=250_000)  # a task exists, but the search never sees it
    llm.search_steps.append([text("Всё")])

    await searcher.run_query(
        "мастерские 3D-печати PETG\nв Ташкенте, нужен Telegram",
        skip=["@print_lab", "+998 90 123 45 67"],
    )

    _, request = llm.requests[0]
    assert request["messages"][0]["content"].split("\n") == [
        "Что найти: мастерские 3D-печати PETG в Ташкенте, нужен Telegram",
        "Город: Ташкент",
        "Сайты владельца — ищи и на них: texnomart.uz",
        "Уже есть — не сохранять: @print_lab, +998 90 123 45 67",
        "Найди до 5.",
    ]
    seen = request["system"] + str(request["messages"])
    assert "Печать корпусов" not in seen and "250" not in seen


async def test_a_query_without_sites_or_known_contacts_has_no_such_lines(
    searcher: Searcher, llm: FakeLLM
) -> None:
    llm.search_steps.append([text("Всё")])

    await searcher.run_query("кто чинит кофемашины", skip=[])

    _, request = llm.requests[0]
    assert request["messages"][0]["content"] == (
        "Что найти: кто чинит кофемашины\nГород: Ташкент\nНайди до 8."
    )


async def test_a_query_s_calls_are_logged_for_its_task_and_wake(
    searcher: Searcher, repo: Repo, llm: FakeLLM, db_path: Path
) -> None:
    task = await a_task(repo)
    wake_id = await a_wake(repo, task.id)
    llm.search_steps.extend([[call("olx_search", "t1", query="3д")], [text("Всё")]])

    await searcher.run_query("3D-печать", skip=[], task_id=task.id, wake_id=wake_id)
    llm.search_steps.append([text("Всё")])
    await searcher.run_query("3D-печать", skip=[])  # no ids: logged as before

    assert logged_calls(db_path) == [
        ("search", task.id, wake_id),
        ("search", task.id, wake_id),
        ("search", None, None),
    ]
    assert await repo.task_cost(task.id) > 0


async def test_a_search_no_longer_wanted_ends_with_what_it_has(
    searcher: Searcher, llm: FakeLLM
) -> None:
    asked: list[int] = []

    async def still_wanted() -> bool:
        asked.append(len(llm.requests))
        return len(llm.requests) < 1  # the task stopped after the first call

    llm.search_steps.extend(
        [
            [
                *fetched("https://catalog.uz", "@print_lab"),
                call(
                    "save_candidate",
                    "t1",
                    name="PrintLab",
                    contact="@print_lab",
                    source="https://catalog.uz",
                    note="PETG",
                ),
            ],
            [text("Всё")],
        ]
    )

    outcome = await searcher.run_query("3D-печать", skip=[], still_wanted=still_wanted)

    assert len(llm.requests) == 1
    assert asked == [0, 1]  # asked before every call
    assert outcome.stopped == "stopped"
    assert [c.contact for c in outcome.candidates] == ["print_lab"]


async def test_no_model_facing_string_names_the_owner(
    repo: Repo, settings: Settings, llm: FakeLLM, gateway: FakeGateway, olx: FakeOlx
) -> None:
    settings = replace(settings, search=replace(settings.search, sites=("prom.uz",)))
    pages: Any = object()  # only offers the site tools; no page is opened
    searcher = Searcher(llm, repo, settings, gateway, olx, pages=pages)
    llm.search_steps.append([text("Всё")])

    await searcher.run_query("3D-печать", skip=["@print_lab"])

    _, request = llm.requests[0]
    names = [tool["name"] for tool in request["tools"]]
    assert {"open_page", "click", "search_site", "save_candidate"} <= set(names)
    for seen in (request["system"], str(request["tools"]), str(request["messages"])):
        assert "Саид" not in seen
    assert "владел" in request["system"] and "владел" in str(request["tools"])
    assert all("Саид" not in note for note in (NOT_LINKED, NOT_SEEN, PAGES_USED))
    # The search finds whom to write to for any request, not only sellers for a purchase.
    assert "исполнител" not in request["system"] + str(request["tools"])


def test_the_search_saves_by_the_need_and_starts_with_the_owner_s_sites() -> None:
    # Fix round 1 (R43, R44): a person the request names is not saved for that; the sites
    # to start with are the line «Сайты владельца», never a site of the request's text.
    from tg_agent.brain.context import load_prompt

    prompt = load_prompt("search")
    assert "запрос называет" not in prompt
    assert "Тех, кто может сделать то, что нужно по запросу." in prompt
    assert "Если в строке «Сайты владельца» названы сайты — начни с них" in prompt
    assert "Если в запросе" not in prompt


# --- one search at a time (fix pass E2.1, review I2) -----------------------------------------


class SlowLLM:
    """A search model whose steps take a while and say when they start and end."""

    def __init__(self, on_step: Any = None) -> None:
        self.log: list[str] = []
        self.on_step = on_step

    async def search_step(
        self, *, system: str, messages: list, tools: list
    ) -> SearchStep:
        name = messages[0]["content"].split("\n")[0]
        self.log.append(f"start {name}")
        for _ in range(5):
            await asyncio.sleep(0)
        if self.on_step is not None:
            self.on_step()
        self.log.append(f"end {name}")
        return SearchStep(
            [text("Всё")], Usage(10, 1), 0, "claude-sonnet-5-5", "end_turn"
        )


async def test_two_searches_at_once_run_one_after_the_other(
    repo: Repo, settings: Settings, gateway: FakeGateway, olx: FakeOlx
) -> None:
    llm = SlowLLM()
    searcher = Searcher(llm, repo, settings, gateway, olx)  # type: ignore[arg-type]

    await asyncio.gather(
        searcher.run_query("первое", skip=[]), searcher.run_query("второе", skip=[])
    )

    assert llm.log == [
        "start Что найти: первое",
        "end Что найти: первое",
        "start Что найти: второе",
        "end Что найти: второе",
    ]


async def test_a_search_that_waited_for_a_task_closed_meanwhile_never_starts(
    repo: Repo, settings: Settings, gateway: FakeGateway, olx: FakeOlx
) -> None:
    closed = False

    def close_the_second_task() -> None:
        nonlocal closed
        closed = True  # while the first search runs, the second one's task is closed

    llm = SlowLLM(on_step=close_the_second_task)
    searcher = Searcher(llm, repo, settings, gateway, olx)  # type: ignore[arg-type]

    async def second_still_wanted() -> bool:
        return not closed

    first, second = await asyncio.gather(
        searcher.run_query("первое", skip=[]),
        searcher.run_query("второе", skip=[], still_wanted=second_still_wanted),
    )

    assert llm.log == ["start Что найти: первое", "end Что найти: первое"]
    assert first.stopped is None
    assert second.stopped == "stopped" and second.cost == 0


# --- a search has a wall-clock limit (fix pass E2.4, review M15) ----------------------------


class PausingLLM:
    """A search model whose every step takes five minutes and asks to go on."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls = 0

    async def search_step(
        self, *, system: str, messages: list, tools: list
    ) -> SearchStep:
        self.calls += 1
        self.clock.advance(timedelta(minutes=5))
        return SearchStep([], Usage(10, 1), 0, "claude-sonnet-5-5", "pause_turn")


async def test_a_search_over_its_minutes_ends_before_the_next_model_call(
    repo: Repo, settings: Settings, gateway: FakeGateway, fake_clock: FakeClock
) -> None:
    llm = PausingLLM(fake_clock)
    searcher = Searcher(llm, repo, settings, gateway, None, clock=fake_clock)  # type: ignore[arg-type]

    outcome = await searcher.run_query("3D-печать", skip=[])

    assert llm.calls == 2  # 5 minutes, then 10: over the 8 before the third call
    assert outcome.stopped == "time"


async def test_a_search_over_its_minutes_ends_after_a_tool_call(
    repo: Repo,
    settings: Settings,
    llm: FakeLLM,
    gateway: FakeGateway,
    olx: FakeOlx,
    fake_clock: FakeClock,
) -> None:
    settings = replace(settings, search=replace(settings.search, max_minutes=3))
    searcher = Searcher(llm, repo, settings, gateway, olx, clock=fake_clock)
    slow_search = olx.search

    async def search_for_four_minutes(query: str) -> list:
        fake_clock.advance(timedelta(minutes=4))
        return await slow_search(query)

    olx.search = search_for_four_minutes  # type: ignore[method-assign]
    llm.search_steps.extend(
        [
            [
                *fetched("https://catalog.uz", "@print_lab"),
                call(
                    "save_candidate",
                    "t1",
                    name="A",
                    contact="@print_lab",
                    source="u",
                    note="n",
                ),
                call("olx_search", "t2", query="3д"),
            ],
            [text("Всё")],
        ]
    )

    outcome = await searcher.run_query("3D-печать", skip=[])

    assert len(llm.requests) == 1
    assert outcome.stopped == "time"
    assert [c.contact for c in outcome.candidates] == ["print_lab"]


async def test_a_search_over_its_minutes_skips_the_rest_of_its_step(
    repo: Repo,
    settings: Settings,
    llm: FakeLLM,
    gateway: FakeGateway,
    olx: FakeOlx,
    fake_clock: FakeClock,
) -> None:
    # Review x07: the slow tool is not the last of its step; what follows it is not done.
    settings = replace(settings, search=replace(settings.search, max_minutes=3))
    searcher = Searcher(llm, repo, settings, gateway, olx, clock=fake_clock)
    slow_search = olx.search

    async def search_for_four_minutes(query: str) -> list:
        fake_clock.advance(timedelta(minutes=4))
        return await slow_search(query)

    olx.search = search_for_four_minutes  # type: ignore[method-assign]
    llm.search_steps.extend(
        [
            [
                *fetched("https://catalog.uz", "@print_lab"),
                call("olx_search", "t1", query="3д"),
                call(
                    "save_candidate",
                    "t2",
                    name="A",
                    contact="@print_lab",
                    source="u",
                    note="n",
                ),
            ],
            [text("Всё")],
        ]
    )

    outcome = await searcher.run_query("3D-печать", skip=[])

    assert outcome.stopped == "time"
    assert outcome.candidates == []
