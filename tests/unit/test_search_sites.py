"""The vendor search on any site, in the agent's own browser with Саид's sign-ins: what it may
open, press and type, and what reaches Саид. The browser itself is a fake here; the real one is
checked against local pages in tests/offline/test_site_browser.py."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tg_agent.brain.context import load_prompt
from tg_agent.brain.llm import FakeLLM
from tg_agent.config import ConfigError, Search, Settings, load_settings
from tg_agent.db import Repo
from tg_agent.gateway.base import PublicPeer
from tg_agent.gateway.fake import FakeGateway
from tg_agent.models import Task
from tg_agent.search.agent import Searcher, Wall
from tg_agent.search.browser import AgentBrowser, BrowserError, agent_profile
from tg_agent.search.sites import (
    Page,
    click_refusal,
    describe,
    domains_in,
    query_refusal,
    url_problem,
)

HOME = "https://printlab.uz/"
PAGE = Page(
    url=HOME,
    title="PrintLab — 3D печать",
    text="Студия 3D-печати в Ташкенте. Печатаем PETG и PLA",
    phones=["+998901234567"],
    telegram=["print_lab"],
    links=[("Контакты", "https://printlab.uz/contacts")],
    buttons=["Показать телефон"],
)


class FakePages:
    """The site browser as the search sees it: what it was asked to open, press and type."""

    def __init__(self) -> None:
        self.opened: list[str] = []
        self.clicked: list[str] = []
        self.searched: list[str] = []
        self.page = PAGE
        self.fail: BrowserError | None = None

    async def open(self, url: str) -> Page:
        self.opened.append(url)
        if self.fail:
            raise self.fail
        return replace(self.page, url=url)

    async def click(self, text: str) -> Page:
        self.clicked.append(text)
        if self.fail:
            raise self.fail
        return self.page

    async def search(self, query: str) -> Page:
        self.searched.append(query)
        if self.fail:
            raise self.fail
        return self.page


def call(tool: str, tool_id: str, /, **arguments: object) -> dict:
    return {"type": "tool_use", "id": tool_id, "name": tool, "input": arguments}


def say(words: str) -> dict:
    return {"type": "text", "text": words}


def web_results(*urls: str) -> list[dict]:
    """A web_search Anthropic ran during the turn: the call and the pages it found."""
    return [
        {
            "type": "server_tool_use",
            "id": "srv9",
            "name": "web_search",
            "input": {"query": "3d печать ташкент"},
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srv9",
            "content": [
                {"type": "web_search_result", "title": "Студия 3D-печати", "url": url}
                for url in urls
            ],
        },
    ]


def fetched(url: str, page: str) -> list[dict]:
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


async def a_task(
    repo: Repo, goal: str = "Узнать цену печати 5 корпусов из PETG"
) -> Task:
    task_id = await repo.create_task(
        "Печать корпусов",
        goal,
        {"what": "корпус для платы", "quantity": 5, "deadline": None},
        250_000,
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
def gateway() -> FakeGateway:
    return FakeGateway()


@pytest.fixture
def pages() -> FakePages:
    return FakePages()


@pytest.fixture
def searcher(
    repo: Repo, settings: Settings, llm: FakeLLM, gateway: FakeGateway, pages: FakePages
) -> Searcher:
    return Searcher(llm, repo, settings, gateway, None, pages=pages)


def results_of(llm: FakeLLM, request: int = -1) -> list[dict]:
    """The tool results a search request sent back."""
    _, sent = llm.requests[request]
    return sent["messages"][-1]["content"]


def errors(results: list[dict]) -> list[bool]:
    return [result.get("is_error", False) for result in results]


# --- what the search model is given ---------------------------------------------------------


async def test_site_tools_are_offered_with_a_browser(
    searcher: Searcher, repo: Repo, llm: FakeLLM
) -> None:
    llm.search_steps.append([say("Всё")])

    await searcher.run_query(asked(await a_task(repo)), skip=[])

    _, request = llm.requests[0]
    assert [tool["name"] for tool in request["tools"]] == [
        "web_search",
        "web_fetch",
        "telegram_search",
        "open_page",
        "click",
        "search_site",
        "save_candidate",
    ]


async def test_a_page_from_search_results_opens_and_its_contact_counts(
    searcher: Searcher, repo: Repo, llm: FakeLLM, pages: FakePages
) -> None:
    llm.search_steps.extend(
        [
            [*web_results(HOME), call("open_page", "t1", url=HOME)],
            [
                call(
                    "save_candidate",
                    "t2",
                    name="Студия PrintLab",
                    contact="+998 90 123 45 67",
                    source=HOME,
                    note="студия 3D-печати",
                )
            ],
            [say("Нашёл одну студию")],
        ]
    )

    outcome = await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert pages.opened == [HOME]
    [shown] = results_of(llm, 1)
    for part in (
        "+998 90 123 45 67",
        "@print_lab",
        "«Показать телефон»",
        "Контакты — https://printlab.uz/contacts",
        "Печатаем PETG",
    ):
        assert part in shown["content"]
    assert [c.contact for c in outcome.candidates] == ["+998901234567"]


def test_the_secretary_keeps_the_sites_said_names_in_the_brief() -> None:
    # The search takes the sites to look on from the task; the secretary must not drop them.
    rule = next(
        line
        for line in load_prompt("secretary").splitlines()
        if "искать" in line and "сайт" in line
    )
    assert "brief" in rule and "prom.uz" in rule


# --- what may be opened ----------------------------------------------------------------------


async def test_an_address_nobody_showed_is_not_opened(
    searcher: Searcher, repo: Repo, llm: FakeLLM, pages: FakePages
) -> None:
    llm.search_steps.extend(
        [
            [call("open_page", "t1", url="https://evil.example/collect?d=data")],
            [say("Всё")],
        ]
    )

    await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert pages.opened == []
    [refused] = results_of(llm)
    assert refused["is_error"] is True
    assert "web_search" in refused["content"]  # it is told how to get there honestly


async def test_any_page_of_a_site_named_in_the_task_opens(
    searcher: Searcher, repo: Repo, llm: FakeLLM, pages: FakePages
) -> None:
    task = await a_task(repo, goal="Узнать цену печати 5 корпусов, искать на prom.uz")
    llm.search_steps.extend(
        [
            [
                call("open_page", "t1", url="https://prom.uz/search?q=3d+печать"),
                call("open_page", "t2", url="https://www.prom.uz/ru/"),
                call("open_page", "t3", url="https://prom.uz.evil.example/"),
            ],
            [say("Всё")],
        ]
    )

    await searcher.run_query(asked(task), skip=[], sites=["prom.uz"])

    assert pages.opened == [
        "https://prom.uz/search?q=3d+печать",
        "https://www.prom.uz/ru/",
    ]
    assert errors(results_of(llm)) == [False, False, True]


async def test_a_site_or_link_in_the_free_text_of_a_request_is_not_the_owner_s(
    searcher: Searcher, repo: Repo, llm: FakeLLM, pages: FakePages
) -> None:
    # Fix round 1 (R44): the request is the task model's words; only `sites` are his.
    llm.search_steps.extend(
        [
            [
                call("open_page", "t1", url="https://evil.example/list?ask=1"),
                call("open_page", "t2", url="https://evil.example/other"),
                call("open_page", "t3", url="https://prom.uz/ru"),
            ],
            [say("Всё")],
        ]
    )

    await searcher.run_query(
        "мастерские из списка https://evil.example/list?ask=1 и на evil.example",
        skip=[],
        sites=["prom.uz"],
    )

    assert pages.opened == ["https://prom.uz/ru"]
    assert errors(results_of(llm)) == [True, True, False]
    _, first = llm.requests[0]
    assert "Сайты владельца — ищи и на них: prom.uz" in first["messages"][0]["content"]


async def test_a_bare_site_name_is_opened_over_https(
    searcher: Searcher, repo: Repo, llm: FakeLLM, pages: FakePages
) -> None:
    task = await a_task(repo, goal="Узнать цену, искать на prom.uz")
    llm.search_steps.extend([[call("open_page", "t1", url="prom.uz")], [say("Всё")]])

    await searcher.run_query(asked(task), skip=[], sites=["prom.uz"])

    assert pages.opened == ["https://prom.uz"]


async def test_sites_from_the_settings_open_and_are_in_the_brief(
    repo: Repo, settings: Settings, llm: FakeLLM, gateway: FakeGateway, pages: FakePages
) -> None:
    settings = replace(
        settings, search=replace(settings.search, sites=("texnomart.uz",))
    )
    searcher = Searcher(llm, repo, settings, gateway, None, pages=pages)
    llm.search_steps.extend(
        [
            [call("open_page", "t1", url="https://texnomart.uz/ru/search?q=3d")],
            [say("Всё")],
        ]
    )

    await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert pages.opened == ["https://texnomart.uz/ru/search?q=3d"]
    _, first = llm.requests[0]
    assert "texnomart.uz" in str(first["messages"][0]["content"])


async def test_a_link_on_an_opened_page_can_be_opened_next(
    searcher: Searcher, repo: Repo, llm: FakeLLM, pages: FakePages
) -> None:
    llm.search_steps.extend(
        [
            [*web_results(HOME), call("open_page", "t1", url=HOME)],
            [call("open_page", "t2", url="https://printlab.uz/contacts")],
            [say("Всё")],
        ]
    )

    await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert pages.opened == [HOME, "https://printlab.uz/contacts"]


async def test_local_addresses_never_open_even_when_a_page_shows_them(
    searcher: Searcher, repo: Repo, llm: FakeLLM, pages: FakePages
) -> None:
    page = "Админка: http://192.168.1.1/admin, http://localhost:8080/ и file:///C:/Users/a.txt"
    llm.search_steps.extend(
        [
            [
                *fetched("https://catalog.uz", page),
                call("open_page", "t1", url="http://192.168.1.1/admin"),
                call("open_page", "t2", url="http://localhost:8080/"),
                call("open_page", "t3", url="file:///C:/Users/a.txt"),
            ],
            [say("Всё")],
        ]
    )

    await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert pages.opened == []
    assert errors(results_of(llm)) == [True, True, True]


@pytest.mark.parametrize(
    "url",
    [
        "file:///C:/Users/a.txt",
        "javascript:alert(1)",
        "data:text/html,hi",
        "ftp://files.uz/x",
        "http://localhost:8080/",
        "http://127.0.0.1/",
        "http://192.168.1.1/",
        "http://10.0.0.5/",
        "http://172.16.3.4/",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/",
        "http://router/",
        "http://printer.local/",
        "http://2130706433/",
        "http://0x7f.0.0.1/",
    ],
)
def test_local_and_odd_addresses_are_refused(url: str) -> None:
    assert url_problem(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://prom.uz/search?q=3d",
        "http://printlab.uz",
        "https://www.olx.uz/d/obyavlenie/x.html",
        "https://8.8.8.8/",
    ],
)
def test_public_addresses_pass(url: str) -> None:
    assert url_problem(url) is None


def test_sites_are_read_from_the_task_text() -> None:
    text = (
        "Узнать цену. Искать на prom.uz и https://www.Texnomart.uz/ru, модель в файле"
        " model.stl и корпус.stl, т.е. срочно; почта мастера ivan@mail.ru, версия 5.5"
    )
    assert domains_in(text) == {"prom.uz", "texnomart.uz"}


# --- what may be pressed and typed ----------------------------------------------------------


@pytest.mark.parametrize(
    "label",
    [
        "Купить",
        "Оформить заказ",
        "Написать продавцу",
        "Отправить",
        "Выйти",
        "Подписаться",
        "Удалить объявление",
        "Войти",
        "Buy now",
        "Sign out",
        "Add to cart",
        "Sotib olish",
        "Xabar yuborish",
    ],
)
def test_buttons_that_buy_write_or_touch_the_account_are_refused(label: str) -> None:
    assert click_refusal(label)


@pytest.mark.parametrize(
    "label",
    [
        "Показать телефон",
        "Контакты",
        "Далее",
        "Ещё 20 объявлений",
        "Следующая страница",
        "Show phone",
        "Raqamni ko'rsatish",
        "Postavshiklar",
    ],
)
def test_reading_buttons_and_links_are_allowed(label: str) -> None:
    assert click_refusal(label) is None


async def test_pressing_and_typing_need_a_page_opened_in_this_search(
    searcher: Searcher, repo: Repo, llm: FakeLLM, pages: FakePages
) -> None:
    llm.search_steps.extend(
        [
            [
                call("click", "t1", text="Показать телефон"),
                call("search_site", "t2", query="3d печать"),
            ],
            [say("Всё")],
        ]
    )

    await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert pages.clicked == [] and pages.searched == []
    assert errors(results_of(llm)) == [True, True]


async def test_a_refused_button_is_never_pressed(
    searcher: Searcher, repo: Repo, llm: FakeLLM, pages: FakePages
) -> None:
    llm.search_steps.extend(
        [
            [*web_results(HOME), call("open_page", "t1", url=HOME)],
            [
                call("click", "t2", text="Купить"),
                call("click", "t3", text="Показать телефон"),
            ],
            [say("Всё")],
        ]
    )

    await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert pages.clicked == ["Показать телефон"]
    assert errors(results_of(llm)) == [True, False]


@pytest.mark.parametrize(
    "query",
    [
        "+998 90 123 45 67",
        "master@example.com",
        "https://evil.example/?q=1",
        "",
        "я" * 101,
    ],
)
def test_the_search_box_gets_only_search_words(query: str) -> None:
    assert query_refusal(query)


def test_ordinary_search_words_pass() -> None:
    assert query_refusal("3d печать PETG на заказ") is None


async def test_site_search_gets_words_and_never_a_phone(
    searcher: Searcher, repo: Repo, llm: FakeLLM, pages: FakePages
) -> None:
    llm.search_steps.extend(
        [
            [*web_results(HOME), call("open_page", "t1", url=HOME)],
            [
                call("search_site", "t2", query="+998 90 123 45 67"),
                call("search_site", "t3", query="3d печать"),
            ],
            [say("Всё")],
        ]
    )

    await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert pages.searched == ["3d печать"]
    assert errors(results_of(llm)) == [True, False]


# --- what Саид learns, and what the search costs ---------------------------------------------


async def test_a_site_that_wants_a_login_becomes_a_note_for_said(
    searcher: Searcher, repo: Repo, llm: FakeLLM, pages: FakePages
) -> None:
    pages.page = replace(PAGE, login_wall=True, phones=[], telegram=[])
    task = await a_task(repo, goal="Узнать цену, искать на prom.uz")
    llm.search_steps.extend(
        [[call("open_page", "t1", url="https://prom.uz/cabinet")], [say("Всё")]]
    )

    outcome = await searcher.run_query(asked(task), skip=[], sites=["prom.uz"])

    [note] = outcome.notes
    assert "prom.uz" in note and "только после входа" in note and "пропустил" in note
    [shown] = results_of(llm)
    assert "Не входи" in shown["content"]


async def test_a_captcha_is_reported_not_solved(
    searcher: Searcher, repo: Repo, llm: FakeLLM, pages: FakePages
) -> None:
    pages.page = replace(PAGE, captcha=True)
    llm.search_steps.extend(
        [[*web_results(HOME), call("open_page", "t1", url=HOME)], [say("Всё")]]
    )

    outcome = await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert any("капч" in note and "printlab.uz" in note for note in outcome.notes)


async def test_a_login_wall_is_a_wall_once_per_site_beside_its_note(
    searcher: Searcher, repo: Repo, llm: FakeLLM, pages: FakePages
) -> None:
    # Stage 4б (Task 4б.2): where the note is written, the wall is kept for the sign-in card.
    pages.page = replace(PAGE, login_wall=True, phones=[], telegram=[])
    task = await a_task(repo, goal="Узнать цену, искать на prom.uz")
    llm.search_steps.extend(
        [
            [call("open_page", "t1", url="https://prom.uz/cabinet")],
            [call("open_page", "t2", url="https://www.prom.uz/other")],
            [say("Всё")],
        ]
    )

    outcome = await searcher.run_query(asked(task), skip=[], sites=["prom.uz"])

    assert outcome.walls == [Wall("prom.uz", "https://prom.uz/cabinet", "login")]
    assert len(outcome.notes) == 1


async def test_a_captcha_is_a_wall_and_a_page_with_both_is_a_login_wall(
    searcher: Searcher, repo: Repo, llm: FakeLLM, pages: FakePages
) -> None:
    # A wall is kept only for a site the owner named (fix round 1, I1): printlab.uz here.
    pages.page = replace(PAGE, captcha=True)
    llm.search_steps.extend(
        [[*web_results(HOME), call("open_page", "t1", url=HOME)], [say("Всё")]]
    )
    outcome = await searcher.run_query(
        asked(await a_task(repo)), skip=[], sites=["printlab.uz"]
    )
    assert outcome.walls == [Wall("printlab.uz", HOME, "captcha")]

    pages.page = replace(PAGE, captcha=True, login_wall=True)
    llm.search_steps.extend(
        [[*web_results(HOME), call("open_page", "t1", url=HOME)], [say("Всё")]]
    )
    outcome = await searcher.run_query(
        asked(await a_task(repo)), skip=[], sites=["printlab.uz"]
    )
    assert outcome.walls == [Wall("printlab.uz", HOME, "login")]

    # Not named by the owner: no wall, so no card; the note still tells him.
    llm.search_steps.extend(
        [[*web_results(HOME), call("open_page", "t1", url=HOME)], [say("Всё")]]
    )
    outcome = await searcher.run_query(asked(await a_task(repo)), skip=[])
    assert outcome.walls == []
    assert any("printlab.uz" in n and "пропустил" in n for n in outcome.notes)


async def test_a_stranger_s_login_page_from_a_web_result_raises_no_card(
    searcher: Searcher, repo: Repo, llm: FakeLLM, pages: FakePages
) -> None:
    # Review 4б.2, I1: a look-alike of OLX found by a web search is not a site to sign in to.
    look_alike = "https://olx-uz.example/account/login"
    pages.page = replace(PAGE, login_wall=True, phones=[], telegram=[])
    llm.search_steps.extend(
        [
            [*web_results(look_alike), call("open_page", "t1", url=look_alike)],
            [say("Всё")],
        ]
    )
    outcome = await searcher.run_query(
        asked(await a_task(repo)), skip=[], sites=["olx.uz"]
    )
    assert outcome.walls == []  # nothing for «Войти» to open
    assert any("olx-uz.example" in n and "пропустил" in n for n in outcome.notes)


async def test_a_page_without_a_wall_leaves_none(
    searcher: Searcher, repo: Repo, llm: FakeLLM
) -> None:
    llm.search_steps.extend(
        [[*web_results(HOME), call("open_page", "t1", url=HOME)], [say("Всё")]]
    )
    outcome = await searcher.run_query(asked(await a_task(repo)), skip=[])
    assert outcome.walls == []


async def test_a_page_that_does_not_open_is_reported_and_the_search_goes_on(
    searcher: Searcher, repo: Repo, llm: FakeLLM, pages: FakePages
) -> None:
    pages.fail = BrowserError("страница не открылась: TimeoutError")
    llm.search_steps.extend(
        [[*web_results(HOME), call("open_page", "t1", url=HOME)], [say("Всё")]]
    )

    outcome = await searcher.run_query(asked(await a_task(repo)), skip=[])

    [failed] = results_of(llm)
    assert failed["is_error"] is True and "не открылась" in failed["content"]
    assert outcome.stopped is None


async def test_pages_per_search_are_capped(
    repo: Repo, settings: Settings, llm: FakeLLM, gateway: FakeGateway, pages: FakePages
) -> None:
    settings = replace(settings, search=replace(settings.search, max_pages=1))
    searcher = Searcher(llm, repo, settings, gateway, None, pages=pages)
    second = "https://maker.uz/"
    llm.search_steps.extend(
        [
            [
                *web_results(HOME, second),
                call("open_page", "t1", url=HOME),
                call("open_page", "t2", url=second),
            ],
            [say("Всё")],
        ]
    )

    await searcher.run_query(asked(await a_task(repo)), skip=[])

    assert pages.opened == [HOME]
    assert errors(results_of(llm)) == [False, True]


async def test_page_opens_are_logged_without_their_text(
    searcher: Searcher, repo: Repo, llm: FakeLLM
) -> None:
    llm.search_steps.extend(
        [
            [
                *web_results(HOME),
                call("open_page", "t1", url=HOME),
                call("open_page", "t2", url="https://evil.example/"),
            ],
            [say("Всё")],
        ]
    )

    await searcher.run_query(asked(await a_task(repo)), skip=[])

    events = await repo.events_of("search_page")
    assert events == [
        {"tool": "open_page", "url": HOME, "result": "ok"},
        {"tool": "open_page", "url": "https://evil.example/", "result": "refused"},
    ]
    assert "Студия" not in str(events)


# --- the browser profile and the sign-in helper ---------------------------------------------


def test_the_old_olx_profile_becomes_the_agent_profile(settings: Settings) -> None:
    old = settings.data_dir / "browser" / "olx"
    (old / "Default").mkdir(parents=True)
    (old / "Default" / "Cookies").write_text("x", encoding="utf-8")

    profile = agent_profile(settings)

    assert profile == settings.data_dir / "browser" / "agent"
    assert (profile / "Default" / "Cookies").read_text(encoding="utf-8") == "x"
    assert not old.exists()


def test_an_existing_agent_profile_is_kept(settings: Settings) -> None:
    old = settings.data_dir / "browser" / "olx"
    new = settings.data_dir / "browser" / "agent"
    old.mkdir(parents=True)
    new.mkdir(parents=True)

    assert agent_profile(settings) == new
    assert old.exists()


def test_the_profile_folder_can_be_set(settings: Settings, tmp_path: Path) -> None:
    custom = tmp_path / "edge-profile"
    settings = replace(
        settings, search=replace(settings.search, browser_profile=custom)
    )

    assert agent_profile(settings) == custom


# --- settings -------------------------------------------------------------------------------


def test_site_settings_come_from_the_settings_file(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "[search]\n"
        'sites = ["https://Prom.uz/ru", "texnomart.uz"]\n'
        "max_pages = 5\n"
        "browse = false\n"
        'browser = "chrome"\n'
        "browser_headless = false\n"
        'browser_profile = "edge"\n',
        encoding="utf-8",
    )

    search = load_settings(path).search

    assert search.sites == ("prom.uz", "texnomart.uz")
    assert search.max_pages == 5
    assert search.browse is False
    assert search.browser == "chrome" and search.browser_headless is False
    assert search.browser_profile == Path("edge")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('[search]\nsites = "prom.uz"\n', "search.sites: expected a list of strings"),
        ('[search]\nsites = ["prom"]\n', "search.sites: 'prom' is not a site"),
        ("[search]\nmax_pages = 0\n", "max_pages must be at least 1"),
        ('[search]\nbrowser = "firefox"\n', "search.browser: msedge, chrome or chromium"),
    ],
)
def test_bad_site_settings_are_refused(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_settings(path)


def test_search_defaults() -> None:
    search = Search()
    assert search.sites == () and search.max_pages == 20 and search.browse is True
    assert search.browser == "msedge" and search.browser_headless is True
    assert search.browser_profile == Path("browser") / "agent"


# --- one browser start at a time (fix pass E2.1, review I2) ----------------------------------


class _FakeTab:
    def __init__(self, url: str = "about:blank") -> None:
        self.url = url
        self.closed = False

    def is_closed(self) -> bool:
        return self.closed

    async def close(self) -> None:
        self.closed = True


class _FakeContext:
    def __init__(self, pages: list[_FakeTab] | None = None) -> None:
        self.pages: list[_FakeTab] = list(pages or [])

    def set_default_timeout(self, ms: int) -> None:
        pass

    async def new_page(self) -> _FakeTab:
        tab = _FakeTab()
        self.pages.append(tab)
        return tab


class _CountedBrowser(AgentBrowser):
    """The agent's browser with a fake launcher that takes a while and counts its starts."""

    def __init__(self, profile_dir: Path) -> None:
        super().__init__(profile_dir)
        self.launches = 0
        self.leftovers: list[_FakeTab] = []

    async def _launch(self, headless: bool) -> tuple[object, object]:
        self.launches += 1
        for _ in range(5):
            await asyncio.sleep(0)
        return object(), _FakeContext(self.leftovers)


async def test_two_first_uses_at_once_start_one_browser(tmp_path: Path) -> None:
    browser = _CountedBrowser(tmp_path / "profile")

    site, olx = await asyncio.gather(browser.tab("site"), browser.tab("olx"))
    again = await browser.tab("site")

    assert browser.launches == 1
    assert site is not olx and again is site


# --- what the search wrote itself is not "seen" (fix pass E2.2, review M4) -------------------


async def test_a_contact_the_model_typed_into_a_query_is_not_saved(
    searcher: Searcher, llm: FakeLLM, gateway: FakeGateway
) -> None:
    gateway.public = [PublicPeer("print3d_tash", "3D печать Ташкент", "channel", "")]
    llm.search_steps.extend(
        [
            [call("telegram_search", "t1", query="@zzz_victim 3d")],
            [
                call(
                    "save_candidate",
                    "t2",
                    name="Жертва",
                    contact="@zzz_victim",
                    source="https://t.me/zzz_victim",
                    note="—",
                )
            ],
            [say("Всё")],
        ]
    )

    outcome = await searcher.run_query("3D-печать", skip=[])

    assert outcome.candidates == []
    assert errors(results_of(llm)) == [True]


async def test_an_address_the_model_typed_into_a_query_is_not_opened(
    searcher: Searcher, llm: FakeLLM, gateway: FakeGateway, pages: FakePages
) -> None:
    gateway.public = [PublicPeer("print3d_tash", "3D печать Ташкент", "channel", "")]
    llm.search_steps.extend(
        [
            [call("telegram_search", "t1", query="3D https://evil-shop.com/x")],
            [call("open_page", "t2", url="https://evil-shop.com/x")],
            [say("Всё")],
        ]
    )

    await searcher.run_query("3D-печать", skip=[])

    assert pages.opened == []
    assert errors(results_of(llm)) == [True]


# --- texts and sites (fix pass E2.3, review M5, M6) ------------------------------------------


def _model_facing_texts() -> list[str]:
    """Every text the search model may read from these modules: their string constants, the
    tool descriptions, a page with a captcha and a login wall, the refusals."""
    from tg_agent.search import agent, sites

    texts = [
        value
        for module in (agent, sites)
        for name, value in vars(module).items()
        if isinstance(value, str) and not name.startswith("__")  # not the docstring
    ]
    texts += [str(agent._CLIENT_TOOLS)]
    texts.append(
        describe(Page("https://a.uz", "A", "текст", login_wall=True, captcha=True))
    )
    texts += [
        click_refusal("Купить") or "",
        *(
            query_refusal(q) or ""
            for q in ("", "x" * 200, "+998901234567", "@a", "https://a.uz")
        ),
    ]
    return texts


def test_no_text_the_search_model_reads_names_the_owner_or_promises_a_sign_in() -> None:
    for text in _model_facing_texts():
        assert "Саид" not in text, text
        assert "где войти" not in text and "где\n войти" not in text, text
        assert "из запроса" not in text and "в запросе" not in text, text


@pytest.mark.parametrize(
    "text",
    [
        "напишите https://t.me/print_lab",
        "или telegram.me/print_lab",
        "сайт telegram.org",
        "Smart.Print делает",
        "PETG.Telegram",
        "версия Node.js",
    ],
)
def test_telegram_links_and_dotted_words_are_not_sites(text: str) -> None:
    assert domains_in(text) == set()


def test_a_site_needs_a_known_top_level_domain() -> None:
    assert domains_in("prom.uz, ozon.ru, example.com, olx.kz, shop.store") == {
        "prom.uz",
        "ozon.ru",
        "example.com",
        "olx.kz",
        "shop.store",
    }


# --- what the browser keeps and hands out (fix pass E2.1, the remote screen's review) -------


async def test_only_a_blank_leftover_page_becomes_a_tab_and_others_are_closed(
    tmp_path: Path,
) -> None:
    popup, blank = _FakeTab("https://evil-shop.com/popup"), _FakeTab()
    browser = _CountedBrowser(tmp_path / "profile")
    browser.leftovers = [popup, blank]

    site = await browser.tab("site")
    olx = await browser.tab("olx")

    assert site is blank
    assert olx is not popup and olx.url == "about:blank"
    assert popup.closed


class _FakeChromium:
    def __init__(self) -> None:
        self.launched: list[dict] = []

    async def launch_persistent_context(
        self, profile: str, **options: object
    ) -> object:
        self.launched.append(options)
        return _FakeContext()


class _FakePlaywright:
    def __init__(self) -> None:
        self.chromium = _FakeChromium()

    async def stop(self) -> None:
        pass


async def test_the_browser_refuses_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import playwright.async_api

    fake = _FakePlaywright()

    class _Starter:
        async def start(self) -> _FakePlaywright:
            return fake

    monkeypatch.setattr(playwright.async_api, "async_playwright", lambda: _Starter())

    await AgentBrowser(tmp_path / "profile").tab("site")

    [options] = fake.chromium.launched
    assert options["accept_downloads"] is False


QUIET = {
    "autofill": {"profile_enabled": False, "credit_card_enabled": False},
    "credentials_enable_service": False,
    "profile": {"password_manager_enabled": False},
}


async def test_a_new_profile_starts_with_autofill_and_passwords_off(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"

    await _CountedBrowser(profile).tab("site")

    prefs = json.loads(
        (profile / "Default" / "Preferences").read_text(encoding="utf-8")
    )
    assert prefs == QUIET


async def test_the_profile_s_other_preferences_and_sign_ins_stay(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    (profile / "Default").mkdir(parents=True)
    (profile / "Default" / "Cookies").write_bytes(b"sign-ins")
    (profile / "Default" / "Preferences").write_text(
        json.dumps(
            {
                "autofill": {"profile_enabled": True, "other": 1},
                "profile": {"name": "agent", "password_manager_enabled": True},
                "intl": {"accept_languages": "ru"},
            }
        ),
        encoding="utf-8",
    )

    await _CountedBrowser(profile).tab("site")

    prefs = json.loads(
        (profile / "Default" / "Preferences").read_text(encoding="utf-8")
    )
    assert prefs == {
        "autofill": {
            "profile_enabled": False,
            "credit_card_enabled": False,
            "other": 1,
        },
        "profile": {"name": "agent", "password_manager_enabled": False},
        "intl": {"accept_languages": "ru"},
        "credentials_enable_service": False,
    }
    assert (profile / "Default" / "Cookies").read_bytes() == b"sign-ins"


async def test_an_unreadable_preferences_file_is_left_alone(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    profile = tmp_path / "profile"
    (profile / "Default").mkdir(parents=True)
    (profile / "Default" / "Preferences").write_text("{not json", encoding="utf-8")
    browser = _CountedBrowser(profile)

    await browser.tab("site")

    assert (profile / "Default" / "Preferences").read_text(
        encoding="utf-8"
    ) == "{not json"
    assert browser.launches == 1
    assert "Preferences" in caplog.text
