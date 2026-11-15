"""The parts of the OLX reader that need no browser: addresses, what a page's text becomes.
The browser part was checked live on olx.uz (2026-11-10): search, listing, phone."""

from dataclasses import replace
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tg_agent.clock import FakeClock
from tg_agent.config import Settings
from tg_agent.db import Repo
from tg_agent.search.olx import (
    _PHONES_JS,
    REVEALS_USED,
    OlxBrowser,
    RevealBudget,
    _description,
    _location,
    is_listing_url,
    phones_from,
    search_url,
)


def test_the_search_page_joins_words_with_dashes() -> None:
    assert search_url("3d печать  на заказ") == (
        "https://www.olx.uz/list/q-3d-%D0%BF%D0%B5%D1%87%D0%B0%D1%82%D1%8C-"
        "%D0%BD%D0%B0-%D0%B7%D0%B0%D0%BA%D0%B0%D0%B7/"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://www.olx.uz/d/obyavlenie/3d-pechat-na-zakaz-ID4lFQK.html",
        "https://olx.uz/d/obyavlenie/3d-pechat-na-zakaz-ID4lFQK.html",
        "https://www.olx.uz/d/oz/obyavlenie/3d-bosma-ID4lFQK.html?reason=search",
    ],
)
def test_listing_pages_are_recognised(url: str) -> None:
    assert is_listing_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/www.olx.uz/d/obyavlenie/x.html",
        "http://www.olx.uz/d/obyavlenie/x.html",
        "https://www.olx.uz/list/q-3d/",
        "https://www.olx.uz.evil.example/d/obyavlenie/x.html",
        "javascript:alert(1)",
    ],
)
def test_nothing_else_is_opened(url: str) -> None:
    assert not is_listing_url(url)


def test_the_place_drops_the_heading() -> None:
    raw = "Местоположение\nТашкент, Яшнабадский район\nЯшнабадский район\nТашкентская область"
    assert _location(raw) == "Ташкент, Яшнабадский район, Яшнабадский район"


def test_the_description_drops_its_heading_and_is_capped() -> None:
    assert _description("ОписаниеПечатаем PETG") == "Печатаем PETG"
    assert len(_description("Описание" + "я" * 5000)) == 1500


def test_phones_come_from_links_and_button_text_once_each() -> None:
    shown = ["+998935550099", "+99 893 5550099", "Сообщение", "+998 90 123 45 67"]
    assert phones_from(shown) == ["+998935550099", "+998901234567"]


# --- the daily cap on «Показать телефон» (stage 4б, R51) --------------------------------------


def with_cap(settings: Settings, cap: int) -> Settings:
    return replace(settings, search=replace(settings.search, olx_reveals_per_day=cap))


async def test_the_presses_are_counted_per_day_of_the_owner_s_time_zone(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    budget = RevealBudget(repo, fake_clock, with_cap(settings, 2))
    # 23:30 in Tashkent: the day ends in half an hour, though in UTC it is still early.
    fake_clock.set(datetime(2026, 11, 10, 23, 30, tzinfo=ZoneInfo("Asia/Tashkent")))

    assert [await budget.take() for _ in range(3)] == [True, True, False]
    fake_clock.advance(timedelta(minutes=29))
    assert await budget.take() is False
    fake_clock.advance(timedelta(minutes=2))  # 00:01: a new day
    assert [await budget.take() for _ in range(3)] == [True, True, False]


async def test_a_cap_of_zero_never_presses(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    assert await RevealBudget(repo, fake_clock, with_cap(settings, 0)).take() is False


class FakeButton:
    def __init__(self) -> None:
        self.clicks = 0

    async def count(self) -> int:
        return 1

    async def click(self) -> None:
        self.clicks += 1


class FakeLocator:
    def __init__(self, button: FakeButton) -> None:
        self.first = button


class FakeListingPage:
    """A listing with «Показать телефон»: the number shows once it is pressed."""

    def __init__(self) -> None:
        self.button = FakeButton()

    def locator(self, selector: str) -> FakeLocator:
        return FakeLocator(self.button)

    async def wait_for_timeout(self, ms: int) -> None:
        return None

    async def evaluate(self, script: str) -> object:
        if script == _PHONES_JS:
            return ["+998 90 123 45 67"] if self.button.clicks else []
        return False  # no captcha, no sign-in


async def test_over_the_cap_a_listing_is_read_without_the_press(
    repo: Repo, fake_clock: FakeClock, settings: Settings
) -> None:
    budget = RevealBudget(repo, fake_clock, with_cap(settings, 1))
    olx = OlxBrowser(None, reveals=budget)  # type: ignore[arg-type]

    first, second = FakeListingPage(), FakeListingPage()
    assert await olx._phones(first) == (["+998901234567"], "")
    assert await olx._phones(second) == ([], REVEALS_USED)

    assert (first.button.clicks, second.button.clicks) == (1, 0)
    assert REVEALS_USED == "дневной предел показов номеров на OLX исчерпан"


async def test_without_a_budget_the_button_is_pressed_as_before() -> None:
    page = FakeListingPage()
    assert await OlxBrowser(None)._phones(page) == (["+998901234567"], "")  # type: ignore[arg-type]
