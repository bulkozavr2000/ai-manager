"""OLX Uzbekistan through the agent's own browser on this computer (`search.browser`), in a tab
of its own.

The search model never drives this tab: it asks for a search or for one listing, and this code
opens the page, reads it and presses only "Показать телефон". Pages are read through OLX's own
data-testid / data-cy marks (checked on olx.uz, 2026-11-10). A captcha stops OLX for the rest of
the search: it is never solved by the program.

Sign-in: from the phone (`agent/site_logins`), in the same browser profile; the session stays
in that profile (encrypted by Windows, outside git).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import quote

from tg_agent.clock import Clock
from tg_agent.config import Settings
from tg_agent.db import Repo
from tg_agent.search.browser import AgentBrowser, BrowserError
from tg_agent.search.contacts import normalize_phone

BASE = "https://www.olx.uz"
LOGIN_URL = f"{BASE}/account/"
MAX_LISTINGS = 15
DESCRIPTION_MAX = 1500
_LISTING_URL = re.compile(
    r"^https://(?:www\.)?olx\.uz/d/(?:[a-z]{2}/)?obyavlenie/[A-Za-z0-9_\-]+\.html(?:[?#].*)?$"
)
NO_PHONE_BUTTON = "продавец не показывает телефон, только чат OLX"
# The page has a hidden copy of each contact button (for the phone layout): only the visible
# one can be pressed.
_PHONE_BUTTON = '[data-testid="ad-contact-phone"]:visible'
_CONTACT_BUTTONS = f'{_PHONE_BUTTON}, [data-testid="ad-contact-message-button"]:visible'
NEEDS_LOGIN = "телефон в этом объявлении виден только после входа — пропустил"
CAPTCHA = "OLX просит капчу — OLX в этот раз пропустил"
PHONE_DID_NOT_SHOW = "телефон не показался"
REVEALS_USED = "дневной предел показов номеров на OLX исчерпан"
OLX_TAB = "olx"
# kv: the presses of «Показать телефон» on one day of the owner's time zone (ruling R51).
REVEALS_KEY = "olx_reveals"

_SEARCH_JS = """() => [...document.querySelectorAll('[data-cy="l-card"]')].map(card => {
  const link = card.querySelector('a[href]');
  const head = card.querySelector('[data-cy="ad-card-title"] h4, [data-cy="ad-card-title"] h6,'
    + ' [data-cy="ad-card-title"] a, h4, h6');
  const text = el => (el ? el.innerText : '').trim();
  return {
    url: link ? new URL(link.getAttribute('href'), location.origin).href : '',
    title: text(head),
    price: text(card.querySelector('[data-testid="ad-price"]')),
    location: text(card.querySelector('[data-testid="location-date"]')),
  };
}).filter(row => row.url)"""

_LISTING_JS = """() => {
  const text = sel => { const el = document.querySelector(sel); return el ? el.innerText.trim() : ''; };
  return {
    title: text('[data-testid="offer_title"]'),
    price: text('[data-testid="ad-price-container"]') || text('[data-testid="prices-wrapper"]'),
    location: text('[data-testid="map-aside-section"]'),
    seller: text('[data-testid="user-profile-user-name"]'),
    description: text('[data-cy="ad_description"]'),
  };
}"""

# Phone numbers on the page once "Показать телефон" was pressed: tel: links and the text of the
# contact buttons and their box.
_PHONES_JS = """() => {
  const found = [];
  document.querySelectorAll('a[href^="tel:"]').forEach(a => found.push(a.getAttribute('href').slice(4)));
  const boxes = [...document.querySelectorAll('[data-testid="ad-action-buttons"],'
    + ' [data-testid="ad-contact-phone"]')];
  for (const box of boxes) for (const m of (box.innerText.match(/\\+?\\d[\\d\\s\\-()]{7,}\\d/g) || [])) found.push(m);
  return found;
}"""

_CAPTCHA_JS = """() => {
  const frames = [...document.querySelectorAll('iframe')].map(f => f.src || '');
  const words = /не робот|captcha|robot emas/i.test(document.body ? document.body.innerText.slice(0, 4000) : '');
  return frames.some(src => /captcha|challenge/i.test(src)) || words;
}"""

_LOGIN_JS = """() => /login|auth/i.test(location.href)
  || !!document.querySelector('input[type="password"]')"""


class OlxError(BrowserError):
    """OLX cannot be used now: a captcha, no browser, a page that is not OLX's."""


@dataclass(frozen=True)
class Listing:
    url: str
    title: str
    price: str
    location: str


@dataclass(frozen=True)
class ListingDetails:
    url: str
    title: str
    price: str
    location: str
    seller: str
    description: str
    phones: list[str] = field(default_factory=list)
    phone_note: str = ""  # why there is no phone, when there is none


class OlxSource(Protocol):
    async def search(self, query: str) -> list[Listing]: ...

    async def listing(self, url: str) -> ListingDetails: ...


def search_url(query: str) -> str:
    """OLX's search page for `query`, all of Uzbekistan: words joined by "-"."""
    slug = "-".join(query.split())
    return f"{BASE}/list/q-{quote(slug)}/"


def is_listing_url(url: str) -> bool:
    return bool(_LISTING_URL.match(url.strip()))


def _location(raw: str) -> str:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    lines = [line for line in lines if line.lower() != "местоположение"]
    return ", ".join(dict.fromkeys(lines[:2]))


def _description(raw: str) -> str:
    text = raw.strip()
    for prefix in ("Описание", "Tavsif"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text[:DESCRIPTION_MAX]


def phones_from(raw: list[str]) -> list[str]:
    """The phones among what the page showed, each once, in one form."""
    phones = [normalize_phone(item) for item in raw]
    return list(dict.fromkeys(phone for phone in phones if phone))


class RevealBudget:
    """At most `search.olx_reveals_per_day` presses of «Показать телефон» a calendar day of
    the owner's time zone (ruling R51), counted in kv so that a restart keeps the count: OLX
    marks an account that asks for many numbers."""

    def __init__(self, repo: Repo, clock: Clock, settings: Settings) -> None:
        self._repo = repo
        self._clock = clock
        self._settings = settings
        self._lock = asyncio.Lock()

    async def take(self) -> bool:
        """One press, if today still has one."""
        async with self._lock:
            day = self._clock.now().astimezone(self._settings.schedule.zone).date()
            raw = await self._repo.get_kv(REVEALS_KEY)
            kept = json.loads(raw) if raw else {}
            count = (
                int(kept.get("count", 0)) if kept.get("day") == day.isoformat() else 0
            )
            if count >= self._settings.search.olx_reveals_per_day:
                return False
            await self._repo.set_kv(
                REVEALS_KEY, json.dumps({"day": day.isoformat(), "count": count + 1})
            )
            return True


class OlxBrowser:
    """OLX in the agent's browser, in a tab of its own. With `reveals`, «Показать телефон»
    is pressed only while the day's presses last."""

    def __init__(
        self, browser: AgentBrowser, *, reveals: RevealBudget | None = None
    ) -> None:
        self._browser = browser
        self._reveals = reveals

    async def _page(self) -> Any:
        try:
            return await self._browser.tab(OLX_TAB)
        except BrowserError as error:
            raise OlxError(str(error)) from None

    async def _open(self, url: str) -> Any:
        page = await self._page()
        try:
            await page.goto(url, wait_until="domcontentloaded")
        except Exception as exc:  # noqa: BLE001 - timeouts, a dropped network
            raise OlxError(f"страница не открылась: {type(exc).__name__}") from None
        if await page.evaluate(_CAPTCHA_JS):
            raise OlxError(CAPTCHA)
        return page

    async def search(self, query: str) -> list[Listing]:
        page = await self._open(search_url(query))
        try:
            await page.wait_for_selector('[data-cy="l-card"]', timeout=8_000)
        except Exception:  # noqa: BLE001 - no cards: nothing found
            return []
        rows = await page.evaluate(_SEARCH_JS)
        return [
            Listing(row["url"], row["title"], row["price"], row["location"])
            for row in rows[:MAX_LISTINGS]
        ]

    async def listing(self, url: str) -> ListingDetails:
        if not is_listing_url(url):
            raise OlxError("это не страница объявления OLX")
        page = await self._open(url)
        try:
            await page.wait_for_selector('[data-testid="offer_title"]', timeout=10_000)
        except Exception:  # noqa: BLE001
            raise OlxError(
                "объявление не открылось: снято или страница другая"
            ) from None
        # The contact buttons and the place are drawn a moment after the title.
        for selector in (_CONTACT_BUTTONS, '[data-testid="map-aside-section"] p'):
            # Not every listing has them: then the wait simply runs out.
            with contextlib.suppress(Exception):
                await page.wait_for_selector(selector, timeout=6_000)
        data = await page.evaluate(_LISTING_JS)
        phones, note = await self._phones(page)
        return ListingDetails(
            url=url,
            title=data["title"],
            price=data["price"],
            location=_location(data["location"]),
            seller=data["seller"],
            description=_description(data["description"]),
            phones=phones,
            phone_note=note,
        )

    async def _phones(self, page: Any) -> tuple[list[str], str]:
        """Press "Показать телефон" and read the number; or why there is none."""
        button = page.locator(_PHONE_BUTTON).first
        if await button.count() == 0:
            return [], NO_PHONE_BUTTON
        if self._reveals is not None and not await self._reveals.take():
            return [], REVEALS_USED  # the rest of the listing is read all the same
        try:
            await button.click()
        except Exception:  # noqa: BLE001
            return [], PHONE_DID_NOT_SHOW
        for _ in range(12):
            await page.wait_for_timeout(500)
            if await page.evaluate(_CAPTCHA_JS):
                raise OlxError(CAPTCHA)
            phones = phones_from(await page.evaluate(_PHONES_JS))
            if phones:
                return phones, ""
            if await page.evaluate(_LOGIN_JS):
                return [], NEEDS_LOGIN
        return [], PHONE_DID_NOT_SHOW
