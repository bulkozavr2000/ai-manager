"""The search: Claude with tools finds whom to write to about a request in the asker's words.

The only model in the program with tools, and the one that writes to nobody. Its tools:
Anthropic's web search and web fetch (run by Anthropic), OLX and any other site through the
agent's own browser on this computer, where the owner signed in himself (`search.olx`,
`search.sites`), Telegram's public search through the work account (read only), and
`save_candidate`. It never sees the task's ceiling, the owner's phone or address, or any
conversation. A candidate is kept only if its contact appears in what a tool showed: a page
fetched or opened, an OLX listing, a Telegram profile — a number the model makes up is refused.
The task's agent that asked for the search decides whom of them it writes to.

The browser holds the owner's sign-ins and reads text anyone may have written, so the model
cannot send it just anywhere: it opens what an earlier result showed, or any page
of a site the owner named, never this computer or the home network; it presses only links and
buttons that show more; it types only search words, only into a site's search box.

The search stops at the first of: the model is done, `max_steps` model calls, `max_cost_usd`
spent (tokens plus $0.01 per web search), `max_minutes` of running, a failed or refused call,
the asker no longer wanting it (its task stopped). What it saved stays.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, timedelta
from typing import Any

from tg_agent.brain.context import load_prompt
from tg_agent.brain.llm import LLM, LLMError, ModelRefusal, SearchStep
from tg_agent.brain.pricing import WEB_SEARCH_USD, cost_usd
from tg_agent.clock import Clock, SystemClock
from tg_agent.config import Settings
from tg_agent.db import Repo
from tg_agent.gateway.base import Gateway, TgActionError
from tg_agent.search.browser import BrowserError
from tg_agent.search.contacts import normalize_contact, seen_in, show
from tg_agent.search.olx import BASE as OLX_BASE
from tg_agent.search.olx import CAPTCHA as OLX_CAPTCHA
from tg_agent.search.olx import LOGIN_URL as OLX_LOGIN_URL
from tg_agent.search.olx import (
    NEEDS_LOGIN,
    OlxError,
    OlxSource,
    is_listing_url,
)
from tg_agent.search.sites import (
    NO_PAGE,
    Page,
    SitePages,
    as_url,
    click_refusal,
    describe,
    on_site,
    query_refusal,
    site_of,
    url_key,
    url_problem,
    urls_in,
)

_log = logging.getLogger(__name__)

NAME_MAX, NOTE_MAX, SOURCE_MAX, QUERY_MAX, CLICK_MAX = 80, 300, 300, 100, 100
TELEGRAM_RESULTS = 10
FETCH_MAX_TOKENS = 6000
_KINDS = {"user": "аккаунт", "channel": "канал", "group": "группа"}
NOT_SEEN = (
    "Этого контакта нет в том, что показали инструменты. Открой страницу с ним"
    " (web_fetch, open_page или olx_listing) и сохрани снова — придумывать контакты нельзя"
)
NOT_LINKED = (
    "Этот адрес не встречался в результатах, и сайт не из строки «Сайты владельца»."
    " Найди страницу через web_search или открой ссылку со страницы, которую уже открыл"
)
PAGES_USED = "Страниц на этот поиск больше нельзя: сохрани, что нашёл, и закончи"
# Саид works from his phone: a site that wants a login or a captcha is skipped, and he is told.
LOGIN_NOTE = "{site} показывает контакты только после входа — этот сайт пропустил"
CAPTCHA_NOTE = "{site} просит капчу — этот сайт пропустил"
_SITE_TOOLS = ("open_page", "click", "search_site")
OLX_SITE = "olx.uz"

_CLIENT_TOOLS: dict[str, dict[str, Any]] = {
    "telegram_search": {
        "name": "telegram_search",
        "description": "Публичный поиск Telegram: аккаунты, каналы и группы по словам."
        " Возвращает @username, название и описание.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "2–4 слова"}},
            "required": ["query"],
        },
    },
    "olx_search": {
        "name": "olx_search",
        "description": "Поиск объявлений на OLX Узбекистана. Возвращает до 15 объявлений:"
        " название, цена, место, ссылка.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "2–4 слова"}},
            "required": ["query"],
        },
    },
    "olx_listing": {
        "name": "olx_listing",
        "description": "Открыть объявление OLX по ссылке из olx_search: описание, продавец,"
        " место и телефон, если продавец его показывает.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    "open_page": {
        "name": "open_page",
        "description": "Открыть страницу сайта в браузере агента, где владелец вошёл на сайты"
        " сам. Возвращает текст страницы, телефоны и Telegram с неё, ссылки и кнопки вроде"
        " «Показать телефон». Открывает ссылки из результатов поиска и с открытых страниц,"
        " а на сайтах из строки «Сайты владельца» — любую страницу (например, их поиск)."
        " Другой адрес сначала найди через web_search.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "ссылка https://…"}
            },
            "required": ["url"],
        },
    },
    "click": {
        "name": "click",
        "description": "Нажать на открытой странице ссылку или кнопку по её тексту:"
        " «Показать телефон», «Контакты», «Далее». Только то, что показывает больше:"
        " покупать, писать, отправлять формы, подписываться, входить и выходить нельзя.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "текст ссылки или кнопки, как на странице",
                }
            },
            "required": ["text"],
        },
    },
    "search_site": {
        "name": "search_site",
        "description": "Поиск по открытому сайту: вписать слова в его строку поиска и нажать"
        " Enter. Больше ничего никуда не вписывает.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "2–5 слов"}},
            "required": ["query"],
        },
    },
    "save_candidate": {
        "name": "save_candidate",
        "description": "Сохранить подходящего человека или компанию. Контакт — @username или"
        " телефон, ровно как он был в результатах инструментов.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "кто это, 2–5 слов"},
                "contact": {"type": "string", "description": "@username или телефон"},
                "source": {
                    "type": "string",
                    "description": "ссылка, где найден контакт",
                },
                "note": {
                    "type": "string",
                    "description": "чем подходит, одной строкой",
                },
            },
            "required": ["name", "contact", "source", "note"],
        },
    },
}


@dataclass(frozen=True)
class Candidate:
    """A vendor the search found: who, how to reach him, where he was found, why he fits."""

    name: str
    contact: str  # a username without "@", or "+<digits>"
    source: str
    note: str


@dataclass(frozen=True)
class Wall:
    """A site that hid its contacts behind a sign-in ("login") or a captcha ("captcha"), and
    the page to start a sign-in from: the owner may pass it from his phone (stage 4б)."""

    site: str
    url: str
    kind: str


@dataclass
class SearchOutcome:
    candidates: list[Candidate] = field(default_factory=list)
    cost: float = 0.0
    # None (done), "budget", "steps", "refusal", "error", "stopped" (no longer wanted),
    # "time" (ran longer than search.max_minutes), "money" (did not start: the day's or the
    # month's money no longer holds one search)
    stopped: str | None = None
    notes: list[str] = field(
        default_factory=list
    )  # for the owner: OLX wants a login, ...
    summary: str = ""  # the model's last words: how many, where it looked
    # The sign-in walls and captchas it met, one per site, where the notes say so.
    walls: list[Wall] = field(default_factory=list)


def _get(block: Any, key: str) -> Any:
    """A field of a content block: an SDK object or, in tests, a dict."""
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def _shown_by_server(block: Any) -> str:
    """The text a server tool showed: a fetched page, the titles and links of search results.

    Code the model writes and runs (dynamic filtering) does not count: it could print anything.
    """
    kind = _get(block, "type")
    content = _get(block, "content")
    if kind == "web_fetch_tool_result" and _get(content, "type") == "web_fetch_result":
        source = _get(_get(content, "content"), "source")
        data = _get(source, "data") if _get(source, "type") == "text" else ""
        return f"{_get(content, 'url') or ''}\n{data or ''}"
    if kind == "web_search_tool_result" and isinstance(content, list):
        return "\n".join(
            f"{_get(r, 'title') or ''} {_get(r, 'url') or ''}" for r in content
        )
    return ""


class Searcher:
    def __init__(
        self,
        llm: LLM,
        repo: Repo,
        settings: Settings,
        gateway: Gateway,
        olx: OlxSource | None,
        *,
        pages: SitePages | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._llm = llm
        self._clock = clock or SystemClock(UTC)
        self._repo = repo
        self._settings = settings
        self._gateway = gateway
        self._olx = olx
        self._pages = pages
        # One search at a time: every search shares the one browser and its tab, so a second
        # task's search would act on the page the first one opened (review I2).
        self._lock = asyncio.Lock()

    def _tools(self) -> list[dict[str, Any]]:
        search = self._settings.search
        tools: list[dict[str, Any]] = [
            {
                "type": "web_search_20260209",
                "name": "web_search",
                "max_uses": search.web_searches_per_step,
                "user_location": {
                    "type": "approximate",
                    "timezone": self._settings.schedule.tz,
                },
            },
            {
                "type": "web_fetch_20260209",
                "name": "web_fetch",
                "max_uses": search.web_fetches_per_step,
                "max_content_tokens": FETCH_MAX_TOKENS,
                # Pages come back to us, so a saved contact can be checked against them.
                "allowed_callers": ["direct"],
            },
            _CLIENT_TOOLS["telegram_search"],
        ]
        if self._olx is not None:
            tools += [_CLIENT_TOOLS["olx_search"], _CLIENT_TOOLS["olx_listing"]]
        if self._pages is not None:
            tools += [_CLIENT_TOOLS[name] for name in _SITE_TOOLS]
        return [*tools, _CLIENT_TOOLS["save_candidate"]]

    def _brief(self, what: str, skip: Sequence[str], sites: Sequence[str]) -> str:
        """The request as the search sees it: what, where, whom it already has. Only the
        asker's words reach it, never a task's ceiling or anything else of the task."""
        lines = [f"Что найти: {' '.join(what.split())}"]
        lines.append(f"Город: {self._settings.search.city}")
        if sites:
            lines.append("Сайты владельца — ищи и на них: " + ", ".join(sites))
        if skip:
            lines.append("Уже есть — не сохранять: " + ", ".join(show(c) for c in skip))
        lines.append(f"Найди до {self._settings.search.max_candidates}.")
        return "\n".join(lines)

    async def run_query(
        self,
        what: str,
        skip: Sequence[str],
        *,
        task_id: int | None = None,
        wake_id: int | None = None,
        still_wanted: Callable[[], Awaitable[bool]] | None = None,
        sites: Sequence[str] = (),
        affordable: Callable[[], Awaitable[bool]] | None = None,
    ) -> SearchOutcome:
        """Search for whom to write to about `what`, in the asker's own words.

        Each model call is logged for `task_id` and `wake_id`, so it counts toward the task's
        cost and the day's limits. `still_wanted` is asked before every model call: once it
        answers False (the task stopped or finished), the search ends with what it saved.
        Searches run one at a time; one that waited behind another asks `still_wanted` only
        after its turn came, so a task closed meanwhile spends nothing. `affordable` is asked
        once, when its turn has come and before its first model call: the search that waited
        while another one spent the money ends at once, with `stopped = "money"` (review I-C).
        """
        async with self._lock:
            if affordable is not None and not await affordable():
                return SearchOutcome(stopped="money")
            return await self._run_query(
                what,
                skip,
                task_id=task_id,
                wake_id=wake_id,
                still_wanted=still_wanted,
                sites=sites,
            )

    async def _run_query(
        self,
        what: str,
        skip: Sequence[str],
        *,
        task_id: int | None,
        wake_id: int | None,
        still_wanted: Callable[[], Awaitable[bool]] | None,
        sites: Sequence[str],
    ) -> SearchOutcome:
        settings = self._settings.search
        known = [c for c in (normalize_contact(raw) for raw in skip) if c]
        # The sites it may browse freely: the settings' and those the asker says the owner
        # gave (`sites`), never a site of the request's free text, which the task's model
        # wrote after reading counterparts (ruling R44).
        owner_sites = list(dict.fromkeys([*settings.sites, *sites]))
        brief = self._brief(what, known, owner_sites)
        run = _Run(known, brief, set(owner_sites))
        system = load_prompt("search").format(
            city=settings.city, max=settings.max_candidates
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": brief}]
        tools = self._tools()
        outcome = run.outcome
        # The minutes count from the search's turn, not from the wait behind another one.
        deadline = self._clock.now() + timedelta(minutes=settings.max_minutes)
        for _ in range(settings.max_steps):
            if self._clock.now() >= deadline:
                outcome.stopped = "time"
                return outcome
            if still_wanted is not None and not await still_wanted():
                outcome.stopped = "stopped"
                return outcome
            try:
                step = await self._llm.search_step(
                    system=system, messages=messages, tools=tools
                )
            except ModelRefusal:
                outcome.stopped = "refusal"
                return outcome
            except LLMError as error:
                _log.warning("search step failed: %s", type(error).__name__)
                outcome.stopped = "error"
                return outcome
            await self._pay(step, outcome, task_id, wake_id)
            for block in step.content:
                run.corpus.append(_shown_by_server(block))
            messages.append({"role": "assistant", "content": step.content})
            texts = [_get(b, "text") for b in step.content if _get(b, "type") == "text"]
            if texts:
                outcome.summary = " ".join(t.strip() for t in texts if t).strip()
            if outcome.cost >= settings.max_cost_usd:
                outcome.stopped = "budget"
                return outcome
            if step.stop_reason == "pause_turn":
                continue
            calls = [b for b in step.content if _get(b, "type") == "tool_use"]
            if not calls:
                return outcome
            results = []
            for c in calls:
                results.append(await self._answer(run, c))
                # A slow site or OLX can hold a search for minutes: it ends with what it has.
                if self._clock.now() >= deadline:
                    outcome.stopped = "time"
                    return outcome
            messages.append({"role": "user", "content": results})
        outcome.stopped = "steps"
        return outcome

    async def _pay(
        self,
        step: SearchStep,
        outcome: SearchOutcome,
        task_id: int | None,
        wake_id: int | None,
    ) -> None:
        cost = cost_usd(step.model, step.usage) + step.web_searches * WEB_SEARCH_USD
        outcome.cost += cost
        await self._repo.log_llm_call(
            None,
            "search",
            step.model,
            step.usage,
            cost,
            step.stop_reason,
            task_id=task_id,
            wake_id=wake_id,
        )

    async def _answer(self, run: _Run, call: Any) -> dict[str, Any]:
        """Do one client tool call; return its tool_result block."""
        name, args = _get(call, "name"), _get(call, "input") or {}
        handler = {
            "telegram_search": self._telegram_search,
            "olx_search": self._olx_search,
            "olx_listing": self._olx_listing,
            "open_page": self._open_page,
            "click": self._click,
            "search_site": self._search_site,
            "save_candidate": self._save,
        }.get(name)
        if (
            handler is None
            or (name.startswith("olx") and self._olx is None)
            or (name in _SITE_TOOLS and self._pages is None)
        ):
            text, error = f"Нет такого инструмента: {name}", True
        else:
            text, error = await handler(run, args)
        result: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": _get(call, "id"),
            "content": text,
        }
        if error:
            result["is_error"] = True
        return result

    async def _telegram_search(self, run: _Run, args: dict) -> tuple[str, bool]:
        query = str(args.get("query") or "").strip()[:QUERY_MAX]
        if not query:
            return "Нужны слова для поиска", True
        if run.telegram_searches >= self._settings.search.telegram_searches:
            return "Поиск в Telegram на сегодня исчерпан: ищи в OLX и в интернете", True
        run.telegram_searches += 1
        try:
            peers = await self._gateway.search_public(query, TELEGRAM_RESULTS)
        except TgActionError as error:
            return f"Telegram не ответил ({error.kind.value})", True
        if not peers:
            return f"В Telegram по «{query}» ничего", False
        lines = [f"Telegram по «{query}»:"]
        for peer in peers:
            about = " ".join(peer.about.split())[:300]
            lines.append(
                f"@{peer.username} — {peer.title} ({_KINDS[peer.kind]})"
                + (f": {about}" if about else "")
            )
        # Only the results count as shown: the header repeats the model's own query, and a
        # contact or an address it typed there was seen by nobody (review M4).
        run.corpus.append("\n".join(lines[1:]))
        return "\n".join(lines), False

    async def _olx_search(self, run: _Run, args: dict) -> tuple[str, bool]:
        query = str(args.get("query") or "").strip()[:QUERY_MAX]
        if not query:
            return "Нужны слова для поиска", True
        try:
            listings = await self._olx.search(query)  # type: ignore[union-attr]
        except OlxError as error:
            run.olx_failed(error)
            return f"OLX недоступен: {error}", True
        if not listings:
            return f"На OLX по «{query}» ничего", False
        lines = [f"OLX по «{query}»:"]
        for i, listing in enumerate(listings, 1):
            lines.append(
                f"{i}. {listing.title} — {listing.price or 'цена не указана'}"
                f" — {listing.location}\n   {listing.url}"
            )
        run.corpus.append("\n".join(lines[1:]))  # not the query: see _telegram_search
        return "\n".join(lines), False

    async def _olx_listing(self, run: _Run, args: dict) -> tuple[str, bool]:
        url = str(args.get("url") or "").strip()
        if not is_listing_url(url):
            return "Открываю только объявления OLX: https://www.olx.uz/d/...", True
        try:
            details = await self._olx.listing(url)  # type: ignore[union-attr]
        except OlxError as error:
            run.olx_failed(error)
            return f"OLX недоступен: {error}", True
        if details.phone_note == NEEDS_LOGIN:
            run.note(f"OLX: {NEEDS_LOGIN}")
            # The sign-in starts on OLX's own page for it, not on the listing.
            run.wall(Wall(OLX_SITE, OLX_LOGIN_URL, "login"))
        phones = (
            ", ".join(show(phone) for phone in details.phones) or details.phone_note
        )
        text = "\n".join(
            [
                f"{details.title} — {details.price or 'цена не указана'}",
                f"Продавец: {details.seller}; место: {details.location}",
                f"Телефон: {phones}",
                f"Описание: {details.description}",
                f"Ссылка: {details.url}",
            ]
        )
        run.corpus.append(text)
        return text, False

    async def _open_page(self, run: _Run, args: dict) -> tuple[str, bool]:
        url = as_url(str(args.get("url") or ""))
        if not url:
            return "Нужна ссылка", True
        problem = url_problem(url) or (None if self._linked(run, url) else NOT_LINKED)
        if problem:
            await self._log_page("open_page", url, "refused")
            return problem, True
        if run.pages >= self._settings.search.max_pages:
            return PAGES_USED, True
        run.pages += 1
        try:
            page = await self._pages.open(url)  # type: ignore[union-attr]
        except BrowserError as error:
            await self._log_page("open_page", url, "error")
            return f"Не вышло: {error}", True
        run.page_open = True
        return await self._shown(run, "open_page", page), False

    async def _click(self, run: _Run, args: dict) -> tuple[str, bool]:
        label = " ".join(str(args.get("text") or "").split())[:CLICK_MAX]
        if not label:
            return "Нужен текст ссылки или кнопки", True
        refusal = click_refusal(label)
        if refusal:
            await self._log_page("click", "", "refused", text=label)
            return refusal, True
        if not run.page_open:
            return NO_PAGE, True
        if run.pages >= self._settings.search.max_pages:
            return PAGES_USED, True
        run.pages += 1
        try:
            page = await self._pages.click(label)  # type: ignore[union-attr]
        except BrowserError as error:
            await self._log_page("click", "", "error", text=label)
            return f"Не вышло: {error}", True
        return await self._shown(run, "click", page, text=label), False

    async def _search_site(self, run: _Run, args: dict) -> tuple[str, bool]:
        query = " ".join(str(args.get("query") or "").split())
        refusal = query_refusal(query)
        if refusal:
            return refusal, True
        if not run.page_open:
            return NO_PAGE, True
        if run.pages >= self._settings.search.max_pages:
            return PAGES_USED, True
        run.pages += 1
        try:
            page = await self._pages.search(query)  # type: ignore[union-attr]
        except BrowserError as error:
            await self._log_page("search_site", "", "error", query=query)
            return f"Не вышло: {error}", True
        return await self._shown(run, "search_site", page, query=query), False

    def _linked(self, run: _Run, url: str) -> bool:
        """May the model open this address: did a result show it, or is it a page of a site
        the owner named? A made-up address, or one written into the request, could carry out
        what a page or a counterpart told the model to send."""
        if url_key(url) in urls_in("\n".join(run.corpus)):
            return True
        return on_site(site_of(url), run.sites)

    async def _shown(self, run: _Run, tool: str, page: Page, **extra: str) -> str:
        """The page as the model is shown it; what Саид must do himself becomes a note."""
        site = site_of(page.url)
        result = "ok"
        if page.captcha:
            run.note(CAPTCHA_NOTE.format(site=site))
            result = "captcha"
        if page.login_wall:
            run.note(LOGIN_NOTE.format(site=site))
            result = "login"
        # A page with both is a sign-in wall: the sign-in passes its captcha too. Only a
        # site the owner named gets a card (review 4б.2, I1): a stranger's page found by a
        # web search could draw a look-alike sign-in. Any other site keeps just its note.
        if result != "ok" and on_site(site, run.sites):
            run.wall(Wall(site, page.url, result))
        await self._log_page(tool, page.url, result, **extra)
        text = describe(page)
        run.corpus.append(text)
        return text

    async def _log_page(self, tool: str, url: str, result: str, **extra: str) -> None:
        """Which page the search opened or was refused, never its text."""
        await self._repo.log_event(
            "search_page", tool=tool, url=url[:300], result=result, **extra
        )

    async def _save(self, run: _Run, args: dict) -> tuple[str, bool]:
        limit = self._settings.search.max_candidates
        contact = normalize_contact(str(args.get("contact") or ""))
        if contact is None:
            return "Контакт — это @username в Telegram или номер телефона", True
        if not seen_in(contact, "\n".join(run.corpus)):
            return NOT_SEEN, True
        key = contact.casefold()
        if key in run.taken:
            return f"Уже есть: {show(contact)}", True
        if len(run.outcome.candidates) >= limit:
            return f"Хватит: уже {limit} — заканчивай", True
        run.taken.add(key)
        run.outcome.candidates.append(
            Candidate(
                name=" ".join(str(args.get("name") or "").split())[:NAME_MAX],
                contact=contact,
                source=str(args.get("source") or "").strip()[:SOURCE_MAX],
                note=" ".join(str(args.get("note") or "").split())[:NOTE_MAX],
            )
        )
        return f"Сохранено: {len(run.outcome.candidates)} из {limit}", False


class _Run:
    """What one search keeps while it runs."""

    def __init__(self, known: Sequence[str], brief: str, sites: set[str]) -> None:
        self.outcome = SearchOutcome()
        # Everything the tools showed: contacts and addresses are checked against it.
        self.corpus: list[str] = []
        self.taken: set[str] = {contact.casefold() for contact in known}
        self.telegram_searches = 0
        self.brief = brief  # the request as the model got it
        self.sites = sites  # sites the owner named: any of their pages may be opened
        self.pages = 0  # pages opened, pressed or searched on in this search
        # A page was opened in this search: click and search_site act on it.
        self.page_open = False

    def note(self, text: str) -> None:
        if text not in self.outcome.notes:
            self.outcome.notes.append(text)

    def wall(self, wall: Wall) -> None:
        """One wall per site in a search: the first page that showed it."""
        if wall.site and all(w.site != wall.site for w in self.outcome.walls):
            self.outcome.walls.append(wall)

    def olx_failed(self, error: OlxError) -> None:
        """OLX could not be used: a note for the owner, and a wall when it was a captcha."""
        self.note(f"OLX: {error}")
        if str(error) == OLX_CAPTCHA:
            self.wall(Wall(OLX_SITE, f"{OLX_BASE}/", "captcha"))
