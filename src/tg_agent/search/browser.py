"""The agent's own browser: an installed Edge (or Chrome) driven by Playwright, with its own profile.

Sign-ins are optional (the search reads what is open to anyone); a sign-in from the phone
(`agent/site_logins`) lets the vendor search read OLX and other sites as him, without ever
seeing a password. One
profile opens in one browser at a time: OLX and the other sites share this one browser, each in
its own tab. It starts on first use and stays until the program ends.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from tg_agent.config import Settings

_log = logging.getLogger(__name__)

OLD_PROFILE = Path("browser") / "olx"  # before 2026-11-09 the profile was OLX's only
BLANK = "about:blank"
# What the profile must never keep of what is typed in it: addresses, cards, passwords. Merged
# into its preferences before every start; its sign-ins (cookies) are other files and stay.
QUIET_PREFERENCES: dict[str, Any] = {
    "autofill": {"profile_enabled": False, "credit_card_enabled": False},
    "credentials_enable_service": False,
    "profile": {"password_manager_enabled": False},
}


class BrowserError(Exception):
    """The browser cannot do it: not installed, the profile is busy, a page did not open, refused."""


def _merge(into: dict[str, Any], values: dict[str, Any]) -> None:
    """Set `values` in `into`, keeping every other key of the nested dictionaries."""
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(into.get(key), dict):
            _merge(into[key], value)
        else:
            into[key] = value


def quiet_profile(profile_dir: Path) -> None:
    """Turn off autofill and the password manager in the profile's preferences.

    A missing file (a new profile) is created; a file that cannot be read as JSON is left
    as it is and logged: the browser keeps it, and a broken file must not stop the search.
    """
    path = profile_dir / "Default" / "Preferences"
    prefs: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            _log.warning(
                "the browser's Preferences were left alone: %s", type(error).__name__
            )
            return
        if not isinstance(loaded, dict):
            _log.warning("the browser's Preferences were left alone: not an object")
            return
        prefs = loaded
    _merge(prefs, QUIET_PREFERENCES)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(prefs, ensure_ascii=False), encoding="utf-8")
    except OSError as error:
        _log.warning(
            "the browser's Preferences were not written: %s", type(error).__name__
        )


def agent_profile(settings: Settings) -> Path:
    """The folder of the agent's browser profile.

    The OLX-only profile of earlier versions becomes it, once, so a sign-in made there is kept.
    If it cannot be moved (a browser still holds it), it is used where it is.
    """
    profile = settings.search.browser_profile
    if not profile.is_absolute():
        profile = settings.data_dir / profile
    old = settings.data_dir / OLD_PROFILE
    if old.is_dir() and not profile.exists():
        try:
            profile.parent.mkdir(parents=True, exist_ok=True)
            old.rename(profile)
        except OSError as error:
            _log.warning(
                "the old browser profile was not moved: %s", type(error).__name__
            )
            return old
    return profile


class AgentBrowser:
    """One persistent browser profile, started on first use; `tab(name)` is a tab of its own for
    each user (OLX, other sites), so one never finds the other's page under its feet."""

    def __init__(
        self,
        profile_dir: Path,
        *,
        channel: str = "msedge",
        headless: bool = True,
        timeout_ms: int = 25_000,
    ) -> None:
        self._profile_dir = profile_dir
        self._channel = channel
        self._headless = headless
        self._timeout_ms = timeout_ms
        self._playwright: Any = None
        self._context: Any = None
        self._tabs: dict[str, Any] = {}
        # Two searches or OLX and a site may want their first tab at once: one profile opens
        # in one browser only, so the start and the choice of a tab happen one at a time.
        self._start_lock = asyncio.Lock()
        self._tab_lock = asyncio.Lock()

    async def _launch(self, headless: bool) -> tuple[Any, Any]:
        """Playwright and a browser on the agent's profile."""
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - the dependency is in pyproject
            raise BrowserError("Playwright не установлен") from exc
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        playwright = await async_playwright().start()
        try:
            # `chromium` is Playwright's own browser, which takes no channel (a server).
            channel = {} if self._channel == "chromium" else {"channel": self._channel}
            context = await playwright.chromium.launch_persistent_context(
                str(self._profile_dir),
                **channel,
                headless=headless,
                # A page may try to save a file: nothing lands on this computer.
                accept_downloads=False,
            )
        except Exception as exc:  # noqa: BLE001 - no browser, a busy profile, anything
            with contextlib.suppress(Exception):
                await playwright.stop()
            raise BrowserError(
                f"браузер не запустился ({type(exc).__name__}): если открыто окно входа"
                " на сайты — закрой его"
            ) from None
        return playwright, context

    async def _start(self) -> Any:
        async with self._start_lock:
            if self._context is not None:
                return self._context
            quiet_profile(self._profile_dir)
            self._playwright, self._context = await self._launch(self._headless)
            self._context.set_default_timeout(self._timeout_ms)
            return self._context

    async def tab(self, name: str) -> Any:
        """The tab kept for `name`; a new one if there is none or it was closed.

        A page of the browser that is nobody's tab is taken only when it is blank (the one a
        new browser opens with). Any other leftover — a popup a site or another session
        opened — is closed, never handed to the search: its page is not one it chose.
        """
        async with self._tab_lock:
            page = self._tabs.get(name)
            if page is not None and not page.is_closed():
                return page
            context = await self._start()
            taken = {id(tab) for tab in self._tabs.values()}
            page = None
            for spare in list(context.pages):
                if id(spare) in taken or spare.is_closed():
                    continue
                if page is None and spare.url == BLANK:
                    page = spare
                elif spare.url != BLANK:
                    with contextlib.suppress(Exception):  # already gone
                        await spare.close()
            if page is None:
                page = await context.new_page()
            self._tabs[name] = page
            return page

    async def close(self) -> None:
        context, playwright = self._context, self._playwright
        self._context = self._playwright = None
        self._tabs.clear()
        if context is not None:
            with contextlib.suppress(Exception):  # a browser that is already gone
                await context.close()
        if playwright is not None:
            with contextlib.suppress(Exception):
                await playwright.stop()
