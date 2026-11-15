"""The search's browser on a server (ruling R85): `chromium` is Playwright's own Chromium."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tg_agent.config import ConfigError, Search
from tg_agent.search.browser import AgentBrowser


class FakeChromium:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def launch_persistent_context(self, profile: str, **kwargs: Any) -> object:
        self.calls.append({"profile": profile, **kwargs})
        return object()


class FakePlaywright:
    def __init__(self) -> None:
        self.chromium = FakeChromium()

    async def stop(self) -> None:
        pass


def fake_playwright(monkeypatch: pytest.MonkeyPatch) -> FakePlaywright:
    playwright = FakePlaywright()

    class Starter:
        async def start(self) -> FakePlaywright:
            return playwright

    import playwright.async_api as api

    monkeypatch.setattr(api, "async_playwright", lambda: Starter())
    return playwright


def test_chromium_is_a_browser_the_settings_accept() -> None:
    assert Search(browser="chromium").browser == "chromium"
    assert Search().browser == "msedge"  # the default stays
    with pytest.raises(ConfigError, match="chromium"):
        Search(browser="firefox")


async def test_chromium_launches_playwrights_own_chromium_without_a_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    playwright = fake_playwright(monkeypatch)

    await AgentBrowser(tmp_path / "profile", channel="chromium")._launch(True)

    (call,) = playwright.chromium.calls
    assert "channel" not in call
    assert call["headless"] is True


async def test_msedge_still_launches_with_its_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    playwright = fake_playwright(monkeypatch)

    await AgentBrowser(tmp_path / "profile", channel="msedge")._launch(True)

    (call,) = playwright.chromium.calls
    assert call["channel"] == "msedge"
