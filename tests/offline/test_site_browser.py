"""The site browser in a real installed browser (Edge), against small pages served from this
computer: reading a page, a phone behind «Показать телефон», the buttons it must not press, a
site's search box, a login wall. Skipped where Edge is not installed."""

from __future__ import annotations

import functools
import os
import threading
from collections.abc import AsyncIterator, Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tg_agent.search.browser import AgentBrowser, BrowserError
from tg_agent.search.sites import SiteBrowser

# Fake numbers only: never the owner's.
PAGES = {
    "index.html": """<!doctype html><html><head><meta charset="utf-8">
<title>Мастерская 3D</title></head><body>
<header><a href="/contacts.html">Контакты</a> <a href="/logout">Выйти</a></header>
<main>
<h1>3D-печать на заказ</h1>
<p>Печатаем PETG и PLA. Цена от 25 000 сум.</p>
<form action="/results.html" role="search">
  <input name="q" placeholder="Поиск по сайту"><button type="submit">Найти</button>
</form>
<form action="/callback" method="post">
  <input name="name"><button type="submit">Показать номер</button>
</form>
<button onclick="document.title = 'КУПЛЕНО'">Купить</button>
<button onclick="document.title = 'СРАВНЕНО'">Сравнить</button>
<button onclick="document.getElementById('more').hidden = false">Показать ещё</button>
<p id="more" hidden>Ещё работы: шестерни, корпуса</p>
</main></body></html>""",
    "contacts.html": """<!doctype html><html><head><meta charset="utf-8">
<title>Контакты — Мастерская 3D</title></head><body>
<p>Telegram: <a href="https://t.me/print_lab">наш канал</a></p>
<p>WhatsApp: <a href="https://wa.me/998935550011">WhatsApp</a></p>
<button id="show" onclick="this.outerHTML =
  '<a href=&quot;tel:+998901234567&quot;>+998 90 123 45 67</a>'">Показать телефон</button>
<p><a href="/index.html">На главную</a></p>
</body></html>""",
    "results.html": """<!doctype html><html><head><meta charset="utf-8">
<title>Результаты</title></head><body><p id="r"></p>
<script>document.getElementById('r').textContent =
  'Результаты по: ' + new URLSearchParams(location.search).get('q')</script>
</body></html>""",
    "login.html": """<!doctype html><html><head><meta charset="utf-8">
<title>Вход</title></head><body>
<form action="/auth" method="post"><input name="login">
<input type="password" name="password"><button>Войти</button></form>
</body></html>""",
    # Like OLX: the contact button is drawn well after the page has loaded, and the phone
    # asks for a sign-in in a window over a long page.
    "listing.html": """<!doctype html><html><head><meta charset="utf-8">
<title>Объявление</title></head><body>
<p>"""
    + "Печатаем на заказ корпуса, шестерни и макеты. " * 60
    + """</p>
<div id="contact"></div>
<script>
setTimeout(() => {
  document.getElementById('contact').innerHTML = '<button id="phone">Показать телефон</button>';
  document.getElementById('phone').onclick = () => {
    document.body.insertAdjacentHTML('beforeend', '<div role="dialog"><p>Войдите</p>'
      + '<input name="login"><input type="password" name="password"></div>');
  };
}, 1000);
</script>
</body></html>""",
}


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass  # the test output stays quiet


@pytest.fixture
def site(tmp_path: Path) -> Iterator[str]:
    root = tmp_path / "site"
    root.mkdir()
    for name, html in PAGES.items():
        (root / name).write_text(html, encoding="utf-8")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_Quiet, directory=str(root))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
async def browser(tmp_path: Path) -> AsyncIterator[AgentBrowser]:
    # The browser the settings would name: Edge here, Playwright's Chromium on a server
    # (TG_AGENT_TEST_BROWSER=chromium). Never the agent's real profile.
    channel = os.environ.get("TG_AGENT_TEST_BROWSER", "msedge")
    agent = AgentBrowser(tmp_path / "profile", channel=channel)
    try:
        await agent.tab("site")
    except BrowserError:
        pytest.skip(f"the browser {channel!r} is not installed here")
    try:
        yield agent
    finally:
        await agent.close()


async def test_a_page_is_read_and_a_hidden_phone_revealed(
    site: str, browser: AgentBrowser
) -> None:
    pages = SiteBrowser(browser, allow_private=True)

    home = await pages.open(f"{site}/index.html")
    assert home.title == "Мастерская 3D"
    assert "Печатаем PETG" in home.text
    assert ("Контакты", f"{site}/contacts.html") in home.links

    contacts = await pages.click("Контакты")
    assert contacts.url == f"{site}/contacts.html"
    assert "print_lab" in contacts.telegram
    assert "+998935550011" in contacts.phones  # from the WhatsApp link
    assert "Показать телефон" in contacts.buttons
    assert "+998901234567" not in contacts.phones

    revealed = await pages.click("Показать телефон")
    assert "+998901234567" in revealed.phones


async def test_buying_and_form_buttons_are_never_pressed(
    site: str, browser: AgentBrowser
) -> None:
    pages = SiteBrowser(browser, allow_private=True)
    await pages.open(f"{site}/index.html")

    # «Купить» and «Выйти»: refused words; «Показать номер»: it would send a form;
    # «Сравнить»: a button that does not reveal anything.
    for label in ("Купить", "Выйти", "Показать номер", "Сравнить"):
        with pytest.raises(BrowserError):
            await pages.click(label)

    page = await pages.click("Показать ещё")
    assert "шестерни" in page.text
    assert page.title == "Мастерская 3D"  # nothing refused was pressed


async def test_the_site_search_box_and_a_login_wall(
    site: str, browser: AgentBrowser
) -> None:
    pages = SiteBrowser(browser, allow_private=True)
    await pages.open(f"{site}/index.html")

    found = await pages.search("3d печать")
    assert "Результаты по: 3d печать" in found.text

    door = await pages.open(f"{site}/login.html")
    assert door.login_wall is True
    with pytest.raises(BrowserError):  # a login form is not a search box
        await pages.search("3d печать")


async def test_a_button_drawn_late_is_seen_and_a_sign_in_window_is_a_login_wall(
    site: str, browser: AgentBrowser
) -> None:
    pages = SiteBrowser(browser, allow_private=True)

    listing = await pages.open(f"{site}/listing.html")
    assert "Показать телефон" in listing.buttons
    assert listing.login_wall is False

    asked = await pages.click("Показать телефон")
    assert asked.login_wall is True


async def test_local_addresses_are_refused_by_default(browser: AgentBrowser) -> None:
    with pytest.raises(BrowserError):
        await SiteBrowser(browser).open("http://127.0.0.1:9/")
