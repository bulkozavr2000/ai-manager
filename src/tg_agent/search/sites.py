"""Any site, read the way a person reads it, in the agent's own browser with Саид's sign-ins.

The vendor search may open a page, press a link or a button that only shows more («Показать
телефон», «Контакты», «Далее»), and type words into a site's search box. Nothing else: it never
buys, writes, sends a form, subscribes, or signs in or out, and it types nowhere but a search box.
What a page says is data for the model, never an instruction.

Which addresses the model may open is decided by the search (`search.agent`): what the task or an
earlier result showed, or any page of a site Саид named. Here: what is never opened at all —
anything but http(s), this computer, and the home network.
"""

from __future__ import annotations

import contextlib
import ipaddress
import re
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

from tg_agent.search.browser import AgentBrowser, BrowserError
from tg_agent.search.contacts import find_phones, normalize_phone, show

SITE_TAB = "site"
TEXT_MAX = 6000  # page text the model is shown
FULL_TEXT_MAX = 60_000  # page text read for phones, beyond what is shown
LINKS_MAX = 40
LINK_TEXT_MAX = 80
URL_MAX = 400
BUTTONS_MAX = 10
QUERY_MAX = 100

NOT_WEB = "Открываю только страницы в интернете: ссылки http:// и https://"
PRIVATE = "Это адрес этого компьютера или домашней сети: такие не открываю"
NO_PAGE = "Сначала открой страницу через open_page"

# Buttons that only show more of the page. Without \b: JavaScript's \b knows no Cyrillic, and the
# same pattern lists the buttons inside the page.
REVEAL = (
    "показа|телефон|номер|контакт|ещё|еще|далее|следующ|подробн|развернуть|поиск|найти"
    "|show|phone|number|contact|more|next|search|details"
    "|ko['‘ʻ`]?rsat|raqam|telefon|keyingi|batafsil|qidir"
)
_REVEAL = re.compile(REVEAL, re.IGNORECASE)

# What is never pressed: buying, paying, writing, sending, subscribing, signing in or out,
# deleting. Verbs, so a category such as «3D печать на заказ» stays a plain link.
_BLOCKED = (
    # Russian
    "купить",
    "купите",
    "корзин",
    "оплатить",
    "оплатите",
    "заказать",
    "закажите",
    "оформить",
    "оформите",
    "отправить",
    "отправьте",
    "написать",
    "напишите",
    "сообщени",
    "чат",
    "подпис",
    "отпис",
    "удали",
    "выйти",
    "выход",
    "войти",
    "вход",
    "регистр",
    "опубликов",
    "жалоб",
    "пожаловаться",
    "избранн",
    "бронир",
    "забронир",
    # English
    "buy",
    "purchase",
    "pay",
    "cart",
    "basket",
    "checkout",
    "check out",
    "order now",
    "place order",
    "my orders",
    "send",
    "message",
    "chat",
    "delete",
    "remove",
    "subscribe",
    "unsubscribe",
    "follow",
    "log ?out",
    "sign ?out",
    "log ?in",
    "sign ?in",
    "sign ?up",
    "register",
    r"post\b",
    "publish",
    "report",
    "favou?rite",
    "book now",
    "reserve",
    # Uzbek, Latin
    "sotib",
    "xarid",
    "savat",
    "to['‘ʻ`]?la(?:sh|ng)",
    "to['‘ʻ`]?lov",
    "buyurtma ber",
    "buyurtmalarim",
    "yubor",
    "xabar",
    "o['‘ʻ`]?chir",
    "chiqish",
    "kirish",
    "ro['‘ʻ`]?yxatdan",
    "obuna",
    "yoz(?:ish|ing)",
)
_BLOCK = re.compile(r"\b(?:" + "|".join(_BLOCKED) + ")", re.IGNORECASE)

_LOCAL_ENDINGS = (
    ".localhost",
    ".local",
    ".lan",
    ".home",
    ".internal",
    ".intranet",
    ".corp",
)
_NUMERIC_HOST = re.compile(
    r"^(?:0x[0-9a-f]+|\d+)(?:\.(?:0x[0-9a-f]+|\d+))*$", re.IGNORECASE
)
# A site written in a text: "prom.uz", "https://www.prom.uz/ru". Not inside an e-mail or a path.
_DOMAIN = re.compile(
    r"(?<![\w@.\-/])(?:https?://)?((?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,24})"
    r"(?![\w\-])",
    re.IGNORECASE,
)
# The endings a site of the owner's words may have: a dotted word with any other ending
# («Smart.Print», «PETG.Telegram», a file name like «model.stl») is not a site (review M6).
_SITE_ENDINGS = frozenset(
    {
        "uz", "ru", "com", "net", "org", "kz", "kg", "tj", "by", "ua", "io", "me", "info",
        "biz", "co", "app", "store", "shop", "site", "online", "pro", "tech",
    }
)  # fmt: skip
# Telegram's own addresses: a t.me link is a contact, not a site for the search to roam.
_TELEGRAM_SITES = ("t.me", "telegram.me", "telegram.org")
_URL = re.compile(r"https?://[^\s<>\"'«»()\[\]{}|\\^`]+", re.IGNORECASE)
_BARE_SITE = re.compile(r"^[a-z0-9.\-]+\.[a-z]{2,24}(?:[/:?#]|$)", re.IGNORECASE)

_TG_LINK = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:t|telegram)\.me/(?:s/)?([A-Za-z0-9_]{5,32})/?(?:[?#].*)?$",
    re.IGNORECASE,
)
_TG_APP = re.compile(
    r"^tg://resolve\?(?:.*&)?domain=([A-Za-z0-9_]{5,32})", re.IGNORECASE
)
_TG_NOT_NAMES = frozenset(
    {
        "share",
        "joinchat",
        "addstickers",
        "addemoji",
        "addtheme",
        "proxy",
        "socks",
        "setlanguage",
        "confirmphone",
        "login",
    }
)
_WHATSAPP = re.compile(
    r"(?:wa\.me/|whatsapp\.com/send/?\?(?:.*&)?phone=)\+?(\d{9,15})", re.IGNORECASE
)
# Phones in a page's text that are surely phones: with "+" or with Uzbekistan's 998. A bare
# "90 000 000" may be a price, so it stays in the text but is not listed as a phone.
_PLUS_PHONE = re.compile(r"\+\d[\d\s\-().]{7,20}\d")
_UZ_PHONE = re.compile(r"(?<![\d+])998[\d\s\-().]{8,14}\d")


@dataclass(frozen=True)
class Page:
    """A page as the search model is shown it."""

    url: str
    title: str
    text: str
    phones: list[str] = field(default_factory=list)  # "+<digits>"
    telegram: list[str] = field(default_factory=list)  # usernames, without "@"
    links: list[tuple[str, str]] = field(
        default_factory=list
    )  # (text, absolute address)
    buttons: list[str] = field(default_factory=list)  # «Показать телефон», «Далее», ...
    login_wall: bool = False
    captcha: bool = False


class SitePages(Protocol):
    async def open(self, url: str) -> Page: ...

    async def click(self, text: str) -> Page: ...

    async def search(self, query: str) -> Page: ...


# --- rules ----------------------------------------------------------------------------------


def url_problem(url: str, *, allow_private: bool = False) -> str | None:
    """Why this address is never opened, or None. Only http(s) on the internet."""
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").rstrip(".").lower()
    except ValueError:
        return NOT_WEB
    if parts.scheme.lower() not in ("http", "https") or not host:
        return NOT_WEB
    if allow_private:
        return None
    if host == "localhost" or host.endswith(_LOCAL_ENDINGS):
        return PRIVATE
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # "router", or 0x7f.0.0.1: what a browser would take for this computer or the network.
        return PRIVATE if "." not in host or _NUMERIC_HOST.match(host) else None
    return None if address.is_global else PRIVATE


def as_url(raw: str) -> str:
    """The address the model gave, with https:// added to a bare site name ("prom.uz/ru")."""
    url = raw.strip()
    return f"https://{url}" if _BARE_SITE.match(url) else url


def site_of(url: str) -> str:
    """The site of an address, as it is compared: "https://www.Prom.uz/ru" -> "prom.uz"."""
    try:
        host = (urlsplit(url).hostname or "").rstrip(".").lower()
    except ValueError:
        return ""
    return host.removeprefix("www.")


def on_site(site: str, sites: set[str] | frozenset[str]) -> bool:
    """Is `site` one of `sites` or a part of one ("m.prom.uz" of "prom.uz")?"""
    return bool(site) and any(site == s or site.endswith(f".{s}") for s in sites)


def domains_in(text: str) -> set[str]:
    """The sites a text names: "искать на prom.uz и https://www.texnomart.uz/ru" -> both.

    Not e-mail addresses, not file names like "корпус.stl", not a dotted word whose ending is
    no known top-level domain, and never Telegram's own addresses.
    """
    found = set()
    for match in _DOMAIN.finditer(text):
        host = match.group(1).lower().removeprefix("www.")
        if host.rsplit(".", 1)[1] not in _SITE_ENDINGS:
            continue
        if on_site(host, frozenset(_TELEGRAM_SITES)):
            continue
        found.add(host)
    return found


def url_key(url: str) -> str:
    """An address as two spellings of the same page compare: no scheme, no "www.", no
    trailing "/", no fragment, %-escapes undone."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        return url.strip()
    host = (parts.hostname or "").rstrip(".").lower().removeprefix("www.")
    if port and port not in (80, 443):
        host += f":{port}"
    query = f"?{unquote(parts.query)}" if parts.query else ""
    return host + unquote(parts.path).rstrip("/") + query


def urls_in(text: str) -> set[str]:
    """The addresses written in a text, as `url_key` spells them."""
    return {url_key(m.group(0).rstrip(".,;:!?…")) for m in _URL.finditer(text)}


def click_refusal(text: str) -> str | None:
    """Why a link or button with this text is never pressed, or None."""
    if _BLOCK.search(text):
        return (
            f"«{text}» не нажимаю: поиск только читает — не покупает, не пишет, не отправляет"
            " и не входит в аккаунты"
        )
    return None


def query_refusal(query: str) -> str | None:
    """Why these words never go into a site's search box, or None: only search words go there."""
    words = " ".join(query.split())
    if not words:
        return "Нужны слова для поиска"
    if len(words) > QUERY_MAX:
        return f"В строку поиска — до {QUERY_MAX} знаков"
    if find_phones(words) or re.search(r"\d{9,}", words):
        return "В строку поиска не вписываю номера телефонов"
    if "@" in words:
        return "В строку поиска не вписываю адреса почты и @имена"
    if re.search(r"https?://|www\.", words, re.IGNORECASE):
        return "В строку поиска не вписываю ссылки"
    return None


# --- what a page becomes --------------------------------------------------------------------


def _telegram(href: str) -> str | None:
    match = _TG_LINK.match(href) or _TG_APP.match(href)
    if match is None:
        return None
    name = match.group(1)
    if name.lower() in _TG_NOT_NAMES or not re.search("[A-Za-z]", name):
        return None
    return name


def _link_phone(href: str) -> str | None:
    if href.lower().startswith("tel:"):
        return normalize_phone(unquote(href[4:]))
    match = _WHATSAPP.search(href)
    return normalize_phone(f"+{match.group(1)}") if match else None


def _text_phones(text: str) -> list[str]:
    found = []
    for pattern in (_PLUS_PHONE, _UZ_PHONE):
        for match in pattern.finditer(text):
            phone = normalize_phone(match.group(0))
            if phone:
                found.append(phone)
    return found


def _tidy(text: str) -> str:
    """Lines without runs of spaces, empty lines and repeats; capped for the model."""
    lines: list[str] = []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if line and (not lines or lines[-1] != line):
            lines.append(line)
    joined = "\n".join(lines)
    return joined if len(joined) <= TEXT_MAX else joined[:TEXT_MAX].rstrip() + " …"


def page_from(data: dict[str, Any]) -> Page:
    """A page read in the browser (`_READ_JS`) as the model is shown it."""
    full = str(data.get("text") or "")
    contacts = [str(href) for href in data.get("contacts") or []]
    phones = [p for p in (_link_phone(h) for h in contacts) if p] + _text_phones(full)
    telegram = [name for name in (_telegram(h) for h in contacts) if name]
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for text, url in data.get("links") or []:
        label, address = " ".join(str(text).split())[:LINK_TEXT_MAX], str(url)
        if label and len(address) <= URL_MAX and address not in seen:
            seen.add(address)
            links.append((label, address))
        if len(links) >= LINKS_MAX:
            break
    buttons = [" ".join(str(b).split())[:60] for b in data.get("buttons") or []]
    return Page(
        url=str(data.get("url") or ""),
        title=" ".join(str(data.get("title") or "").split())[:200],
        text=_tidy(full),
        phones=list(dict.fromkeys(phones)),
        telegram=list(dict.fromkeys(telegram)),
        links=links,
        buttons=list(dict.fromkeys(b for b in buttons if b))[:BUTTONS_MAX],
        login_wall=bool(data.get("login")),
        captcha=bool(data.get("captcha")),
    )


def describe(page: Page) -> str:
    """The page as the search model reads it in a tool result."""
    lines = [f"Страница: {page.title or 'без названия'}", f"Адрес: {page.url}"]
    if page.captcha:
        lines.append(
            "⚠ Сайт просит капчу. Не проходи её: программа сама сообщит, что сайт просит"
            " капчу. Ищи в другом месте."
        )
    if page.login_wall:
        lines.append(
            "⚠ Сайт просит вход. Не входи и ничего не вписывай: программа сама сообщит, что"
            " сайт просит вход. Ищи в другом месте."
        )
    if page.phones:
        lines.append("Телефоны: " + ", ".join(show(phone) for phone in page.phones))
    if page.telegram:
        lines.append("Telegram: " + ", ".join(f"@{name}" for name in page.telegram))
    if page.buttons:
        lines.append("Кнопки: " + ", ".join(f"«{button}»" for button in page.buttons))
    lines += ["Текст:", page.text or "(пусто)"]
    if page.links:
        lines.append("Ссылки:")
        lines += [f"- {text} — {url}" for text, url in page.links]
    return "\n".join(lines)


# --- the browser ----------------------------------------------------------------------------

_VISIBLE_JS = (
    "const visible = el => el.getClientRects().length > 0"
    " && getComputedStyle(el).visibility !== 'hidden';"
    " const clean = s => (s || '').replace(/\\s+/g, ' ').trim();"
)

_READ_JS = (
    "(opts) => {"
    + _VISIBLE_JS
    + """
  const text = document.body ? document.body.innerText : '';
  const contactHost = /^(?:www\\.)?(?:t\\.me|telegram\\.me|wa\\.me|api\\.whatsapp\\.com|web\\.whatsapp\\.com|whatsapp\\.com)$/i;
  const contacts = [];
  const links = [];
  const seen = new Set();
  const here = location.href.split('#')[0];
  for (const a of document.querySelectorAll('a[href]')) {
    if (!visible(a)) continue;
    let u;
    try { u = new URL(a.href); } catch (e) { continue; }
    if (u.protocol === 'tel:' || u.protocol === 'tg:' || contactHost.test(u.hostname)) contacts.push(a.href);
    if (!/^https?:$/.test(u.protocol) || links.length >= opts.links) continue;
    const href = u.href.split('#')[0];
    const label = clean(a.innerText || a.getAttribute('aria-label') || a.title);
    if (href === here || seen.has(href) || !label) continue;
    seen.add(href);
    links.push([label, href]);
  }
  const reveal = new RegExp(opts.reveal, 'iu');
  const buttons = [];
  for (const el of document.querySelectorAll('button, [role=button], a, input[type=button], input[type=submit], summary')) {
    if (buttons.length >= opts.buttons) break;
    if (!visible(el)) continue;
    const label = clean(el.innerText || el.value || el.getAttribute('aria-label') || el.title);
    if (label && label.length <= 60 && reveal.test(label) && !buttons.includes(label)) buttons.push(label);
  }
  const size = clean(text).length;
  const passwords = [...document.querySelectorAll('input[type=password]')].filter(visible);
  const password = passwords.length > 0;
  // A sign-in window over the page, as OLX opens one when a phone needs a login.
  const modal = passwords.some(el => el.closest('[role=dialog], [aria-modal="true"], dialog'));
  const loginUrl = /(^|[\\/?#=&_.-])(login|signin|sign-in|sign_in|log-in|auth|authorize|kirish|vhod|voiti)([\\/?#=&_.-]|$)/i
    .test(location.pathname + location.search);
  const frames = [...document.querySelectorAll('iframe')].map(f => f.src || '');
  const robot = /не робот|captcha|robot emas|verify you are human|подтвердите, что вы человек|проверка браузера|checking your browser|just a moment/i;
  const captcha = /just a moment|checking your browser|attention required/i.test(document.title)
    || (size < 800 && (frames.some(s => /captcha|challenge|turnstile/i.test(s)) || robot.test(text.slice(0, 3000))));
  return {
    url: location.href, title: document.title, text: text.slice(0, opts.full),
    contacts, links, buttons, login: loginUrl || modal || (password && size < 1500), captcha,
  };
}"""
)

# The element to press for a text: an exact label first, then one that starts with it, then one
# that holds it; the shortest wins. It is marked for the click and described for the checks.
_TARGET_JS = (
    "(want) => {"
    + _VISIBLE_JS
    + """
  const label = el => clean(el.innerText || el.value || el.getAttribute('aria-label') || el.title);
  const w = clean(want).toLowerCase();
  const all = [...document.querySelectorAll('a[href], button, [role=button], [role=link], input[type=button], input[type=submit], summary')]
    .filter(visible).map(el => [el, label(el).toLowerCase()]).filter(([, l]) => l);
  const pick = test => all.filter(([, l]) => test(l)).sort((a, b) => a[1].length - b[1].length)[0];
  const hit = pick(l => l === w) || pick(l => l.startsWith(w)) || pick(l => l.includes(w));
  document.querySelectorAll('[data-agent-target]').forEach(e => e.removeAttribute('data-agent-target'));
  if (!hit) return null;
  const el = hit[0];
  const tag = el.tagName.toLowerCase();
  const form = el.closest('form');
  const type = (el.getAttribute('type') || (tag === 'button' ? 'submit' : '')).toLowerCase();
  const submit = !!form && (tag === 'button' || tag === 'input') && type === 'submit';
  const fields = form ? [...form.querySelectorAll('input, textarea, select')] : [];
  const searchy = f => (f.type || '').toLowerCase() === 'search' || f.getAttribute('role') === 'searchbox'
    || /^(q|query|search|s|k|keyword|keywords|text_search|search_query|searchtext|search_text)$/i.test(f.name || '')
    || /search|поиск|найти|qidir/i.test([f.placeholder, f.getAttribute('aria-label'), f.title].join(' '));
  const risky = f => f.tagName.toLowerCase() === 'textarea'
    || ['password', 'email', 'tel', 'file'].includes((f.type || '').toLowerCase());
  const searchForm = !!form && (form.getAttribute('role') === 'search' || fields.some(searchy)) && !fields.some(risky);
  el.setAttribute('data-agent-target', '1');
  if (tag === 'a') el.removeAttribute('target');
  let link = '';
  if (tag === 'a') { try { const u = new URL(el.href); if (/^https?:$/.test(u.protocol)) link = u.href; } catch (e) {} }
  return {label: label(el), link, submit, searchForm};
}"""
)

# A site's search box: a search field, or a text field named or labelled as one. Never a
# password, e-mail, phone or message field.
_SEARCH_BOX_JS = (
    "() => {"
    + _VISIBLE_JS
    + """
  document.querySelectorAll('[data-agent-search]').forEach(e => e.removeAttribute('data-agent-search'));
  const named = /^(q|query|search|s|k|keyword|keywords|text_search|search_query|searchtext|search_text)$/i;
  const words = /search|поиск|найти|qidir|izla/i;
  const box = [...document.querySelectorAll('input')].filter(visible).find(el => {
    if (el.disabled || el.readOnly) return false;
    const type = (el.getAttribute('type') || 'text').toLowerCase();
    if (type === 'search' || el.getAttribute('role') === 'searchbox') return true;
    if (type !== 'text') return false;
    const hint = [el.placeholder, el.getAttribute('aria-label'), el.title].join(' ');
    return named.test(el.name || '') || named.test(el.id || '')
      || words.test(hint) || words.test(el.name || '') || words.test(el.id || '');
  });
  if (!box) return false;
  box.setAttribute('data-agent-search', '1');
  return true;
}"""
)


# Resolves once the page has not changed for `quiet` ms, or after `most` ms at the latest.
_QUIET_JS = """([quiet, most]) => new Promise(resolve => {
  let timer = null;
  const observer = new MutationObserver(() => { clearTimeout(timer); timer = setTimeout(done, quiet); });
  function done() { observer.disconnect(); resolve(true); }
  observer.observe(document, {subtree: true, childList: true, characterData: true});
  timer = setTimeout(done, quiet);
  setTimeout(done, most);
})"""


async def _settle(page: Any) -> None:
    """Let what a navigation or a click started finish, within a few seconds: the page loaded,
    the network quiet, the page drawn (OLX draws its contact buttons seconds after loading)."""
    await page.wait_for_timeout(300)
    with contextlib.suppress(Exception):
        await page.wait_for_load_state("domcontentloaded", timeout=10_000)
    with contextlib.suppress(Exception):  # some sites never fall quiet
        await page.wait_for_load_state("networkidle", timeout=3_000)
    with contextlib.suppress(Exception):  # a page that went away meanwhile
        await page.evaluate(_QUIET_JS, [1_200, 4_000])


class SiteBrowser:
    """SitePages in the agent's browser, in a tab of its own (OLX has another)."""

    def __init__(self, browser: AgentBrowser, *, allow_private: bool = False) -> None:
        self._browser = browser
        self._allow_private = allow_private  # tests serve pages from this computer

    async def open(self, url: str) -> Page:
        problem = url_problem(url, allow_private=self._allow_private)
        if problem:
            raise BrowserError(problem)
        page = await self._browser.tab(SITE_TAB)
        try:
            await page.goto(url, wait_until="domcontentloaded")
        except Exception as exc:  # noqa: BLE001 - timeouts, a dropped network, a bad address
            raise BrowserError(f"страница не открылась: {type(exc).__name__}") from None
        await _settle(page)
        return await self._read(page)

    async def click(self, text: str) -> Page:
        want = " ".join(text.split())
        refusal = click_refusal(want) if want else "Нужен текст ссылки или кнопки"
        if refusal:
            raise BrowserError(refusal)
        page = await self._current()
        target = await page.evaluate(_TARGET_JS, want)
        if not target:
            raise BrowserError(
                f"На странице нет «{want}»: возьми текст из «Кнопки» или «Ссылки»"
            )
        label = target["label"]
        refusal = click_refusal(label)
        if refusal:
            raise BrowserError(refusal)
        if target["submit"] and not target["searchForm"]:
            raise BrowserError(f"«{label}» отправляет форму: такое не нажимаю")
        if target["link"]:
            problem = url_problem(target["link"], allow_private=self._allow_private)
            if problem:
                raise BrowserError(problem)
        elif not _REVEAL.search(label):
            raise BrowserError(
                f"«{label}» — кнопка, а нажимаю только ссылки и кнопки, которые показывают"
                " больше: «Показать телефон», «Контакты», «Далее»"
            )
        try:
            await page.locator('[data-agent-target="1"]').first.click()
        except Exception as exc:  # noqa: BLE001 - covered, gone, not clickable
            raise BrowserError(f"«{label}» не нажалась: {type(exc).__name__}") from None
        await _settle(page)
        return await self._read(page)

    async def search(self, query: str) -> Page:
        refusal = query_refusal(query)
        if refusal:
            raise BrowserError(refusal)
        page = await self._current()
        if not await page.evaluate(_SEARCH_BOX_JS):
            raise BrowserError(
                "На этой странице нет строки поиска: открой главную или страницу поиска сайта"
            )
        box = page.locator('[data-agent-search="1"]').first
        try:
            await box.fill(" ".join(query.split()))
            await box.press("Enter")
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(
                f"строка поиска не сработала: {type(exc).__name__}"
            ) from None
        await _settle(page)
        return await self._read(page)

    async def _current(self) -> Any:
        page = await self._browser.tab(SITE_TAB)
        if not str(page.url).startswith(("http://", "https://")):
            raise BrowserError(NO_PAGE)
        return page

    async def _read(self, page: Any) -> Page:
        options = {
            "reveal": REVEAL,
            "links": LINKS_MAX,
            "buttons": BUTTONS_MAX,
            "full": FULL_TEXT_MAX,
        }
        try:
            data = await page.evaluate(_READ_JS, options)
        except Exception:  # noqa: BLE001 - the page moved on while it was read: once more
            await _settle(page)
            try:
                data = await page.evaluate(_READ_JS, options)
            except Exception as exc:  # noqa: BLE001
                raise BrowserError(
                    f"страница не прочиталась: {type(exc).__name__}"
                ) from None
        shown = page_from(data)
        if url_problem(shown.url, allow_private=self._allow_private):
            raise BrowserError(
                "страница увела на адрес этого компьютера или домашней сети"
            )
        return shown
