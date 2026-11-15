"""Settings: frozen dataclasses holding the spec's defaults, overridable from a TOML file.

Secrets are not settings: they live only in the OS keyring (see secrets.py).
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, get_type_hints
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# src layout, run from the checkout (uv installs the project in editable mode):
# <project>/src/tg_agent/config.py -> <project>. Anchoring here, not on the working
# directory, keeps runtime data in one place if a command is run from elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


class ConfigError(ValueError):
    """The settings file is malformed: unknown key, wrong type or impossible value."""


@dataclass(frozen=True)
class Owner:
    full_name: str = "Саид Ахмедов"
    short_name: str = "Саид"
    # The short name in the genitive case («помощник Саида»): Russian needs it in the agent's
    # rules, and code cannot derive it from the name.
    short_name_gen: str = "Саида"
    city: str = ""
    # Telegram user ids the control bot obeys. Empty means nobody can control it.
    control_user_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class Limits:
    # Spend caps, USD.
    daily_usd: float = 5.0
    monthly_usd: float = 40.0
    # One task, until the owner presses «Продолжить» on its card (+ this much again).
    task_usd: float = 3.0
    # One wake of the task agent: model calls, and money (USD).
    wake_max_steps: int = 12
    wake_max_usd: float = 0.50
    # New counterparts per day: the ramp-up rate during the first `ramp_days` days
    # counted from `Settings.live_since`, the normal rate afterwards.
    ramp_new_contacts_per_day: int = 3
    ramp_days: int = 7
    new_contacts_per_day: int = 5
    # Waiting and pauses.
    followup_after_hours: int = 24
    max_followups: int = 2
    owner_pause_hours: int = 12  # silence after the owner writes by hand
    batch_window_seconds: int = 45  # wait to collect a burst of incoming messages
    batch_max_seconds: int = 180  # ... but never longer than this
    send_delay_min_seconds: int = 60  # human-like pause before answering (R70)
    send_delay_max_seconds: int = 180
    # What the model sees and what we keep.
    history_messages: int = 12
    raw_retention_days: int = 90
    # The owner's free texts that follow each other within this many seconds are read by
    # the secretary together, as one turn (ruling R71).
    owner_burst_seconds: int = 5
    # People one task may hold, everyone ever attached to it (closed ones too).
    task_max_people: int = 12
    # People one task may add on a counterpart's word without the owner (ruling R69).
    task_referrals: int = 3

    def __post_init__(self) -> None:
        if self.task_max_people < 1:
            raise ConfigError("limits.task_max_people must be at least 1")
        if self.task_referrals < 0:
            raise ConfigError("limits.task_referrals must be at least 0")


@dataclass(frozen=True)
class Schedule:
    tz: str = "Asia/Tashkent"
    work_start: time = time(9)  # the agent sends only inside [work_start, work_end)
    work_end: time = time(21)
    quiet_start: time = time(
        21
    )  # the owner is not disturbed from quiet_start to quiet_end
    quiet_end: time = time(6)
    digest_at: time = time(20, 30)
    # Round the clock, a first message to a new person still goes only from this time to
    # midnight: strangers are not written to at night (scenario 32, 2026-11-10).
    new_contacts_start: time = time(9)

    def __post_init__(self) -> None:
        try:
            ZoneInfo(self.tz)
        except (ZoneInfoNotFoundError, ValueError):
            raise ConfigError(f"unknown time zone {self.tz!r}") from None

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.tz)


@dataclass(frozen=True)
class Models:
    decide: str = "claude-sonnet-5-5"  # decisions in conversations
    intake: str = "claude-sonnet-5-5"  # the secretary: the owner's chat with the bot
    judge: str = "claude-haiku-5-5"  # checks the agent's outgoing drafts
    simulate: str = "claude-haiku-5-5"  # plays counterparts in evals
    search: str = (
        "claude-sonnet-5-5"  # finds vendors for a task (the only model with tools)
    )


@dataclass(frozen=True)
class Search:
    """The search for people the task's agent runs (its tool `research`)."""

    city: str = "Ташкент"  # where vendors are looked for
    max_candidates: int = 8
    max_cost_usd: float = 0.50  # one search stops once it has cost this much
    max_steps: int = 16  # model calls in one search
    max_minutes: float = (
        8.0  # one search ends after running this long (a slow site, OLX)
    )
    per_task: int = 3  # searches one task may run
    web_searches_per_step: int = 3
    web_fetches_per_step: int = 4
    telegram_searches: int = (
        4  # Telegram's public search, per search: it is the work account
    )
    olx: bool = True  # OLX listings, through the agent's own browser
    # Any site through the agent's own browser: read pages, press «Показать телефон», use a
    # site's search box. Read only.
    browse: bool = True
    # Sites to search every time, besides the ones a task names: ["prom.uz", "uzum.uz"].
    sites: tuple[str, ...] = ()
    max_pages: int = 20  # site pages one search may open, press or search on
    # The browser Playwright drives: msedge or chrome (installed), or chromium (Playwright's
    # own, for a server, which has neither).
    browser: str = "msedge"
    browser_headless: bool = True
    # The agent's browser profile, with the sign-ins the owner made from his phone
    # (`agent/site_logins`); relative to data_dir.
    browser_profile: Path = Path("browser") / "agent"
    # Presses of OLX's «Показать телефон» a day of the owner's time zone (ruling R51): over
    # it a listing is read without the press.
    olx_reveals_per_day: int = 30

    def __post_init__(self) -> None:
        if self.olx_reveals_per_day < 0:
            raise ConfigError("search.olx_reveals_per_day must be at least 0")
        if self.max_candidates < 1 or self.max_steps < 1 or self.max_cost_usd <= 0:
            raise ConfigError(
                "search: max_candidates, max_steps and max_cost_usd must be positive"
            )
        if self.max_pages < 1:
            raise ConfigError("max_pages must be at least 1")
        if self.max_minutes <= 0:
            raise ConfigError("search.max_minutes must be positive")
        if self.per_task < 1:
            raise ConfigError("search.per_task must be at least 1")
        if self.browser not in ("msedge", "chrome", "chromium"):
            raise ConfigError(
                f"search.browser: msedge, chrome or chromium, got {self.browser!r}"
            )
        sites = []
        for raw in self.sites:
            site = _site(raw)
            if site is None:
                raise ConfigError(f"search.sites: {raw!r} is not a site")
            sites.append(site)
        # Frozen: the one place the value is written, in its single spelling.
        object.__setattr__(self, "sites", tuple(dict.fromkeys(sites)))


TUNNELS = ("none", "cloudflared")


@dataclass(frozen=True)
class Remote:
    """The owner signs in to a site from his phone, in the bot's own browser (stage 4б).

    Off until he chooses a way in from the internet (ruling R50): `public_url` (a server with
    its own https address), or `tunnel` = "cloudflared" (Cloudflare's quick tunnel, the one
    tunnel (ruling R73); the program `cloudflared` must already be on the computer). The
    program never downloads anything.
    """

    tunnel: str = "none"
    public_url: str = ""
    port: int = 8765  # the page server on 127.0.0.1 that the way in leads to
    session_minutes: int = 10  # one sign-in's link lives this long
    # Where cloudflared is; relative to the project folder.
    cloudflared: Path = Path("tools") / "cloudflared.exe"

    def __post_init__(self) -> None:
        if self.tunnel not in TUNNELS:
            raise ConfigError(
                'remote.tunnel must be "cloudflared" (the Cloudflare quick tunnel) or'
                f' "none", got {self.tunnel!r}'
            )
        if not 1 <= self.port <= 65535:
            raise ConfigError("remote.port must be from 1 to 65535")
        if not 1 <= self.session_minutes <= 60:
            raise ConfigError("remote.session_minutes must be from 1 to 60")
        # The page encrypts with crypto.subtle, which browsers give only to https pages.
        if self.public_url and not re.match(
            r"^https://[^/\s?#]+", self.public_url.strip()
        ):
            raise ConfigError("remote.public_url must start with https://")


_SITE = re.compile(
    r"^(?:https?://)?(?:www\.)?((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,24})"
    r"(?::\d+)?(?:[/?#].*)?$",
    re.IGNORECASE,
)


def _site(raw: str) -> str | None:
    """A site as the owner may write it ("https://www.Prom.uz/ru") in one spelling: "prom.uz"."""
    match = _SITE.match(raw.strip())
    return match.group(1).lower() if match else None


@dataclass(frozen=True)
class Settings:
    owner: Owner = field(default_factory=Owner)
    limits: Limits = field(default_factory=Limits)
    schedule: Schedule = field(default_factory=Schedule)
    models: Models = field(default_factory=Models)
    search: Search = field(default_factory=Search)
    remote: Remote = field(default_factory=Remote)
    data_dir: Path = DEFAULT_DATA_DIR  # agent.db, logs/, files/, evals/
    live_since: date | None = None  # first day of real conversations
    mute_chats: bool = True  # the chats of the agent's people make no sound (R105)
    chat_folder: str = "ИИ агент"  # the folder they all join; "" = none (R106)


# Keys a settings file may still hold from an older example, per table: read and ignored, so
# that an owner's config.toml copied before they were removed still loads. (The settings only
# the old purchase engine read, removed in stage 3; the late replies' ones, stage 9.2.)
_RETIRED: dict[str, frozenset[str]] = {
    "owner": frozenset({"short_name_latn"}),
    "limits": frozenset(
        {
            "per_conversation_usd",
            "max_agent_messages",
            "offtopic_strikes_to_close",
            "no_response_after_hours",
            "context_tokens",
            "summary_max_chars",
            "late_reply_days",
        }
    ),
}


def load_settings(path: Path) -> Settings:
    """Read settings from a TOML file.

    Keys that are not in the file keep their defaults, and so does everything when the file does
    not exist. Unknown keys and values of the wrong type raise ConfigError (a typo in a spending
    limit must not pass silently); a retired key (`_RETIRED`) is ignored. A relative `data_dir`
    is resolved against the file's folder.
    """
    path = Path(path)
    try:
        # utf-8-sig: some Windows editors prepend a BOM, which tomllib itself would reject.
        raw = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raw = {}
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc

    settings = _build(Settings, raw, "")
    if not settings.data_dir.is_absolute():
        settings = replace(settings, data_dir=path.resolve().parent / settings.data_dir)
    return settings


# --- TOML -> dataclass ---------------------------------------------------------------------


def _build(cls: type, table: dict[str, Any], where: str) -> Any:
    label = where or "config"
    retired = _RETIRED.get(where, frozenset())
    table = {name: value for name, value in table.items() if name not in retired}
    unknown = sorted(set(table) - {f.name for f in fields(cls)})
    if unknown:
        raise ConfigError(f"{label}: unknown key(s): {', '.join(unknown)}")

    hints = get_type_hints(cls)
    kwargs = {
        name: _coerce(hints[name], value, f"{where}.{name}" if where else name)
        for name, value in table.items()
    }
    try:
        return cls(**kwargs)
    except ConfigError as exc:
        raise ConfigError(f"{label}: {exc}") from exc


def _coerce(tp: Any, value: Any, where: str) -> Any:
    if is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a table, got {value!r}")
        return _build(tp, value, where)
    try:
        return _PARSERS[tp](value)
    except ValueError as exc:
        raise ConfigError(f"{where}: {exc}, got {value!r}") from None


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("expected a string")
    return value


def _integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("expected a whole number")
    return value


def _boolean(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("expected true or false")  # noqa: TRY004 - _coerce turns it into ConfigError
    return value


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("expected a number")
    return float(value)


def _time_of_day(value: Any) -> time:
    message = 'expected a time of day like "09:00"'
    if isinstance(value, str):
        try:
            value = time.fromisoformat(value)
        except ValueError:
            raise ValueError(message) from None
    if not isinstance(value, time) or value.tzinfo is not None:
        raise ValueError(message)
    return value


def _path(value: Any) -> Path:
    if not isinstance(value, str):
        raise ValueError("expected a path string")
    return Path(value)


def _date(value: Any) -> date:
    message = "expected a date like 2026-11-10"
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            raise ValueError(message) from None
    if not isinstance(value, date) or isinstance(value, datetime):
        raise ValueError(message)
    return value


def _int_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        raise ValueError("expected a list of whole numbers")
    return tuple(value)


def _str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("expected a list of strings")
    return tuple(value)


_PARSERS: dict[Any, Callable[[Any], Any]] = {
    str: _text,
    bool: _boolean,
    int: _integer,
    float: _number,
    time: _time_of_day,
    Path: _path,
    date | None: _date,
    tuple[int, ...]: _int_tuple,
    tuple[str, ...]: _str_tuple,
}
