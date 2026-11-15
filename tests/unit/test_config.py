from __future__ import annotations

import dataclasses
import importlib.util
import re
from datetime import date, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tg_agent.config import (
    DEFAULT_DATA_DIR,
    TUNNELS,
    ConfigError,
    Models,
    Owner,
    Remote,
    Schedule,
    Settings,
    load_settings,
)
from tg_agent.search import tunnel as tunnel_module
from tg_agent.search.tunnel import CloudflaredTunnel, PublicUrl, tunnel_from

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


# --- defaults -----------------------------------------------------------------------------


def test_defaults_match_spec():
    settings = Settings()

    assert settings.limits.daily_usd == 5.0
    assert settings.limits.monthly_usd == 40.0
    assert settings.limits.task_usd == 3.0
    assert settings.schedule.work_start == time(9)
    assert settings.schedule.digest_at == time(20, 30)
    assert settings.models.decide == "claude-sonnet-5-5"
    assert settings.models.judge == "claude-haiku-5-5"
    # No signature: the agent says it is an AI in its first reply (tasks.templates.INTRO).
    assert not hasattr(settings, "signature")


def test_defaults_cover_the_rest_of_the_global_constraints():
    settings = Settings()

    assert settings.owner == Owner(
        full_name="Саид Ахмедов", short_name="Саид", city="", control_user_ids=()
    )
    assert settings.live_since is None

    schedule = settings.schedule
    assert schedule.tz == "Asia/Tashkent"
    assert schedule.zone == ZoneInfo("Asia/Tashkent")
    assert (schedule.work_start, schedule.work_end) == (time(9), time(21))
    assert (schedule.quiet_start, schedule.quiet_end) == (time(21), time(6))
    assert schedule.digest_at == time(20, 30)

    limits = settings.limits
    assert limits.ramp_new_contacts_per_day == 3
    assert limits.ramp_days == 7
    assert limits.new_contacts_per_day == 5
    assert limits.followup_after_hours == 24
    assert limits.max_followups == 2
    assert limits.owner_pause_hours == 12
    assert limits.batch_window_seconds == 45
    assert limits.batch_max_seconds == 180
    assert limits.send_delay_min_seconds == 60
    assert limits.send_delay_max_seconds == 180
    assert limits.history_messages == 12
    assert limits.raw_retention_days == 90

    models = settings.models
    assert models.intake == "claude-sonnet-5-5"
    assert models.simulate == "claude-haiku-5-5"


def test_settings_are_frozen():
    settings = Settings()

    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.live_since = None  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.limits.daily_usd = 99.0  # type: ignore[misc]


def test_default_data_dir_is_the_project_data_folder():
    assert Settings().data_dir == DEFAULT_DATA_DIR
    assert DEFAULT_DATA_DIR == PROJECT_ROOT / "data"
    assert (DEFAULT_DATA_DIR.parent / "pyproject.toml").is_file()


def test_settings_fixture_is_isolated_and_owned_by_111(settings, tmp_path):
    assert settings.owner.control_user_ids == (111,)
    assert settings.data_dir == tmp_path


def test_schedule_rejects_unknown_time_zone():
    with pytest.raises(ConfigError, match="unknown time zone"):
        Schedule(tz="Mars/Olympus")


# --- loading ------------------------------------------------------------------------------


def test_load_settings_overrides_from_toml(tmp_path):
    path = write_config(tmp_path, "[owner]\ncontrol_user_ids = [5]\n\n[limits]\ndaily_usd = 1.5\n")

    loaded = load_settings(path)

    assert loaded.owner.control_user_ids == (5,)
    assert loaded.limits.daily_usd == 1.5
    # everything the file does not mention keeps its default
    assert loaded.limits.monthly_usd == 40.0
    assert loaded.owner.full_name == "Саид Ахмедов"
    assert loaded.schedule == Schedule()
    assert loaded.models == Models()
    assert loaded.live_since == Settings().live_since
    assert loaded.data_dir == DEFAULT_DATA_DIR


def test_load_settings_without_a_file_gives_defaults(tmp_path):
    assert load_settings(tmp_path / "absent.toml") == Settings()


def test_load_settings_with_an_empty_file_gives_defaults(tmp_path):
    assert load_settings(write_config(tmp_path, "")) == Settings()


def test_load_settings_reads_utf8_values(tmp_path):
    path = write_config(
        tmp_path,
        '[owner]\nfull_name = "Салим Рахимов"\nshort_name = "Салим"\ncity = "Ташкент"\n',
    )

    loaded = load_settings(path)

    assert loaded.owner.short_name == "Салим"
    assert loaded.owner.full_name == "Салим Рахимов"
    assert loaded.owner.city == "Ташкент"


def test_load_settings_tolerates_a_utf8_bom(tmp_path):
    # Some Windows editors prepend a BOM; tomllib alone would reject the file.
    path = tmp_path / "config.toml"
    path.write_bytes(b"\xef\xbb\xbf" + '[owner]\ncity = "Ташкент"\n'.encode("utf-8"))

    assert load_settings(path).owner.city == "Ташкент"


def test_load_settings_rejects_a_file_that_is_not_utf8(tmp_path):
    path = tmp_path / "config.toml"
    path.write_bytes('[owner]\ncity = "Ташкент"\n'.encode("cp1251"))

    with pytest.raises(ConfigError, match="config.toml"):
        load_settings(path)


def test_load_settings_parses_times_and_dates(tmp_path):
    path = write_config(
        tmp_path,
        'live_since = 2026-11-10\n\n[schedule]\nwork_start = "08:30"\ndigest_at = 19:45:00\n',
    )

    loaded = load_settings(path)

    assert loaded.live_since == date(2026, 11, 10)
    assert loaded.schedule.work_start == time(8, 30)
    assert loaded.schedule.digest_at == time(19, 45)
    assert loaded.schedule.work_end == time(21)


def test_load_settings_accepts_live_since_as_a_string(tmp_path):
    loaded = load_settings(write_config(tmp_path, 'live_since = "2026-11-10"\n'))

    assert loaded.live_since == date(2026, 11, 10)


def test_load_settings_turns_whole_money_numbers_into_floats(tmp_path):
    loaded = load_settings(write_config(tmp_path, "[limits]\nmonthly_usd = 40\n"))

    assert loaded.limits.monthly_usd == 40.0
    assert isinstance(loaded.limits.monthly_usd, float)


def test_relative_data_dir_is_resolved_against_the_config_folder(tmp_path):
    loaded = load_settings(write_config(tmp_path, 'data_dir = "var/agent"\n'))

    assert loaded.data_dir == tmp_path.resolve() / "var" / "agent"


def test_absolute_data_dir_is_kept(tmp_path):
    target = tmp_path / "elsewhere"

    loaded = load_settings(write_config(tmp_path, f"data_dir = '{target.as_posix()}'\n"))

    assert loaded.data_dir == target


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[limits]\ndaly_usd = 1.0\n", "limits: unknown key(s): daly_usd"),
        ("[limitz]\ndaily_usd = 1.0\n", "unknown key(s): limitz"),
        ("sinature = 'x'\n", "unknown key(s): sinature"),
    ],
)
def test_load_settings_rejects_unknown_keys(tmp_path, text, message):
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_settings(write_config(tmp_path, text))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('[limits]\ndaily_usd = "cheap"\n', "limits.daily_usd"),
        ("[limits]\nmax_followups = true\n", "limits.max_followups"),
        ("[limits]\nmax_followups = 2.5\n", "limits.max_followups"),
        ('[owner]\ncontrol_user_ids = "5"\n', "owner.control_user_ids"),
        ('[owner]\ncontrol_user_ids = [1, "x"]\n', "owner.control_user_ids"),
        ("[owner]\ncity = 5\n", "owner.city"),
        ('[schedule]\nwork_start = "9am"\n', "schedule.work_start"),
        ('[schedule]\nwork_start = "09:00+05:00"\n', "schedule.work_start"),
        ("[schedule]\nwork_start = 9\n", "schedule.work_start"),
        ('[schedule]\ntz = "Mars/Olympus"\n', "schedule: unknown time zone"),
        ('live_since = "soon"\n', "live_since"),
        ("live_since = 2026-11-10T09:00:00\n", "live_since"),
        ("data_dir = 5\n", "data_dir"),
        ("limits = 5\n", "limits: expected a table"),
        ("[limits\ndaily_usd = 1\n", "config.toml"),
    ],
)
def test_load_settings_rejects_bad_values(tmp_path, text, message):
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_settings(write_config(tmp_path, text))


def test_example_config_documents_exactly_the_defaults():
    # config.example.toml spells out every default; if a default changes, the example must follow.
    assert load_settings(PROJECT_ROOT / "config.example.toml") == Settings()


def test_every_command_the_example_names_exists():
    # L27: the cleanup deleted `tg_agent.search.login`; the example must not send him there.
    example = (PROJECT_ROOT / "config.example.toml").read_text(encoding="utf-8")
    for module in re.findall(r"python -m (tg_agent[\w.]*)", example):
        assert importlib.util.find_spec(module) is not None, module


# The settings only the old purchase engine read (removed in stage 3).
RETIRED_LIMITS = (
    "per_conversation_usd",
    "max_agent_messages",
    "offtopic_strikes_to_close",
    "no_response_after_hours",
    "context_tokens",
    "summary_max_chars",
)


def test_the_old_engine_s_settings_are_gone_and_the_example_lists_none():
    limits = {field.name for field in dataclasses.fields(Settings().limits)}
    assert not limits & set(RETIRED_LIMITS)
    example = (PROJECT_ROOT / "config.example.toml").read_text(encoding="utf-8")
    for key in RETIRED_LIMITS:
        assert not re.search(rf"^\s*#?\s*{key}\b", example, re.MULTILINE), key


def test_a_retired_key_in_a_config_file_does_not_stop_the_load(tmp_path):
    # The owner's config.toml was copied from an older example: its old keys are ignored.
    retired = "".join(f"{key} = 1\n" for key in RETIRED_LIMITS)
    path = write_config(tmp_path, f"[limits]\n{retired}daily_usd = 1.5\n")

    loaded = load_settings(path)

    assert loaded.limits == dataclasses.replace(Settings().limits, daily_usd=1.5)


def test_the_people_caps_have_their_defaults_and_the_example_lists_them():
    # Stage 4 (R27, R29): searches one task may run, people one task may hold.
    settings = Settings()

    assert (settings.search.per_task, settings.limits.task_max_people) == (3, 12)
    example = (PROJECT_ROOT / "config.example.toml").read_text(encoding="utf-8")
    assert re.search(r"^per_task = 3$", example, re.MULTILINE)
    assert re.search(r"^task_max_people = 12$", example, re.MULTILINE)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[search]\nper_task = 0\n", "per_task"),
        ("[limits]\ntask_max_people = 0\n", "task_max_people"),
    ],
)
def test_the_people_caps_are_at_least_one(tmp_path, text, message):
    with pytest.raises(ConfigError, match=message):
        load_settings(write_config(tmp_path, text))


def test_the_cap_of_referrals_has_its_default_and_the_example_lists_it(tmp_path):
    # Fix pass G (ruling R69): people a task adds on a counterpart's word, without the owner.
    assert Settings().limits.task_referrals == 3
    example = (PROJECT_ROOT / "config.example.toml").read_text(encoding="utf-8")
    assert re.search(r"^task_referrals = 3$", example, re.MULTILINE)
    loaded = load_settings(write_config(tmp_path, "[limits]\ntask_referrals = 0\n"))
    assert loaded.limits.task_referrals == 0
    with pytest.raises(ConfigError, match="task_referrals"):
        load_settings(write_config(tmp_path, "[limits]\ntask_referrals = -1\n"))


def test_the_remote_settings_have_their_defaults_and_the_example_lists_them():
    # Stage 4б (R50, R51): the way in from the internet is off until the owner chooses one.
    settings = Settings()

    assert settings.remote == Remote(
        tunnel="none",
        public_url="",
        port=8765,
        session_minutes=10,
        cloudflared=Path("tools") / "cloudflared.exe",
    )
    assert settings.search.olx_reveals_per_day == 30
    example = (PROJECT_ROOT / "config.example.toml").read_text(encoding="utf-8")
    for line in (
        r'^tunnel = "none"$',
        r'^public_url = ""$',
        r"^port = 8765$",
        r"^session_minutes = 10$",
        r'^cloudflared = "tools/cloudflared.exe"$',
        r"^olx_reveals_per_day = 30$",
    ):
        assert re.search(line, example, re.MULTILINE), line


def test_the_remote_settings_are_read(tmp_path):
    loaded = load_settings(
        write_config(
            tmp_path,
            '[remote]\ntunnel = "cloudflared"\nport = 9000\nsession_minutes = 5\n'
            '[search]\nolx_reveals_per_day = 0\n',
        )
    )
    assert (loaded.remote.tunnel, loaded.remote.port) == ("cloudflared", 9000)
    assert loaded.remote.session_minutes == 5
    assert loaded.search.olx_reveals_per_day == 0


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('[remote]\ntunnel = "ngrok"\n', "tunnel"),
        # The ssh way is gone (R73): an old config says what to write instead of loading
        # with the feature silently off.
        ('[remote]\ntunnel = "ssh"\n', '"cloudflared"'),
        ("[remote]\nport = 0\n", "port"),
        ("[remote]\nport = 70000\n", "port"),
        ("[remote]\nsession_minutes = 0\n", "session_minutes"),
        ("[remote]\nsession_minutes = 61\n", "session_minutes"),
        ('[remote]\npublic_url = "http://agent.example"\n', "public_url"),
        ('[remote]\npublic_url = "agent.example"\n', "public_url"),
        ("[search]\nolx_reveals_per_day = -1\n", "olx_reveals_per_day"),
    ],
)
def test_the_remote_settings_are_checked(tmp_path, text, message):
    with pytest.raises(ConfigError, match=message):
        load_settings(write_config(tmp_path, text))


def test_tunnel_from_gives_the_way_in_the_owner_chose(tmp_path):
    def remote(**values) -> Settings:
        return dataclasses.replace(Settings(), remote=Remote(**values))

    assert tunnel_from(Settings()) is None  # the feature is off

    public = tunnel_from(remote(public_url="https://agent.example/"))
    assert isinstance(public, PublicUrl)
    # A server with its own address needs no tunnel, whatever else is chosen.
    both = tunnel_from(remote(tunnel="cloudflared", public_url="https://a.example"))
    assert isinstance(both, PublicUrl)

    cloud = tunnel_from(remote(tunnel="cloudflared"))
    assert isinstance(cloud, CloudflaredTunnel)
    assert cloud._program == str(PROJECT_ROOT / "tools" / "cloudflared.exe")
    elsewhere = tmp_path / "cf.exe"
    cloud = tunnel_from(remote(tunnel="cloudflared", cloudflared=elsewhere))
    assert cloud._program == str(elsewhere)


def test_cloudflared_is_the_one_tunnel():
    # R73: Cloudflare's quick tunnel is the way in; the ssh way is gone from the code.
    assert TUNNELS == ("none", "cloudflared")
    assert {"SshTunnel", "find_ssh"}.isdisjoint(vars(tunnel_module))


def test_the_wake_caps_have_their_defaults_and_the_example_lists_them():
    limits = Settings().limits

    assert (limits.wake_max_steps, limits.wake_max_usd) == (12, 0.50)
    example = (PROJECT_ROOT / "config.example.toml").read_text(encoding="utf-8")
    assert re.search(r"^wake_max_steps = 12$", example, re.MULTILINE)
    assert re.search(r"^wake_max_usd = 0\.50?$", example, re.MULTILINE)


def test_an_old_config_keeps_its_own_money_limits(tmp_path):
    # Task 6.1: the owner's config.toml with his own numbers keeps them.
    path = write_config(tmp_path, "[limits]\ndaily_usd = 2.0\nmonthly_usd = 30.0\n")

    loaded = load_settings(path)

    assert (loaded.limits.daily_usd, loaded.limits.monthly_usd) == (2.0, 30.0)
    assert loaded.limits.task_usd == 3.0
