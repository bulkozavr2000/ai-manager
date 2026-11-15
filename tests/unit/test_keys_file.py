"""Keys on a server (ruling R84): where the system has no key store, `keys.toml` in the data folder.

The key store stays the first place; the file is read only for what the store does not give.
Every value here is made up; no test touches the real key store or the real data folder.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

import keyring
import pytest
from keyring.backends import fail

from tg_agent import secrets
from tg_agent.secrets import (
    Key,
    MissingSecret,
    get_secret,
    keys_file,
    require_secret,
    set_secret,
    use_data_dir,
    write_keys_file,
)
from tg_agent.setup.__main__ import ask_secrets, export_keys

FAKE = {
    Key.ANTHROPIC_API_KEY: "sk-ant-made-up-1",
    Key.BOT_TOKEN: "7000000001:made-up-bot",
    Key.TG_API_ID: "424242",
    Key.TG_API_HASH: "made-up-hash-0f0f",
    Key.TG_SESSION: 'made-up-session "quoted" \\ back',
}

POSIX = pytest.mark.skipif(os.name == "nt", reason="file modes do not apply on Windows")


@pytest.fixture
def no_key_store():
    """A system with no key store at all, as in a Linux container."""
    previous = keyring.get_keyring()
    keyring.set_keyring(fail.Keyring())
    try:
        yield
    finally:
        keyring.set_keyring(previous)


def test_the_keys_file_lives_in_the_data_folder(tmp_path: Path) -> None:
    use_data_dir(tmp_path / "data")
    assert keys_file() == tmp_path / "data" / "keys.toml"


def test_the_file_is_read_when_there_is_no_key_store(no_key_store) -> None:
    write_keys_file(keys_file(), FAKE)

    assert get_secret(Key.BOT_TOKEN) == FAKE[Key.BOT_TOKEN]
    assert require_secret(Key.TG_SESSION) == FAKE[Key.TG_SESSION]
    assert get_secret(Key.OWNER_PHONE) is None
    with pytest.raises(MissingSecret, match="keys.toml"):
        require_secret(Key.OWNER_ADDRESS)


def test_the_file_is_read_for_what_the_key_store_does_not_have() -> None:
    write_keys_file(keys_file(), {Key.BOT_TOKEN: FAKE[Key.BOT_TOKEN]})

    assert get_secret(Key.BOT_TOKEN) == FAKE[Key.BOT_TOKEN]


def test_the_key_store_wins_when_both_have_the_key() -> None:
    write_keys_file(keys_file(), {Key.BOT_TOKEN: "7000000002:from-the-file"})
    keyring.set_password(secrets.SERVICE, Key.BOT_TOKEN.value, FAKE[Key.BOT_TOKEN])

    assert get_secret(Key.BOT_TOKEN) == FAKE[Key.BOT_TOKEN]


def test_without_a_key_store_or_a_file_a_key_is_simply_missing(no_key_store) -> None:
    assert get_secret(Key.BOT_TOKEN) is None
    with pytest.raises(MissingSecret, match="setup secrets"):
        require_secret(Key.BOT_TOKEN)


def test_set_secret_writes_the_file_when_there_is_no_key_store(no_key_store) -> None:
    assert set_secret(Key.BOT_TOKEN, FAKE[Key.BOT_TOKEN]) == keys_file()
    assert set_secret(Key.TG_API_ID, FAKE[Key.TG_API_ID]) == keys_file()

    assert get_secret(Key.BOT_TOKEN) == FAKE[Key.BOT_TOKEN]
    assert get_secret(Key.TG_API_ID) == FAKE[Key.TG_API_ID]


def test_set_secret_keeps_using_the_key_store_where_there_is_one() -> None:
    assert set_secret(Key.BOT_TOKEN, FAKE[Key.BOT_TOKEN]) is None
    assert not keys_file().exists()


@POSIX
def test_the_file_is_written_for_its_owner_only(tmp_path: Path) -> None:
    path = write_keys_file(tmp_path / "keys.toml", FAKE)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@POSIX
def test_a_file_others_can_read_stops_the_start(no_key_store) -> None:
    path = write_keys_file(keys_file(), FAKE)
    path.chmod(0o644)

    with pytest.raises(MissingSecret, match="chmod 600") as caught:
        require_secret(Key.BOT_TOKEN)
    assert not any(value in str(caught.value) for value in FAKE.values())


def test_a_broken_file_is_named_without_its_content(no_key_store) -> None:
    keys_file().parent.mkdir(parents=True, exist_ok=True)
    keys_file().write_text(f'bot_token = "{FAKE[Key.BOT_TOKEN]}', encoding="utf-8")
    if os.name != "nt":
        keys_file().chmod(0o600)

    with pytest.raises(MissingSecret, match="keys.toml") as caught:
        get_secret(Key.BOT_TOKEN)
    assert FAKE[Key.BOT_TOKEN] not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


def test_no_key_reaches_a_log_line(no_key_store, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    for key, value in FAKE.items():
        set_secret(key, value)
    for key in FAKE:
        get_secret(key)
    text = caplog.text + "".join(str(r.args) for r in caplog.records)
    assert not any(value in text for value in FAKE.values())


# --- the setup commands ---------------------------------------------------------------------


def test_secrets_writes_the_file_when_there_is_no_key_store(no_key_store) -> None:
    answers = iter(
        [FAKE[Key.ANTHROPIC_API_KEY], FAKE[Key.BOT_TOKEN], "424242", "made-up-hash", "", ""]
    )
    said: list[str] = []

    saved = ask_secrets(force=False, ask=lambda _: next(answers), say=said.append)

    assert saved == [Key.ANTHROPIC_API_KEY, Key.BOT_TOKEN, Key.TG_API_ID, Key.TG_API_HASH]
    assert get_secret(Key.BOT_TOKEN) == FAKE[Key.BOT_TOKEN]
    assert str(keys_file()) in said[-1]
    assert not any(value in "\n".join(said) for value in FAKE.values())


def test_export_keys_writes_the_file_and_prints_no_value(tmp_path: Path) -> None:
    for key, value in FAKE.items():
        set_secret(key, value)
    said: list[str] = []
    target = tmp_path / "out" / "keys.toml"

    assert export_keys(target, say=said.append) == 0

    use_data_dir(target.parent)
    keyring.set_keyring(fail.Keyring())
    for key, value in FAKE.items():
        assert get_secret(key) == value
    assert get_secret(Key.OWNER_PHONE) is None
    printed = "\n".join(said)
    assert not any(value in printed for value in FAKE.values())
    assert "5" in printed and str(target) in printed
    if os.name != "nt":
        assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_export_keys_with_an_empty_key_store_writes_nothing(tmp_path: Path) -> None:
    said: list[str] = []
    target = tmp_path / "keys.toml"

    assert export_keys(target, say=said.append) == 1
    assert not target.exists()
    assert said


def test_the_export_keys_command_line(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from tg_agent.setup.__main__ import main

    set_secret(Key.BOT_TOKEN, FAKE[Key.BOT_TOKEN])
    target = tmp_path / "keys.toml"

    assert main(["export-keys", str(target)]) == 0

    assert target.is_file()
    assert FAKE[Key.BOT_TOKEN] not in capsys.readouterr().out
