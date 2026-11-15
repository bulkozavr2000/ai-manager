from __future__ import annotations

import keyring
import pytest

from tg_agent.secrets import (
    SERVICE,
    Key,
    MissingSecret,
    get_secret,
    load_owner_secrets,
    require_secret,
    set_secret,
)


def test_secret_roundtrip(memory_keyring):
    assert get_secret(Key.BOT_TOKEN) is None

    set_secret(Key.BOT_TOKEN, "123456:test-token")

    assert get_secret(Key.BOT_TOKEN) == "123456:test-token"
    assert require_secret(Key.BOT_TOKEN) == "123456:test-token"

    with pytest.raises(MissingSecret, match="setup secrets"):
        require_secret(Key.ANTHROPIC_API_KEY)


def test_service_and_keys_match_spec():
    assert SERVICE == "tg-agent"
    assert {key.name: key.value for key in Key} == {
        "ANTHROPIC_API_KEY": "anthropic_api_key",
        "BOT_TOKEN": "bot_token",
        "TG_API_ID": "tg_api_id",
        "TG_API_HASH": "tg_api_hash",
        "TG_SESSION": "tg_session",
        "OWNER_PHONE": "owner_phone",
        "OWNER_ADDRESS": "owner_address",
    }


def test_secrets_land_in_the_keyring_under_the_service_name(memory_keyring):
    set_secret(Key.TG_API_HASH, "abc123")

    assert memory_keyring.store == {("tg-agent", "tg_api_hash"): "abc123"}


def test_empty_value_counts_as_missing(memory_keyring):
    set_secret(Key.OWNER_PHONE, "")

    with pytest.raises(MissingSecret, match="owner_phone"):
        require_secret(Key.OWNER_PHONE)


def test_missing_secret_message_gives_the_setup_command():
    with pytest.raises(MissingSecret) as excinfo:
        require_secret(Key.TG_SESSION)

    assert "uv run python -m tg_agent.setup secrets" in str(excinfo.value)


def test_every_test_runs_against_the_in_memory_keyring():
    # This test does not ask for the fixture: it must be installed anyway (autouse),
    # so no test can ever read or overwrite the owner's real credentials.
    assert type(keyring.get_keyring()).__name__ == "MemoryKeyring"


def test_owner_secrets_come_from_the_keyring_without_blanks(memory_keyring):
    set_secret(Key.OWNER_PHONE, "+998901112233")
    set_secret(Key.OWNER_ADDRESS, "   ")

    assert load_owner_secrets() == ["+998901112233"]
