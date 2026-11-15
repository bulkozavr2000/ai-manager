"""Secrets live in the OS keyring (Windows Credential Manager), under one service name.

Where the system has no key store (a Linux server in a container) they live in `keys.toml` in
the data folder instead, readable by its owner only (ruling R84). The key store stays the first
place: the file is read only for a key the store does not give.

Never write a secret to a log, an exception text or git.
"""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path

import keyring
from keyring.errors import NoKeyringError

from tg_agent.config import DEFAULT_DATA_DIR

SERVICE = "tg-agent"
KEYS_FILE = "keys.toml"


class Key(StrEnum):
    ANTHROPIC_API_KEY = "anthropic_api_key"
    BOT_TOKEN = "bot_token"
    TG_API_ID = "tg_api_id"
    TG_API_HASH = "tg_api_hash"
    TG_SESSION = "tg_session"
    OWNER_PHONE = "owner_phone"
    OWNER_ADDRESS = "owner_address"


class MissingSecret(RuntimeError):
    """A required secret is not in the keyring or the keys file, or the file cannot be used."""


_keys_file = DEFAULT_DATA_DIR / KEYS_FILE


def use_data_dir(data_dir: Path) -> None:
    """Look for the keys file in this data folder (the program's settings name it)."""
    global _keys_file
    _keys_file = data_dir / KEYS_FILE


def keys_file() -> Path:
    return _keys_file


def get_secret(key: Key) -> str | None:
    try:
        value = keyring.get_password(SERVICE, key.value)
    except NoKeyringError:  # no key store on this system
        value = None
    return value or _read_keys_file().get(key.value)


def set_secret(key: Key, value: str) -> Path | None:
    """Keep the key in the key store; where there is none, in the keys file, whose path is
    returned then."""
    try:
        keyring.set_password(SERVICE, key.value, value)
        return None
    except NoKeyringError:
        pass
    stored = _read_keys_file()
    stored[key.value] = value
    return write_keys_file(_keys_file, stored)


def stored_in_key_store() -> dict[Key, str]:
    """Every key the key store holds (for `setup export-keys`); none where there is no store."""
    found: dict[Key, str] = {}
    for key in Key:
        try:
            value = keyring.get_password(SERVICE, key.value)
        except NoKeyringError:
            return {}
        if value:
            found[key] = value
    return found


def write_keys_file(path: Path, values: Mapping[str, str]) -> Path:
    """Write the keys to `path` for its owner only (0600 where the system has file modes).

    The file is written next to its place and then moved there, so a half-written one never
    stands in for it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # A JSON string is a valid TOML basic string: quotes, backslashes and control signs escaped.
    lines = [
        f"{str(name)} = {json.dumps(value, ensure_ascii=False)}\n"
        for name, value in values.items()
        if value
    ]
    part = path.with_name(path.name + ".part")
    part.unlink(missing_ok=True)
    descriptor = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as out:
        out.writelines(lines)
    os.replace(part, path)
    return path


def _read_keys_file() -> dict[str, str]:
    path = _keys_file
    if not path.is_file():
        return {}
    if os.name != "nt" and path.stat().st_mode & 0o077:
        raise MissingSecret(
            f"Файл ключей {path} могут читать другие пользователи: запусти chmod 600 {path}"
        )
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        # Only the error's kind: its text may quote a line of the file, which is a key.
        raise MissingSecret(
            f"Файл ключей {path} не читается ({type(error).__name__})"
        ) from None
    return {name: value for name, value in data.items() if isinstance(value, str)}


def load_owner_secrets() -> list[str]:
    """The owner's real phone and address from the keyring, for the leak check; blanks dropped."""
    values = (get_secret(Key.OWNER_PHONE), get_secret(Key.OWNER_ADDRESS))
    return [value for value in values if value and value.strip()]


def require_secret(key: Key) -> str:
    value = get_secret(key)
    if not value:
        raise MissingSecret(
            f"Secret {key.value!r} is not set in the keyring (service {SERVICE!r})"
            f" or in {_keys_file}. Set it with: uv run python -m tg_agent.setup secrets"
        )
    return value
