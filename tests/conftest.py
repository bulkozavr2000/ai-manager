"""Shared fixtures.

Tests never touch the real OS keyring (Windows Credential Manager) or the real data folder.
The `repo` fixture opens a database file inside the test's temporary folder.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import keyring
import pytest
from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError

from tg_agent import secrets
from tg_agent.clock import FakeClock
from tg_agent.config import Owner, Settings
from tg_agent.db import Repo

TASHKENT = ZoneInfo("Asia/Tashkent")


class MemoryKeyring(KeyringBackend):
    """Keyring backend that keeps secrets in a dict, so nothing reaches the real credential store."""

    priority = 1

    def __init__(self) -> None:
        super().__init__()
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        try:
            del self.store[(service, username)]
        except KeyError:
            raise PasswordDeleteError("Password not found") from None


@pytest.fixture(autouse=True)
def memory_keyring():
    """Install an in-memory keyring for the test and restore the previous backend afterwards.

    Autouse on purpose: a test that forgets to ask for it must still never read or overwrite
    the owner's real secrets.
    """
    previous = keyring.get_keyring()
    backend = MemoryKeyring()
    keyring.set_keyring(backend)
    try:
        yield backend
    finally:
        keyring.set_keyring(previous)


@pytest.fixture(autouse=True)
def keys_file_in_tmp(tmp_path: Path):
    """The keys file (`keys.toml`, for a system with no key store) is looked for in the
    test's own folder, never in the real data folder."""
    previous = secrets.keys_file()
    secrets.use_data_dir(tmp_path / "keys-data")
    try:
        yield
    finally:
        secrets.use_data_dir(previous.parent)


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock(datetime(2026, 11, 10, 10, 0, tzinfo=TASHKENT))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(owner=Owner(control_user_ids=(111,)), data_dir=tmp_path)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Where the `repo` fixture keeps its database; tests that read the file directly use it."""
    return tmp_path / "agent.db"


@pytest.fixture
async def repo(db_path: Path, fake_clock: FakeClock) -> AsyncIterator[Repo]:
    """A repository on a fresh database that takes its time from `fake_clock`."""
    repo = await Repo.open(db_path, clock=fake_clock)
    try:
        yield repo
    finally:
        await repo.close()
