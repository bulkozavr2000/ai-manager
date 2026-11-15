"""Live tests call the real Claude API and spend real money.

They run only when TG_AGENT_LIVE=1 is set, and only then read the real keyring: the autouse
in-memory keyring from tests/conftest.py is replaced here by a fixture that does nothing.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def memory_keyring() -> None:
    """Shadow the root fixture: live tests need the real API key."""


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if os.environ.get("TG_AGENT_LIVE") == "1":
        return
    skip = pytest.mark.skip(
        reason="live test: set TG_AGENT_LIVE=1 to run (spends money)"
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
