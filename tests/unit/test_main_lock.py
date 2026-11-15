from __future__ import annotations

import socket

import pytest

from tg_agent.main import AlreadyRunning, acquire_single_instance


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_second_instance_rejected() -> None:
    port = free_port()
    first = acquire_single_instance(port)
    try:
        with pytest.raises(AlreadyRunning):
            acquire_single_instance(port)
    finally:
        first.close()


def test_the_lock_is_free_again_once_the_first_copy_ends() -> None:
    port = free_port()
    acquire_single_instance(port).close()

    acquire_single_instance(port).close()
