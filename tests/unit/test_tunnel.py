"""The ways in from the internet, on a fake process: no test starts cloudflared."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tg_agent.search import tunnel as tunnel_module
from tg_agent.search.tunnel import (
    CloudflaredTunnel,
    PublicUrl,
    TunnelError,
)


class FakeProcess:
    def __init__(self, lines: list[str], *, ends: bool = False) -> None:
        self.stdout = asyncio.StreamReader()
        for line in lines:
            self.stdout.feed_data(line.encode())
        if ends:
            self.stdout.feed_eof()
            self.returncode: int | None = 255
        else:
            self.returncode = None
        self.terminated = 0
        self.killed = 0
        self.waited = 0
        self.pid = 4242

    def say(self, line: str) -> None:
        self.stdout.feed_data(line.encode())

    def terminate(self) -> None:
        self.terminated += 1
        if self.returncode is None:
            self.returncode = 1
            self.stdout.feed_eof()

    def kill(self) -> None:
        self.killed += 1
        if self.returncode is None:
            self.returncode = -9
            self.stdout.feed_eof()

    async def wait(self) -> int:
        self.waited += 1
        while self.returncode is None:
            await asyncio.sleep(0.01)
        return self.returncode


class Spawner:
    def __init__(
        self, process: FakeProcess | None = None, error: OSError | None = None
    ):
        self.process = process
        self.error = error
        self.calls: list[tuple[str, tuple[str, ...], dict[str, Any]]] = []

    async def __call__(self, program: str, *args: str, **kwargs: Any) -> FakeProcess:
        self.calls.append((program, args, kwargs))
        if self.error is not None:
            raise self.error
        assert self.process is not None
        return self.process


CLOUDFLARED_OUTPUT = [
    "2026-11-11T08:00:00Z INF Thank you for trying Cloudflare Tunnel.\n",
    "2026-11-11T08:00:00Z INF Requesting new quick Tunnel on trycloudflare.com...\n",
    "2026-11-11T08:00:01Z INF see https://evil.example/x.trycloudflare.com for more\n",
    "2026-11-11T08:00:01Z INF http://plain-words-here.trycloudflare.com\n",
    "2026-11-11T08:00:01Z INF https://api.trycloudflare.com\n",
    "2026-11-11T08:00:01Z INF https://lure-words.trycloudflare.com.evil.example\n",
    "2026-11-11T08:00:02Z INF +--------------------------------------------------------+\n",
    "2026-11-11T08:00:02Z INF |  Your quick Tunnel has been created! Visit it at:      |\n",
    "2026-11-11T08:00:02Z INF |  https://seasonal-deck-organisms-sf.trycloudflare.com  |\n",
    "2026-11-11T08:00:02Z INF +--------------------------------------------------------+\n",
]


async def test_cloudflared_finds_its_address_among_look_alikes() -> None:
    process = FakeProcess(CLOUDFLARED_OUTPUT)
    spawn = Spawner(process)
    tunnel = CloudflaredTunnel("tools/cloudflared.exe", spawn=spawn)
    assert (
        await tunnel.open(8765)
        == "https://seasonal-deck-organisms-sf.trycloudflare.com"
    )
    program, args, kwargs = spawn.calls[0]
    assert program == "tools/cloudflared.exe"
    assert args == ("tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8765")
    assert "shell" not in kwargs
    process.say("2026-11-11T08:00:05Z INF Registered tunnel connection\n" * 1000)
    await asyncio.sleep(
        0.05
    )  # what it prints later is read and dropped, so it never stalls
    assert process.stdout.at_eof() or len(process.stdout._buffer) == 0  # type: ignore[attr-defined]
    await tunnel.close()
    assert process.terminated == 1 and process.waited >= 1


async def test_no_address_in_time_is_an_error_and_the_process_is_stopped() -> None:
    process = FakeProcess(
        ["https://evil.example/x.trycloudflare.com\n", "connecting...\n"]
    )
    tunnel = CloudflaredTunnel("cf.exe", spawn=Spawner(process), timeout=0.1)
    with pytest.raises(TunnelError, match="30|0.1|не открылся"):
        await tunnel.open(8765)
    assert process.terminated == 1


async def test_a_process_that_ends_without_an_address_is_an_error() -> None:
    process = FakeProcess(
        ["2026-11-11T08:00:00Z ERR failed to request quick Tunnel\n"], ends=True
    )
    with pytest.raises(TunnelError, match="cloudflared"):
        await CloudflaredTunnel("cf.exe", spawn=Spawner(process)).open(8765)


async def test_a_missing_program_is_an_error_that_names_it() -> None:
    spawn = Spawner(error=FileNotFoundError(2, "not found"))
    with pytest.raises(TunnelError, match="tools/cloudflared.exe"):
        await CloudflaredTunnel("tools/cloudflared.exe", spawn=spawn).open(8765)


async def test_close_may_be_called_twice_and_before_open() -> None:
    await CloudflaredTunnel("cf.exe", spawn=Spawner()).close()
    process = FakeProcess(CLOUDFLARED_OUTPUT)
    tunnel = CloudflaredTunnel("cf.exe", spawn=Spawner(process))
    await tunnel.open(8765)
    await tunnel.close()
    await tunnel.close()
    assert process.terminated == 1


async def test_a_process_that_will_not_stop_is_killed() -> None:
    process = FakeProcess(CLOUDFLARED_OUTPUT)
    process.terminate = lambda: None  # type: ignore[method-assign]
    tunnel = CloudflaredTunnel("cf.exe", spawn=Spawner(process), stop_timeout=0.05)
    await tunnel.open(8765)
    await tunnel.close()
    assert process.killed == 1


async def test_a_public_url_opens_nothing() -> None:
    tunnel = PublicUrl("https://agent.example.com/")
    assert await tunnel.open(8765) == "https://agent.example.com"
    await tunnel.close()
    with pytest.raises(TunnelError):
        await PublicUrl("http://agent.example.com").open(8765)  # the page needs https
    with pytest.raises(TunnelError):
        await PublicUrl("not an address").open(8765)


# --- fix round 1 (F6) -------------------------------------------------------------------------


async def test_a_running_tunnel_is_stopped_when_the_program_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hooks: list[Any] = []
    monkeypatch.setattr(tunnel_module.atexit, "register", hooks.append)
    monkeypatch.setattr(tunnel_module.atexit, "unregister", hooks.remove)
    process = FakeProcess(CLOUDFLARED_OUTPUT)
    tunnel = CloudflaredTunnel("cf.exe", spawn=Spawner(process))
    assert tunnel.pid is None
    await tunnel.open(8765)
    assert tunnel.pid == 4242  # for 4б.2 to clean up after a crash
    assert len(hooks) == 1
    hooks[0]()  # what atexit would run
    assert process.terminated == 1
    await tunnel.close()
    assert hooks == [] and tunnel.pid is None


async def test_a_line_over_the_reader_limit_is_skipped() -> None:
    process = FakeProcess(["x" * 100_000 + "\n"] + CLOUDFLARED_OUTPUT[-3:], ends=False)
    tunnel = CloudflaredTunnel("cf.exe", spawn=Spawner(process))
    assert (
        await tunnel.open(8765)
        == "https://seasonal-deck-organisms-sf.trycloudflare.com"
    )
    await tunnel.close()


@pytest.mark.parametrize(
    "line",
    [
        "xhttps://a-b.trycloudflare.com\n",
        "see:https://a-b.trycloudflare.com\n",
        "https://a-b.trycloudflare.com/path\n",
        "https://user@a-b.trycloudflare.com\n",
    ],
)
async def test_a_look_alike_with_something_glued_on_is_not_the_address(
    line: str,
) -> None:
    process = FakeProcess([line])
    with pytest.raises(TunnelError):
        await CloudflaredTunnel("cf.exe", spawn=Spawner(process), timeout=0.1).open(1)
