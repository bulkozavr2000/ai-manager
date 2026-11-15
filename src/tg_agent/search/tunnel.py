"""The way in from the internet to the remote session's page on this computer (R50).

The owner chooses it: a server with its own address (`PublicUrl`), or Cloudflare's quick
tunnel (`CloudflaredTunnel`: a program that must already be on the computer; ruling R73). The
program never downloads or installs anything: a missing program is an error that names it.

A tunnel process is started without a shell, its output is read line by line for its public
address, and only an address of the expected shape is taken, so a look-alike in its output
(an `http://` address, `https://evil.example/x.trycloudflare.com`) is never the link.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import logging
import re
import subprocess
import sys
from collections.abc import Awaitable, Callable
from typing import Any, Protocol
from urllib.parse import urlsplit

from tg_agent.config import PROJECT_ROOT, Settings

_log = logging.getLogger(__name__)

OPEN_TIMEOUT = 30.0  # seconds for a tunnel to show its address
STOP_TIMEOUT = 5.0  # seconds for a process to end before it is killed

_CLOUDFLARE = re.compile(
    r"(?<![\w.:/-])(https://(?!api\.)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.trycloudflare\.com)"
    r"(?![\w.:/-])"
)


class TunnelError(Exception):
    """The way in did not open; the text is for the owner, in plain words."""


class Tunnel(Protocol):
    async def open(self, port: int) -> str:
        """The public `https://…` base that leads to 127.0.0.1:`port`."""
        ...

    async def close(self) -> None: ...


Spawn = Callable[..., Awaitable[Any]]


class PublicUrl:
    """A server with its own address: nothing to open."""

    def __init__(self, base: str) -> None:
        self._base = base.strip().rstrip("/")

    async def open(self, port: int) -> str:
        try:
            parts = urlsplit(self._base)
        except ValueError:
            parts = None
        # The page encrypts with crypto.subtle, which a browser gives only to https pages.
        if parts is None or parts.scheme != "https" or not parts.hostname:
            raise TunnelError("Адрес для входа с телефона должен начинаться с https://")
        return self._base

    async def close(self) -> None:
        return None


class _ProcessTunnel:
    """A program that opens the tunnel while it runs and prints its address."""

    name = ""

    def __init__(
        self,
        program: str,
        pattern: re.Pattern[str],
        *,
        spawn: Spawn | None = None,
        timeout: float = OPEN_TIMEOUT,
        stop_timeout: float = STOP_TIMEOUT,
    ) -> None:
        self._program = program
        self._pattern = pattern
        self._spawn: Spawn = spawn or asyncio.create_subprocess_exec
        self._timeout = timeout
        self._stop_timeout = stop_timeout
        self._process: Any = None
        self._drain: asyncio.Task[None] | None = None

    @property
    def pid(self) -> int | None:
        """The running tunnel process, so a crash can be cleaned up after (4б.2)."""
        return None if self._process is None else self._process.pid

    def _args(self, port: int) -> list[str]:
        raise NotImplementedError

    def _missing(self) -> TunnelError:
        return TunnelError(
            f"Не найдена программа {self.name}: {self._program or self.name}. Сам я её"
            " не скачиваю и не устанавливаю."
        )

    def _at_exit(self) -> None:
        """When the program ends without closing the tunnel: on Windows a child process
        outlives its parent, and the public address would stay open."""
        process = self._process
        if process is not None and process.returncode is None:
            with contextlib.suppress(Exception):
                process.terminate()

    async def open(self, port: int) -> str:
        options: dict[str, Any] = {
            # stdin is a pipe of its own: the tunnel never reads the agent's console.
            "stdin": asyncio.subprocess.PIPE,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.STDOUT,
        }
        if sys.platform == "win32":  # no console window flashing up beside the agent
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            self._process = await self._spawn(
                self._program, *self._args(port), **options
            )
        except OSError:
            raise self._missing() from None
        atexit.register(self._at_exit)
        try:
            address = await asyncio.wait_for(self._find(), self._timeout)
        except TimeoutError:
            await self.close()
            raise TunnelError(
                f"{self.name}: туннель не открылся за {self._timeout:g} с"
            ) from None
        except BaseException:
            await self.close()
            raise
        if address is None:
            await self.close()
            raise TunnelError(f"{self.name} завершилась, не дав адреса для входа")
        # What it prints later is read and dropped, so a full pipe never stalls it.
        self._drain = asyncio.create_task(self._read_rest())
        _log.info("tunnel %s is open", self.name)
        return address

    async def _find(self) -> str | None:
        stdout = self._process.stdout
        while True:
            try:
                line = await stdout.readline()
            except ValueError:  # a line over the reader's limit: it is skipped
                continue
            if not line:
                return None
            match = self._pattern.search(line.decode("utf-8", "replace"))
            if match:
                return match.group(1)

    async def _read_rest(self) -> None:
        with contextlib.suppress(Exception):
            while await self._process.stdout.read(65536):
                pass

    async def close(self) -> None:
        process, self._process = self._process, None
        drain, self._drain = self._drain, None
        if process is None:
            return
        with contextlib.suppress(Exception):
            atexit.unregister(self._at_exit)
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), self._stop_timeout)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError, OSError):
                    process.kill()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(process.wait(), self._stop_timeout)
        else:
            with contextlib.suppress(Exception):
                await process.wait()
        if drain is not None:
            drain.cancel()
            with contextlib.suppress(BaseException):
                await drain


class CloudflaredTunnel(_ProcessTunnel):
    """Cloudflare's quick tunnel: `cloudflared` from the given path, no account needed."""

    name = "cloudflared"

    def __init__(self, path: str, **options: Any) -> None:
        super().__init__(path, _CLOUDFLARE, **options)

    def _args(self, port: int) -> list[str]:
        return ["tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"]


def tunnel_from(settings: Settings) -> Tunnel | None:
    """The way in the owner chose (`[remote]`), or None: the feature is off (ruling R50).

    A server with its own address (`public_url`) needs no tunnel, whatever `tunnel` says.
    Nothing is started or made here: the tunnel opens with its `open`.
    """
    remote = settings.remote
    if remote.public_url.strip():
        return PublicUrl(remote.public_url)
    if remote.tunnel == "cloudflared":
        path = remote.cloudflared
        return CloudflaredTunnel(
            str(path if path.is_absolute() else PROJECT_ROOT / path)
        )
    return None
