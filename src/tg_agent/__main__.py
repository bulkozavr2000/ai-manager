"""`uv run python -m tg_agent`: run the agent (Ctrl+C stops it)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from tg_agent.config import PROJECT_ROOT, ConfigError
from tg_agent.gateway.userbot import SessionNotAuthorized
from tg_agent.main import AlreadyRunning, UserbotLost, amain
from tg_agent.secrets import MissingSecret

# Problems the owner can fix: said in one line, without a traceback.
_EXPECTED = (
    ConfigError,
    MissingSecret,
    AlreadyRunning,
    SessionNotAuthorized,
    UserbotLost,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tg_agent", description="ИИ-помощник Саида в Telegram"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.toml",
        help="файл настроек (по умолчанию config.toml в корне проекта)",
    )
    args = parser.parse_args(argv)
    try:
        asyncio.run(amain(args.config))
    except KeyboardInterrupt:
        print("Агент остановлен", file=sys.stderr)
    except _EXPECTED as error:
        print(error, file=sys.stderr)
        return 2
    except Exception:
        logging.getLogger("tg_agent").exception("the agent stopped on an error")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
