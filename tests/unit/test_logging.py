from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from tg_agent.logging_setup import RedactFilter, configure

API_KEY = "sk-ant-api03-abcdefghijklmnop"
BOT_TOKEN = "7123456789:AAH-secret-token-value"


def record(msg: str, *args: object, exc_info: object = None) -> logging.LogRecord:
    return logging.LogRecord(
        "tg_agent.test", logging.INFO, __file__, 1, msg, args, exc_info
    )


@pytest.fixture
def clean_logging() -> Iterator[None]:
    """`configure` changes the process-wide logging: put it back after the test."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    levels = {name: logging.getLogger(name).level for name in ("telethon", "httpx")}
    yield
    for handler in root.handlers:
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)
    for name, value in levels.items():
        logging.getLogger(name).setLevel(value)


def file_log(data_dir: Path) -> str:
    for handler in logging.getLogger().handlers:
        handler.flush()
    return (data_dir / "logs" / "agent.log").read_text(encoding="utf-8")


def test_redact_filter_masks_secrets_and_phones() -> None:
    rec = record(
        "key %s, token %s, call +998 90 123-45-67 or 998901234567; total 250000",
        API_KEY,
        BOT_TOKEN,
    )

    assert RedactFilter([API_KEY, BOT_TOKEN, ""]).filter(rec) is True

    assert rec.getMessage() == "key ***, token ***, call *** or ***; total 250000"


def test_redact_filter_masks_the_traceback_too() -> None:
    try:
        raise RuntimeError(f"rejected key {API_KEY}")
    except RuntimeError:
        rec = record("call failed", exc_info=sys.exc_info())

    RedactFilter([API_KEY]).filter(rec)

    text = logging.Formatter().format(rec)
    assert API_KEY not in text and "rejected key ***" in text


def test_a_message_that_cannot_be_formatted_is_still_logged() -> None:
    rec = record("price %d for %s", "not a number")

    assert RedactFilter([API_KEY]).filter(rec) is True

    assert rec.getMessage().startswith("price %d for %s")


def test_configure_writes_a_rotating_redacted_file_and_quiets_telethon(
    tmp_path: Path, clean_logging: None
) -> None:
    configure(tmp_path, [API_KEY])

    logging.getLogger("tg_agent.engine").warning("using %s", API_KEY)
    # Telethon logs whole requests, outgoing text included, at WARNING and below.
    logging.getLogger("telethon.network.mtprotosender").warning("request: the text")

    (rotating,) = [
        h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)
    ]
    assert (rotating.maxBytes, rotating.backupCount) == (5 * 1024 * 1024, 5)
    log = file_log(tmp_path)
    assert "using ***" in log and API_KEY not in log
    assert "the text" not in log


def test_configure_twice_does_not_double_the_lines(
    tmp_path: Path, clean_logging: None
) -> None:
    configure(tmp_path, [API_KEY])
    configure(tmp_path, [API_KEY])

    logging.getLogger("tg_agent.engine").warning("once")

    assert file_log(tmp_path).count("once") == 1
