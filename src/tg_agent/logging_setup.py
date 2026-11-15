"""Logging: a rotating file under `data/logs` and the console, secrets and phone numbers masked.

Telethon logs whole requests, the text of outgoing messages included, at WARNING and below, so
its loggers are kept at ERROR. The mask works on the formatted line and on the traceback, so a
secret passed as an argument or carried in an exception's text is masked as well.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterable
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TextIO

MASK = "***"
LOG_FILE = "agent.log"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 5
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# Loggers that say more than may be kept, and the level they are held at.
_QUIET = {"telethon": logging.ERROR, "httpx": logging.WARNING}
# A run of digits written the way phone numbers are: nine digits or more is masked. Telegram
# user ids look the same and are masked with them.
_DIGIT_RUN = re.compile(r"\+?\d(?:[ \t\-()]{0,3}\d)+")
_MIN_PHONE_DIGITS = 9
_OURS = "_tg_agent_handler"  # marks the handlers `configure` installed


class RedactFilter(logging.Filter):
    """Masks the given secret values and anything that looks like a phone number.

    It rewrites the record (message, traceback, stack) and never drops one.
    """

    def __init__(self, secrets: Iterable[str]) -> None:
        super().__init__()
        # Longest first, so a secret that contains another one is masked whole.
        self._secrets = sorted(
            {value for value in secrets if value}, key=len, reverse=True
        )

    def redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, MASK)
        return _DIGIT_RUN.sub(_mask_phone, text)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a broken format string must not cost the line
            message = f"{record.msg} {record.args!r}"
        record.msg, record.args = self.redact(message), None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = self.redact(record.exc_text)
        if record.stack_info:
            record.stack_info = self.redact(record.stack_info)
        return True


def _mask_phone(match: re.Match[str]) -> str:
    digits = sum(char.isdigit() for char in match.group())
    return MASK if digits >= _MIN_PHONE_DIGITS else match.group()


def configure(data_dir: Path, secrets: Iterable[str]) -> None:
    """Write the log to `data_dir/logs/agent.log` (5 files of 5 MB) and to the console.

    Calling it again replaces what the earlier call installed.
    """
    logs = Path(data_dir) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _OURS, False)]:
        root.removeHandler(handler)
        handler.close()

    redact = RedactFilter(secrets)
    formatter = logging.Formatter(_FORMAT)
    file = RotatingFileHandler(
        logs / LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    console = logging.StreamHandler(_console())
    for handler in (file, console):
        handler.setFormatter(formatter)
        # On the handler, not on a logger: records from child loggers pass here too.
        handler.addFilter(redact)
        setattr(handler, _OURS, True)
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    for name, level in _QUIET.items():
        logging.getLogger(name).setLevel(level)


def _console() -> TextIO:
    """stderr, set so a character its encoding lacks (a redirected Windows console) is escaped
    instead of failing the line."""
    stream = sys.stderr
    try:
        stream.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError, OSError):
        pass
    return stream
