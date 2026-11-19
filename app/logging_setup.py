"""
Structured logging configuration.

Configures Python's stdlib logging with JSON-style formatting suitable for
production log aggregators (ELK, Loki, CloudWatch) while remaining readable
during local development.

Usage::

    from app.config.logging import setup_logging

    setup_logging("INFO")
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone


class _JsonFormatter(logging.Formatter):
    """Lightweight JSON-ish formatter — no external dependencies."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        msg = record.getMessage()
        base = (
            f'{{"ts":"{ts}","level":"{record.levelname}",'
            f'"logger":"{record.name}","msg":"{msg}"}}'
        )
        if record.exc_info and record.exc_info[1]:
            exc = self.formatException(record.exc_info)
            base = base[:-1] + f',"exc":"{exc}"}}'
        return base


class _DevFormatter(logging.Formatter):
    """Human-readable coloured formatter for development."""

    COLOURS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        c = self.COLOURS.get(record.levelname, "")
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%H:%M:%S"
        )
        return f"{c}{ts} [{record.levelname:<8}]{self.RESET} {record.name}: {record.getMessage()}"


def setup_logging(level: str = "INFO", *, json_format: bool = False) -> None:
    """
    Configure the root logger.

    Parameters
    ----------
    level : str
        Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    json_format : bool
        If True, use JSON formatter. If False, use the coloured dev formatter.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter() if json_format else _DevFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).setLevel(logging.WARNING)
