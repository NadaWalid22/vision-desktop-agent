"""
Structured logger configuration using loguru.

loguru advantages over stdlib logging:
- Colourised, human-readable output by default
- Structured JSON sink for production log aggregation
- Thread-safe with no extra configuration
- Lazy string interpolation (no cost if log level disabled)

The module exposes a single get_logger() factory so all modules share
the same loguru instance while retaining per-module name tagging.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _root_logger

_configured = False


def configure_logging(
    level: str = "INFO",
    log_file: Path | str | None = None,
    json_logs: bool = False,
) -> None:
    """
    Configure loguru sinks.  Call once at application startup.

    Args:
        level:     Minimum log level ("DEBUG", "INFO", "WARNING", "ERROR").
        log_file:  If provided, also write logs to this file (rotating 10 MB).
        json_logs: If True, emit JSON-structured logs to the file sink.
    """
    global _configured

    _root_logger.remove()  # Remove default sink

    # ── Console sink ──────────────────────────────────────────────────────────
    fmt_console = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )
    _root_logger.add(
        sys.stderr,
        format=fmt_console,
        level=level,
        colorize=True,
        enqueue=True,  # Thread-safe async queue
    )

    # ── File sink (optional) ──────────────────────────────────────────────────
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if json_logs:
            _root_logger.add(
                str(file_path),
                format="{message}",
                level=level,
                rotation="10 MB",
                retention="7 days",
                serialize=True,   # loguru native JSON
                enqueue=True,
            )
        else:
            _root_logger.add(
                str(file_path),
                format=(
                    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                    "{name}:{line} — {message}"
                ),
                level=level,
                rotation="10 MB",
                retention="7 days",
                enqueue=True,
            )

    _configured = True


def get_logger(name: str):
    """
    Return a loguru logger bound to *name*.

    Usage::

        logger = get_logger(__name__)
        logger.info("Module initialised")
        logger.debug(f"Result: {value}")

    Returns:
        A loguru bound logger with extra {"name": name} context.
    """
    if not _configured:
        configure_logging()
    return _root_logger.bind(name=name)
