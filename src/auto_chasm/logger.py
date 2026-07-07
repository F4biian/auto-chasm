"""Centralized logging utilities.

This module provides a thin proxy around the standard library ``logging``
module so that the *backend* can be swapped globally without editing every
file that logs.

Usage
-----
Always import ``get_logger`` from here instead of calling
``logging.getLogger`` directly::

    from auto_chasm.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Something happened")

If the logging backend is ever changed (e.g. to ``structlog`` or a custom
handler), only this file needs to be updated.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    pass


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger with *name*.

    This is a drop-in replacement for ``logging.getLogger`` that lets us
    change the logger factory globally if needed in the future.

    Args:
        name: The logger name; conventionally ``__name__``.
            If *None* (or omitted), the root ``auto_chasm`` logger is
            returned.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    return logging.getLogger(name or "auto_chasm")


def configure_logging(
    level: int = logging.INFO,
    fmt: str = "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream: TextIO = sys.stderr,
) -> None:
    """Set up a basic logging configuration for the application.

    This is idempotent: calling it multiple times is safe.

    Args:
        level: The minimum log level to emit. Defaults to ``logging.INFO``.
        fmt: The log record format string.
        stream: The output stream (default: ``sys.stderr``).
    """
    handler = logging.StreamHandler(stream=stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))

    # Get the root auto_chasm logger
    root = get_logger("auto_chasm")
    root.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
