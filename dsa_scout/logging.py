"""Structured logging helpers for DSA-Scout."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a Rich-backed logger.

    Args:
        name: Logger name, usually ``__name__``.

    Returns:
        Configured standard-library logger.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
        _CONFIGURED = True
    return logging.getLogger(name)
