"""Structured logging for MedAlign.

Provides a configured logger that emits JSON-structured logs including
request IDs, execution times, module names, and status codes.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Generator


class StructuredFormatter(logging.Formatter):
    """Emit log records as structured key-value pairs."""

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        # Attach extra fields set via `extra=` on the logger call.
        for key in ("request_id", "execution_time_ms", "status", "component"):
            value = getattr(record, key, None)
            if value is not None:
                base[key] = value
        return str(base)


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with structured formatting."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())

    root = logging.getLogger("medalign")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``medalign`` namespace."""
    return logging.getLogger(f"medalign.{name}")


def generate_request_id() -> str:
    """Generate a unique request identifier."""
    return uuid.uuid4().hex[:12]


@contextmanager
def log_execution(
    logger: logging.Logger,
    operation: str,
    *,
    request_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Generator[None, None, None]:
    """Context manager that logs the start, end, and duration of an operation.

    Usage::

        with log_execution(logger, "investigate_prediction", request_id=rid):
            result = investigator.investigate(...)
    """
    rid = request_id or generate_request_id()
    logger.info(
        f"[START] {operation}",
        extra={"request_id": rid, "status": "started", **(extra or {})},
    )
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        logger.exception(
            f"[FAIL] {operation}",
            extra={
                "request_id": rid,
                "execution_time_ms": round(elapsed, 2),
                "status": "failed",
                **(extra or {}),
            },
        )
        raise
    else:
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"[DONE] {operation}",
            extra={
                "request_id": rid,
                "execution_time_ms": round(elapsed, 2),
                "status": "completed",
                **(extra or {}),
            },
        )
