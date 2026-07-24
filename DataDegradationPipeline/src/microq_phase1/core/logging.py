"""Structured application logging with human and machine-readable sinks."""

from __future__ import annotations

import logging
import logging.handlers
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator
from uuid import uuid4

import structlog
from rich.logging import RichHandler


def configure_logging(
    *,
    level: str,
    json_log_path: Path,
    rotate_megabytes: int,
    backup_count: int,
    run_id: str | None = None,
) -> str:
    """Configure rich console logs and a rotating JSON event stream."""
    resolved_run_id = run_id or uuid4().hex
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=True),
        foreign_pre_chain=shared_processors,
    )
    json_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(sort_keys=True),
        foreign_pre_chain=shared_processors,
    )

    console_handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        markup=False,
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(numeric_level)

    json_log_path.parent.mkdir(parents=True, exist_ok=True)
    json_handler = logging.handlers.RotatingFileHandler(
        json_log_path,
        maxBytes=rotate_megabytes * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8",
    )
    json_handler.setFormatter(json_formatter)
    json_handler.setLevel(numeric_level)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric_level)
    root.addHandler(console_handler)
    root.addHandler(json_handler)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(run_id=resolved_run_id)
    return resolved_run_id


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structured logger using the process-wide configuration."""
    return structlog.get_logger(name)


@contextmanager
def logged_stage(
    logger: structlog.stdlib.BoundLogger,
    stage: str,
    **context: object,
) -> Iterator[None]:
    """Log stage start, completion, duration, and exceptions uniformly."""
    started = perf_counter()
    bound = logger.bind(stage=stage, **context)
    bound.info("stage_started")
    try:
        yield
    except Exception:
        bound.exception("stage_failed", duration_seconds=perf_counter() - started)
        raise
    bound.info("stage_completed", duration_seconds=perf_counter() - started)

