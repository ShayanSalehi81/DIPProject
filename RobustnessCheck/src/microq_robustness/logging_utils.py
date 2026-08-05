"""Readable console logging plus a rotating machine-readable JSON log."""

from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path

UTC_ZONE = timezone.utc  # noqa: UP017 -- keeps diagnostic use possible on Python 3.10


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC_ZONE).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for name in ("run_id", "sample_id", "condition", "stage"):
            if hasattr(record, name):
                payload[name] = getattr(record, name)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def configure_logging(project_root: Path, *, level: int = logging.INFO) -> None:
    log_path = project_root / "outputs" / "logs" / "robustness.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    root.addHandler(console)

    structured = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=20 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    structured.setFormatter(JsonFormatter())
    root.addHandler(structured)
