"""Atomic JSON/JSONL utilities and portable Phase 1 path resolution."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from microq_robustness.errors import ManifestError


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise ManifestError(f"JSONL file does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ManifestError(f"Expected an object at {path}:{line_number}.")
            yield payload


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def resolve_phase1_path(raw_path: str, pipeline_root: Path) -> Path:
    """Resolve absolute Phase 1 paths even after moving the repository to a new server."""
    candidate = Path(raw_path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    parts = candidate.parts
    if pipeline_root.name in parts:
        index = parts.index(pipeline_root.name)
        relocated = pipeline_root.joinpath(*parts[index + 1 :])
        if relocated.is_file():
            return relocated.resolve()
    if not candidate.is_absolute():
        relocated = pipeline_root / candidate
        if relocated.is_file():
            return relocated.resolve()
    raise ManifestError(
        f"Image path does not exist: {raw_path}. Checked relocation under {pipeline_root}."
    )

