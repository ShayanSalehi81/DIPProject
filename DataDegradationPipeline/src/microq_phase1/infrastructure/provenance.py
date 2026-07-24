"""Run-level provenance snapshots for reproducible academic artifacts."""

from __future__ import annotations

import hashlib
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from microq_phase1 import __version__
from microq_phase1.infrastructure.jsonl import write_json_atomic


class RunProvenanceRecorder:
    """Record code/config/runtime identity without collecting sensitive environment data."""

    def record(
        self,
        *,
        run_id: str,
        command: str,
        config_path: Path,
        output_directory: Path,
    ) -> Path:
        payload = config_path.read_bytes()
        output_path = output_directory / f"{run_id}.json"
        write_json_atomic(
            output_path,
            {
                "run_id": run_id,
                "command": command,
                "started_at_utc": datetime.now(UTC).isoformat(),
                "microq_phase1_version": __version__,
                "config_path": str(config_path.resolve()),
                "config_sha256": hashlib.sha256(payload).hexdigest(),
                "python_version": sys.version,
                "platform": platform.platform(),
            },
        )
        return output_path

