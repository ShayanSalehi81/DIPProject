"""Configuration loading and path anchoring."""

from __future__ import annotations

from pathlib import Path

import yaml

from microq_phase1.core.config import ProjectConfig
from microq_phase1.core.errors import ConfigurationError

PATH_KEYS = {
    "catalog_path",
    "output_root",
    "cache_dir",
    "json_log_path",
    "recipe_manifest",
    "generated_manifest",
    "output_jsonl",
    "output_csv",
    "restored_output_jsonl",
    "restored_output_csv",
    "baseline_output_dir",
    "manifest_path",
    "report_path",
    "overlap_report_path",
}


def load_config(path: Path) -> ProjectConfig:
    """Load, validate, and resolve a YAML configuration."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = ProjectConfig.model_validate(raw)
        payload = _resolve_paths(config.model_dump(), path.resolve().parent)
        return ProjectConfig.model_validate(payload)
    except Exception as exc:
        raise ConfigurationError(f"Unable to load configuration: {path}") from exc


def _resolve_paths(value: object, base: Path, key: str | None = None) -> object:
    if isinstance(value, dict):
        return {
            child_key: _resolve_paths(child, base, child_key)
            for child_key, child in value.items()
        }
    if isinstance(value, list):
        return [_resolve_paths(child, base, key) for child in value]
    if key in PATH_KEYS and value is not None:
        path = Path(str(value))
        return path if path.is_absolute() else (base / path).resolve()
    return value
