"""Environment-backed, validated configuration for all Phase 1.5 stages."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from microq_robustness.errors import ConfigurationError


def _read_env_file(path: Path) -> dict[str, str]:
    """Read the small dotenv subset used by this project without extra dependencies."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigurationError(f"Invalid .env entry at {path}:{line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true or false, got {value!r}.")


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated Phase 1.5 settings; OS variables override values in `.env`."""

    project_root: Path
    sample_count: int
    random_seed: int
    data_pipeline_root: Path
    generated_manifest: Path
    restored_manifest: Path
    sample_manifest: Path
    results_path: Path
    analysis_output_dir: Path
    question_types: tuple[str, ...]
    severities: tuple[str, ...]
    include_mixed_degradations: bool
    allow_synthetic_benchmark_rows: bool
    verify_images: bool
    model_id: str
    api_base: str
    api_key: str
    request_timeout_seconds: float
    max_retries: int
    concurrency: int
    max_tokens: int
    temperature: float
    strict_completeness: bool
    bootstrap_resamples: int
    confidence_level: float
    figure_dpi: int

    @classmethod
    def load(cls, project_root: Path | None = None) -> Settings:
        root = (project_root or Path.cwd()).resolve()
        file_values = _read_env_file(root / ".env")

        def value(name: str, default: str) -> str:
            return os.environ.get(name, file_values.get(name, default))

        def path(name: str, default: str) -> Path:
            candidate = Path(value(name, default)).expanduser()
            return (candidate if candidate.is_absolute() else root / candidate).resolve()

        settings = cls(
            project_root=root,
            sample_count=int(value("SAMPLE_COUNT", "100")),
            random_seed=int(value("RANDOM_SEED", "20260805")),
            data_pipeline_root=path("DATA_PIPELINE_ROOT", "../DataDegradationPipeline"),
            generated_manifest=path(
                "GENERATED_MANIFEST",
                "../DataDegradationPipeline/outputs/manifests/generated.jsonl",
            ),
            restored_manifest=path(
                "RESTORED_MANIFEST",
                "../DataDegradationPipeline/outputs/manifests/restored.jsonl",
            ),
            sample_manifest=path("SAMPLE_MANIFEST", "outputs/samples.jsonl"),
            results_path=path("RESULTS_PATH", "outputs/evaluation_results.jsonl"),
            analysis_output_dir=path("ANALYSIS_OUTPUT_DIR", "outputs/analysis"),
            question_types=_csv(
                value(
                    "QUESTION_TYPES",
                    "domain,modality,classification,subdomain,submodality,stain",
                )
            ),
            severities=_csv(value("SEVERITIES", "weak,medium,strong")),
            include_mixed_degradations=_bool(
                value("INCLUDE_MIXED_DEGRADATIONS", "false"),
                name="INCLUDE_MIXED_DEGRADATIONS",
            ),
            allow_synthetic_benchmark_rows=_bool(
                value("ALLOW_SYNTHETIC_BENCHMARK_ROWS", "false"),
                name="ALLOW_SYNTHETIC_BENCHMARK_ROWS",
            ),
            verify_images=_bool(value("VERIFY_IMAGES", "true"), name="VERIFY_IMAGES"),
            model_id=value("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct"),
            api_base=value("VLLM_API_BASE", "http://localhost:8007/v1").rstrip("/"),
            api_key=value("VLLM_API_KEY", "EMPTY"),
            request_timeout_seconds=float(value("REQUEST_TIMEOUT_SECONDS", "180")),
            max_retries=int(value("MAX_RETRIES", "3")),
            concurrency=int(value("CONCURRENCY", "4")),
            max_tokens=int(value("MAX_TOKENS", "32")),
            temperature=float(value("TEMPERATURE", "0")),
            strict_completeness=_bool(
                value("STRICT_COMPLETENESS", "true"), name="STRICT_COMPLETENESS"
            ),
            bootstrap_resamples=int(value("BOOTSTRAP_RESAMPLES", "10000")),
            confidence_level=float(value("CONFIDENCE_LEVEL", "0.95")),
            figure_dpi=int(value("FIGURE_DPI", "300")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.sample_count < 1:
            raise ConfigurationError("SAMPLE_COUNT must be positive.")
        if not self.question_types:
            raise ConfigurationError("QUESTION_TYPES cannot be empty.")
        allowed_tasks = {
            "classification",
            "domain",
            "modality",
            "stain",
            "subdomain",
            "submodality",
        }
        unknown = set(self.question_types) - allowed_tasks
        if unknown:
            raise ConfigurationError(f"Unsupported QUESTION_TYPES: {sorted(unknown)}")
        if not self.severities or set(self.severities) - {"weak", "medium", "strong"}:
            raise ConfigurationError("SEVERITIES must contain weak, medium, and/or strong.")
        if self.concurrency < 1 or self.max_retries < 1:
            raise ConfigurationError("CONCURRENCY and MAX_RETRIES must be positive.")
        if self.max_tokens < 1 or self.request_timeout_seconds <= 0:
            raise ConfigurationError("MAX_TOKENS and REQUEST_TIMEOUT_SECONDS must be positive.")
        if self.temperature < 0:
            raise ConfigurationError("TEMPERATURE cannot be negative.")
        if self.bootstrap_resamples < 100:
            raise ConfigurationError("BOOTSTRAP_RESAMPLES must be at least 100.")
        if not 0.5 < self.confidence_level < 1.0:
            raise ConfigurationError("CONFIDENCE_LEVEL must be between 0.5 and 1.0.")
        if self.figure_dpi < 72:
            raise ConfigurationError("FIGURE_DPI must be at least 72.")

    def public_snapshot(self) -> dict[str, Any]:
        """Return serializable provenance while excluding the API key."""
        payload = asdict(self)
        payload.pop("api_key", None)
        return {
            key: str(item) if isinstance(item, Path) else item
            for key, item in payload.items()
        }

