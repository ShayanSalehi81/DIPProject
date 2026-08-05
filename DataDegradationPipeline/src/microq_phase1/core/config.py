"""Pydantic models for the complete Phase 1 experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from microq_phase1.domain.models import DegradationKind, Modality, PoolRole, Severity


def _all_modalities() -> set[Modality]:
    return set(Modality)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"
    json_log_path: Path = Path("outputs/logs/phase1.jsonl")
    rotate_megabytes: int = Field(default=50, ge=1)
    backup_count: int = Field(default=5, ge=1)


class PoolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: PoolRole
    catalog_path: Path
    official_benchmark: bool = False
    max_images: int | None = Field(default=None, gt=0)


class MicroBenchExportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    dataset_id: str = "jnirschl/uBench"
    revision: str
    split: str = "test"
    output_root: Path = Path("data/microbench")
    catalog_path: Path = Path("data/catalogs/microbench_evaluation.jsonl")
    cache_dir: Path = Path("data/cache/huggingface")
    max_records: int | None = Field(default=None, gt=0)
    included_licenses: set[str] | None = None

    @model_validator(mode="after")
    def require_pinned_revision(self) -> MicroBenchExportConfig:
        forbidden = {"main", "master", "latest", "REPLACE_WITH_PINNED_COMMIT_SHA"}
        if self.enabled and self.revision in forbidden:
            raise ValueError("Micro-Bench revision must be an immutable commit SHA.")
        return self


class DegradationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: DegradationKind
    enabled: bool = True
    applicable_modalities: set[Modality] = Field(default_factory=_all_modalities)
    requires_multichannel: bool = False
    severity_parameters: dict[Severity, dict[str, Any]]

    @model_validator(mode="after")
    def require_all_severities(self) -> DegradationConfig:
        required = {Severity.WEAK, Severity.MEDIUM, Severity.STRONG}
        if self.enabled and set(self.severity_parameters) != required:
            raise ValueError(f"{self.kind} must define weak, medium, and strong parameters.")
        return self


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    master_seed: int = Field(default=20260724, ge=0)
    variants_per_image: int = Field(default=6, ge=1)
    clean_control_fraction: float = Field(default=0.10, ge=0.0, le=0.5)
    mixed_degradation_fraction: float = Field(default=0.10, ge=0.0, le=0.5)
    output_root: Path = Path("outputs/generated")
    recipe_manifest: Path = Path("outputs/manifests/recipes.jsonl")
    generated_manifest: Path = Path("outputs/manifests/generated.jsonl")
    output_format: Literal["png"] = "png"
    progress_interval: int = Field(default=100, ge=1)


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_jsonl: Path = Path("outputs/analysis/degradation_analysis.jsonl")
    output_csv: Path = Path("outputs/analysis/degradation_analysis.csv")
    restored_output_jsonl: Path = Path("outputs/analysis/restored_analysis.jsonl")
    restored_output_csv: Path = Path("outputs/analysis/restored_analysis.csv")
    baseline_output_dir: Path = Path("outputs/baseline")
    wavelet: str = "db2"
    wavelet_level: int = Field(default=2, ge=1, le=4)
    jpeg_block_size: int = Field(default=8, ge=2)


class RestorationRuleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    degradation: DegradationKind
    algorithm: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class RestorationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    output_root: Path = Path("outputs/restored")
    manifest_path: Path = Path("outputs/manifests/restored.jsonl")
    rules: list[RestorationRuleConfig]


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    perceptual_hamming_threshold: int = Field(default=4, ge=0, le=16)
    minimum_images_per_modality: int = Field(default=20, ge=1)
    severity_imbalance_tolerance: float = Field(default=0.20, ge=0.0, le=1.0)
    minimum_monotonic_fraction: float = Field(default=0.80, ge=0.0, le=1.0)
    fail_on_near_duplicate: bool = True
    license_allowlist: set[str] = Field(
        default_factory=lambda: {
            "CC-BY-4.0",
            "CC-BY-SA-4.0",
            "CC0-1.0",
            "MIT",
            "Public Domain",
        }
    )
    report_path: Path = Path("outputs/reports/validation.json")
    overlap_report_path: Path = Path("outputs/reports/pool_overlap.json")


class ReportingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_root: Path = Path("outputs/reports")
    contact_sheet_examples_per_group: int = Field(default=2, ge=1, le=10)
    figure_dpi: int = Field(default=180, ge=72, le=600)
    bootstrap_resamples: int = Field(default=2000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)


class ProjectConfig(BaseModel):
    """Complete, immutable experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    config_version: Literal["1.0"] = "1.0"
    project_name: str = "micro-q-instruct-phase1"
    pools: list[PoolConfig]
    microbench_export: MicroBenchExportConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    degradations: list[DegradationConfig]
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    restoration: RestorationConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    @model_validator(mode="after")
    def require_unique_entities(self) -> ProjectConfig:
        pool_names = [pool.name for pool in self.pools]
        kinds = [item.kind for item in self.degradations]
        if len(pool_names) != len(set(pool_names)):
            raise ValueError("Pool names must be unique.")
        if len(kinds) != len(set(kinds)):
            raise ValueError("Each degradation kind may be configured once.")
        return self
