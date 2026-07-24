"""Dependency composition and guarded CLI execution."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import typer

from microq_phase1.analysis import CompositeFeatureExtractor, QualityMetricCalculator
from microq_phase1.analysis.morphology import MorphologyMetricCalculator
from microq_phase1.application.analysis import AnalysisService
from microq_phase1.application.audit import PoolAuditService
from microq_phase1.application.baseline import FingerprintBaselineService
from microq_phase1.application.configuration import ExperimentConfigurationValidator
from microq_phase1.application.generation import GenerationService
from microq_phase1.application.overlap import CrossPoolOverlapDetector
from microq_phase1.application.planning import RecipePlanner
from microq_phase1.application.quality_gates import DatasetQualityGateEvaluator
from microq_phase1.application.recipe_selection import RecipeCandidateSelector
from microq_phase1.application.restoration import RestorationService
from microq_phase1.application.validation import DatasetValidationService
from microq_phase1.core.config import ProjectConfig
from microq_phase1.core.config_io import load_config
from microq_phase1.core.errors import MicroQError
from microq_phase1.core.logging import configure_logging, get_logger, logged_stage
from microq_phase1.degradations import DegradationRegistry
from microq_phase1.domain.policy import StrictTwoPoolPolicy
from microq_phase1.infrastructure.hashing import ImageHasher
from microq_phase1.infrastructure.images import PillowImageRepository
from microq_phase1.infrastructure.provenance import RunProvenanceRecorder
from microq_phase1.reporting import ReportingService
from microq_phase1.restorations import RestorationRegistry

ResultT = TypeVar("ResultT")


def build_services(config: ProjectConfig) -> dict[str, Any]:
    """Compose concrete adapters at the outer application boundary."""
    images = PillowImageRepository()
    hasher = ImageHasher()
    restoration_registry = RestorationRegistry()
    return {
        "config": ExperimentConfigurationValidator(
            StrictTwoPoolPolicy(),
            restoration_registry,
        ),
        "audit": PoolAuditService(
            StrictTwoPoolPolicy(),
            hasher,
            CrossPoolOverlapDetector(),
        ),
        "planner": RecipePlanner(
            StrictTwoPoolPolicy(),
            hasher,
            RecipeCandidateSelector(),
        ),
        "generation": GenerationService(images, DegradationRegistry(), hasher),
        "analysis": AnalysisService(
            images,
            CompositeFeatureExtractor(
                wavelet=config.analysis.wavelet,
                wavelet_level=config.analysis.wavelet_level,
                jpeg_block_size=config.analysis.jpeg_block_size,
            ),
            QualityMetricCalculator(),
            MorphologyMetricCalculator(),
        ),
        "restoration": RestorationService(images, restoration_registry, hasher),
        "validation": DatasetValidationService(hasher, DatasetQualityGateEvaluator()),
        "baseline": FingerprintBaselineService(),
        "reporting": ReportingService(),
    }


def execute(
    config_path: Path,
    command: str,
    action: Callable[[ProjectConfig, dict[str, Any]], ResultT],
) -> ResultT:
    """Load configuration, bind provenance/log context, and normalize expected errors."""
    try:
        config = load_config(config_path)
        run_id = configure_logging(
            level=config.logging.level,
            json_log_path=config.logging.json_log_path,
            rotate_megabytes=config.logging.rotate_megabytes,
            backup_count=config.logging.backup_count,
        )
        logger = get_logger(__name__).bind(command=command)
        RunProvenanceRecorder().record(
            run_id=run_id,
            command=command,
            config_path=config_path,
            output_directory=config.logging.json_log_path.parent / "runs",
        )
        services = build_services(config)
        services["config"].validate(config)
        with logged_stage(logger, command):
            result = action(config, services)
        typer.echo(f"{command} completed successfully.")
        return result
    except (MicroQError, ValueError, FileNotFoundError) as exc:
        typer.echo(f"{command} failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
