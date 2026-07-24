"""Analyze generated images into a structured microscopy degradation fingerprint."""

from __future__ import annotations

import csv
import os
from collections.abc import Iterator
from pathlib import Path

from microq_phase1.analysis import CompositeFeatureExtractor, QualityMetricCalculator
from microq_phase1.analysis.morphology import MorphologyMetricCalculator
from microq_phase1.core.config import ProjectConfig
from microq_phase1.core.logging import get_logger
from microq_phase1.domain.models import AnalysisRecord, RecipeRecord
from microq_phase1.infrastructure.images import PillowImageRepository
from microq_phase1.infrastructure.jsonl import read_jsonl, write_jsonl_atomic

LOGGER = get_logger(__name__)


class AnalysisService:
    """Compute no-reference evidence and paired quality measurements."""

    def __init__(
        self,
        images: PillowImageRepository,
        features: CompositeFeatureExtractor,
        quality: QualityMetricCalculator,
        morphology: MorphologyMetricCalculator,
    ) -> None:
        self._images = images
        self._features = features
        self._quality = quality
        self._morphology = morphology

    def analyze(
        self,
        config: ProjectConfig,
        *,
        manifest_path: Path | None = None,
        output_jsonl: Path | None = None,
        output_csv: Path | None = None,
    ) -> int:
        source_manifest = manifest_path or config.generation.generated_manifest
        jsonl_path = output_jsonl or config.analysis.output_jsonl
        csv_path = output_csv or config.analysis.output_csv
        collected: list[AnalysisRecord] = []
        for index, recipe in enumerate(read_jsonl(source_manifest, RecipeRecord), start=1):
            try:
                collected.append(self._analyze_one(recipe))
            except Exception:
                LOGGER.exception(
                    "analysis_record_failed",
                    recipe_id=recipe.recipe_id,
                    source_image_id=recipe.source.image_id,
                )
                raise
            if index % config.generation.progress_interval == 0:
                LOGGER.info("analysis_progress", completed=index)
        write_jsonl_atomic(jsonl_path, collected)
        self._write_csv(csv_path, collected)
        LOGGER.info(
            "analysis_completed",
            records=len(collected),
            jsonl=str(jsonl_path),
            csv=str(csv_path),
        )
        return len(collected)

    def _analyze_one(self, recipe: RecipeRecord) -> AnalysisRecord:
        if not recipe.generated:
            raise ValueError(f"Recipe has not been materialized: {recipe.recipe_id}")
        clean = self._images.load(recipe.source)
        degraded_record = recipe.source.model_copy(update={"image_path": recipe.output_path})
        degraded = self._images.load(degraded_record)
        morphology: dict[str, float] = {}
        if recipe.source.mask_path is not None:
            mask = self._images.load_mask(recipe.source.mask_path)
            morphology = self._morphology.calculate(degraded, mask)
        degradation = "+".join(step.kind for step in recipe.steps) or "clean"
        severity = "+".join(step.severity for step in recipe.steps) or "clean"
        return AnalysisRecord(
            recipe_id=recipe.recipe_id,
            source_image_id=recipe.source.image_id,
            pool_role=recipe.pool_role,
            modality=recipe.source.modality,
            source_dataset=recipe.source.source_dataset,
            degradation=degradation,
            severity=severity,
            step_count=len(recipe.steps),
            restoration_algorithms=[step.algorithm for step in recipe.restoration_steps],
            features=self._features.extract(degraded),
            quality_metrics=self._quality.calculate(clean, degraded),
            morphology_metrics=morphology,
        )

    @staticmethod
    def _write_csv(path: Path, records: list[AnalysisRecord]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        rows = list(_flatten_records(records))
        fieldnames = sorted({key for row in rows for key in row})
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def _flatten_records(records: list[AnalysisRecord]) -> Iterator[dict[str, object]]:
    for record in records:
        row: dict[str, object] = {
            "recipe_id": record.recipe_id,
            "source_image_id": record.source_image_id,
            "pool_role": record.pool_role,
            "modality": record.modality,
            "source_dataset": record.source_dataset,
            "degradation": record.degradation,
            "severity": record.severity,
            "step_count": record.step_count,
            "restoration_algorithms": "+".join(record.restoration_algorithms),
        }
        row.update({f"feature.{key}": value for key, value in record.features.items()})
        row.update({f"quality.{key}": value for key, value in record.quality_metrics.items()})
        row.update(
            {f"morphology.{key}": value for key, value in record.morphology_metrics.items()}
        )
        yield row
