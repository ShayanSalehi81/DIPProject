"""Quality gates for generated manifests and calibrated severity behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from microq_phase1.core.config import ProjectConfig
from microq_phase1.core.errors import ValidationFailure
from microq_phase1.core.logging import get_logger
from microq_phase1.application.quality_gates import DatasetQualityGateEvaluator
from microq_phase1.domain.models import AnalysisRecord, RecipeRecord
from microq_phase1.infrastructure.hashing import ImageHasher
from microq_phase1.infrastructure.jsonl import read_jsonl, write_json_atomic

LOGGER = get_logger(__name__)


class DatasetValidationService:
    """Validate completeness, balance, provenance, and severity monotonicity."""

    def __init__(
        self,
        hasher: ImageHasher,
        quality_gates: DatasetQualityGateEvaluator,
    ) -> None:
        self._hasher = hasher
        self._quality_gates = quality_gates

    def validate(
        self,
        config: ProjectConfig,
        *,
        manifest_path: Path | None = None,
        analysis_path: Path | None = None,
    ) -> dict[str, Any]:
        recipes = list(
            read_jsonl(
                manifest_path or config.generation.generated_manifest,
                RecipeRecord,
            )
        )
        analyses = list(
            read_jsonl(
                analysis_path or config.analysis.output_jsonl,
                AnalysisRecord,
            )
        )
        errors: list[str] = []
        warnings: list[str] = []
        self._validate_recipes(recipes, errors, warnings)
        self._validate_checksums(recipes, errors)
        self._quality_gates.validate_modality_coverage(config, recipes, warnings)
        self._quality_gates.validate_severity_balance(config, recipes, warnings)
        monotonicity = self._quality_gates.severity_monotonicity(analyses)
        if monotonicity["eligible_groups"] == 0:
                warnings.append(
                    "No complete single-degradation severity triplets were available."
                )
        elif (
            monotonicity["monotonic_fraction"]
            < config.validation.minimum_monotonic_fraction
        ):
            errors.append(
                "Severity monotonicity is below the configured minimum: "
                f"{monotonicity['monotonic_fraction']:.3f} < "
                f"{config.validation.minimum_monotonic_fraction:.3f}."
            )
        report = {
            "status": "failed" if errors else "passed",
            "errors": errors,
            "warnings": warnings,
            "recipe_count": len(recipes),
            "analysis_count": len(analyses),
            "severity_monotonicity": monotonicity,
            "distribution": self._quality_gates.distribution(recipes),
        }
        write_json_atomic(config.validation.report_path, report)
        LOGGER.info(
            "dataset_validation_completed",
            status=report["status"],
            errors=len(errors),
            warnings=len(warnings),
            report=str(config.validation.report_path),
        )
        if errors:
            raise ValidationFailure("Dataset validation failed; inspect the validation report.")
        return report

    @staticmethod
    def _validate_recipes(
        recipes: list[RecipeRecord],
        errors: list[str],
        warnings: list[str],
    ) -> None:
        recipe_ids = [recipe.recipe_id for recipe in recipes]
        if len(recipe_ids) != len(set(recipe_ids)):
            errors.append("Recipe IDs are not unique.")
        missing = [
            str(recipe.output_path)
            for recipe in recipes
            if not recipe.output_path.is_file()
        ]
        if missing:
            errors.append(f"{len(missing)} materialized image files are missing.")
        incomplete = [recipe.recipe_id for recipe in recipes if not recipe.generated]
        if incomplete:
            errors.append(f"{len(incomplete)} recipes are not marked generated.")
        if not recipes:
            errors.append("Generated manifest is empty.")
            return
        clean_fraction = sum(recipe.is_clean_control for recipe in recipes) / len(recipes)
        if clean_fraction == 0:
            warnings.append("No clean controls are present.")

    def _validate_checksums(
        self,
        recipes: list[RecipeRecord],
        errors: list[str],
    ) -> None:
        source_hashes: dict[tuple[str, str], str] = {}
        output_mismatches = 0
        source_mismatches = 0
        for recipe in recipes:
            if recipe.output_path.is_file():
                actual = self._hasher.file_sha256(recipe.output_path)
                if recipe.output_sha256 is None or actual != recipe.output_sha256:
                    output_mismatches += 1
            source_key = (recipe.pool_name, recipe.source.image_id)
            if source_key not in source_hashes:
                signature = self._hasher.signature(
                    recipe.source.image_id,
                    recipe.pool_name,
                    recipe.source.image_path,
                )
                source_hashes[source_key] = signature.pixel_sha256
            if recipe.clean_sha256 != source_hashes[source_key]:
                source_mismatches += 1
        if output_mismatches:
            errors.append(f"{output_mismatches} output checksums do not match.")
        if source_mismatches:
            errors.append(f"{source_mismatches} recipes have changed clean-source pixels.")
