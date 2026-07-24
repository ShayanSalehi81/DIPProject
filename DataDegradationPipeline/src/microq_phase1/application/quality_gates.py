"""Distribution, coverage, and severity-calibration quality gates."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from microq_phase1.core.config import ProjectConfig
from microq_phase1.domain.models import (
    AnalysisRecord,
    DegradationKind,
    Modality,
    RecipeRecord,
    Severity,
)


class DatasetQualityGateEvaluator:
    """Evaluate scientific balance separately from filesystem integrity."""

    def validate_modality_coverage(
        self,
        config: ProjectConfig,
        recipes: list[RecipeRecord],
        warnings: list[str],
    ) -> None:
        unique_sources: dict[tuple[object, Modality], set[str]] = defaultdict(set)
        for recipe in recipes:
            unique_sources[(recipe.pool_role, recipe.source.modality)].add(
                recipe.source.image_id
            )
        roles = {recipe.pool_role for recipe in recipes}
        for role in roles:
            for modality in (Modality.LIGHT, Modality.FLUORESCENCE, Modality.ELECTRON):
                count = len(unique_sources[(role, modality)])
                if count < config.validation.minimum_images_per_modality:
                    warnings.append(
                        f"Role '{role}', modality '{modality}' has {count} unique images; "
                        f"target is {config.validation.minimum_images_per_modality}."
                    )

    def validate_severity_balance(
        self,
        config: ProjectConfig,
        recipes: list[RecipeRecord],
        warnings: list[str],
    ) -> None:
        counts: dict[DegradationKind, Counter[Severity]] = defaultdict(Counter)
        for recipe in recipes:
            if len(recipe.steps) != 1:
                continue
            step = recipe.steps[0]
            counts[step.kind][step.severity] += 1
        tolerance = config.validation.severity_imbalance_tolerance
        for kind, severity_counts in counts.items():
            values = [severity_counts[level] for level in Severity]
            mean = sum(values) / len(values)
            if mean == 0:
                continue
            maximum_deviation = max(abs(value - mean) / mean for value in values)
            if maximum_deviation > tolerance:
                warnings.append(
                    f"Severity imbalance for '{kind}' is {maximum_deviation:.3f}; "
                    f"tolerance is {tolerance:.3f}."
                )

    @staticmethod
    def severity_monotonicity(analyses: list[AnalysisRecord]) -> dict[str, Any]:
        grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        for item in analyses:
            if item.step_count != 1 or item.degradation == "clean":
                continue
            grouped[(item.source_image_id, item.degradation)][item.severity] = (
                item.quality_metrics["psnr"]
            )
        order = [Severity.WEAK, Severity.MEDIUM, Severity.STRONG]
        eligible = 0
        monotonic = 0
        for values in grouped.values():
            if not all(level in values for level in order):
                continue
            eligible += 1
            if values[Severity.WEAK] >= values[Severity.MEDIUM] >= values[Severity.STRONG]:
                monotonic += 1
        return {
            "eligible_groups": eligible,
            "monotonic_groups": monotonic,
            "monotonic_fraction": monotonic / eligible if eligible else None,
            "criterion": "PSNR(weak) >= PSNR(medium) >= PSNR(strong)",
        }

    @staticmethod
    def distribution(recipes: list[RecipeRecord]) -> dict[str, object]:
        return {
            "pool_roles": dict(Counter(recipe.pool_role for recipe in recipes)),
            "modalities": dict(Counter(recipe.source.modality for recipe in recipes)),
            "degradations": dict(
                Counter(
                    "+".join(step.kind for step in recipe.steps) or "clean"
                    for recipe in recipes
                )
            ),
            "severities": dict(
                Counter(
                    "+".join(step.severity for step in recipe.steps) or "clean"
                    for recipe in recipes
                )
            ),
        }

