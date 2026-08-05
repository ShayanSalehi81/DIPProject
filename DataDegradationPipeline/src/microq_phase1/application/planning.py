"""Balanced deterministic planning of clean, single, and mixed degradation recipes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from uuid import NAMESPACE_URL, uuid5

from microq_phase1.application.recipe_selection import RecipeCandidateSelector
from microq_phase1.core.config import DegradationConfig, PoolConfig, ProjectConfig
from microq_phase1.core.logging import get_logger
from microq_phase1.core.randomness import derive_seed
from microq_phase1.domain.models import (
    DegradationStep,
    ImageRecord,
    RecipeRecord,
    Severity,
)
from microq_phase1.domain.policy import StrictTwoPoolPolicy
from microq_phase1.infrastructure.catalogs import JsonlImageCatalog
from microq_phase1.infrastructure.hashing import ImageHasher
from microq_phase1.infrastructure.jsonl import write_jsonl_atomic

LOGGER = get_logger(__name__)


class RecipePlanner:
    """Create a balanced incomplete factorial design without reading image pixels."""

    def __init__(
        self,
        policy: StrictTwoPoolPolicy,
        hasher: ImageHasher,
        selector: RecipeCandidateSelector,
    ) -> None:
        self._policy = policy
        self._hasher = hasher
        self._selector = selector

    def plan(self, config: ProjectConfig) -> int:
        self._policy.validate_pool_configs(config.pools)
        records = self._iter_all_recipes(config)
        count = 0

        def counted() -> Iterator[RecipeRecord]:
            nonlocal count
            for record in records:
                count += 1
                yield record

        write_jsonl_atomic(config.generation.recipe_manifest, counted())
        LOGGER.info(
            "recipes_planned",
            count=count,
            manifest=str(config.generation.recipe_manifest),
        )
        return count

    def _iter_all_recipes(self, config: ProjectConfig) -> Iterator[RecipeRecord]:
        for pool in config.pools:
            source_records = list(JsonlImageCatalog(pool.catalog_path).records())
            self._policy.validate_records(pool, source_records)
            limit = pool.max_images
            if limit is not None:
                source_records = self._selector.stratified_limit(
                    source_records,
                    limit,
                    derive_seed(config.generation.master_seed, pool.name, "sampling"),
                )
            for image_index, source in enumerate(source_records):
                yield from self._recipes_for_image(config, pool, source, image_index)

    def _recipes_for_image(
        self,
        config: ProjectConfig,
        pool: PoolConfig,
        source: ImageRecord,
        image_index: int,
    ) -> Iterator[RecipeRecord]:
        clean_hash = self._hasher.signature(source.image_id, pool.name, source.image_path)
        eligible = self._selector.applicable(config.degradations, source)
        if not eligible:
            return
        if self._include_clean_control(config, source):
            yield self._make_recipe(config, pool, source, [], clean_hash.pixel_sha256, "clean")
        total = config.generation.variants_per_image
        family_start = image_index % len(eligible)
        candidates = [
            (eligible[(family_start + family_offset) % len(eligible)], severity)
            for family_offset in range((total + len(Severity) - 1) // len(Severity))
            for severity in Severity
        ]
        for variant_index, primary in enumerate(candidates[:total]):
            steps = [self._make_step(config, source, primary, variant_index)]
            if self._include_mixed(config, source, variant_index) and len(eligible) > 1:
                secondary_config = self._selector.different_kind(
                    eligible,
                    primary[0],
                    family_start + variant_index + 1,
                )
                secondary = (secondary_config, primary[1])
                steps.append(self._make_step(config, source, secondary, variant_index + total))
            label = "__".join(f"{step.kind}-{step.severity}" for step in steps)
            yield self._make_recipe(
                config,
                pool,
                source,
                steps,
                clean_hash.pixel_sha256,
                label,
            )

    @staticmethod
    def _make_step(
        config: ProjectConfig,
        source: ImageRecord,
        candidate: tuple[DegradationConfig, Severity],
        variant_index: int,
    ) -> DegradationStep:
        item, severity = candidate
        seed = derive_seed(
            config.generation.master_seed,
            source.image_id,
            item.kind,
            severity,
            variant_index,
        )
        return DegradationStep(
            kind=item.kind,
            severity=severity,
            parameters=item.severity_parameters[severity],
            seed=seed,
        )

    @staticmethod
    def _include_clean_control(config: ProjectConfig, source: ImageRecord) -> bool:
        seed = derive_seed(config.generation.master_seed, source.image_id, "clean-control")
        return (seed / (2**32)) < config.generation.clean_control_fraction

    @staticmethod
    def _include_mixed(
        config: ProjectConfig,
        source: ImageRecord,
        variant_index: int,
    ) -> bool:
        seed = derive_seed(
            config.generation.master_seed,
            source.image_id,
            "mixed",
            variant_index,
        )
        return (seed / (2**32)) < config.generation.mixed_degradation_fraction

    @staticmethod
    def _make_recipe(
        config: ProjectConfig,
        pool: PoolConfig,
        source: ImageRecord,
        steps: list[DegradationStep],
        clean_sha256: str,
        label: str,
    ) -> RecipeRecord:
        serialized_steps = json.dumps(
            [step.model_dump(mode="json") for step in steps],
            sort_keys=True,
            separators=(",", ":"),
        )
        identity = f"{pool.name}:{source.image_id}:{label}:{serialized_steps}"
        recipe_id = str(uuid5(NAMESPACE_URL, identity))
        filename = f"{source.image_id}__{recipe_id[:12]}.png"
        output_path = config.generation.output_root / pool.name / filename
        return RecipeRecord(
            recipe_id=recipe_id,
            pool_name=pool.name,
            pool_role=pool.role,
            source=source,
            steps=steps,
            output_path=output_path,
            clean_sha256=clean_sha256,
        )
