"""Materialize deterministic degradation recipes."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from microq_phase1.core.config import ProjectConfig
from microq_phase1.core.errors import DegradationError
from microq_phase1.core.logging import get_logger
from microq_phase1.degradations import DegradationRegistry
from microq_phase1.domain.models import RecipeRecord
from microq_phase1.infrastructure.hashing import ImageHasher
from microq_phase1.infrastructure.images import PillowImageRepository
from microq_phase1.infrastructure.jsonl import read_jsonl, write_jsonl_atomic

LOGGER = get_logger(__name__)


class GenerationService:
    """Apply planned strategies and persist immutable generated artifacts."""

    def __init__(
        self,
        images: PillowImageRepository,
        registry: DegradationRegistry,
        hasher: ImageHasher,
    ) -> None:
        self._images = images
        self._registry = registry
        self._hasher = hasher

    def materialize(self, config: ProjectConfig) -> int:
        count = 0

        def generated() -> Iterator[RecipeRecord]:
            nonlocal count
            for recipe in read_jsonl(config.generation.recipe_manifest, RecipeRecord):
                completed = self._materialize_one(recipe, config.generation.output_root)
                count += 1
                if count % config.generation.progress_interval == 0:
                    LOGGER.info("generation_progress", completed=count)
                yield completed

        write_jsonl_atomic(config.generation.generated_manifest, generated())
        LOGGER.info(
            "generation_completed",
            count=count,
            manifest=str(config.generation.generated_manifest),
        )
        return count

    def _materialize_one(self, recipe: RecipeRecord, output_root: Path) -> RecipeRecord:
        resolved_output = recipe.output_path.resolve()
        if not resolved_output.is_relative_to(output_root.resolve()):
            raise DegradationError(f"Recipe output escapes configured root: {resolved_output}")
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        if recipe.is_clean_control:
            shutil.copyfile(recipe.source.image_path, resolved_output)
        else:
            frame = self._images.load(recipe.source)
            for step in recipe.steps:
                strategy = self._registry.get(step.kind)
                if not strategy.supports(frame):
                    raise DegradationError(
                        f"{step.kind} does not support source image {recipe.source.image_id}."
                    )
                try:
                    frame = strategy.apply(
                        frame,
                        step.parameters,
                        np.random.default_rng(step.seed),
                    )
                except Exception as exc:
                    raise DegradationError(
                        f"Failed {step.kind} for recipe {recipe.recipe_id} "
                        f"(source {recipe.source.image_id})."
                    ) from exc
            self._images.save(frame, resolved_output)
        digest = self._hasher.file_sha256(resolved_output)
        LOGGER.debug(
            "recipe_materialized",
            recipe_id=recipe.recipe_id,
            source_image_id=recipe.source.image_id,
            output=str(resolved_output),
            degradations=[step.kind for step in recipe.steps],
        )
        return recipe.model_copy(
            update={
                "output_path": resolved_output,
                "output_sha256": digest,
                "generated": True,
            }
        )
