"""Apply degradation-matched classical restoration in reverse corruption order."""

from __future__ import annotations

import shutil
from collections.abc import Iterator

from microq_phase1.core.config import ProjectConfig, RestorationRuleConfig
from microq_phase1.core.logging import get_logger
from microq_phase1.domain.models import DegradationKind, RecipeRecord, RestorationStep
from microq_phase1.infrastructure.hashing import ImageHasher
from microq_phase1.infrastructure.images import PillowImageRepository
from microq_phase1.infrastructure.jsonl import read_jsonl, write_jsonl_atomic
from microq_phase1.restorations import RestorationRegistry

LOGGER = get_logger(__name__)


class RestorationService:
    """Create restored counterparts while retaining degradation provenance."""

    def __init__(
        self,
        images: PillowImageRepository,
        registry: RestorationRegistry,
        hasher: ImageHasher,
    ) -> None:
        self._images = images
        self._registry = registry
        self._hasher = hasher

    def restore(self, config: ProjectConfig) -> int:
        rules = {rule.degradation: rule for rule in config.restoration.rules}
        count = 0

        def restored() -> Iterator[RecipeRecord]:
            nonlocal count
            for recipe in read_jsonl(config.generation.generated_manifest, RecipeRecord):
                result = self._restore_one(recipe, config, rules)
                count += 1
                if count % config.generation.progress_interval == 0:
                    LOGGER.info("restoration_progress", completed=count)
                yield result

        write_jsonl_atomic(config.restoration.manifest_path, restored())
        LOGGER.info(
            "restoration_completed",
            count=count,
            manifest=str(config.restoration.manifest_path),
        )
        return count

    def _restore_one(
        self,
        recipe: RecipeRecord,
        config: ProjectConfig,
        rules: dict[DegradationKind, RestorationRuleConfig],
    ) -> RecipeRecord:
        output_path = (
            config.restoration.output_root
            / recipe.pool_name
            / f"{recipe.output_path.stem}__restored.png"
        ).resolve()
        if not output_path.is_relative_to(config.restoration.output_root.resolve()):
            raise ValueError(f"Restoration output escapes configured root: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if recipe.is_clean_control:
            shutil.copyfile(recipe.output_path, output_path)
            steps: list[RestorationStep] = []
        else:
            input_record = recipe.source.model_copy(update={"image_path": recipe.output_path})
            frame = self._images.load(input_record)
            steps = []
            for degradation in reversed(recipe.steps):
                rule = rules.get(degradation.kind)
                if rule is None:
                    LOGGER.warning(
                        "restoration_rule_missing",
                        recipe_id=recipe.recipe_id,
                        degradation=degradation.kind,
                    )
                    continue
                strategy = self._registry.get(rule.algorithm)
                frame = strategy.restore(frame, degradation.parameters, rule.parameters)
                steps.append(
                    RestorationStep(
                        degradation=degradation.kind,
                        algorithm=rule.algorithm,
                        parameters=rule.parameters,
                    )
                )
            self._images.save(frame, output_path)
        digest = self._hasher.file_sha256(output_path)
        return recipe.model_copy(
            update={
                "output_path": output_path,
                "output_sha256": digest,
                "restoration_steps": steps,
                "generated": True,
            }
        )
