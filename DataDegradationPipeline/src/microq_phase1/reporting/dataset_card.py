"""Generate a release-oriented dataset card from immutable manifests."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

from microq_phase1.core.config import ProjectConfig
from microq_phase1.domain.models import RecipeRecord
from microq_phase1.infrastructure.jsonl import read_jsonl


class DatasetCardBuilder:
    """Summarize provenance, composition, intended use, and limitations."""

    def build(self, config: ProjectConfig, output_path: Path) -> None:
        recipes = list(read_jsonl(config.generation.generated_manifest, RecipeRecord))
        unique_sources = {
            (recipe.pool_name, recipe.source.image_id): recipe.source for recipe in recipes
        }
        roles = Counter(record.pool_role for record in unique_sources.values())
        modalities = Counter(record.modality for record in unique_sources.values())
        licenses = Counter(record.license for record in unique_sources.values())
        degradations = Counter(
            "+".join(step.kind for step in recipe.steps) or "clean" for recipe in recipes
        )
        body = self._render(
            config=config,
            recipe_count=len(recipes),
            source_count=len(unique_sources),
            roles=roles,
            modalities=modalities,
            licenses=licenses,
            degradations=degradations,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(body, encoding="utf-8", newline="\n")
        os.replace(temporary, output_path)

    @staticmethod
    def _render(
        *,
        config: ProjectConfig,
        recipe_count: int,
        source_count: int,
        roles: Counter[Any],
        modalities: Counter[Any],
        licenses: Counter[Any],
        degradations: Counter[Any],
    ) -> str:
        return f"""# Micro-Q-Instruct Phase 1 Dataset Card

## Version and provenance

- Configuration version: `{config.config_version}`
- Master seed: `{config.generation.master_seed}`
- Micro-Bench dataset ID: `{config.microbench_export.dataset_id}`
- Micro-Bench immutable revision: `{config.microbench_export.revision}`
- Pool policy: strict external-train / official-evaluation separation

## Composition

- Unique clean source images: {source_count}
- Materialized recipe records: {recipe_count}
- Source roles: `{_serializable_counts(roles)}`
- Source modalities: `{_serializable_counts(modalities)}`
- Source licenses: `{_serializable_counts(licenses)}`
- Degradation records: `{_serializable_counts(degradations)}`

## Intended uses

This release is intended for classical microscopy degradation analysis, severity calibration,
restoration evaluation, and later degradation-aware vision-language instruction construction.
Official Micro-Bench-derived records are evaluation-only.

## Out-of-scope uses

The release is not a diagnostic medical device, does not establish biological ground truth, and
must not be used to claim independent Micro-Bench performance after benchmark images have entered
training.

## Known limitations

Degradations are controlled approximations. Photobleaching is represented as retained signal,
defocus uses a disk PSF, periodic restoration uses the known synthetic frequency, and reference
morphology is available only for catalog records with aligned masks. Severity parameters are
empirical and require recalibration when the source or acquisition distribution changes.

## Licensing

Code licensing does not relicense source images. Each record retains its original source dataset,
license, and metadata. Consult the source license before redistributing generated derivatives.

## Reproduction

Archive this card with the frozen YAML configuration, generated manifest, pool-overlap report,
validation report, structured logs, and run-provenance JSON.
"""


def _serializable_counts(values: Counter[Any]) -> dict[str, int]:
    ordered = sorted(values.items(), key=lambda item: str(item[0]))
    return {str(key): count for key, count in ordered}
