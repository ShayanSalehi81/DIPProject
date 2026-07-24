"""Deterministic group-aware partitioning for external microscopy catalogs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from microq_phase1.core.logging import get_logger
from microq_phase1.domain.models import ImageRecord, PoolRole
from microq_phase1.infrastructure.catalogs import JsonlImageCatalog
from microq_phase1.infrastructure.jsonl import write_jsonl_atomic

LOGGER = get_logger(__name__)


class ExternalCatalogSplitter:
    """Split external images without allowing groups to cross train/development."""

    def split(
        self,
        *,
        input_catalog: Path,
        training_output: Path,
        development_output: Path,
        development_fraction: float,
        seed: int,
        group_by_source_dataset: bool,
    ) -> tuple[int, int]:
        if development_fraction <= 0 or development_fraction >= 0.5:
            raise ValueError("development_fraction must be in (0, 0.5).")
        records = list(JsonlImageCatalog(input_catalog).records())
        grouped: dict[str, list[ImageRecord]] = defaultdict(list)
        for record in records:
            group = (
                record.source_dataset
                if group_by_source_dataset
                else (record.group_id or record.image_id)
            )
            grouped[group].append(record)
        keys = sorted(grouped)
        order = np.random.default_rng(seed).permutation(len(keys))
        keys = [keys[int(index)] for index in order]
        development_target = round(len(records) * development_fraction)
        development_keys: set[str] = set()
        development_count = 0
        for key in keys:
            if development_count >= development_target:
                break
            development_keys.add(key)
            development_count += len(grouped[key])
        training: list[ImageRecord] = []
        development: list[ImageRecord] = []
        for key, items in grouped.items():
            destination = development if key in development_keys else training
            role = PoolRole.DEVELOPMENT if key in development_keys else PoolRole.TRAINING
            destination.extend(item.model_copy(update={"pool_role": role}) for item in items)
        write_jsonl_atomic(training_output, training)
        write_jsonl_atomic(development_output, development)
        LOGGER.info(
            "external_catalog_split",
            training=len(training),
            development=len(development),
            groups=len(grouped),
            grouping="source_dataset" if group_by_source_dataset else "group_id",
        )
        return len(training), len(development)
