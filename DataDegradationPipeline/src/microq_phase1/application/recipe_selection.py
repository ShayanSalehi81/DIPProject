"""Source stratification and modality-aware degradation selection."""

from __future__ import annotations

import numpy as np

from microq_phase1.core.config import DegradationConfig
from microq_phase1.domain.models import ImageRecord


class RecipeCandidateSelector:
    """Keep pilot samples representative and enforce degradation applicability."""

    def stratified_limit(
        self,
        records: list[ImageRecord],
        limit: int,
        seed: int,
    ) -> list[ImageRecord]:
        if limit >= len(records):
            return records
        buckets: dict[tuple[str, str], list[ImageRecord]] = {}
        for record in records:
            key = (str(record.modality), record.source_dataset)
            buckets.setdefault(key, []).append(record)
        rng = np.random.default_rng(seed)
        for key, items in buckets.items():
            order = rng.permutation(len(items))
            buckets[key] = [items[int(index)] for index in order]
        selected: list[ImageRecord] = []
        ordered_keys = sorted(buckets)
        while len(selected) < limit and ordered_keys:
            remaining: list[tuple[str, str]] = []
            for key in ordered_keys:
                if not buckets[key]:
                    continue
                selected.append(buckets[key].pop())
                if buckets[key]:
                    remaining.append(key)
                if len(selected) == limit:
                    break
            ordered_keys = remaining
        return selected

    @staticmethod
    def applicable(
        configured: list[DegradationConfig],
        source: ImageRecord,
    ) -> list[DegradationConfig]:
        candidates: list[DegradationConfig] = []
        for item in configured:
            if not item.enabled or source.modality not in item.applicable_modalities:
                continue
            if item.requires_multichannel and not source.is_multichannel:
                continue
            candidates.append(item)
        return candidates

    @staticmethod
    def different_kind(
        eligible: list[DegradationConfig],
        primary: DegradationConfig,
        start: int,
    ) -> DegradationConfig:
        for offset in range(len(eligible)):
            candidate = eligible[(start + offset) % len(eligible)]
            if candidate.kind != primary.kind:
                return candidate
        return primary
