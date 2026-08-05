"""Exact and perceptual train/evaluation overlap detection."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from microq_phase1.core.config import PoolConfig
from microq_phase1.domain.models import ImageRecord, PoolRole
from microq_phase1.infrastructure.hashing import ImageSignature


class CrossPoolOverlapDetector:
    """Compare external and evaluation pools without quadratic Python loops."""

    def exact(
        self,
        signatures: dict[str, list[ImageSignature]],
        pools: dict[str, tuple[PoolConfig, list[ImageRecord]]],
    ) -> list[dict[str, object]]:
        by_hash: dict[str, list[ImageSignature]] = defaultdict(list)
        for items in signatures.values():
            for item in items:
                by_hash[item.pixel_sha256].append(item)
        results: list[dict[str, object]] = []
        for pixel_hash, items in by_hash.items():
            roles = {pools[item.pool_name][0].role for item in items}
            if PoolRole.EVALUATION not in roles or len(roles) == 1:
                continue
            results.append(
                {
                    "pixel_sha256": pixel_hash,
                    "matches": [
                        {"pool": item.pool_name, "image_id": item.image_id} for item in items
                    ],
                }
            )
        return results

    def near(
        self,
        signatures: dict[str, list[ImageSignature]],
        pools: dict[str, tuple[PoolConfig, list[ImageRecord]]],
        threshold: int,
    ) -> list[dict[str, object]]:
        evaluation, non_evaluation = self._separate_roles(signatures, pools)
        if not evaluation or not non_evaluation:
            return []
        eval_bits = np.stack([_hash_bits(item.perceptual_hash) for item in evaluation])
        other_bits = np.stack([_hash_bits(item.perceptual_hash) for item in non_evaluation])
        neighbors = NearestNeighbors(n_neighbors=1, metric="hamming", algorithm="brute")
        neighbors.fit(eval_bits)
        distances, indices = neighbors.kneighbors(other_bits)
        results: list[dict[str, object]] = []
        for source, distance, index in zip(
            non_evaluation,
            distances[:, 0],
            indices[:, 0],
            strict=True,
        ):
            hamming = int(round(float(distance) * eval_bits.shape[1]))
            if hamming > threshold:
                continue
            target = evaluation[int(index)]
            results.append(
                {
                    "hamming_distance": hamming,
                    "non_evaluation": {"pool": source.pool_name, "image_id": source.image_id},
                    "evaluation": {"pool": target.pool_name, "image_id": target.image_id},
                }
            )
        return results

    @staticmethod
    def _separate_roles(
        signatures: dict[str, list[ImageSignature]],
        pools: dict[str, tuple[PoolConfig, list[ImageRecord]]],
    ) -> tuple[list[ImageSignature], list[ImageSignature]]:
        evaluation: list[ImageSignature] = []
        non_evaluation: list[ImageSignature] = []
        for pool_name, items in signatures.items():
            destination = (
                evaluation
                if pools[pool_name][0].role == PoolRole.EVALUATION
                else non_evaluation
            )
            destination.extend(items)
        return evaluation, non_evaluation


def _hash_bits(value: str) -> NDArray[np.uint8]:
    integer = int(value, 16)
    return np.asarray([(integer >> offset) & 1 for offset in range(64)], dtype=np.uint8)
