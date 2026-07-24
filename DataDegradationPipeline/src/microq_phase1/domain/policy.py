"""Strict two-pool safety policy."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from microq_phase1.core.errors import PoolPolicyError
from microq_phase1.domain.models import ImageRecord, PoolRole


class PoolDescriptor(Protocol):
    name: str
    role: PoolRole
    official_benchmark: bool


class StrictTwoPoolPolicy:
    """Prevent official benchmark data from influencing train/development artifacts."""

    forbidden_benchmark_roles = {PoolRole.TRAINING, PoolRole.DEVELOPMENT}

    def validate_pool_configs(self, pools: Iterable[PoolDescriptor]) -> None:
        configured = list(pools)
        if not any(pool.role == PoolRole.TRAINING for pool in configured):
            raise PoolPolicyError("At least one external training pool is required.")
        if not any(pool.role == PoolRole.EVALUATION for pool in configured):
            raise PoolPolicyError("At least one evaluation pool is required.")
        violations = [
            pool.name
            for pool in configured
            if pool.official_benchmark and pool.role in self.forbidden_benchmark_roles
        ]
        if violations:
            names = ", ".join(sorted(violations))
            raise PoolPolicyError(f"Official benchmark pools cannot train or tune models: {names}")

    def validate_records(
        self,
        pool: PoolDescriptor,
        records: Iterable[ImageRecord],
    ) -> None:
        for record in records:
            if record.pool_role != pool.role:
                raise PoolPolicyError(
                    f"Record {record.image_id} role {record.pool_role} "
                    f"disagrees with pool {pool.name}."
                )
            if pool.official_benchmark and record.pool_role != PoolRole.EVALUATION:
                raise PoolPolicyError(
                    f"Benchmark record {record.image_id} is not evaluation-only."
                )
