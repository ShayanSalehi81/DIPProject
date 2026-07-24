from __future__ import annotations

from pathlib import Path

import pytest

from microq_phase1.core.config import PoolConfig
from microq_phase1.core.errors import PoolPolicyError
from microq_phase1.domain.models import PoolRole
from microq_phase1.domain.policy import StrictTwoPoolPolicy


def test_policy_accepts_external_training_and_benchmark_evaluation() -> None:
    pools = [
        PoolConfig(
            name="external",
            role=PoolRole.TRAINING,
            catalog_path=Path("training.jsonl"),
            official_benchmark=False,
        ),
        PoolConfig(
            name="microbench",
            role=PoolRole.EVALUATION,
            catalog_path=Path("evaluation.jsonl"),
            official_benchmark=True,
        ),
    ]
    StrictTwoPoolPolicy().validate_pool_configs(pools)


def test_policy_rejects_benchmark_training() -> None:
    pools = [
        PoolConfig(
            name="microbench-train",
            role=PoolRole.TRAINING,
            catalog_path=Path("training.jsonl"),
            official_benchmark=True,
        ),
        PoolConfig(
            name="evaluation",
            role=PoolRole.EVALUATION,
            catalog_path=Path("evaluation.jsonl"),
            official_benchmark=True,
        ),
    ]
    with pytest.raises(PoolPolicyError):
        StrictTwoPoolPolicy().validate_pool_configs(pools)

