from __future__ import annotations

import pandas as pd

from microq_robustness.statistics import (
    accuracy_table,
    paired_tests,
    transition_table,
)


def _wide() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d", "e", "f"],
            "task_type": ["domain", "domain", "modality", "classification", "stain", "stain"],
            "task_group": [
                "coarse_perception",
                "coarse_perception",
                "coarse_perception",
                "fine_perception",
                "fine_perception",
                "fine_perception",
            ],
            "clean": [1, 1, 1, 1, 0, 0],
            "degraded": [0, 1, 0, 0, 0, 0],
            "restored": [1, 1, 0, 1, 0, 0],
        }
    )


def test_accuracy_table_preserves_condition_order() -> None:
    result = accuracy_table(_wide(), resamples=100, confidence_level=0.95, seed=4)
    overall = result.loc[result["scope"] == "all"].set_index("condition")
    assert overall.loc["clean", "accuracy"] > overall.loc["degraded", "accuracy"]
    assert overall.loc["restored", "accuracy"] > overall.loc["degraded", "accuracy"]


def test_transition_table_counts_each_sample_once_per_scope() -> None:
    transitions = transition_table(_wide())
    overall = transitions.loc[transitions["scope"] == "all"]
    assert overall["count"].sum() == 6
    assert abs(overall["fraction"].sum() - 1.0) < 1e-9


def test_mcnemar_uses_discordant_pairs() -> None:
    tests = paired_tests(_wide()).set_index("comparison")
    row = tests.loc["clean_vs_degraded"]
    assert row["discordant_pairs"] == 3
    assert 0 <= row["p_value_holm"] <= 1

