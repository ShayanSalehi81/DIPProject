"""Paired, cluster-aware statistical summaries for the robustness experiment."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

CONDITION_LABELS = {"clean": "Clean", "degraded": "Degraded", "restored": "Restored"}
SCOPE_LABELS = {
    "all": "Overall",
    "coarse_perception": "Coarse perception",
    "fine_perception": "Fine perception",
}


def macro_accuracy(frame: pd.DataFrame, condition: str) -> float:
    """Macro-average task accuracies so frequent question types cannot dominate."""
    if frame.empty:
        return float("nan")
    return float(frame.groupby("task_type", observed=True)[condition].mean().mean())


def percentile_interval(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    *,
    resamples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Cluster bootstrap over base image-question samples (the pairing unit)."""
    estimate = metric(frame)
    if frame.empty:
        return estimate, float("nan"), float("nan")
    values = np.empty(resamples, dtype=float)
    row_count = len(frame)
    for index in range(resamples):
        draw = rng.integers(0, row_count, size=row_count)
        values[index] = metric(frame.iloc[draw])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return estimate, float("nan"), float("nan")
    alpha = 1.0 - confidence_level
    low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return estimate, float(low), float(high)


def accuracy_table(
    wide: pd.DataFrame,
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    scopes = ["all", "coarse_perception", "fine_perception"]
    for scope in scopes:
        subset = wide if scope == "all" else wide.loc[wide["task_group"] == scope]
        if subset.empty:
            continue
        for condition in ("clean", "degraded", "restored"):
            estimate, low, high = percentile_interval(
                subset,
                lambda data, current=condition: macro_accuracy(data, current),
                resamples=resamples,
                confidence_level=confidence_level,
                rng=rng,
            )
            rows.append(
                {
                    "scope": scope,
                    "scope_label": SCOPE_LABELS[scope],
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "n": len(subset),
                    "task_types": subset["task_type"].nunique(),
                    "accuracy": estimate,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def _effect(frame: pd.DataFrame, left: str, right: str) -> float:
    return macro_accuracy(frame, right) - macro_accuracy(frame, left)


def paired_effect_table(
    wide: pd.DataFrame,
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    definitions = {
        "degradation_effect": ("clean", "degraded"),
        "restoration_gain": ("degraded", "restored"),
        "residual_vs_clean": ("clean", "restored"),
    }
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 17)
    for scope in ("all", "coarse_perception", "fine_perception"):
        subset = wide if scope == "all" else wide.loc[wide["task_group"] == scope]
        if subset.empty:
            continue
        for effect_name, (left, right) in definitions.items():
            estimate, low, high = percentile_interval(
                subset,
                lambda data, a=left, b=right: _effect(data, a, b),
                resamples=resamples,
                confidence_level=confidence_level,
                rng=rng,
            )
            rows.append(
                {
                    "scope": scope,
                    "scope_label": SCOPE_LABELS[scope],
                    "effect": effect_name,
                    "left_condition": left,
                    "right_condition": right,
                    "n": len(subset),
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                }
            )

        def recovery(data: pd.DataFrame) -> float:
            clean = macro_accuracy(data, "clean")
            degraded = macro_accuracy(data, "degraded")
            restored = macro_accuracy(data, "restored")
            loss = clean - degraded
            return (restored - degraded) / loss if abs(loss) > 1e-12 else float("nan")

        estimate, low, high = percentile_interval(
            subset,
            recovery,
            resamples=resamples,
            confidence_level=confidence_level,
            rng=rng,
        )
        rows.append(
            {
                "scope": scope,
                "scope_label": SCOPE_LABELS[scope],
                "effect": "recovery_fraction",
                "left_condition": "degraded",
                "right_condition": "restored",
                "n": len(subset),
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def factor_breakdown(
    wide: pd.DataFrame,
    factor: str,
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + sum(ord(char) for char in factor))
    for value, subset in wide.groupby(factor, observed=True, sort=True):
        row: dict[str, Any] = {"factor": factor, "value": str(value), "n": len(subset)}
        for condition in ("clean", "degraded", "restored"):
            estimate, low, high = percentile_interval(
                subset,
                lambda data, current=condition: macro_accuracy(data, current),
                resamples=resamples,
                confidence_level=confidence_level,
                rng=rng,
            )
            row[f"{condition}_accuracy"] = estimate
            row[f"{condition}_ci_low"] = low
            row[f"{condition}_ci_high"] = high
        row["degradation_effect"] = row["degraded_accuracy"] - row["clean_accuracy"]
        row["restoration_gain"] = row["restored_accuracy"] - row["degraded_accuracy"]
        row["residual_vs_clean"] = row["restored_accuracy"] - row["clean_accuracy"]
        loss = row["clean_accuracy"] - row["degraded_accuracy"]
        row["recovery_fraction"] = (
            row["restoration_gain"] / loss if abs(loss) > 1e-12 else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(wide: pd.DataFrame) -> pd.DataFrame:
    """Exact two-sided McNemar tests plus Holm family-wise correction."""
    comparisons = [
        ("clean", "degraded"),
        ("degraded", "restored"),
        ("clean", "restored"),
    ]
    rows: list[dict[str, Any]] = []
    for left, right in comparisons:
        left_only = int(((wide[left] == 1) & (wide[right] == 0)).sum())
        right_only = int(((wide[left] == 0) & (wide[right] == 1)).sum())
        discordant = left_only + right_only
        p_value = (
            float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue)
            if discordant
            else 1.0
        )
        rows.append(
            {
                "comparison": f"{left}_vs_{right}",
                "left_correct_right_wrong": left_only,
                "left_wrong_right_correct": right_only,
                "discordant_pairs": discordant,
                "p_value": p_value,
            }
        )
    adjusted = holm_adjust([row["p_value"] for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["p_value_holm"] = value
    return pd.DataFrame(rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[int(index)])
        running = max(running, candidate)
        adjusted[int(index)] = running
    return adjusted.tolist()


def transition_table(wide: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "111": "Stable correct",
        "101": "Restored after failure",
        "100": "Unrecovered failure",
        "110": "Harmed by restoration",
        "011": "Degradation helped, retained",
        "010": "Transient degradation gain",
        "001": "Only restored correct",
        "000": "Always incorrect",
    }
    rows: list[dict[str, Any]] = []
    for scope in ("all", "coarse_perception", "fine_perception"):
        subset = wide if scope == "all" else wide.loc[wide["task_group"] == scope]
        patterns = subset.apply(
            lambda item: f"{int(item['clean'])}{int(item['degraded'])}{int(item['restored'])}",
            axis=1,
        )
        counts = patterns.value_counts()
        for pattern, label in labels.items():
            count = int(counts.get(pattern, 0))
            rows.append(
                {
                    "scope": scope,
                    "scope_label": SCOPE_LABELS[scope],
                    "pattern": pattern,
                    "transition": label,
                    "count": count,
                    "fraction": count / len(subset) if len(subset) else float("nan"),
                }
            )
    return pd.DataFrame(rows)

