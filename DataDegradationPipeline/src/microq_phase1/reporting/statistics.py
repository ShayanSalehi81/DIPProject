"""Deterministic bootstrap summaries for degradation and restoration metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from microq_phase1.core.config import ProjectConfig
from microq_phase1.core.randomness import derive_seed


def degradation_summary(frame: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    single = frame[frame["step_count"] == 1]
    metrics = ["psnr", "ssim", "edge_retention_ratio", "histogram_js_divergence"]
    available = [metric for metric in metrics if metric in single.columns]
    group_columns = ["pool_role", "modality", "degradation", "severity"]
    rows: list[dict[str, object]] = []
    for group, subset in single.groupby(group_columns):
        for metric in available:
            values = subset[metric].dropna().to_numpy(dtype=np.float64)
            low, high = _bootstrap_mean_ci(
                values,
                config.reporting.bootstrap_resamples,
                config.reporting.confidence_level,
                derive_seed(config.generation.master_seed, *group, metric),
            )
            rows.append(
                {
                    **dict(zip(group_columns, group, strict=True)),
                    "metric": metric,
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "ci_low": low,
                    "ci_high": high,
                    "confidence_level": config.reporting.confidence_level,
                }
            )
    return pd.DataFrame(rows)


def restoration_delta_summary(
    degraded: pd.DataFrame,
    restored: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    metrics = ["psnr", "ssim", "edge_retention_ratio"]
    joined = degraded.merge(restored, on="recipe_id", suffixes=("_degraded", "_restored"))
    for metric in metrics:
        joined[f"delta_{metric}"] = joined[f"{metric}_restored"] - joined[
            f"{metric}_degraded"
        ]
    group_columns = [
        "modality_degraded",
        "degradation_degraded",
        "severity_degraded",
    ]
    rows: list[dict[str, object]] = []
    for group, subset in joined.groupby(group_columns):
        for metric in metrics:
            values = subset[f"delta_{metric}"].dropna().to_numpy(dtype=np.float64)
            low, high = _bootstrap_mean_ci(
                values,
                config.reporting.bootstrap_resamples,
                config.reporting.confidence_level,
                derive_seed(config.generation.master_seed, "restoration", *group, metric),
            )
            rows.append(
                {
                    **dict(zip(group_columns, group, strict=True)),
                    "metric": metric,
                    "count": len(values),
                    "mean_delta": float(np.mean(values)),
                    "std_delta": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median_delta": float(np.median(values)),
                    "ci_low": low,
                    "ci_high": high,
                    "confidence_level": config.reporting.confidence_level,
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_mean_ci(
    values: np.ndarray,
    resamples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = np.mean(values[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)
