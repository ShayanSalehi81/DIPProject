"""End-to-end result validation, paired statistics, tables, and paper figures."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from microq_robustness.config import Settings
from microq_robustness.errors import AnalysisError
from microq_robustness.evaluator import CONDITIONS, load_samples
from microq_robustness.io_utils import read_jsonl, write_json_atomic
from microq_robustness.statistics import (
    accuracy_table,
    factor_breakdown,
    paired_effect_table,
    paired_tests,
    transition_table,
)
from microq_robustness.visualization import PaperFigureFactory

LOGGER = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Validate a complete paired run and produce reproducible research artifacts."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self) -> dict[str, Any]:
        samples = load_samples(self.settings.sample_manifest)
        long, wide, excluded = self._load_frames(samples)
        output = self.settings.analysis_output_dir
        figures = output / "figures"
        tables = output / "tables"
        figures.mkdir(parents=True, exist_ok=True)
        tables.mkdir(parents=True, exist_ok=True)

        accuracy = accuracy_table(
            wide,
            resamples=self.settings.bootstrap_resamples,
            confidence_level=self.settings.confidence_level,
            seed=self.settings.random_seed,
        )
        effects = paired_effect_table(
            wide,
            resamples=self.settings.bootstrap_resamples,
            confidence_level=self.settings.confidence_level,
            seed=self.settings.random_seed,
        )
        breakdown = pd.concat(
            [
                factor_breakdown(
                    wide,
                    factor,
                    resamples=self.settings.bootstrap_resamples,
                    confidence_level=self.settings.confidence_level,
                    seed=self.settings.random_seed,
                )
                for factor in (
                    "degradation_kind",
                    "severity",
                    "task_type",
                    "modality",
                    "restoration_algorithm",
                    "restoration_changed_pixels",
                )
            ],
            ignore_index=True,
        )
        tests = paired_tests(wide)
        transitions = transition_table(wide)
        response_quality = self._response_quality(long)
        latency = self._latency_summary(long)

        frames = {
            "accuracy_with_bootstrap_ci.csv": accuracy,
            "paired_effects_with_bootstrap_ci.csv": effects,
            "factor_breakdown.csv": breakdown,
            "paired_mcnemar_tests.csv": tests,
            "correctness_transitions.csv": transitions,
            "response_quality.csv": response_quality,
            "latency_summary.csv": latency,
        }
        for filename, frame in frames.items():
            frame.to_csv(tables / filename, index=False, float_format="%.8f")

        latex_path = tables / "macro_accuracy_table.tex"
        latex_path.write_text(self._latex_table(accuracy), encoding="utf-8")

        figure_files = PaperFigureFactory(
            figures, dpi=self.settings.figure_dpi
        ).render_all(
            accuracy=accuracy,
            breakdown=breakdown,
            transitions=transitions,
            wide=wide,
        )
        summary = self._summary(
            samples=samples,
            long=long,
            wide=wide,
            accuracy=accuracy,
            effects=effects,
            tests=tests,
            response_quality=response_quality,
            excluded=excluded,
            figure_files=figure_files,
            table_files=[str(tables / name) for name in frames] + [str(latex_path)],
        )
        write_json_atomic(output / "analysis_summary.json", summary)
        LOGGER.info("Analysis and four figure sets written to %s", output)
        return summary

    def _load_frames(
        self, samples: list[Any]
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        sample_ids = {sample.sample_id for sample in samples}
        latest: dict[tuple[str, str], dict[str, Any]] = {}
        for record in read_jsonl(self.settings.results_path):
            sample_id = str(record.get("sample_id") or "")
            condition = str(record.get("condition") or "")
            if sample_id not in sample_ids:
                raise AnalysisError(
                    f"Result sample_id {sample_id!r} is not in the frozen sample manifest."
                )
            if condition not in CONDITIONS:
                raise AnalysisError(f"Invalid result condition: {condition!r}")
            latest[(sample_id, condition)] = record

        complete_ids: list[str] = []
        excluded: list[str] = []
        for sample_id in sorted(sample_ids):
            records = [latest.get((sample_id, condition)) for condition in CONDITIONS]
            complete = all(record is not None for record in records)
            no_api_errors = complete and all(
                record.get("scoring_status") != "api_error" for record in records if record
            )
            if complete and no_api_errors:
                complete_ids.append(sample_id)
            else:
                excluded.append(sample_id)
        if excluded and self.settings.strict_completeness:
            raise AnalysisError(
                f"{len(excluded)} samples are missing a condition or contain an API error. "
                "Run run_evaluation.py again; transport errors are retried automatically."
            )
        if not complete_ids:
            raise AnalysisError("No complete clean/degraded/restored triplets are available.")

        records = [
            latest[(sample_id, condition)]
            for sample_id in complete_ids
            for condition in CONDITIONS
        ]
        long = pd.DataFrame(records)
        long["correct"] = long["correct"].astype(bool)
        models = sorted(long["model_id"].dropna().astype(str).unique())
        if len(models) != 1:
            raise AnalysisError(f"Expected one model_id, found: {models}")

        correct = (
            long.pivot(index="sample_id", columns="condition", values="correct")
            .reindex(columns=list(CONDITIONS))
            .astype(int)
        )
        sample_frame = pd.DataFrame([sample.to_dict() for sample in samples]).set_index("sample_id")
        metadata_columns = [
            "image_id",
            "recipe_id",
            "task_type",
            "task_group",
            "degradation_kind",
            "severity",
            "restoration_algorithm",
            "restoration_changed_pixels",
            "modality",
            "submodality",
            "source_dataset",
            "dataset_revision",
        ]
        wide = sample_frame.loc[correct.index, metadata_columns].join(correct).reset_index()
        if wide["image_id"].nunique() != len(wide):
            raise AnalysisError("Paired analysis requires exactly one question per unique image.")
        return long, wide, excluded

    @staticmethod
    def _response_quality(long: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for (condition, task_group), subset in long.groupby(
            ["condition", "task_group"], observed=True, sort=True
        ):
            rows.append(
                {
                    "condition": condition,
                    "task_group": task_group,
                    "n": len(subset),
                    "parse_rate": float((subset["scoring_status"] == "parsed").mean()),
                    "unparseable_rate": float(
                        (subset["scoring_status"] == "unparseable").mean()
                    ),
                    "retry_rate": float((subset["attempts"] > 1).mean()),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _latency_summary(long: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for condition, subset in long.groupby("condition", observed=True, sort=True):
            values = subset["latency_seconds"].astype(float).to_numpy()
            rows.append(
                {
                    "condition": condition,
                    "n": len(values),
                    "mean_seconds": float(np.mean(values)),
                    "median_seconds": float(np.median(values)),
                    "p95_seconds": float(np.quantile(values, 0.95)),
                    "total_seconds": float(np.sum(values)),
                }
            )
        return pd.DataFrame(rows)

    def _summary(
        self,
        *,
        samples: list[Any],
        long: pd.DataFrame,
        wide: pd.DataFrame,
        accuracy: pd.DataFrame,
        effects: pd.DataFrame,
        tests: pd.DataFrame,
        response_quality: pd.DataFrame,
        excluded: list[str],
        figure_files: list[str],
        table_files: list[str],
    ) -> dict[str, Any]:
        overall_accuracy = accuracy.loc[accuracy["scope"] == "all"].set_index("condition")
        overall_effects = effects.loc[effects["scope"] == "all"].set_index("effect")
        return {
            "schema_version": "1.0",
            "study": "paired clean/degraded/restored Micro-Bench robustness",
            "model_id": str(long["model_id"].iloc[0]),
            "configured_sample_count": len(samples),
            "analyzed_complete_samples": len(wide),
            "analyzed_requests": len(long),
            "excluded_incomplete_sample_ids": excluded,
            "bootstrap_resamples": self.settings.bootstrap_resamples,
            "confidence_level": self.settings.confidence_level,
            "macro_accuracy": {
                condition: {
                    "estimate": float(overall_accuracy.loc[condition, "accuracy"]),
                    "ci_low": float(overall_accuracy.loc[condition, "ci_low"]),
                    "ci_high": float(overall_accuracy.loc[condition, "ci_high"]),
                }
                for condition in CONDITIONS
            },
            "paired_effects": {
                effect: {
                    "estimate": self._finite_or_none(overall_effects.loc[effect, "estimate"]),
                    "ci_low": self._finite_or_none(overall_effects.loc[effect, "ci_low"]),
                    "ci_high": self._finite_or_none(overall_effects.loc[effect, "ci_high"]),
                }
                for effect in overall_effects.index
            },
            "paired_tests": tests.to_dict(orient="records"),
            "response_quality": response_quality.to_dict(orient="records"),
            "benchmark_scope_note": (
                "The completed Phase 1 artifacts contain Micro-Bench perception annotations "
                "only. Results are reported for coarse and fine perception; no cognition "
                "category is inferred or fabricated."
            ),
            "figure_files": figure_files,
            "table_files": table_files,
            "output_directory": str(self.settings.analysis_output_dir),
        }

    @staticmethod
    def _finite_or_none(value: Any) -> float | None:
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None

    @staticmethod
    def _latex_table(accuracy: pd.DataFrame) -> str:
        pivot = accuracy.set_index(["scope_label", "condition"])

        def cell(scope: str, condition: str) -> str:
            row = pivot.loc[(scope, condition)]
            return (
                f"{row['accuracy'] * 100:.2f} "
                f"[{row['ci_low'] * 100:.2f}, {row['ci_high'] * 100:.2f}]"
            )

        scopes = [
            scope
            for scope in ("Overall", "Coarse perception", "Fine perception")
            if scope in set(accuracy["scope_label"])
        ]
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            (
                r"\caption{Macro-average accuracy (\%) with paired bootstrap "
                r"95\% confidence intervals.}"
            ),
            r"\label{tab:paired_robustness}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Scope & Clean & Degraded & Restored \\",
            r"\midrule",
        ]
        for scope in scopes:
            escaped = scope.replace("_", r"\_")
            lines.append(
                f"{escaped} & {cell(scope, 'clean')} & {cell(scope, 'degraded')} & "
                f"{cell(scope, 'restored')} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
        return "\n".join(lines)
