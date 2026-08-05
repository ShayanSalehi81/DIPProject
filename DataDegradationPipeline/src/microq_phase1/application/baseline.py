"""Interpretable DIP-fingerprint readout for degradation and severity recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from microq_phase1.core.logging import get_logger
from microq_phase1.domain.models import AnalysisRecord, PoolRole
from microq_phase1.infrastructure.jsonl import read_jsonl, write_json_atomic

LOGGER = get_logger(__name__)


class FingerprintBaselineService:
    """Fit lightweight classifiers on external data and evaluate only on benchmark data."""

    def run(
        self,
        *,
        analysis_path: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        records = [
            item
            for item in read_jsonl(analysis_path, AnalysisRecord)
            if item.step_count == 1 and not item.restoration_algorithms
        ]
        training = [item for item in records if item.pool_role == PoolRole.TRAINING]
        evaluation = [item for item in records if item.pool_role == PoolRole.EVALUATION]
        if not training or not evaluation:
            raise ValueError("Baseline requires both training and evaluation analysis records.")
        feature_names = sorted(training[0].features)
        self._require_feature_schema(records, feature_names)
        x_train = self._matrix(training, feature_names)
        x_eval = self._matrix(evaluation, feature_names)
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "degradation": self._fit_task(
                x_train,
                [item.degradation for item in training],
                x_eval,
                [item.degradation for item in evaluation],
                output_dir / "degradation_classifier.joblib",
            ),
            "severity": self._fit_task(
                x_train,
                [item.severity for item in training],
                x_eval,
                [item.severity for item in evaluation],
                output_dir / "severity_classifier.joblib",
            ),
            "feature_names": feature_names,
            "training_samples": len(training),
            "evaluation_samples": len(evaluation),
            "policy": "fit=external_training; evaluate=official_benchmark",
        }
        write_json_atomic(output_dir / "fingerprint_baseline_report.json", report)
        LOGGER.info(
            "fingerprint_baseline_completed",
            training_samples=len(training),
            evaluation_samples=len(evaluation),
            output_dir=str(output_dir),
        )
        return report

    @staticmethod
    def _fit_task(
        x_train: NDArray[np.float64],
        y_train: list[str],
        x_eval: NDArray[np.float64],
        y_eval: list[str],
        model_path: Path,
    ) -> dict[str, Any]:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=20260724,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_eval)
        labels = sorted(set(y_train) | set(y_eval))
        joblib.dump(model, model_path)
        return {
            "accuracy": float(accuracy_score(y_eval, predictions)),
            "macro_f1": float(f1_score(y_eval, predictions, average="macro")),
            "labels": labels,
            "confusion_matrix": confusion_matrix(y_eval, predictions, labels=labels).tolist(),
            "model_path": str(model_path),
        }

    @staticmethod
    def _matrix(
        records: list[AnalysisRecord], feature_names: list[str]
    ) -> NDArray[np.float64]:
        return np.asarray(
            [[record.features[name] for name in feature_names] for record in records],
            dtype=np.float64,
        )

    @staticmethod
    def _require_feature_schema(
        records: list[AnalysisRecord],
        feature_names: list[str],
    ) -> None:
        expected = set(feature_names)
        inconsistent = [item.recipe_id for item in records if set(item.features) != expected]
        if inconsistent:
            raise ValueError(f"Inconsistent feature schema in {len(inconsistent)} records.")
