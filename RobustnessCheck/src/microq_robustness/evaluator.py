"""Concurrent, resumable evaluation over blinded clean/degraded/restored conditions."""

from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from microq_robustness.client import VllmClient
from microq_robustness.config import Settings
from microq_robustness.errors import ManifestError, ServerError
from microq_robustness.io_utils import append_jsonl, read_jsonl, write_json_atomic
from microq_robustness.models import Condition, EvaluationRecord, SampleSpec
from microq_robustness.prompting import build_question_prompt
from microq_robustness.scoring import MultipleChoiceScorer

LOGGER = logging.getLogger(__name__)
CONDITIONS: tuple[Condition, ...] = ("clean", "degraded", "restored")
UTC_ZONE = timezone.utc  # noqa: UP017 -- keeps diagnostic use possible on Python 3.10


def load_samples(path: Path) -> list[SampleSpec]:
    samples = [SampleSpec.from_dict(payload) for payload in read_jsonl(path)]
    if not samples:
        raise ManifestError(f"No samples found in {path}")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ManifestError("Sample manifest contains duplicate sample_id values.")
    return samples


class RobustnessEvaluator:
    """Evaluate every paired condition exactly once, with safe resume behavior."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = VllmClient(settings)
        self.scorer = MultipleChoiceScorer()

    def run(self, *, restart: bool = False) -> dict[str, Any]:
        samples = load_samples(self.settings.sample_manifest)
        if len(samples) != self.settings.sample_count:
            raise ManifestError(
                f"SAMPLE_COUNT={self.settings.sample_count}, but the frozen sample manifest "
                f"contains {len(samples)} rows. Re-run preparation intentionally if needed."
            )
        if restart and self.settings.results_path.exists():
            backup = self.settings.results_path.with_name(
                f"{self.settings.results_path.stem}."
                f"{datetime.now(UTC_ZONE).strftime('%Y%m%dT%H%M%SZ')}.bak.jsonl"
            )
            os.replace(self.settings.results_path, backup)
            LOGGER.warning("Moved the prior result file to %s", backup)

        completed = self._completed_keys(samples)
        pending = [
            (sample, condition)
            for sample in samples
            for condition in CONDITIONS
            if (sample.sample_id, condition) not in completed
        ]
        rng = random.Random(self.settings.random_seed)
        rng.shuffle(pending)
        run_id = uuid4().hex
        started_at = datetime.now(UTC_ZONE)

        if pending:
            self.client.assert_ready()
        else:
            LOGGER.info(
                "All %d condition requests are already present; nothing to do.",
                len(completed),
            )

        status_counts: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=self.settings.concurrency) as executor:
            futures: dict[Future[EvaluationRecord], tuple[SampleSpec, Condition]] = {
                executor.submit(self._evaluate_one, sample, condition, run_id): (sample, condition)
                for sample, condition in pending
            }
            for position, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                append_jsonl(self.settings.results_path, record.to_dict())
                status_counts[record.scoring_status] = (
                    status_counts.get(record.scoring_status, 0) + 1
                )
                if position == 1 or position % 10 == 0 or position == len(pending):
                    LOGGER.info(
                        "Evaluation progress: %d/%d new requests (%d/%d total complete)",
                        position,
                        len(pending),
                        len(completed) + position,
                        len(samples) * len(CONDITIONS),
                    )

        metadata = {
            "schema_version": "1.0",
            "run_id": run_id,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(UTC_ZONE).isoformat(),
            "sample_count": len(samples),
            "expected_requests": len(samples) * len(CONDITIONS),
            "preexisting_requests": len(completed),
            "new_requests": len(pending),
            "new_status_counts": dict(sorted(status_counts.items())),
            "sample_manifest_sha256": self._sha256(self.settings.sample_manifest),
            "settings": self.settings.public_snapshot(),
        }
        metadata_path = (
            self.settings.results_path.parent / "run_metadata" / f"{run_id}.json"
        )
        write_json_atomic(metadata_path, metadata)
        return metadata

    def _completed_keys(self, samples: list[SampleSpec]) -> set[tuple[str, str]]:
        if not self.settings.results_path.exists():
            return set()
        valid_ids = {sample.sample_id for sample in samples}
        completed: set[tuple[str, str]] = set()
        for record in read_jsonl(self.settings.results_path):
            sample_id = str(record.get("sample_id") or "")
            condition = str(record.get("condition") or "")
            if sample_id not in valid_ids:
                raise ManifestError(
                    "Existing results refer to a different sample cohort. Use --restart to "
                    "archive them before starting this cohort."
                )
            if record.get("model_id") != self.settings.model_id:
                raise ManifestError(
                    "Existing results were produced by a different MODEL_ID. Use --restart."
                )
            if condition not in CONDITIONS:
                raise ManifestError(f"Existing result has invalid condition: {condition!r}")
            # Transport failures are deliberately retried on the next invocation. A later
            # successful record supersedes the earlier error during analysis.
            if record.get("scoring_status") != "api_error":
                completed.add((sample_id, condition))
        return completed

    def _evaluate_one(
        self, sample: SampleSpec, condition: Condition, run_id: str
    ) -> EvaluationRecord:
        image_path = Path(sample.image_path(condition))
        # Reuse one decoding seed across all three images in a pair. This couples any
        # stochastic decoding noise instead of confounding the image-condition effect.
        seed = self._sample_seed(sample.sample_id)
        started = time.perf_counter()
        try:
            completion = self.client.complete(
                image_path=image_path,
                user_prompt=build_question_prompt(sample),
                seed=seed,
            )
            scored = self.scorer.score(completion.text, sample.options)
            return self._record(
                sample=sample,
                condition=condition,
                run_id=run_id,
                response_text=completion.text,
                predicted_index=scored.predicted_index,
                correct=scored.predicted_index == sample.gold_index,
                scoring_status=scored.scoring_status,
                parse_method=scored.parse_method,
                latency_seconds=completion.latency_seconds,
                attempts=completion.attempts,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                error=None,
            )
        except (ServerError, OSError, ValueError) as exc:
            LOGGER.error(
                "Request failed for sample=%s condition=%s: %s",
                sample.sample_id,
                condition,
                exc,
            )
            return self._record(
                sample=sample,
                condition=condition,
                run_id=run_id,
                response_text="",
                predicted_index=None,
                correct=False,
                scoring_status="api_error",
                parse_method="none",
                latency_seconds=time.perf_counter() - started,
                attempts=self.settings.max_retries,
                prompt_tokens=None,
                completion_tokens=None,
                error=str(exc)[:2000],
            )

    def _record(
        self,
        *,
        sample: SampleSpec,
        condition: Condition,
        run_id: str,
        response_text: str,
        predicted_index: int | None,
        correct: bool,
        scoring_status: str,
        parse_method: str,
        latency_seconds: float,
        attempts: int,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        error: str | None,
    ) -> EvaluationRecord:
        return EvaluationRecord(
            schema_version="1.0",
            run_id=run_id,
            sample_id=sample.sample_id,
            image_id=sample.image_id,
            recipe_id=sample.recipe_id,
            condition=condition,
            task_type=sample.task_type,
            task_group=sample.task_group,
            question_id=sample.question_id,
            gold_index=sample.gold_index,
            gold_answer=sample.gold_answer,
            predicted_index=predicted_index,
            correct=correct,
            scoring_status=scoring_status,
            parse_method=parse_method,
            response_text=response_text,
            model_id=self.settings.model_id,
            latency_seconds=latency_seconds,
            attempts=attempts,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=error,
            image_path=sample.image_path(condition),
            degradation_kind=sample.degradation_kind,
            severity=sample.severity,
            restoration_algorithm=sample.restoration_algorithm,
            restoration_changed_pixels=sample.restoration_changed_pixels,
            modality=sample.modality,
            source_dataset=sample.source_dataset,
            timestamp_utc=datetime.now(UTC_ZONE).isoformat(),
        )

    def _sample_seed(self, sample_id: str) -> int:
        key = f"{self.settings.random_seed}|{sample_id}".encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
