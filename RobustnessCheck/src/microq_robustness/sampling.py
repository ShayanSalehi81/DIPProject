"""Deterministic, balanced construction of paired Micro-Bench image triplets."""

from __future__ import annotations

import hashlib
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from microq_robustness.config import Settings
from microq_robustness.errors import ManifestError
from microq_robustness.io_utils import (
    read_jsonl,
    resolve_phase1_path,
    write_json_atomic,
    write_jsonl_atomic,
)
from microq_robustness.models import SampleSpec

LOGGER = logging.getLogger(__name__)
TASK_GROUPS = {
    "domain": "coarse_perception",
    "modality": "coarse_perception",
    "classification": "fine_perception",
    "subdomain": "fine_perception",
    "submodality": "fine_perception",
    "stain": "fine_perception",
}


@dataclass(frozen=True, slots=True)
class _Candidate:
    generated: dict[str, Any]
    restored: dict[str, Any]
    task_type: str
    question: dict[str, Any]
    random_tie_breaker: float

    @property
    def image_id(self) -> str:
        return str(self.generated["source"]["image_id"])

    @property
    def steps(self) -> list[dict[str, Any]]:
        return list(self.generated["steps"])

    @property
    def degradation_kind(self) -> str:
        return "+".join(sorted(str(step["kind"]) for step in self.steps))

    @property
    def severity(self) -> str:
        values = {str(step["severity"]) for step in self.steps}
        return next(iter(values)) if len(values) == 1 else "mixed"

    @property
    def modality(self) -> str:
        return str(self.generated["source"].get("modality") or "unknown")


class PairedSampleSelector:
    """Select unique image-question triplets with balanced experimental factors."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def prepare(self, *, force: bool = False) -> list[SampleSpec]:
        if self.settings.sample_manifest.exists() and not force:
            raise ManifestError(
                f"Sample manifest already exists: {self.settings.sample_manifest}. "
                "Use --force only when intentionally drawing a new cohort."
            )
        restored = self._load_restored_records()
        candidates = self._load_candidates(restored)
        selected = self._balanced_select(candidates)
        samples = [self._materialize(candidate) for candidate in selected]
        self._validate_samples(samples)
        write_jsonl_atomic(self.settings.sample_manifest, (sample.to_dict() for sample in samples))
        report = self._sampling_report(samples, candidate_count=len(candidates))
        report_path = self.settings.sample_manifest.with_name("sampling_report.json")
        write_json_atomic(report_path, report)
        LOGGER.info(
            "Prepared %d paired samples from %d candidates at %s",
            len(samples),
            len(candidates),
            self.settings.sample_manifest,
        )
        return samples

    def _load_restored_records(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for record in read_jsonl(self.settings.restored_manifest):
            if record.get("pool_name") != "microbench_evaluation":
                continue
            if not record.get("generated"):
                continue
            recipe_id = str(record.get("recipe_id") or "")
            if not recipe_id:
                raise ManifestError("A restored Micro-Bench record is missing recipe_id.")
            if recipe_id in records:
                raise ManifestError(f"Duplicate restored recipe_id: {recipe_id}")
            records[recipe_id] = record
        if not records:
            raise ManifestError("No completed Micro-Bench restoration records were found.")
        return records

    def _load_candidates(
        self, restored_records: dict[str, dict[str, Any]]
    ) -> list[_Candidate]:
        rng = random.Random(self.settings.random_seed)
        candidates: list[_Candidate] = []
        for generated in read_jsonl(self.settings.generated_manifest):
            if generated.get("pool_name") != "microbench_evaluation":
                continue
            if not generated.get("generated"):
                continue
            recipe_id = str(generated.get("recipe_id") or "")
            restored = restored_records.get(recipe_id)
            if restored is None:
                continue
            steps = list(generated.get("steps") or [])
            if not steps:
                continue
            if not self.settings.include_mixed_degradations and len(steps) != 1:
                continue
            severities = {str(step.get("severity")) for step in steps}
            severity = next(iter(severities)) if len(severities) == 1 else "mixed"
            if severity not in self.settings.severities:
                continue
            questions = self._questions_by_type(generated)
            for task_type in self.settings.question_types:
                question = questions.get(task_type)
                if question is not None:
                    candidates.append(
                        _Candidate(
                            generated=generated,
                            restored=restored,
                            task_type=task_type,
                            question=question,
                            random_tie_breaker=rng.random(),
                        )
                    )
        if not candidates:
            raise ManifestError("No eligible paired Micro-Bench candidates matched the policy.")
        return candidates

    def _questions_by_type(self, generated: dict[str, Any]) -> dict[str, dict[str, Any]]:
        source = generated.get("source") or {}
        metadata = source.get("metadata") or {}
        annotations = sorted(
            metadata.get("benchmark_annotations") or [],
            key=lambda item: int(item.get("source_row_index", 0)),
        )
        result: dict[str, dict[str, Any]] = {}
        for annotation in annotations:
            if annotation.get("synthetic") and not self.settings.allow_synthetic_benchmark_rows:
                continue
            questions = annotation.get("questions") or {}
            for task_type in self.settings.question_types:
                value = questions.get(task_type)
                if task_type in result or not isinstance(value, dict):
                    continue
                options = value.get("options")
                answer_idx = value.get("answer_idx")
                if not value.get("question") or not isinstance(options, list):
                    continue
                try:
                    index = int(answer_idx)
                except (TypeError, ValueError):
                    continue
                if not 0 <= index < len(options):
                    continue
                normalized = dict(value)
                normalized["answer_idx"] = index
                normalized["source_row_index"] = annotation.get("source_row_index")
                result[task_type] = normalized
        return result

    def _balanced_select(self, candidates: list[_Candidate]) -> list[_Candidate]:
        unique_images = {candidate.image_id for candidate in candidates}
        if len(unique_images) < self.settings.sample_count:
            raise ManifestError(
                f"Requested {self.settings.sample_count} unique images but only "
                f"{len(unique_images)} are eligible."
            )

        dimensions = {
            "task_type": sorted({item.task_type for item in candidates}),
            "degradation_kind": sorted({item.degradation_kind for item in candidates}),
            "severity": sorted({item.severity for item in candidates}),
            "modality": sorted({item.modality for item in candidates}),
        }
        counts: Counter[tuple[str, str]] = Counter()
        selected: list[_Candidate] = []
        used_images: set[str] = set()

        def score(item: _Candidate) -> tuple[float, float]:
            values = {
                "task_type": item.task_type,
                "degradation_kind": item.degradation_kind,
                "severity": item.severity,
                "modality": item.modality,
            }
            imbalance = sum(
                counts[(name, value)] * len(dimensions[name])
                for name, value in values.items()
            )
            return float(imbalance), item.random_tie_breaker

        while len(selected) < self.settings.sample_count:
            eligible = (item for item in candidates if item.image_id not in used_images)
            try:
                chosen = min(eligible, key=score)
            except ValueError as exc:
                raise ManifestError("Balanced selection exhausted eligible unique images.") from exc
            selected.append(chosen)
            used_images.add(chosen.image_id)
            counts[("task_type", chosen.task_type)] += 1
            counts[("degradation_kind", chosen.degradation_kind)] += 1
            counts[("severity", chosen.severity)] += 1
            counts[("modality", chosen.modality)] += 1
        return selected

    def _materialize(self, candidate: _Candidate) -> SampleSpec:
        generated = candidate.generated
        restored = candidate.restored
        source = generated["source"]
        metadata = source.get("metadata") or {}
        clean_path = resolve_phase1_path(
            str(source["image_path"]), self.settings.data_pipeline_root
        )
        degraded_path = resolve_phase1_path(
            str(generated["output_path"]), self.settings.data_pipeline_root
        )
        restored_path = resolve_phase1_path(
            str(restored["output_path"]), self.settings.data_pipeline_root
        )
        if self.settings.verify_images:
            for image_path in (clean_path, degraded_path, restored_path):
                self._verify_image(image_path)

        question = candidate.question
        gold_index = int(question["answer_idx"])
        options = tuple(str(option) for option in question["options"])
        question_id = str(
            question.get("id")
            or f"row-{question.get('source_row_index', 'unknown')}-{candidate.task_type}"
        )
        stable_key = "|".join(
            [candidate.image_id, str(generated["recipe_id"]), question_id, candidate.task_type]
        )
        sample_id = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:20]
        restoration_steps = list(restored.get("restoration_steps") or [])
        algorithms = "+".join(
            str(step.get("algorithm") or "unknown") for step in restoration_steps
        ) or "none"
        restoration_parameters = {
            str(step.get("algorithm") or f"step_{index}"): step.get("parameters") or {}
            for index, step in enumerate(restoration_steps)
        }
        degradation_parameters = {
            str(step["kind"]): step.get("parameters") or {} for step in candidate.steps
        }
        return SampleSpec(
            sample_id=sample_id,
            image_id=candidate.image_id,
            recipe_id=str(generated["recipe_id"]),
            question_id=question_id,
            task_type=candidate.task_type,
            task_group=TASK_GROUPS[candidate.task_type],
            question=str(question["question"]),
            options=options,
            gold_index=gold_index,
            gold_answer=str(question.get("answer") or options[gold_index]),
            clean_path=str(clean_path),
            degraded_path=str(degraded_path),
            restored_path=str(restored_path),
            degradation_kind=candidate.degradation_kind,
            severity=candidate.severity,
            degradation_parameters=degradation_parameters,
            restoration_algorithm=algorithms,
            restoration_parameters=restoration_parameters,
            restoration_changed_pixels=(
                restored.get("output_sha256") != generated.get("output_sha256")
            ),
            modality=candidate.modality,
            submodality=source.get("submodality"),
            source_dataset=str(source.get("source_dataset") or "unknown"),
            dataset_revision=str(metadata.get("dataset_revision") or "unknown"),
        )

    @staticmethod
    def _verify_image(path: Path) -> None:
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            raise ManifestError(f"Unreadable image: {path}") from exc

    def _validate_samples(self, samples: list[SampleSpec]) -> None:
        if len(samples) != self.settings.sample_count:
            raise ManifestError("The selected cohort does not match SAMPLE_COUNT.")
        for field_name, values in {
            "sample_id": [sample.sample_id for sample in samples],
            "image_id": [sample.image_id for sample in samples],
            "recipe_id": [sample.recipe_id for sample in samples],
        }.items():
            if len(values) != len(set(values)):
                raise ManifestError(f"Selected cohort has duplicate {field_name} values.")

    def _sampling_report(
        self, samples: list[SampleSpec], *, candidate_count: int
    ) -> dict[str, Any]:
        def distribution(attribute: str) -> dict[str, int]:
            return dict(sorted(Counter(str(getattr(item, attribute)) for item in samples).items()))

        revisions = sorted({item.dataset_revision for item in samples})
        return {
            "schema_version": "1.0",
            "design": "paired clean/degraded/restored",
            "requested_samples": self.settings.sample_count,
            "selected_samples": len(samples),
            "inference_requests": len(samples) * 3,
            "candidate_question_recipe_pairs": candidate_count,
            "unique_images": len({sample.image_id for sample in samples}),
            "dataset_revisions": revisions,
            "distributions": {
                "task_group": distribution("task_group"),
                "task_type": distribution("task_type"),
                "degradation_kind": distribution("degradation_kind"),
                "severity": distribution("severity"),
                "modality": distribution("modality"),
                "source_dataset": distribution("source_dataset"),
                "restoration_algorithm": distribution("restoration_algorithm"),
            },
            "settings": self.settings.public_snapshot(),
        }
