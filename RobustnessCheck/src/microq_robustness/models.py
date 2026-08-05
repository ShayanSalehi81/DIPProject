"""Serializable domain models for paired samples and model responses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Condition = Literal["clean", "degraded", "restored"]


@dataclass(frozen=True, slots=True)
class SampleSpec:
    """One Micro-Bench question attached to an exact image triplet."""

    sample_id: str
    image_id: str
    recipe_id: str
    question_id: str
    task_type: str
    task_group: str
    question: str
    options: tuple[str, ...]
    gold_index: int
    gold_answer: str
    clean_path: str
    degraded_path: str
    restored_path: str
    degradation_kind: str
    severity: str
    degradation_parameters: dict[str, Any]
    restoration_algorithm: str
    restoration_parameters: dict[str, Any]
    restoration_changed_pixels: bool
    modality: str
    submodality: str | None
    source_dataset: str
    dataset_revision: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["options"] = list(self.options)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SampleSpec:
        copied = dict(payload)
        copied["options"] = tuple(copied["options"])
        return cls(**copied)

    def image_path(self, condition: Condition) -> str:
        return {
            "clean": self.clean_path,
            "degraded": self.degraded_path,
            "restored": self.restored_path,
        }[condition]


@dataclass(frozen=True, slots=True)
class ScoredAnswer:
    predicted_index: int | None
    parse_method: str
    scoring_status: str


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One VLM response for one sample-condition pair."""

    schema_version: str
    run_id: str
    sample_id: str
    image_id: str
    recipe_id: str
    condition: Condition
    task_type: str
    task_group: str
    question_id: str
    gold_index: int
    gold_answer: str
    predicted_index: int | None
    correct: bool
    scoring_status: str
    parse_method: str
    response_text: str
    model_id: str
    latency_seconds: float
    attempts: int
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str | None
    image_path: str
    degradation_kind: str
    severity: str
    restoration_algorithm: str
    restoration_changed_pixels: bool
    modality: str
    source_dataset: str
    timestamp_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

