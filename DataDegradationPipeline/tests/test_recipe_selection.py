from __future__ import annotations

from pathlib import Path

from microq_phase1.application.recipe_selection import RecipeCandidateSelector
from microq_phase1.domain.models import ImageRecord, Modality, PoolRole


def test_stratified_limit_covers_available_modalities() -> None:
    records = [
        _record(f"{modality}-{index}", modality, f"source-{modality}")
        for modality in (Modality.LIGHT, Modality.FLUORESCENCE, Modality.ELECTRON)
        for index in range(4)
    ]
    selected = RecipeCandidateSelector().stratified_limit(records, limit=6, seed=7)
    assert len(selected) == 6
    assert {record.modality for record in selected} == {
        Modality.LIGHT,
        Modality.FLUORESCENCE,
        Modality.ELECTRON,
    }


def _record(image_id: str, modality: Modality, source: str) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        image_path=Path(f"{image_id}.png"),
        source_dataset=source,
        license="CC-BY-4.0",
        pool_role=PoolRole.TRAINING,
        modality=modality,
    )
