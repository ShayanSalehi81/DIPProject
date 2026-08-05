from __future__ import annotations

from pathlib import Path

from microq_phase1.domain.models import DegradationKind, ImageRecord, Modality, PoolRole, Severity


def test_string_enums_expose_their_values_as_strings() -> None:
    members = (
        PoolRole.TRAINING,
        Modality.FLUORESCENCE,
        Severity.MEDIUM,
        DegradationKind.POISSON_GAUSSIAN,
    )

    for member in members:
        assert isinstance(member, str)
        assert str(member) == member.value


def test_image_record_paths_validate_from_json() -> None:
    record = ImageRecord.model_validate_json(
        '{"image_id":"example","image_path":"images/example.tif",'
        '"source_dataset":"external","license":"CC0-1.0",'
        '"pool_role":"training","modality":"light",'
        '"is_multichannel":false,"bit_depth":16}'
    )

    assert record.image_path == Path("images/example.tif")
