from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from microq_phase1.core.config import MicroBenchExportConfig
from microq_phase1.infrastructure.microbench import MicroBenchExporter


def _config(tmp_path: Path) -> MicroBenchExportConfig:
    return MicroBenchExportConfig(
        revision="a" * 40,
        output_root=tmp_path / "microbench",
        catalog_path=tmp_path / "catalog.jsonl",
        cache_dir=tmp_path / "cache",
    )


def _row(
    image: Image.Image,
    *,
    dataset: str,
    label: str,
    image_id: str = "shared-image",
    license_name: str = "CC-BY-4.0",
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "image": image,
        "dataset": dataset,
        "license": license_name,
        "modality": "fluorescence microscopy",
        "submodality": "confocal",
        "domain": "cell biology",
        "subdomain": "cells",
        "stain": "DAPI",
        "microns_per_pixel": 0.25,
        "synthetic": False,
        "label": 1,
        "label_name": label,
        "questions": {"classification": {"answer": label}},
        "captions": {},
        "bbox": [],
        "polygon": [],
    }


def test_exporter_consolidates_task_rows_for_the_same_image(tmp_path: Path) -> None:
    image = Image.new("L", (4, 4), color=17)
    dataset = [
        _row(image.copy(), dataset="contour-task", label="smooth"),
        _row(image.copy(), dataset="texture-task", label="fine"),
    ]

    records = list(MicroBenchExporter()._iter_records(dataset, _config(tmp_path)))

    assert len(records) == 1
    record = records[0]
    assert record.image_id == "shared-image"
    assert record.source_dataset == "contour-task"
    assert record.metadata["source_datasets"] == ["contour-task", "texture-task"]
    assert [
        annotation["label_name"] for annotation in record.metadata["benchmark_annotations"]
    ] == ["smooth", "fine"]
    assert record.image_path.is_file()


def test_exporter_rejects_reused_id_with_different_pixels(tmp_path: Path) -> None:
    dataset = [
        _row(Image.new("L", (4, 4), color=17), dataset="task-a", label="a"),
        _row(Image.new("L", (4, 4), color=18), dataset="task-b", label="b"),
    ]

    with pytest.raises(ValueError, match="different image pixels"):
        list(MicroBenchExporter()._iter_records(dataset, _config(tmp_path)))


def test_lossless_export_reuses_a_matching_existing_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = MicroBenchExporter()
    image = Image.new("L", (4, 4), color=17)
    path = tmp_path / "image.png"
    digest = exporter._image_digest(image)
    exporter._save_lossless(image, path, expected_digest=digest)

    def fail_save(*args: object, **kwargs: object) -> None:
        raise AssertionError("matching image should not be rewritten")

    monkeypatch.setattr(Image.Image, "save", fail_save)

    exporter._save_lossless(image, path, expected_digest=digest)


def test_exporter_excludes_unapproved_licenses(tmp_path: Path) -> None:
    config = _config(tmp_path).model_copy(update={"included_licenses": {"CC-BY-4.0"}})
    dataset = [
        _row(
            Image.new("L", (4, 4)),
            dataset="allowed",
            label="a",
            image_id="allowed",
        ),
        _row(
            Image.new("L", (4, 4)),
            dataset="excluded",
            label="b",
            image_id="excluded",
            license_name="Unknown",
        ),
    ]

    records = list(MicroBenchExporter()._iter_records(dataset, config))

    assert [record.image_id for record in records] == ["allowed"]
