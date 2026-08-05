from __future__ import annotations

from pathlib import Path

from PIL import Image

from microq_robustness.io_utils import write_jsonl_atomic
from microq_robustness.sampling import PairedSampleSelector


def _image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(value, value, value)).save(path)


def test_selector_builds_unique_paired_triplets(tmp_path: Path, settings_factory) -> None:
    pipeline = tmp_path / "DataDegradationPipeline"
    generated_rows = []
    restored_rows = []
    task_names = ["domain", "modality", "classification", "domain", "modality", "classification"]
    kinds = ["sampling", "jpeg", "motion", "sampling", "jpeg", "motion"]
    severities = ["weak", "medium", "strong", "medium", "strong", "weak"]
    for index in range(6):
        image_id = f"image-{index}"
        recipe_id = f"recipe-{index}"
        clean = pipeline / "data" / f"{image_id}.png"
        degraded = pipeline / "outputs" / "generated" / f"{recipe_id}.png"
        restored = pipeline / "outputs" / "restored" / f"{recipe_id}.png"
        _image(clean, 20 + index)
        _image(degraded, 40 + index)
        _image(restored, 30 + index)
        task = task_names[index]
        question = {
            "id": f"question-{index}",
            "question": "Which option is correct?",
            "options": ["first", "second", "none"],
            "answer_idx": 1,
            "answer": "second",
        }
        source = {
            "image_id": image_id,
            "image_path": str(clean),
            "source_dataset": "fixture",
            "modality": ["light", "electron", "fluorescence"][index % 3],
            "submodality": None,
            "metadata": {
                "dataset_revision": "abc123",
                "benchmark_annotations": [
                    {
                        "source_row_index": index,
                        "synthetic": False,
                        "questions": {task: question},
                    }
                ],
            },
        }
        base = {
            "generated": True,
            "pool_name": "microbench_evaluation",
            "recipe_id": recipe_id,
            "source": source,
            "steps": [
                {
                    "kind": kinds[index],
                    "severity": severities[index],
                    "parameters": {"strength": index + 1},
                }
            ],
        }
        generated_rows.append({**base, "output_path": str(degraded), "output_sha256": "a"})
        restored_rows.append(
            {
                **base,
                "output_path": str(restored),
                "output_sha256": "b",
                "restoration_steps": [
                    {"algorithm": "fixture_restore", "parameters": {"value": 1}}
                ],
            }
        )
    settings = settings_factory(
        data_pipeline_root=pipeline,
        question_types=("domain", "modality", "classification"),
    )
    write_jsonl_atomic(settings.generated_manifest, generated_rows)
    write_jsonl_atomic(settings.restored_manifest, restored_rows)

    samples = PairedSampleSelector(settings).prepare()

    assert len(samples) == 3
    assert len({sample.image_id for sample in samples}) == 3
    assert all(Path(sample.clean_path).is_file() for sample in samples)
    assert all(sample.restoration_changed_pixels for sample in samples)

