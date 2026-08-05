from __future__ import annotations

from pathlib import Path

import pytest

from microq_robustness.config import Settings


@pytest.fixture
def settings_factory(tmp_path: Path):
    def factory(**overrides: object) -> Settings:
        defaults: dict[str, object] = {
            "project_root": tmp_path,
            "sample_count": 3,
            "random_seed": 42,
            "data_pipeline_root": tmp_path / "DataDegradationPipeline",
            "generated_manifest": tmp_path / "generated.jsonl",
            "restored_manifest": tmp_path / "restored.jsonl",
            "sample_manifest": tmp_path / "outputs" / "samples.jsonl",
            "results_path": tmp_path / "outputs" / "results.jsonl",
            "analysis_output_dir": tmp_path / "outputs" / "analysis",
            "question_types": ("domain", "modality", "classification"),
            "severities": ("weak", "medium", "strong"),
            "include_mixed_degradations": False,
            "allow_synthetic_benchmark_rows": False,
            "verify_images": True,
            "model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "api_base": "http://localhost:8007/v1",
            "api_key": "EMPTY",
            "request_timeout_seconds": 10.0,
            "max_retries": 2,
            "concurrency": 2,
            "max_tokens": 16,
            "temperature": 0.0,
            "strict_completeness": True,
            "bootstrap_resamples": 100,
            "confidence_level": 0.95,
            "figure_dpi": 100,
        }
        defaults.update(overrides)
        settings = Settings(**defaults)  # type: ignore[arg-type]
        settings.validate()
        return settings

    return factory

