from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from microq_phase1.application.configuration import ExperimentConfigurationValidator
from microq_phase1.core.config import DegradationConfig, ProjectConfig
from microq_phase1.core.errors import ConfigurationError
from microq_phase1.domain.models import DegradationKind, Modality, Severity


def test_invalid_noise_parameters_are_rejected() -> None:
    degradation = DegradationConfig(
        kind=DegradationKind.POISSON_GAUSSIAN,
        applicable_modalities={Modality.FLUORESCENCE},
        severity_parameters={
            Severity.WEAK: {"peak_photons": 80.0, "read_noise_std": 0.0},
            Severity.MEDIUM: {"peak_photons": 30.0, "read_noise_std": 0.01},
            Severity.STRONG: {"peak_photons": -1.0, "read_noise_std": 0.02},
        },
    )
    with pytest.raises(ConfigurationError):
        ExperimentConfigurationValidator._validate_degradation(degradation)


def test_microbench_export_licenses_must_be_reviewed() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "phase1.example.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["microbench_export"]["revision"] = "a" * 40
    config = ProjectConfig.model_validate(payload)
    config = config.model_copy(
        update={
            "microbench_export": config.microbench_export.model_copy(
                update={"included_licenses": {"unreviewed-license"}}
            )
        }
    )

    with pytest.raises(ConfigurationError, match="absent from the reviewed allowlist"):
        ExperimentConfigurationValidator._validate_benchmark_binding(config)
