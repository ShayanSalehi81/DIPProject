from __future__ import annotations

import pytest

from microq_phase1.application.configuration import ExperimentConfigurationValidator
from microq_phase1.core.config import DegradationConfig
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
