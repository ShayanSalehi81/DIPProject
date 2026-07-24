"""Physically motivated photon shot noise plus electronic read noise."""

from __future__ import annotations

from typing import Any

import numpy as np

from microq_phase1.domain.models import DegradationKind
from microq_phase1.domain.strategies import DegradationStrategy, ImageFrame


class PoissonGaussianNoise(DegradationStrategy):
    kind = DegradationKind.POISSON_GAUSSIAN

    def supports(self, frame: ImageFrame) -> bool:
        return frame.pixels.size > 0

    def apply(
        self,
        frame: ImageFrame,
        parameters: dict[str, Any],
        rng: np.random.Generator,
    ) -> ImageFrame:
        peak_photons = float(parameters["peak_photons"])
        read_noise_std = float(parameters["read_noise_std"])
        if peak_photons <= 0 or read_noise_std < 0:
            raise ValueError("Noise parameters must have positive photons and nonnegative sigma.")
        shot = rng.poisson(frame.pixels.astype(np.float64) * peak_photons) / peak_photons
        read = rng.normal(0.0, read_noise_std, size=frame.pixels.shape)
        output = np.clip(shot + read, 0.0, 1.0)
        return ImageFrame(output.astype(np.float32), frame.record)
