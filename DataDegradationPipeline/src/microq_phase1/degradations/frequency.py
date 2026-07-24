"""Frequency-domain periodic interference degradation."""

from __future__ import annotations

from typing import Any

import numpy as np

from microq_phase1.domain.models import DegradationKind
from microq_phase1.domain.strategies import DegradationStrategy, ImageFrame


class PeriodicInterference(DegradationStrategy):
    kind = DegradationKind.PERIODIC

    def supports(self, frame: ImageFrame) -> bool:
        return min(frame.pixels.shape[:2]) >= 16

    def apply(
        self,
        frame: ImageFrame,
        parameters: dict[str, Any],
        rng: np.random.Generator,
    ) -> ImageFrame:
        height, width = frame.pixels.shape[:2]
        cycles = float(parameters["cycles_per_image"])
        angle = np.deg2rad(float(parameters["angle_degrees"]))
        amplitude = float(parameters["amplitude"])
        phase = (
            float(parameters["phase_radians"])
            if "phase_radians" in parameters
            else float(rng.uniform(0.0, 2.0 * np.pi))
        )
        yy, xx = np.mgrid[0:height, 0:width]
        coordinate = (xx / width) * np.cos(angle) + (yy / height) * np.sin(angle)
        pattern = amplitude * np.sin((2.0 * np.pi * cycles * coordinate) + phase)
        if frame.pixels.ndim == 3:
            pattern = pattern[..., None]
        output = np.clip(frame.pixels + pattern, 0.0, 1.0)
        return ImageFrame(output.astype(np.float32), frame.record)
