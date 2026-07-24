"""Microscopy-specific intensity and channel degradations."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import shift

from microq_phase1.domain.models import DegradationKind, Modality
from microq_phase1.domain.strategies import DegradationStrategy, ImageFrame


class Photobleaching(DegradationStrategy):
    kind = DegradationKind.PHOTOBLEACHING

    def supports(self, frame: ImageFrame) -> bool:
        return frame.record.modality == Modality.FLUORESCENCE

    def apply(
        self,
        frame: ImageFrame,
        parameters: dict[str, Any],
        rng: np.random.Generator,
    ) -> ImageFrame:
        del rng
        retained = float(parameters["retained_fraction"])
        background = float(parameters.get("background", 0.0))
        if retained <= 0 or retained > 1:
            raise ValueError("Photobleaching retained_fraction must be in (0, 1].")
        output = background + retained * (frame.pixels - background)
        return ImageFrame(np.clip(output, 0.0, 1.0).astype(np.float32), frame.record)


class ChromaticMisregistration(DegradationStrategy):
    kind = DegradationKind.CHROMATIC_MISREGISTRATION

    def supports(self, frame: ImageFrame) -> bool:
        return (
            frame.record.modality == Modality.FLUORESCENCE
            and frame.record.is_multichannel
            and frame.pixels.ndim == 3
        )

    def apply(
        self,
        frame: ImageFrame,
        parameters: dict[str, Any],
        rng: np.random.Generator,
    ) -> ImageFrame:
        del rng
        fraction = float(parameters["shift_fraction"])
        angle = np.deg2rad(float(parameters.get("angle_degrees", 0.0)))
        magnitude = max(1, round(min(frame.pixels.shape[:2]) * fraction))
        dy = float(magnitude * np.sin(angle))
        dx = float(magnitude * np.cos(angle))
        channels = [
            shift(frame.pixels[..., 0], (dy, dx), order=1, mode="reflect"),
            frame.pixels[..., 1].copy(),
            shift(frame.pixels[..., 2], (-dy, -dx), order=1, mode="reflect"),
        ]
        return ImageFrame(np.stack(channels, axis=-1).astype(np.float32), frame.record)


class SaturationClipping(DegradationStrategy):
    kind = DegradationKind.SATURATION

    def supports(self, frame: ImageFrame) -> bool:
        return frame.pixels.size > 0

    def apply(
        self,
        frame: ImageFrame,
        parameters: dict[str, Any],
        rng: np.random.Generator,
    ) -> ImageFrame:
        del rng
        gain = float(parameters.get("gain", 1.0))
        black_level = float(parameters.get("black_level", 0.0))
        output = np.clip((frame.pixels - black_level) * gain, 0.0, 1.0)
        return ImageFrame(output.astype(np.float32), frame.record)
