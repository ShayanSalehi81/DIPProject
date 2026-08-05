"""Spatial convolution degradations representing focus and acquisition drift."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve

from microq_phase1.degradations.kernels import disk_psf, motion_psf
from microq_phase1.domain.models import DegradationKind
from microq_phase1.domain.strategies import DegradationStrategy, ImageFrame


def convolve_channels(
    pixels: NDArray[np.float32], kernel: NDArray[np.float32]
) -> NDArray[np.float32]:
    if pixels.ndim == 2:
        return np.asarray(convolve(pixels, kernel, mode="reflect"), dtype=np.float32)
    channels = [convolve(pixels[..., index], kernel, mode="reflect") for index in range(3)]
    return np.asarray(np.stack(channels, axis=-1), dtype=np.float32)


class DefocusBlur(DegradationStrategy):
    kind = DegradationKind.DEFOCUS

    def supports(self, frame: ImageFrame) -> bool:
        return frame.pixels.size > 0

    def apply(
        self,
        frame: ImageFrame,
        parameters: dict[str, Any],
        rng: np.random.Generator,
    ) -> ImageFrame:
        del rng
        radius_fraction = float(parameters["radius_fraction"])
        radius = max(1, round(min(frame.pixels.shape[:2]) * radius_fraction))
        output = convolve_channels(frame.pixels, disk_psf(radius))
        return ImageFrame(np.clip(output, 0.0, 1.0).astype(np.float32), frame.record)


class MotionBlur(DegradationStrategy):
    kind = DegradationKind.MOTION

    def supports(self, frame: ImageFrame) -> bool:
        return frame.pixels.size > 0

    def apply(
        self,
        frame: ImageFrame,
        parameters: dict[str, Any],
        rng: np.random.Generator,
    ) -> ImageFrame:
        del rng
        length_fraction = float(parameters["length_fraction"])
        length = max(2, round(min(frame.pixels.shape[:2]) * length_fraction))
        angle = float(parameters["angle_degrees"])
        output = convolve_channels(frame.pixels, motion_psf(length, angle))
        return ImageFrame(np.clip(output, 0.0, 1.0).astype(np.float32), frame.record)
