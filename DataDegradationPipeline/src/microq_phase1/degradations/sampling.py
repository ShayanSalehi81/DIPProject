"""Sampling and quantization degradations from Digital Image Fundamentals."""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.transform import resize

from microq_phase1.domain.models import DegradationKind
from microq_phase1.domain.strategies import DegradationStrategy, ImageFrame


class SamplingDegradation(DegradationStrategy):
    kind = DegradationKind.SAMPLING

    def supports(self, frame: ImageFrame) -> bool:
        return min(frame.pixels.shape[:2]) >= 16

    def apply(
        self,
        frame: ImageFrame,
        parameters: dict[str, Any],
        rng: np.random.Generator,
    ) -> ImageFrame:
        del rng
        factor = float(parameters["factor"])
        if factor <= 1:
            raise ValueError("Sampling factor must be greater than one.")
        height, width = frame.pixels.shape[:2]
        target = (max(1, round(height / factor)), max(1, round(width / factor)))
        if frame.pixels.ndim == 3:
            target = (*target, frame.pixels.shape[2])
        reduced = resize(
            frame.pixels,
            target,
            order=1,
            anti_aliasing=bool(parameters.get("anti_aliasing", True)),
            preserve_range=True,
        )
        restored = resize(
            reduced,
            frame.pixels.shape,
            order=3,
            anti_aliasing=False,
            preserve_range=True,
        )
        return ImageFrame(np.clip(restored, 0.0, 1.0).astype(np.float32), frame.record)


class BitDepthQuantization(DegradationStrategy):
    kind = DegradationKind.QUANTIZATION

    def supports(self, frame: ImageFrame) -> bool:
        return frame.pixels.size > 0

    def apply(
        self,
        frame: ImageFrame,
        parameters: dict[str, Any],
        rng: np.random.Generator,
    ) -> ImageFrame:
        del rng
        bits = int(parameters["bits"])
        if bits < 1 or bits > 8:
            raise ValueError("Quantization bits must be between one and eight.")
        levels = (2**bits) - 1
        output = np.rint(frame.pixels * levels) / levels
        return ImageFrame(output.astype(np.float32), frame.record)
