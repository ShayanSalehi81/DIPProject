"""Lossy JPEG round-trip degradation."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from microq_phase1.domain.models import DegradationKind
from microq_phase1.domain.strategies import DegradationStrategy, ImageFrame


class JpegCompression(DegradationStrategy):
    kind = DegradationKind.JPEG

    def supports(self, frame: ImageFrame) -> bool:
        return min(frame.pixels.shape[:2]) >= 8

    def apply(
        self,
        frame: ImageFrame,
        parameters: dict[str, Any],
        rng: np.random.Generator,
    ) -> ImageFrame:
        del rng
        quality = int(parameters["quality"])
        if quality < 1 or quality > 100:
            raise ValueError("JPEG quality must be between one and 100.")
        pixels = np.rint(frame.pixels * 255.0).astype(np.uint8)
        mode = "L" if pixels.ndim == 2 else "RGB"
        with BytesIO() as buffer:
            Image.fromarray(pixels, mode=mode).save(
                buffer,
                format="JPEG",
                quality=quality,
                optimize=False,
                subsampling=int(parameters.get("subsampling", 0)),
            )
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                output = np.asarray(decoded.convert(mode), dtype=np.float32) / 255.0
        return ImageFrame(output.astype(np.float32), frame.record)
