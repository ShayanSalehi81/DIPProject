"""Known-frequency notch rejection for synthetic periodic interference."""

from __future__ import annotations

from typing import Any

import numpy as np

from microq_phase1.domain.strategies import ImageFrame, RestorationStrategy


class NotchRejectRestoration(RestorationStrategy):
    name = "notch_reject"

    def restore(
        self,
        frame: ImageFrame,
        degradation_parameters: dict[str, Any],
        algorithm_parameters: dict[str, Any],
    ) -> ImageFrame:
        cycles = float(degradation_parameters["cycles_per_image"])
        angle = np.deg2rad(float(degradation_parameters["angle_degrees"]))
        radius = float(algorithm_parameters.get("notch_radius_bins", 2.0))
        output = self._filter(frame.pixels, cycles, angle, radius)
        return ImageFrame(np.clip(output, 0.0, 1.0).astype(np.float32), frame.record)

    def _filter(
        self,
        pixels: np.ndarray,
        cycles: float,
        angle: float,
        radius: float,
    ) -> np.ndarray:
        if pixels.ndim == 2:
            return self._filter_channel(pixels, cycles, angle, radius)
        return np.stack(
            [
                self._filter_channel(pixels[..., index], cycles, angle, radius)
                for index in range(3)
            ],
            axis=-1,
        )

    @staticmethod
    def _filter_channel(
        channel: np.ndarray,
        cycles: float,
        angle: float,
        radius: float,
    ) -> np.ndarray:
        height, width = channel.shape
        spectrum = np.fft.fftshift(np.fft.fft2(channel))
        yy, xx = np.mgrid[0:height, 0:width]
        center_y, center_x = height // 2, width // 2
        offset_x = cycles * np.cos(angle)
        offset_y = cycles * np.sin(angle)
        mask = np.ones((height, width), dtype=np.float32)
        for sign in (-1.0, 1.0):
            distance = np.hypot(
                xx - (center_x + sign * offset_x),
                yy - (center_y + sign * offset_y),
            )
            mask[distance <= radius] = 0.0
        restored = np.fft.ifft2(np.fft.ifftshift(spectrum * mask)).real
        return restored.astype(np.float32)

