"""Phase-correlation correction of fluorescence-channel misregistration."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation

from microq_phase1.domain.strategies import ImageFrame, RestorationStrategy


class ChannelRegistrationRestoration(RestorationStrategy):
    name = "channel_registration"

    def restore(
        self,
        frame: ImageFrame,
        degradation_parameters: dict[str, Any],
        algorithm_parameters: dict[str, Any],
    ) -> ImageFrame:
        if frame.pixels.ndim != 3 or frame.pixels.shape[2] < 3:
            raise ValueError("Channel registration requires an RGB image.")
        upsample = int(algorithm_parameters.get("upsample_factor", 20))
        injected_fraction = float(degradation_parameters["shift_fraction"])
        injected_pixels = max(1, round(min(frame.pixels.shape[:2]) * injected_fraction))
        maximum_factor = float(algorithm_parameters.get("maximum_displacement_factor", 2.0))
        if maximum_factor < 1.0:
            raise ValueError("maximum_displacement_factor must be at least one.")
        maximum_displacement = injected_pixels * maximum_factor
        reference = frame.pixels[..., 1]
        channels = []
        for index in range(3):
            if index == 1:
                channels.append(reference.copy())
                continue
            moving = frame.pixels[..., index]
            if not self._has_registration_signal(reference) or not self._has_registration_signal(
                moving
            ):
                channels.append(moving.copy())
                continue
            displacement, _, _ = phase_cross_correlation(
                reference,
                moving,
                upsample_factor=upsample,
            )
            if not np.all(np.isfinite(displacement)):
                channels.append(moving.copy())
                continue
            if float(np.linalg.norm(displacement)) > maximum_displacement:
                channels.append(moving.copy())
                continue
            channels.append(shift(moving, displacement, order=1, mode="reflect"))
        output = np.stack(channels, axis=-1)
        return ImageFrame(np.clip(output, 0.0, 1.0).astype(np.float32), frame.record)

    @staticmethod
    def _has_registration_signal(channel: NDArray[np.float32]) -> bool:
        return float(np.std(channel)) >= 1e-8
