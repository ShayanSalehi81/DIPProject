"""Phase-correlation correction of fluorescence-channel misregistration."""

from __future__ import annotations

from typing import Any

import numpy as np
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
        del degradation_parameters
        if frame.pixels.ndim != 3 or frame.pixels.shape[2] < 3:
            raise ValueError("Channel registration requires an RGB image.")
        upsample = int(algorithm_parameters.get("upsample_factor", 20))
        reference = frame.pixels[..., 1]
        channels = []
        for index in range(3):
            if index == 1:
                channels.append(reference.copy())
                continue
            displacement, _, _ = phase_cross_correlation(
                reference,
                frame.pixels[..., index],
                upsample_factor=upsample,
            )
            channels.append(
                shift(frame.pixels[..., index], displacement, order=1, mode="reflect")
            )
        output = np.stack(channels, axis=-1)
        return ImageFrame(np.clip(output, 0.0, 1.0).astype(np.float32), frame.record)

