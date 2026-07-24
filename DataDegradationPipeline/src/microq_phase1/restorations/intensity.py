"""Intensity-domain enhancement for dimming and detector saturation."""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.exposure import adjust_gamma, equalize_adapthist

from microq_phase1.domain.strategies import ImageFrame, RestorationStrategy


class GammaClaheRestoration(RestorationStrategy):
    name = "gamma_clahe"

    def restore(
        self,
        frame: ImageFrame,
        degradation_parameters: dict[str, Any],
        algorithm_parameters: dict[str, Any],
    ) -> ImageFrame:
        retained = float(degradation_parameters.get("retained_fraction", 1.0))
        default_gamma = max(0.35, min(1.0, retained))
        gamma = float(algorithm_parameters.get("gamma", default_gamma))
        clip_limit = float(algorithm_parameters.get("clip_limit", 0.01))
        corrected = adjust_gamma(frame.pixels, gamma=gamma)
        if corrected.ndim == 2:
            output = equalize_adapthist(corrected, clip_limit=clip_limit)
        else:
            output = np.stack(
                [
                    equalize_adapthist(corrected[..., index], clip_limit=clip_limit)
                    for index in range(3)
                ],
                axis=-1,
            )
        return ImageFrame(np.clip(output, 0.0, 1.0).astype(np.float32), frame.record)

