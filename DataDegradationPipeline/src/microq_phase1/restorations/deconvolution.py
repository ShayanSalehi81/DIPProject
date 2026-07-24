"""Wiener deconvolution using recorded point-spread-function parameters."""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.restoration import wiener

from microq_phase1.degradations.kernels import disk_psf, motion_psf
from microq_phase1.domain.strategies import ImageFrame, RestorationStrategy


class WienerDeconvolution(RestorationStrategy):
    name = "wiener_deconvolution"

    def restore(
        self,
        frame: ImageFrame,
        degradation_parameters: dict[str, Any],
        algorithm_parameters: dict[str, Any],
    ) -> ImageFrame:
        psf = self._psf(frame, degradation_parameters)
        balance = float(algorithm_parameters.get("balance", 0.02))
        clip = bool(algorithm_parameters.get("clip", True))
        if frame.pixels.ndim == 2:
            output = wiener(frame.pixels, psf, balance=balance, clip=clip)
        else:
            output = np.stack(
                [
                    wiener(frame.pixels[..., index], psf, balance=balance, clip=clip)
                    for index in range(3)
                ],
                axis=-1,
            )
        return ImageFrame(np.clip(output, 0.0, 1.0).astype(np.float32), frame.record)

    @staticmethod
    def _psf(frame: ImageFrame, parameters: dict[str, Any]) -> np.ndarray:
        scale = min(frame.pixels.shape[:2])
        if "radius_fraction" in parameters:
            radius = max(1, round(scale * float(parameters["radius_fraction"])))
            return disk_psf(radius)
        if "length_fraction" in parameters:
            length = max(2, round(scale * float(parameters["length_fraction"])))
            return motion_psf(length, float(parameters["angle_degrees"]))
        raise ValueError("Wiener deconvolution requires recorded blur PSF parameters.")

