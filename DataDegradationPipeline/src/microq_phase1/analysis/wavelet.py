"""Multiresolution wavelet-energy fingerprint."""

from __future__ import annotations

import numpy as np
import pywt

from microq_phase1.domain.strategies import FeatureExtractorStrategy, ImageFrame
from microq_phase1.infrastructure.images import grayscale


class WaveletFeatureExtractor(FeatureExtractorStrategy):
    name = "wavelet"

    def __init__(self, wavelet: str, level: int) -> None:
        self._wavelet = wavelet
        self._level = level

    def extract(self, frame: ImageFrame) -> dict[str, float]:
        gray = grayscale(frame.pixels)
        maximum = pywt.dwtn_max_level(gray.shape, self._wavelet)
        level = min(self._level, maximum)
        if level < 1:
            return {"wavelet_detail_energy_ratio": 0.0, "wavelet_diagonal_fraction": 0.0}
        coefficients = pywt.wavedec2(gray, self._wavelet, level=level)
        approximation_energy = float(np.sum(np.square(coefficients[0])))
        detail_energies: list[float] = []
        diagonal_energy = 0.0
        for horizontal, vertical, diagonal in coefficients[1:]:
            detail_energies.extend(
                [
                    float(np.sum(np.square(horizontal))),
                    float(np.sum(np.square(vertical))),
                    float(np.sum(np.square(diagonal))),
                ]
            )
            diagonal_energy += float(np.sum(np.square(diagonal)))
        detail_total = sum(detail_energies)
        total = approximation_energy + detail_total + np.finfo(np.float64).eps
        return {
            "wavelet_detail_energy_ratio": float(detail_total / total),
            "wavelet_diagonal_fraction": float(
                diagonal_energy / (detail_total + np.finfo(np.float64).eps)
            ),
        }
