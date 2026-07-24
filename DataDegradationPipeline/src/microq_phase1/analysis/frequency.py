"""Fourier-domain energy and periodic-interference features."""

from __future__ import annotations

import numpy as np

from microq_phase1.domain.strategies import FeatureExtractorStrategy, ImageFrame
from microq_phase1.infrastructure.images import grayscale


class FrequencyFeatureExtractor(FeatureExtractorStrategy):
    name = "frequency"

    def extract(self, frame: ImageFrame) -> dict[str, float]:
        gray = grayscale(frame.pixels)
        centered = gray - float(np.mean(gray))
        power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
        radius = _normalized_radius(power.shape)
        total = float(np.sum(power)) + np.finfo(np.float64).eps
        high_frequency = float(np.sum(power[radius >= 0.50]) / total)
        mid_frequency = float(np.sum(power[(radius >= 0.20) & (radius < 0.50)]) / total)
        searchable = power[radius >= 0.08]
        median = float(np.median(searchable)) + np.finfo(np.float64).eps
        peak = float(np.quantile(searchable, 0.999))
        return {
            "fourier_high_frequency_ratio": high_frequency,
            "fourier_mid_frequency_ratio": mid_frequency,
            "fourier_spectral_centroid": float(np.sum(radius * power) / total),
            "periodic_peak_to_median": peak / median,
        }


def _normalized_radius(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    denominator = np.hypot(max(cy, 1.0), max(cx, 1.0))
    return np.hypot(yy - cy, xx - cx) / denominator

