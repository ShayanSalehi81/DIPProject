"""Histogram, contrast, gradient, and sharpening-domain features."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import laplace
from skimage.filters import sobel
from skimage.measure import shannon_entropy

from microq_phase1.domain.strategies import FeatureExtractorStrategy, ImageFrame
from microq_phase1.infrastructure.images import grayscale


class SpatialFeatureExtractor(FeatureExtractorStrategy):
    name = "spatial"

    def extract(self, frame: ImageFrame) -> dict[str, float]:
        gray = grayscale(frame.pixels)
        gradient = sobel(gray)
        low, high = np.quantile(gray, [0.01, 0.99])
        return {
            "intensity_mean": float(np.mean(gray)),
            "intensity_std": float(np.std(gray)),
            "rms_contrast": float(np.std(gray)),
            "dynamic_range_p99_p01": float(high - low),
            "entropy": float(shannon_entropy(np.rint(gray * 255).astype(np.uint8))),
            "gradient_mean": float(np.mean(np.abs(gradient))),
            "gradient_energy": float(np.mean(np.square(gradient))),
            "laplacian_variance": float(np.var(laplace(gray))),
            "black_clip_fraction": float(np.mean(gray <= (1.0 / 255.0))),
            "white_clip_fraction": float(np.mean(gray >= (254.0 / 255.0))),
        }

