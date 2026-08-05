"""Full-reference fidelity and edge-retention metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import jensenshannon
from skimage.filters import sobel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from microq_phase1.domain.strategies import ImageFrame
from microq_phase1.infrastructure.images import grayscale


class QualityMetricCalculator:
    """Compute paired measurements against the undegraded source."""

    def calculate(self, clean: ImageFrame, candidate: ImageFrame) -> dict[str, float]:
        if clean.pixels.shape != candidate.pixels.shape:
            raise ValueError("Full-reference metrics require identical image shapes.")
        channel_axis = -1 if clean.pixels.ndim == 3 else None
        window = _ssim_window(clean.pixels.shape[:2])
        clean_gray = grayscale(clean.pixels)
        candidate_gray = grayscale(candidate.pixels)
        clean_edge = float(np.mean(np.abs(sobel(clean_gray))))
        candidate_edge = float(np.mean(np.abs(sobel(candidate_gray))))
        mse = float(np.mean(np.square(clean.pixels - candidate.pixels)))
        psnr = (
            100.0
            if mse == 0.0
            else float(peak_signal_noise_ratio(clean.pixels, candidate.pixels, data_range=1.0))
        )
        return {
            "mse": mse,
            "psnr": psnr,
            "ssim": float(
                structural_similarity(
                    clean.pixels,
                    candidate.pixels,
                    data_range=1.0,
                    channel_axis=channel_axis,
                    win_size=window,
                )
            ),
            "edge_retention_ratio": float(
                candidate_edge / (clean_edge + np.finfo(np.float32).eps)
            ),
            "histogram_js_divergence": self._histogram_divergence(
                clean_gray,
                candidate_gray,
            ),
        }

    @staticmethod
    def _histogram_divergence(
        clean: NDArray[np.float32], candidate: NDArray[np.float32]
    ) -> float:
        clean_hist, _ = np.histogram(clean, bins=256, range=(0.0, 1.0), density=False)
        candidate_hist, _ = np.histogram(candidate, bins=256, range=(0.0, 1.0), density=False)
        clean_probability = (clean_hist + 1e-12) / (clean_hist.sum() + 256e-12)
        candidate_probability = (candidate_hist + 1e-12) / (
            candidate_hist.sum() + 256e-12
        )
        return float(jensenshannon(clean_probability, candidate_probability, base=2.0) ** 2)


def _ssim_window(shape: tuple[int, int]) -> int:
    smallest = min(shape)
    window = min(7, smallest if smallest % 2 == 1 else smallest - 1)
    if window < 3:
        raise ValueError("Images must be at least 3x3 for SSIM.")
    return window
