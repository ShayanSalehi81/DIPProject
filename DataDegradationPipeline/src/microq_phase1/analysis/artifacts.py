"""Compression-block and chromatic-registration features."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage.registration import phase_cross_correlation

from microq_phase1.domain.strategies import FeatureExtractorStrategy, ImageFrame
from microq_phase1.infrastructure.images import grayscale


class ArtifactFeatureExtractor(FeatureExtractorStrategy):
    name = "artifacts"

    def __init__(self, block_size: int) -> None:
        self._block_size = block_size

    def extract(self, frame: ImageFrame) -> dict[str, float]:
        gray = grayscale(frame.pixels)
        blockiness = self._blockiness(gray)
        shifts = self._channel_shifts(frame.pixels)
        return {
            "jpeg_blockiness": blockiness,
            "chromatic_max_shift": max(shifts, default=0.0),
            "chromatic_mean_shift": float(np.mean(shifts)) if shifts else 0.0,
        }

    def _blockiness(self, gray: NDArray[np.float32]) -> float:
        horizontal = np.abs(np.diff(gray, axis=1))
        vertical = np.abs(np.diff(gray, axis=0))
        boundary_h = horizontal[:, self._block_size - 1 :: self._block_size]
        boundary_v = vertical[self._block_size - 1 :: self._block_size, :]
        nonboundary_h = np.delete(
            horizontal,
            np.arange(self._block_size - 1, horizontal.shape[1], self._block_size),
            axis=1,
        )
        nonboundary_v = np.delete(
            vertical,
            np.arange(self._block_size - 1, vertical.shape[0], self._block_size),
            axis=0,
        )
        boundary_mean = _safe_mean(boundary_h) + _safe_mean(boundary_v)
        interior_mean = _safe_mean(nonboundary_h) + _safe_mean(nonboundary_v)
        return float(boundary_mean / (interior_mean + np.finfo(np.float32).eps))

    @staticmethod
    def _channel_shifts(pixels: NDArray[np.float32]) -> list[float]:
        if pixels.ndim != 3 or pixels.shape[2] < 3:
            return []
        reference = pixels[..., 1]
        shifts: list[float] = []
        for index in (0, 2):
            if np.std(reference) < 1e-8 or np.std(pixels[..., index]) < 1e-8:
                shifts.append(0.0)
                continue
            displacement, _, _ = phase_cross_correlation(
                reference,
                pixels[..., index],
                upsample_factor=10,
            )
            shifts.append(float(np.linalg.norm(displacement)))
        return shifts


def _safe_mean(array: NDArray[np.float32]) -> float:
    return float(np.mean(array)) if array.size else 0.0
