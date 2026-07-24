"""Morphological structure retention against optional reference masks."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, disk

from microq_phase1.domain.strategies import ImageFrame
from microq_phase1.infrastructure.images import grayscale


class MorphologyMetricCalculator:
    """Measure whether degradation preserves mask-scale biological structure."""

    def calculate(self, frame: ImageFrame, reference_mask: np.ndarray) -> dict[str, float]:
        gray = grayscale(frame.pixels)
        if gray.shape != reference_mask.shape:
            raise ValueError(
                f"Reference mask shape {reference_mask.shape} does not match image {gray.shape}."
            )
        threshold = threshold_otsu(gray)
        candidate = binary_closing(binary_opening(gray > threshold, disk(1)), disk(1))
        if self._dice(~candidate, reference_mask) > self._dice(candidate, reference_mask):
            candidate = ~candidate
        intersection = int(np.logical_and(candidate, reference_mask).sum())
        union = int(np.logical_or(candidate, reference_mask).sum())
        reference_area = int(reference_mask.sum())
        _, candidate_components = label(candidate)
        _, reference_components = label(reference_mask)
        return {
            "morphology_dice": self._dice(candidate, reference_mask),
            "morphology_iou": intersection / max(union, 1),
            "morphology_area_ratio": float(candidate.sum()) / max(reference_area, 1),
            "morphology_component_ratio": candidate_components / max(reference_components, 1),
        }

    @staticmethod
    def _dice(left: np.ndarray, right: np.ndarray) -> float:
        intersection = int(np.logical_and(left, right).sum())
        return (2.0 * intersection) / max(int(left.sum()) + int(right.sum()), 1)
