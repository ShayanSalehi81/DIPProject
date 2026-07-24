"""Abstract strategies for degradation, restoration, and feature extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from microq_phase1.domain.models import DegradationKind, ImageRecord

FloatImage = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class ImageFrame:
    """Normalized grayscale or RGB image passed between strategies."""

    pixels: FloatImage
    record: ImageRecord

    def __post_init__(self) -> None:
        if self.pixels.dtype != np.float32:
            raise TypeError("ImageFrame pixels must be float32.")
        if self.pixels.ndim not in (2, 3):
            raise ValueError("ImageFrame must contain a 2-D grayscale or 3-D color image.")
        if self.pixels.size == 0:
            raise ValueError("ImageFrame cannot be empty.")
        if not np.all(np.isfinite(self.pixels)):
            raise ValueError("ImageFrame pixels must all be finite.")
        if float(self.pixels.min()) < 0.0 or float(self.pixels.max()) > 1.0:
            raise ValueError("ImageFrame pixels must be normalized to [0, 1].")


class DegradationStrategy(ABC):
    """Interface implemented by every degradation operator."""

    kind: DegradationKind

    @abstractmethod
    def supports(self, frame: ImageFrame) -> bool:
        """Return whether the operation is physically applicable."""

    @abstractmethod
    def apply(
        self,
        frame: ImageFrame,
        parameters: dict[str, Any],
        rng: np.random.Generator,
    ) -> ImageFrame:
        """Apply the degradation without mutating the input frame."""


class RestorationStrategy(ABC):
    """Interface implemented by every classical restoration algorithm."""

    name: str

    @abstractmethod
    def restore(
        self,
        frame: ImageFrame,
        degradation_parameters: dict[str, Any],
        algorithm_parameters: dict[str, Any],
    ) -> ImageFrame:
        """Restore an image without mutating the input frame."""


class FeatureExtractorStrategy(ABC):
    """Interface for one interpretable group of DIP features."""

    name: str

    @abstractmethod
    def extract(self, frame: ImageFrame) -> dict[str, float]:
        """Return finite scalar features."""
