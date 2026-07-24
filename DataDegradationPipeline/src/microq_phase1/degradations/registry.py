"""Registry/factory for degradation strategies."""

from __future__ import annotations

from microq_phase1.core.errors import DegradationError
from microq_phase1.degradations.compression import JpegCompression
from microq_phase1.degradations.convolution import DefocusBlur, MotionBlur
from microq_phase1.degradations.frequency import PeriodicInterference
from microq_phase1.degradations.microscopy import (
    ChromaticMisregistration,
    Photobleaching,
    SaturationClipping,
)
from microq_phase1.degradations.noise import PoissonGaussianNoise
from microq_phase1.degradations.sampling import BitDepthQuantization, SamplingDegradation
from microq_phase1.domain.models import DegradationKind
from microq_phase1.domain.strategies import DegradationStrategy


class DegradationRegistry:
    """Resolve configured degradation kinds without conditional pipelines."""

    def __init__(self) -> None:
        strategies: list[DegradationStrategy] = [
            SamplingDegradation(),
            BitDepthQuantization(),
            PoissonGaussianNoise(),
            DefocusBlur(),
            MotionBlur(),
            PeriodicInterference(),
            JpegCompression(),
            Photobleaching(),
            ChromaticMisregistration(),
            SaturationClipping(),
        ]
        self._strategies = {strategy.kind: strategy for strategy in strategies}

    def get(self, kind: DegradationKind) -> DegradationStrategy:
        try:
            return self._strategies[kind]
        except KeyError as exc:
            raise DegradationError(f"No degradation strategy is registered for {kind}.") from exc
