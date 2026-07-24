"""Registry/factory for classical restoration strategies."""

from __future__ import annotations

from microq_phase1.core.errors import RestorationError
from microq_phase1.domain.strategies import RestorationStrategy
from microq_phase1.restorations.deconvolution import WienerDeconvolution
from microq_phase1.restorations.frequency import NotchRejectRestoration
from microq_phase1.restorations.intensity import GammaClaheRestoration
from microq_phase1.restorations.registration import ChannelRegistrationRestoration
from microq_phase1.restorations.spatial import (
    LocalWienerRestoration,
    MedianRestoration,
    WaveletDenoiseRestoration,
)


class RestorationRegistry:
    """Resolve algorithms independently from degradation policy."""

    def __init__(self) -> None:
        strategies: list[RestorationStrategy] = [
            LocalWienerRestoration(),
            MedianRestoration(),
            WaveletDenoiseRestoration(),
            WienerDeconvolution(),
            NotchRejectRestoration(),
            GammaClaheRestoration(),
            ChannelRegistrationRestoration(),
        ]
        self._strategies = {strategy.name: strategy for strategy in strategies}

    def get(self, name: str) -> RestorationStrategy:
        try:
            return self._strategies[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._strategies))
            raise RestorationError(
                f"Unknown restoration algorithm '{name}'. Available: {available}"
            ) from exc

