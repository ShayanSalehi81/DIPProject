"""Composite microscopy degradation fingerprint."""

from __future__ import annotations

import math

from microq_phase1.analysis.artifacts import ArtifactFeatureExtractor
from microq_phase1.analysis.frequency import FrequencyFeatureExtractor
from microq_phase1.analysis.spatial import SpatialFeatureExtractor
from microq_phase1.analysis.wavelet import WaveletFeatureExtractor
from microq_phase1.domain.strategies import FeatureExtractorStrategy, ImageFrame


class CompositeFeatureExtractor:
    """Aggregate independent, syllabus-aligned feature groups."""

    def __init__(self, *, wavelet: str, wavelet_level: int, jpeg_block_size: int) -> None:
        self._extractors: list[FeatureExtractorStrategy] = [
            SpatialFeatureExtractor(),
            FrequencyFeatureExtractor(),
            WaveletFeatureExtractor(wavelet, wavelet_level),
            ArtifactFeatureExtractor(jpeg_block_size),
        ]

    def extract(self, frame: ImageFrame) -> dict[str, float]:
        features: dict[str, float] = {}
        for extractor in self._extractors:
            current = extractor.extract(frame)
            for key, value in current.items():
                if key in features:
                    raise ValueError(f"Duplicate feature name: {key}")
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite feature {key}: {value}")
                features[key] = value
        return features

