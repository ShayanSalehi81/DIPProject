from __future__ import annotations

import numpy as np

from microq_phase1.analysis import CompositeFeatureExtractor, QualityMetricCalculator
from microq_phase1.domain.strategies import ImageFrame


def test_clean_reference_has_identity_metrics(color_frame: ImageFrame) -> None:
    metrics = QualityMetricCalculator().calculate(color_frame, color_frame)
    assert metrics["mse"] == 0.0
    assert metrics["psnr"] == 100.0
    assert metrics["ssim"] == 1.0


def test_composite_features_are_finite(color_frame: ImageFrame) -> None:
    extractor = CompositeFeatureExtractor(wavelet="db2", wavelet_level=2, jpeg_block_size=8)
    features = extractor.extract(color_frame)
    assert "laplacian_variance" in features
    assert "fourier_high_frequency_ratio" in features
    assert "wavelet_detail_energy_ratio" in features
    assert "jpeg_blockiness" in features
    assert all(np.isfinite(value) for value in features.values())

