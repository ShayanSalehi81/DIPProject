from __future__ import annotations

import warnings

import numpy as np

from microq_phase1.domain.models import ImageRecord
from microq_phase1.domain.strategies import ImageFrame
from microq_phase1.restorations.registration import ChannelRegistrationRestoration
from microq_phase1.restorations.spatial import WaveletDenoiseRestoration


def test_wavelet_denoise_preserves_constant_frame(
    fluorescence_record: ImageRecord,
) -> None:
    pixels = np.zeros((32, 32), dtype=np.float32)
    frame = ImageFrame(pixels, fluorescence_record)

    restored = WaveletDenoiseRestoration().restore(
        frame,
        degradation_parameters={"bits": 3},
        algorithm_parameters={"wavelet": "db2", "method": "BayesShrink", "mode": "soft"},
    )

    np.testing.assert_array_equal(restored.pixels, pixels)
    assert np.all(np.isfinite(restored.pixels))


def test_channel_registration_preserves_empty_channel_without_warning(
    fluorescence_record: ImageRecord,
) -> None:
    gradient = np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float32), (32, 1))
    pixels = np.stack([np.zeros_like(gradient), gradient, np.roll(gradient, 2, axis=1)], axis=-1)
    frame = ImageFrame(pixels, fluorescence_record)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored = ChannelRegistrationRestoration().restore(
            frame,
            degradation_parameters={"shift_fraction": 2 / 32},
            algorithm_parameters={"upsample_factor": 20},
        )

    np.testing.assert_array_equal(restored.pixels[..., 0], pixels[..., 0])
    assert np.all(np.isfinite(restored.pixels))


def test_channel_registration_rejects_implausibly_large_shift(
    fluorescence_record: ImageRecord,
) -> None:
    rng = np.random.default_rng(7)
    reference = rng.random((32, 32), dtype=np.float32)
    far_channel = np.roll(reference, 12, axis=1)
    near_channel = np.roll(reference, -2, axis=1)
    pixels = np.stack([far_channel, reference, near_channel], axis=-1)
    frame = ImageFrame(pixels, fluorescence_record)

    restored = ChannelRegistrationRestoration().restore(
        frame,
        degradation_parameters={"shift_fraction": 2 / 32},
        algorithm_parameters={"upsample_factor": 20, "maximum_displacement_factor": 2.0},
    )

    np.testing.assert_array_equal(restored.pixels[..., 0], far_channel)
    assert np.mean(np.abs(restored.pixels[..., 2] - reference)) < 0.03
