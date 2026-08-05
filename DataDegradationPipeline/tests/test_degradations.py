from __future__ import annotations

import numpy as np
import pytest

from microq_phase1.degradations.compression import JpegCompression
from microq_phase1.degradations.convolution import DefocusBlur, MotionBlur
from microq_phase1.degradations.frequency import PeriodicInterference
from microq_phase1.degradations.kernels import motion_psf
from microq_phase1.degradations.microscopy import (
    ChromaticMisregistration,
    Photobleaching,
    SaturationClipping,
)
from microq_phase1.degradations.noise import PoissonGaussianNoise
from microq_phase1.degradations.sampling import BitDepthQuantization, SamplingDegradation
from microq_phase1.domain.strategies import ImageFrame


@pytest.mark.parametrize(
    ("strategy", "parameters"),
    [
        (SamplingDegradation(), {"factor": 2.0, "anti_aliasing": True}),
        (BitDepthQuantization(), {"bits": 4}),
        (PoissonGaussianNoise(), {"peak_photons": 30.0, "read_noise_std": 0.01}),
        (DefocusBlur(), {"radius_fraction": 0.02}),
        (MotionBlur(), {"length_fraction": 0.05, "angle_degrees": 45.0}),
        (
            PeriodicInterference(),
            {
                "cycles_per_image": 8.0,
                "angle_degrees": 30.0,
                "amplitude": 0.05,
                "phase_radians": 0.0,
            },
        ),
        (JpegCompression(), {"quality": 20, "subsampling": 0}),
        (Photobleaching(), {"retained_fraction": 0.5, "background": 0.0}),
        (ChromaticMisregistration(), {"shift_fraction": 0.02, "angle_degrees": 0.0}),
        (SaturationClipping(), {"gain": 1.5, "black_level": 0.0}),
    ],
)
def test_degradation_preserves_contract(
    strategy: object,
    parameters: dict[str, object],
    color_frame: ImageFrame,
) -> None:
    assert strategy.supports(color_frame)
    result = strategy.apply(color_frame, parameters, np.random.default_rng(7))
    assert result.pixels.shape == color_frame.pixels.shape
    assert result.pixels.dtype == np.float32
    assert float(result.pixels.min()) >= 0.0
    assert float(result.pixels.max()) <= 1.0
    assert not np.shares_memory(result.pixels, color_frame.pixels)


def test_noise_is_deterministic_for_same_seed(color_frame: ImageFrame) -> None:
    strategy = PoissonGaussianNoise()
    parameters = {"peak_photons": 20.0, "read_noise_std": 0.02}
    first = strategy.apply(color_frame, parameters, np.random.default_rng(42))
    second = strategy.apply(color_frame, parameters, np.random.default_rng(42))
    np.testing.assert_array_equal(first.pixels, second.pixels)


def test_two_pixel_motion_psf_is_not_identity() -> None:
    kernel = motion_psf(2, 15.0)

    assert np.count_nonzero(kernel) == 3
    assert float(kernel[1, 1]) == pytest.approx(0.5)
    assert float(kernel.sum()) == pytest.approx(1.0)


def test_photobleaching_rejects_non_fluorescence(color_frame: ImageFrame) -> None:
    light_record = color_frame.record.model_copy(update={"modality": "light"})
    light_frame = ImageFrame(color_frame.pixels.copy(), light_record)
    assert not Photobleaching().supports(light_frame)
