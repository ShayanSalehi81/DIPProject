"""Spatial and multiresolution denoising strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import medfilt2d, wiener
from skimage.restoration import denoise_wavelet

from microq_phase1.domain.strategies import ImageFrame, RestorationStrategy


class LocalWienerRestoration(RestorationStrategy):
    name = "local_wiener"

    def restore(
        self,
        frame: ImageFrame,
        degradation_parameters: dict[str, Any],
        algorithm_parameters: dict[str, Any],
    ) -> ImageFrame:
        del degradation_parameters
        size = int(algorithm_parameters.get("window_size", 5))
        output = _per_channel(frame.pixels, lambda channel: wiener(channel, (size, size)))
        output = np.where(np.isfinite(output), output, frame.pixels)
        return ImageFrame(np.clip(output, 0.0, 1.0).astype(np.float32), frame.record)


class MedianRestoration(RestorationStrategy):
    name = "median"

    def restore(
        self,
        frame: ImageFrame,
        degradation_parameters: dict[str, Any],
        algorithm_parameters: dict[str, Any],
    ) -> ImageFrame:
        del degradation_parameters
        size = int(algorithm_parameters.get("kernel_size", 3))
        output = _per_channel(frame.pixels, lambda channel: medfilt2d(channel, size))
        return ImageFrame(np.clip(output, 0.0, 1.0).astype(np.float32), frame.record)


class WaveletDenoiseRestoration(RestorationStrategy):
    name = "wavelet_denoise"

    def restore(
        self,
        frame: ImageFrame,
        degradation_parameters: dict[str, Any],
        algorithm_parameters: dict[str, Any],
    ) -> ImageFrame:
        del degradation_parameters
        wavelet = str(algorithm_parameters.get("wavelet", "db2"))
        mode = str(algorithm_parameters.get("mode", "soft"))
        method = str(algorithm_parameters.get("method", "BayesShrink"))

        def denoise(channel: NDArray[np.float32]) -> NDArray[np.float32]:
            if float(np.ptp(channel)) == 0.0:
                return channel.copy()
            return np.asarray(
                denoise_wavelet(
                    channel,
                    wavelet=wavelet,
                    mode=mode,
                    method=method,
                    channel_axis=None,
                    rescale_sigma=True,
                ),
                dtype=np.float32,
            )

        output = _per_channel(frame.pixels, denoise)
        output = np.where(np.isfinite(output), output, frame.pixels)
        return ImageFrame(np.clip(output, 0.0, 1.0).astype(np.float32), frame.record)


def _per_channel(
    pixels: NDArray[np.float32],
    operation: Any,
) -> NDArray[np.float32]:
    if pixels.ndim == 2:
        return np.asarray(operation(pixels), dtype=np.float32)
    return np.asarray(
        np.stack([operation(pixels[..., index]) for index in range(3)], axis=-1),
        dtype=np.float32,
    )
