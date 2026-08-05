"""Normalized point-spread functions shared by degradation and restoration."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage.draw import disk, line


def disk_psf(radius: int) -> NDArray[np.float32]:
    if radius < 1:
        raise ValueError("Disk PSF radius must be positive.")
    size = radius * 2 + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    rows, columns = disk((radius, radius), radius + 0.25, shape=kernel.shape)
    kernel[rows, columns] = 1.0
    return _normalize(kernel)


def motion_psf(length: int, angle_degrees: float) -> NDArray[np.float32]:
    if length < 2:
        raise ValueError("Motion PSF length must be at least two pixels.")
    size = length if length % 2 == 1 else length + 1
    center = size // 2
    angle = np.deg2rad(angle_degrees)
    offset_x = int(round(np.cos(angle) * (length - 1) / 2))
    offset_y = int(round(np.sin(angle) * (length - 1) / 2))
    kernel = np.zeros((size, size), dtype=np.float32)
    if offset_x == 0 and offset_y == 0:
        # A two-pixel request previously collapsed to a one-tap identity PSF because
        # Python rounds 0.5 to zero. A centered three-tap approximation keeps the
        # smallest configured motion subtle, measurable, and free of a half-pixel shift.
        direction_x = int(np.sign(np.cos(angle)))
        direction_y = int(np.sign(np.sin(angle)))
        if abs(np.cos(angle)) >= 2 * abs(np.sin(angle)):
            direction_y = 0
        elif abs(np.sin(angle)) >= 2 * abs(np.cos(angle)):
            direction_x = 0
        kernel[center, center] = 0.5
        kernel[center - direction_y, center - direction_x] = 0.25
        kernel[center + direction_y, center + direction_x] = 0.25
        return _normalize(kernel)
    rows, columns = line(
        center - offset_y,
        center - offset_x,
        center + offset_y,
        center + offset_x,
    )
    kernel[rows, columns] = 1.0
    return _normalize(kernel)


def _normalize(kernel: NDArray[np.float32]) -> NDArray[np.float32]:
    total = float(kernel.sum())
    if total <= 0:
        raise ValueError("PSF must have positive mass.")
    return (kernel / total).astype(np.float32)
