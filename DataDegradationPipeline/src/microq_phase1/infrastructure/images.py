"""Lossless image loading, normalized conversion, and atomic persistence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from microq_phase1.domain.models import ImageRecord
from microq_phase1.domain.strategies import FloatImage, ImageFrame


class PillowImageRepository:
    """Filesystem repository preserving grayscale/RGB structure and source resolution."""

    def load(self, record: ImageRecord) -> ImageFrame:
        if record.bit_depth == 16:
            return self._load_sixteen_bit(record)
        with Image.open(record.image_path) as opened:
            image = cast(Image.Image, ImageOps.exif_transpose(opened))
            if image.mode.startswith("I;16"):
                raise ValueError(
                    f"Catalog declares 8-bit but image is 16-bit: {record.image_path}"
                )
            if image.mode not in {"L", "RGB"}:
                grayscale_modes = {"1", "I", "F"}
                image = image.convert("L" if image.mode in grayscale_modes else "RGB")
            array = np.asarray(image, dtype=np.float32) / 255.0
        return ImageFrame(pixels=array.astype(np.float32), record=record)

    def save(self, frame: ImageFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        if frame.record.bit_depth == 16:
            self._save_sixteen_bit(frame, path, temporary)
            return
        pixels = np.rint(np.clip(frame.pixels, 0.0, 1.0) * 255.0).astype(np.uint8)
        mode = "L" if pixels.ndim == 2 else "RGB"
        file_format = "JPEG" if path.suffix.lower() in {".jpg", ".jpeg"} else "PNG"
        Image.fromarray(pixels, mode=mode).save(temporary, format=file_format)
        os.replace(temporary, path)

    def load_mask(self, path: Path) -> NDArray[np.bool_]:
        with Image.open(path) as opened:
            return np.asarray(opened.convert("L"), dtype=np.uint8) > 0

    @staticmethod
    def _load_sixteen_bit(record: ImageRecord) -> ImageFrame:
        decoded = cv2.imread(str(record.image_path), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(f"OpenCV could not decode 16-bit image: {record.image_path}")
        if decoded.dtype != np.uint16:
            raise ValueError(
                f"Catalog declares 16-bit but decoded dtype is {decoded.dtype}: {record.image_path}"
            )
        if decoded.ndim == 3:
            if decoded.shape[2] == 4:
                decoded = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB)
            elif decoded.shape[2] == 3:
                decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"Unsupported 16-bit channel count: {decoded.shape[2]}")
        pixels = decoded.astype(np.float32) / 65535.0
        return ImageFrame(pixels=pixels, record=record)

    @staticmethod
    def _save_sixteen_bit(frame: ImageFrame, path: Path, temporary: Path) -> None:
        pixels = np.rint(np.clip(frame.pixels, 0.0, 1.0) * 65535.0).astype(np.uint16)
        if pixels.ndim == 3:
            pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".png", pixels)
        if not success:
            raise ValueError(f"OpenCV could not encode 16-bit PNG: {path}")
        temporary.write_bytes(encoded.tobytes())
        os.replace(temporary, path)


def grayscale(pixels: FloatImage) -> FloatImage:
    """Convert normalized RGB to luminance while leaving grayscale unchanged."""
    if pixels.ndim == 2:
        return pixels
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.tensordot(pixels[..., :3], weights, axes=([-1], [0])).astype(np.float32)
