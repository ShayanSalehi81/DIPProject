"""Exact decoded-pixel and perceptual image signatures."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import imagehash
import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True, slots=True)
class ImageSignature:
    image_id: str
    pool_name: str
    pixel_sha256: str
    perceptual_hash: str


class ImageHasher:
    """Compute format-independent pixel hashes and 64-bit perceptual hashes."""

    def signature(self, image_id: str, pool_name: str, path: Path) -> ImageSignature:
        decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(f"Unable to decode image for hashing: {path}")
        if decoded.ndim == 3 and decoded.shape[2] == 4:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB)
        elif decoded.ndim == 3 and decoded.shape[2] == 3:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened)
            pixel_hash = hashlib.sha256()
            pixel_hash.update(str(decoded.shape).encode("ascii"))
            pixel_hash.update(str(decoded.dtype).encode("ascii"))
            pixel_hash.update(decoded.tobytes(order="C"))
            perceptual = str(imagehash.phash(image.convert("L"), hash_size=8))
        return ImageSignature(
            image_id=image_id,
            pool_name=pool_name,
            pixel_sha256=pixel_hash.hexdigest(),
            perceptual_hash=perceptual,
        )

    def file_sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
