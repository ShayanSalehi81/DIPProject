"""Stable seed derivation independent of process order and Python hash randomization."""

from __future__ import annotations

import hashlib


def derive_seed(master_seed: int, *parts: object) -> int:
    """Derive a stable unsigned 32-bit seed from semantic recipe identifiers."""
    payload = "::".join([str(master_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % (2**32)

