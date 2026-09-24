"""Uses the configured workflow."""

from __future__ import annotations

import numpy as np
from PIL import Image

HASH_SIZE = 8
PHASH_HIGHFREQ = 4
DHASH_HAMMING_GATE = 12
PHASH_HAMMING_CONFIRM = 8


def dhash64(image: Image.Image, *, hash_size: int = HASH_SIZE) -> str:
    """Difference hash: 64 bits as 16 lowercase hex characters."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    return _bits_to_hex(bits)


def phash64(image: Image.Image, *, hash_size: int = HASH_SIZE) -> str:
    """DCT perceptual hash: 64 bits as 16 lowercase hex characters."""
    side = hash_size * PHASH_HIGHFREQ
    gray = image.convert("L").resize((side, side), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    transformed = _dct2(pixels)
    low = transformed[:hash_size, :hash_size]
    # Exclude DC so overall brightness does not dominate the hash.
    ac = low.copy()
    ac[0, 0] = np.median(low)
    bits = low > np.median(ac)
    return _bits_to_hex(bits)


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("Perceptual hashes must be the same length.")
    return (int(left, 16) ^ int(right, 16)).bit_count()


def near_duplicate(
    dhash_a: str,
    dhash_b: str,
    phash_a: str,
    phash_b: str,
    *,
    dhash_gate: int = DHASH_HAMMING_GATE,
    phash_confirm: int = PHASH_HAMMING_CONFIRM,
) -> bool:
    """Three-layer confirm: dHash gate then pHash. Exact match is handled upstream."""
    if hamming_distance(dhash_a, dhash_b) > dhash_gate:
        return False
    return hamming_distance(phash_a, phash_b) <= phash_confirm


def _bits_to_hex(bits: np.ndarray) -> str:
    flat = np.packbits(bits.astype(np.uint8).reshape(-1))
    return bytes(flat).hex()


def _dct2(block: np.ndarray) -> np.ndarray:
    """Orthonormal 2-D type-II DCT, implemented with NumPy only."""
    n = block.shape[0]
    matrix = _dct_matrix(n)
    return matrix @ block @ matrix.T


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)[:, None]
    i = np.arange(n, dtype=np.float64)[None, :]
    matrix = np.cos(np.pi / n * (i + 0.5) * k)
    matrix[0] *= 1.0 / np.sqrt(2.0)
    return matrix * np.sqrt(2.0 / n)
