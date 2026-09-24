"""Deterministic embeddings for the mock provider.

``hash()`` is salted per process, so it is reproducible within a run and
different across runs — the worst possible fixture. blake2s is stable.
"""

from __future__ import annotations

import hashlib
import math

PERSON = b"pa-vis01"
DEFAULT_DIM = 384
DUPLICATE_COSINE = 0.92


def blake2s_digest(payload: bytes, *, counter: int = 0) -> bytes:
    return hashlib.blake2s(
        payload + counter.to_bytes(4, "big"),
        digest_size=32,
        person=PERSON,
    ).digest()


def deterministic_embedding(
    payload: bytes,
    *,
    dim: int = DEFAULT_DIM,
    personalize: bytes = b"",
) -> list[float]:
    """Unit-length vector derived from blake2s. Independent of PYTHONHASHSEED."""
    if dim < 2:
        raise ValueError("dim must be at least 2")
    material = payload if not personalize else payload + b"\x1e" + personalize
    values: list[float] = []
    counter = 0
    while len(values) < dim:
        digest = blake2s_digest(material, counter=counter)
        for offset in range(0, 32, 4):
            unsigned = int.from_bytes(digest[offset : offset + 4], "big")
            values.append((unsigned / 0xFFFFFFFF) * 2.0 - 1.0)
        counter += 1
    return l2_normalize(values[:dim])


def controlled_pair(
    payload: bytes, *, cosine: float = DUPLICATE_COSINE, dim: int = DEFAULT_DIM
) -> tuple[list[float], list[float]]:
    """Two unit vectors with a known cosine, used to test duplicate-mug ambiguity."""
    first = deterministic_embedding(payload, dim=dim)
    extra = deterministic_embedding(payload, dim=dim, personalize=b"ortho")
    dot = _dot(first, extra)
    residual = [extra[i] - dot * first[i] for i in range(dim)]
    ortho = l2_normalize(residual)
    mix = cosine
    complement = math.sqrt(max(0.0, 1.0 - mix * mix))
    second = l2_normalize([mix * first[i] + complement * ortho[i] for i in range(dim)])
    return first, second


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return _dot(left, right)


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return [v / norm for v in values]
