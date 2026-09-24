"""Cosine similarity over unit-length float vectors."""

from __future__ import annotations

import math


def l2_normalise(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 0.0:
        return list(values)
    return [v / norm for v in values]


def cosine(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None or right is None:
        return None
    if len(left) != len(right) or not left:
        return None
    return float(sum(a * b for a, b in zip(left, right, strict=True)))


def running_mean(current: list[float] | None, incoming: list[float], count: int) -> list[float]:
    unit = l2_normalise(incoming)
    if current is None or count <= 0:
        return unit
    blended = [(c * count + n) / (count + 1) for c, n in zip(current, unit, strict=True)]
    return l2_normalise(blended)
