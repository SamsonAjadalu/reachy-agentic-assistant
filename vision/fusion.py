"""Multi-view fusion: log-odds with a correlation discount. No midpoint of conflict."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vision.enums import ScanOutcomeKind


@dataclass(frozen=True)
class FusionResult:
    belief: float
    contradicted: bool


def fuse_confidences(values: list[float], *, independent: bool = False) -> tuple[float, bool]:
    """Return ``(belief, contradicted)``.

    Same-scan views are correlated: the discount starts at ``1/sqrt(N)``.
    Bimodal disagreement is not averaged.
    """
    if not values:
        return 0.0, False
    clipped = [min(0.99, max(0.01, v)) for v in values]
    if _bimodal(clipped):
        return 0.0, True
    n = len(clipped)
    discount = 1.0 if independent else 1.0 / math.sqrt(n)
    log_odds = 0.0
    for value in clipped:
        odds = value / (1.0 - value)
        log_odds += discount * math.log(odds)
    log_odds = max(-4.0, min(4.0, log_odds))
    belief = 1.0 / (1.0 + math.exp(-log_odds))
    return round(belief, 4), False


def fuse_probabilities(values: list[float], *, independent: bool = False) -> FusionResult:
    belief, contradicted = fuse_confidences(values, independent=independent)
    return FusionResult(belief=belief, contradicted=contradicted)


def is_bimodal(values: list[float], *, gap: float = 0.5) -> bool:
    if not values:
        return False
    clipped = [min(0.99, max(0.01, v)) for v in values]
    return _bimodal(clipped, gap=gap)


def kendall_tau(left: list[str], right: list[str]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    rank = {item: index for index, item in enumerate(left)}
    perm = [rank[item] for item in right if item in rank]
    n = len(perm)
    if n < 2:
        return None
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] < perm[j]:
                concordant += 1
            elif perm[i] > perm[j]:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return 1.0
    return (concordant - discordant) / denom


def apply_scan_outcome(prior: float, kind: ScanOutcomeKind | str) -> float:
    """Uses the configured workflow."""
    value = kind.value if isinstance(kind, ScanOutcomeKind) else str(kind)
    if value in {ScanOutcomeKind.BLOCKED.value, ScanOutcomeKind.DEGRADED.value}:
        return prior
    if value == ScanOutcomeKind.RESOLVED_FOUND.value:
        return min(0.99, prior + 0.15)
    if value == ScanOutcomeKind.RESOLVED_ABSENT.value:
        return max(0.01, prior * 0.5)
    if value == ScanOutcomeKind.CONTRADICTED.value:
        return 0.0
    if value in {
        ScanOutcomeKind.IMPROVED_NOT_RESOLVED.value,
        ScanOutcomeKind.IMPROVED_BUT_NOT_RESOLVED.value,
    }:
        return prior
    return max(0.01, prior * 0.7)


def _bimodal(values: list[float], *, gap: float = 0.5) -> bool:
    if len(values) < 2:
        return False
    lo, hi = min(values), max(values)
    return (hi - lo) >= gap and lo <= 0.35 and hi >= 0.65
