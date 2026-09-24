"""Eval metrics. Scene-clustered intervals. No p-value stars. No MOTA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vision.geometry import Box
from vision.hungarian import linear_sum_assignment
from vision.labels import labels_compatible


@dataclass
class MetricValue:
    name: str
    value: float | None
    n: int
    ci_low: float | None = None
    ci_high: float | None = None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "n": self.n,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "note": self.note,
            "undefined": self.undefined,
        }

    @property
    def undefined(self) -> bool:
        return self.value is None


def detection_pr(
    predicted: list[tuple[str, Box]],
    truth: list[tuple[str, Box]],
    *,
    iou_threshold: float = 0.5,
) -> tuple[float, float, int, int, int]:
    """Hungarian match at IoU ≥ threshold. Returns P, R, tp, fp, fn."""
    if not truth and not predicted:
        return 1.0, 1.0, 0, 0, 0
    if not truth:
        return 0.0, 1.0, 0, len(predicted), 0
    if not predicted:
        return 1.0, 0.0, 0, 0, len(truth)
    cost = np.full((len(predicted), len(truth)), np.inf)
    for i, (p_label, p_box) in enumerate(predicted):
        for j, (t_label, t_box) in enumerate(truth):
            if not labels_compatible(p_label, t_label):
                continue
            iou = p_box.iou(t_box)
            if iou >= iou_threshold:
                cost[i, j] = 1.0 - iou
    row_ind, col_ind = linear_sum_assignment(cost)
    tp = sum(1 for i, j in zip(row_ind, col_ind, strict=True) if np.isfinite(cost[int(i), int(j)]))
    fp = len(predicted) - tp
    fn = len(truth) - tp
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return precision, recall, tp, fp, fn


def abstention_rates(decisions: list[str], *, truth_ambiguous: list[bool]) -> tuple[float, float]:
    """Return (abstention_rate, confident_wrong_rate)."""
    n = len(decisions)
    if n == 0:
        return 0.0, 0.0
    abstain = sum(1 for d in decisions if d == "abstain")
    confident_wrong = 0
    for decision, ambiguous in zip(decisions, truth_ambiguous, strict=True):
        if decision in {"matched", "new"} and ambiguous:
            confident_wrong += 1
    return abstain / n, confident_wrong / n


def scene_cluster_bootstrap(
    values_by_scene: dict[str, list[float]],
    *,
    draws: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Resample scenes, not frames. Wide intervals are the honest output."""
    scenes = [name for name, vals in values_by_scene.items() if vals]
    if not scenes:
        return None, None, None
    generator = rng or np.random.default_rng(0)
    means: list[float] = []
    for _ in range(draws):
        chosen = generator.choice(scenes, size=len(scenes), replace=True)
        pooled: list[float] = []
        for name in chosen:
            pooled.extend(values_by_scene[str(name)])
        if pooled:
            means.append(float(np.mean(pooled)))
    if not means:
        return None, None, None
    point = float(np.mean([np.mean(v) for v in values_by_scene.values() if v]))
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return point, lo, hi


def within_noise(delta: float, noise_floor: float) -> bool:
    return abs(delta) < noise_floor
