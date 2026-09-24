"""Uses the configured workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NeedLook:
    total: float
    terms: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.55

    @property
    def score(self) -> float:
        return self.total

    @property
    def should_look(self) -> bool:
        return self.total >= self.threshold

    def as_dict(self) -> dict[str, float]:
        return dict(self.terms)


def compute_need_look(
    detections: list[Any] | int | None = None,
    tracks: list[Any] | None = None,
    quality: Any | None = None,
    *,
    query_label: str | None = None,
    threshold: float = 0.55,
    n_similar: int = 0,
    occlusion_ratio: float | None = None,
    detection_confidence: float | None = None,
    blur: bool = False,
    exposure: bool = False,
    stale_frame: bool = False,
    pose_unsettled: bool = False,
    quality_failed: bool = False,
    previous_unresolved_scans: int = 0,
) -> NeedLook:
    _ = tracks
    if isinstance(detections, int):
        n_similar = detections
        detections = None
    terms: dict[str, float] = {}
    if n_similar >= 2:
        terms["identical_instances"] = 0.45
    if occlusion_ratio is not None and occlusion_ratio >= 0.35:
        terms["occlusion"] = min(0.4, occlusion_ratio)
    if detection_confidence is not None and detection_confidence < 0.55:
        terms["low_confidence"] = 0.25
    if blur:
        terms["blur"] = 0.2
    if exposure:
        terms["exposure"] = 0.2
    if stale_frame:
        terms["stale_frame"] = 0.3
    if pose_unsettled:
        terms["pose_unsettled"] = 0.2
    if quality is not None:
        quality_failed = quality_failed or bool(getattr(quality, "blur_flag", False))
        quality_failed = quality_failed or bool(getattr(quality, "low_texture", False))
        flags = getattr(quality, "flags", ())
        if "blur" in flags or getattr(quality, "quality_score", 1.0) < 0.5:
            quality_failed = True
    if quality_failed:
        terms["quality"] = 0.15
    if previous_unresolved_scans:
        terms["prior_unresolved"] = min(0.3, 0.1 * previous_unresolved_scans)
    if query_label and not detections:
        terms["query_unmatched"] = 0.4
    elif query_label and detections is not None:
        wanted = query_label.strip().lower()
        if wanted and not any(
            getattr(item, "label", "").strip().lower() == wanted for item in detections
        ):
            terms["query_unmatched"] = 0.4
    total = min(1.0, sum(terms.values()))
    return NeedLook(total=total, terms=terms, threshold=threshold)


def should_scan(need: NeedLook, *, threshold: float, hourly_count: int, hourly_cap: int) -> bool:
    if hourly_count >= hourly_cap:
        return False
    return need.total >= threshold
