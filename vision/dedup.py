"""Near-duplicate suppression for static scenes.

Event-producing frames bypass this entirely. Dedup is scoped to
``(zone, pose_bucket, model_version)`` within 24 hours. A different detection
Uses the configured workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from vision.hashing import near_duplicate
from vision.schemas import DetectionSet

DEDUP_WINDOW = timedelta(hours=24)


@dataclass(frozen=True)
class DedupCandidate:
    content_hash: str
    dhash64: str
    phash64: str
    zone_id: str | None
    pose_bucket: str | None
    model_version: str | None
    captured_at: datetime
    detection_set: DetectionSet
    produced_event: bool = False


@dataclass(frozen=True)
class DedupResult:
    duplicate: bool
    matched_hash: str | None = None


def first_duplicate(
    incoming: DedupCandidate,
    existing: list[DedupCandidate],
    *,
    now: datetime,
    event_bypass: bool = False,
    window: timedelta = DEDUP_WINDOW,
) -> DedupResult:
    if event_bypass or incoming.produced_event:
        return DedupResult(False)
    since = now - window
    incoming_fp = incoming.detection_set.fingerprint()
    for candidate in existing:
        if candidate.captured_at < since:
            continue
        if incoming.zone_id is not None and candidate.zone_id != incoming.zone_id:
            continue
        if incoming.pose_bucket and candidate.pose_bucket != incoming.pose_bucket:
            continue
        if incoming.model_version and candidate.model_version != incoming.model_version:
            continue
        if incoming.content_hash == candidate.content_hash:
            return DedupResult(True, candidate.content_hash)
        other_fp = candidate.detection_set.fingerprint()
        if incoming_fp and other_fp and incoming_fp != other_fp:
            continue
        if near_duplicate(incoming.dhash64, candidate.dhash64, incoming.phash64, candidate.phash64):
            return DedupResult(True, candidate.content_hash)
    return DedupResult(False)
