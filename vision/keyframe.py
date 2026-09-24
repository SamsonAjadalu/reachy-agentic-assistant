"""Uses the configured workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from vision.enums import IngestSkipReason, ObservationTrigger
from vision.hashing import near_duplicate
from vision.schemas import DetectionSet, FrameIngest

FORCE_TRIGGERS = frozenset(
    {
        ObservationTrigger.QUERY,
        ObservationTrigger.EVENT,
        ObservationTrigger.SCAN,
        ObservationTrigger.MANUAL,
        ObservationTrigger.EVALUATION,
    }
)


@dataclass(frozen=True)
class LastKeyframe:
    persisted_at: datetime
    dhash64: str
    phash64: str
    detection_set: DetectionSet
    zone_id: str | None = None
    pose_bucket: str | None = None


@dataclass(frozen=True)
class KeyframeVerdict:
    persist: bool
    reason: str | None = None


def evaluate_keyframe(
    *,
    visual_enabled: bool,
    trigger: ObservationTrigger,
    is_event: bool,
    now: datetime,
    min_persist_interval: timedelta,
    current_dhash: str,
    current_phash: str,
    current_detections: DetectionSet,
    last: LastKeyframe | None,
) -> KeyframeVerdict:
    if not visual_enabled:
        return KeyframeVerdict(False, IngestSkipReason.DISABLED.value)

    force = is_event or trigger in FORCE_TRIGGERS
    if last is None:
        return KeyframeVerdict(True, None)

    same_detections = current_detections.fingerprint() == last.detection_set.fingerprint()
    same_look = current_dhash == last.dhash64 and current_phash == last.phash64
    if not same_look:
        same_look = near_duplicate(current_dhash, last.dhash64, current_phash, last.phash64)

    if same_look and same_detections and not force:
        return KeyframeVerdict(False, IngestSkipReason.UNCHANGED.value)

    if not force and (now - last.persisted_at) < min_persist_interval:
        return KeyframeVerdict(False, IngestSkipReason.RATE_LIMITED.value)

    return KeyframeVerdict(True, None)


def is_force_persist(ingest: FrameIngest) -> bool:
    return ingest.is_event or ingest.trigger in FORCE_TRIGGERS
