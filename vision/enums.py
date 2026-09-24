"""String enums for visual memory.

Stored as ``String(n)`` with no CHECK constraint, matching the rest of the schema.
"""

from __future__ import annotations

from enum import StrEnum


class ProvenanceStatus(StrEnum):
    MEASURED = "measured"
    CALIBRATED = "calibrated"
    NOMINAL = "nominal"
    DERIVED = "derived"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class WorldFrameStatus(StrEnum):
    """Phase 1 has no map frame. This enum has a single honest value."""

    UNKNOWN = "unknown"


class BodyYawSource(StrEnum):
    MEASURED = "measured"
    COMMANDED = "commanded"
    UNKNOWN = "unknown"


class DepthSpace(StrEnum):
    RELATIVE_NORMALISED = "relative_normalised"


class DepthOutputKind(StrEnum):
    DIRECT_DEPTH = "direct_depth"
    INVERSE_DEPTH = "inverse_depth"


class DistortionModel(StrEnum):
    NONE = "none"
    BROWN_CONRADY = "brown_conrady"
    FISHEYE = "fisheye"
    UNKNOWN = "unknown"


class RetentionClass(StrEnum):
    EPHEMERAL = "ephemeral"
    THUMBNAIL = "thumbnail"
    EVIDENCE = "evidence"
    PINNED = "pinned"
    EVALUATION = "evaluation"


class EvidenceKind(StrEnum):
    FULL_FRAME = "full_frame"
    THUMBNAIL = "thumbnail"
    CROP = "crop"
    MASK = "mask"
    CANONICAL = "canonical"


class EvidenceState(StrEnum):
    ACTIVE = "active"
    EXPIRED = "expired"
    MISSING = "missing"
    QUARANTINED = "quarantined"


class ObservationTrigger(StrEnum):
    QUERY = "query"
    WATCH = "watch"
    EVENT = "event"
    SCAN = "scan"
    MANUAL = "manual"
    CHANGE = "change"
    CALIBRATION = "calibration"
    EVALUATION = "evaluation"
    PASSIVE = "passive"


class TrackState(StrEnum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    TERMINATED = "terminated"


class EntityStatus(StrEnum):
    ACTIVE = "active"
    MERGED = "merged"
    SPLIT = "split"
    FORGOTTEN = "forgotten"
    STALE = "stale"
    CONTRADICTED = "contradicted"


class IdentityRevisionKind(StrEnum):
    MERGE = "merge"
    SPLIT = "split"
    RELABEL = "relabel"


class SnapshotKind(StrEnum):
    SCAN = "scan"
    QUERY = "query"
    WATCH = "watch"
    PASSIVE = "passive"
    CALIBRATION = "calibration"
    EVALUATION = "evaluation"


class SnapshotState(StrEnum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABORTED = "aborted"
    BLOCKED = "blocked"


class VisualEventType(StrEnum):
    APPEARED = "appeared"
    DISAPPEARED = "disappeared"
    MOVED = "moved"
    CHANGED = "changed"


class VisualWatchStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERRORED = "errored"


class SpatialPredicate(StrEnum):
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    OCCLUDES = "occludes"
    SUPPORTED_BY = "supported_by"
    ON = "on"
    INSIDE = "inside"
    NEAR = "near"
    PART_OF = "part_of"
    IN_REGION = "in_region"
    CO_VISIBLE_WITH = "co_visible_with"


class FrameClass(StrEnum):
    EGOCENTRIC = "egocentric"
    ALLOCENTRIC = "allocentric"
    INTRINSIC = "intrinsic"


class ChurnClass(StrEnum):
    STATIC = "static"
    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"
    CARRIED = "carried"


class CameraSourceKind(StrEnum):
    REACHY_HEAD = "reachy_head"
    MOCK = "mock"
    FILE = "file"


class ObservationRequestStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    BLOCKED = "blocked"


class ProvenanceState(StrEnum):
    CURRENT = "current"
    STALE = "stale"
    UNRECOVERABLE = "unrecoverable"


class EmbeddingOwnerKind(StrEnum):
    ENTITY = "entity"
    DETECTION = "detection"
    ZONE = "zone"


class CalibrationStatus(StrEnum):
    NOMINAL = "nominal"
    CALIBRATED = "calibrated"
    STALE = "stale"
    UNKNOWN = "unknown"


class IngestDecision(StrEnum):
    PERSISTED = "persisted"
    UNCHANGED = "unchanged"
    DUPLICATE = "duplicate"
    RATE_LIMITED = "rate_limited"
    BUDGET_EXCEEDED = "budget_exceeded"
    DISABLED = "disabled"
    DISK_FULL = "disk_full"


class IngestSkipReason(StrEnum):
    DISABLED = "disabled"
    UNCHANGED = "unchanged"
    DUPLICATE = "duplicate"
    RATE_LIMITED = "min_capture_interval"
    BUDGET_EXCEEDED = "budget_exceeded"
    DISK_FULL = "disk_full"
    ZONE_CAPTURE_DISALLOWED = "zone_capture_disallowed"
    IDEMPOTENT_REPLAY = "idempotent_replay"
    DAILY_CAP = "daily_observation_cap"
    CAPTURE_REFUSED = "daily_observation_cap"


class Presence(StrEnum):
    """Schema-level distinction: a missed detection is not an empty scene."""

    PRESENT = "present"
    NOTHING_DETECTED = "nothing_detected"
    NOTHING_PRESENT = "nothing_present"
    UNKNOWN = "unknown"
    ABSTAINED = "abstained"


class AssociationDecision(StrEnum):
    MATCHED = "matched"
    NEW = "new"
    NO_MATCH = "no_match"
    ABSTAIN = "abstain"
    BLOCKED_PERSON = "blocked_person"


class LookOutcome(StrEnum):
    ANSWERED = "answered"
    SCAN_QUEUED = "scan_queued"
    SCAN_REJECTED = "scan_rejected"
    DISABLED = "disabled"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class ScanOutcomeKind(StrEnum):
    RESOLVED_FOUND = "resolved_found"
    RESOLVED_ABSENT = "resolved_absent"
    IMPROVED_BUT_NOT_RESOLVED = "improved_but_not_resolved"
    IMPROVED_NOT_RESOLVED = "improved_not_resolved"
    NOT_RESOLVED = "not_resolved"
    CONTRADICTED = "contradicted"
    BLOCKED = "blocked"
    DEGRADED = "degraded"


class PresenceKind(StrEnum):
    """Schema-level split: empty detections are not the same as an empty scene."""

    PRESENT = "present"
    ABSENT = "absent"
    NOTHING_DETECTED = "nothing_detected"
    UNKNOWN = "unknown"
    CAMERA_UNAVAILABLE = "camera_unavailable"
    ABSTAINED = "abstained"
    QUALITY_GATED = "quality_gated"


class WatchTrigger(StrEnum):
    SCHEDULE = "schedule"
    APPEAR = "appear"
    DISAPPEAR = "disappear"
    CHANGE = "change"
    MOVE = "move"


PERSON_LABELS = frozenset({"person", "human", "man", "woman", "child", "people", "face"})


WORLD_FRAME_STATUS_VALUE = WorldFrameStatus.UNKNOWN.value

FORBIDDEN_METRIC_COLUMNS = frozenset(
    {
        "position_xyz",
        "distance_m",
        "depth_m",
        "bearing_deg",
        "size_cm",
        "point_cloud",
        "volume",
    }
)
