"""Scan gating: preset_id only, rate limits, refuse identical failed retries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from integrations.reachy.camera.models import PROPOSED_PRESET_IDS
from vision.enums import ObservationRequestStatus, ScanOutcomeKind

DEFAULT_HOURLY_CAP = 6
FAILED_RETRY_WINDOW = timedelta(minutes=30)


@dataclass(frozen=True)
class ScanGate:
    allowed: bool
    reason: str | None = None
    preset_id: str | None = None


def validate_preset(preset_id: str) -> str:
    cleaned = preset_id.strip()
    if not cleaned:
        raise ValueError("preset_id is required")
    if any(ch in cleaned for ch in "{};/\\"):
        raise ValueError("preset_id contains forbidden characters")
    return cleaned


def gate_scan(
    *,
    preset_id: str,
    visual_enabled: bool,
    hourly_count: int,
    hourly_cap: int = DEFAULT_HOURLY_CAP,
    last_failed_preset: str | None,
    last_failed_at: datetime | None,
    now: datetime,
    voice_active: bool = False,
) -> ScanGate:
    if not visual_enabled:
        return ScanGate(False, "visual_disabled")
    if voice_active:
        return ScanGate(False, "voice_active")
    try:
        preset = validate_preset(preset_id)
    except ValueError as exc:
        return ScanGate(False, str(exc))
    if hourly_count >= hourly_cap:
        return ScanGate(False, "rate_limited")
    if (
        last_failed_preset == preset
        and last_failed_at is not None
        and now - last_failed_at < FAILED_RETRY_WINDOW
    ):
        return ScanGate(False, "identical_retry_refused")
    return ScanGate(True, None, preset)


def outcome_from_request_status(
    status: str, *, found: bool | None, improved: bool
) -> ScanOutcomeKind:
    if status in {ObservationRequestStatus.BLOCKED.value, ObservationRequestStatus.REJECTED.value}:
        return ScanOutcomeKind.BLOCKED
    if status == ObservationRequestStatus.ABORTED.value:
        return ScanOutcomeKind.DEGRADED
    if found is True:
        return ScanOutcomeKind.RESOLVED_FOUND
    if found is False:
        return ScanOutcomeKind.RESOLVED_ABSENT
    if improved:
        return ScanOutcomeKind.IMPROVED_NOT_RESOLVED
    return ScanOutcomeKind.NOT_RESOLVED


KNOWN_PRESETS = PROPOSED_PRESET_IDS
