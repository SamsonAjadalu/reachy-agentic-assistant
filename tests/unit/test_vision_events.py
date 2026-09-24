"""Uses the configured workflow."""

from __future__ import annotations

from datetime import UTC, datetime

from vision.enums import ScanOutcomeKind, VisualEventType
from vision.events import detect_events
from vision.fusion import apply_scan_outcome, fuse_probabilities, is_bimodal, kendall_tau
from vision.need_look import compute_need_look
from vision.quality import QualityReport
from vision.scan_policy import gate_scan
from vision.tracking import hypothesis_from_write


def test_appearance_and_disappearance() -> None:
    previous = [hypothesis_from_write("mug", box_xyxy=(0.1, 0.1, 0.3, 0.3))]
    current = [hypothesis_from_write("keyboard", box_xyxy=(0.4, 0.4, 0.7, 0.6))]
    events = detect_events(
        previous=previous,
        current=current,
        occurred_at=datetime(2026, 9, 2, 18, 0, tzinfo=UTC),
        zone_id="z1",
    )
    types = {event.event_type for event in events}
    assert VisualEventType.APPEARED in types
    assert VisualEventType.DISAPPEARED in types
    assert all(event.dedup_key for event in events if event.event_type is not VisualEventType.MOVED)


def test_failed_scan_lowers_belief() -> None:
    prior = 0.8
    after = apply_scan_outcome(prior, ScanOutcomeKind.NOT_RESOLVED)
    assert after < prior
    blocked = apply_scan_outcome(prior, ScanOutcomeKind.BLOCKED)
    assert blocked == prior
    found = apply_scan_outcome(0.4, ScanOutcomeKind.RESOLVED_FOUND)
    assert found > 0.4


def test_bimodal_is_contradicted() -> None:
    result = fuse_probabilities([0.9, 0.1])
    assert result.contradicted
    assert is_bimodal([0.9, 0.15])
    tau = kendall_tau(["a", "b", "c"], ["a", "c", "b"])
    assert tau is not None
    assert tau < 1.0


def test_need_look_is_additive() -> None:
    quality = QualityReport(
        mean_luma=20,
        laplacian_variance=4,
        blur_flag=True,
        exposure_flag=False,
        saturated_fraction=0.0,
        low_texture=True,
        quality_score=0.3,
        flags=("blur",),
    )
    look = compute_need_look([], [], quality, query_label="mug", threshold=0.4)
    assert "quality" in look.terms
    assert "query_unmatched" in look.terms
    assert look.score == min(1.0, sum(look.terms.values()))
    assert look.should_look


def test_identical_retry_refused() -> None:
    now = datetime(2026, 9, 2, 18, 0, tzinfo=UTC)
    gate = gate_scan(
        preset_id="REGION_SWEEP",
        visual_enabled=True,
        hourly_count=0,
        last_failed_preset="REGION_SWEEP",
        last_failed_at=now,
        now=now,
    )
    assert gate.allowed is False
    assert gate.reason == "identical_retry_refused"


def test_voice_active_blocks_scan() -> None:
    now = datetime(2026, 9, 2, 18, 0, tzinfo=UTC)
    gate = gate_scan(
        preset_id="CLOSE_LOOK",
        visual_enabled=True,
        hourly_count=0,
        last_failed_preset=None,
        last_failed_at=None,
        now=now,
        voice_active=True,
    )
    assert gate.reason == "voice_active"
