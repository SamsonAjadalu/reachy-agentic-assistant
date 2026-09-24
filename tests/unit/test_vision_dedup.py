"""Duplicate suppression with event bypass and detection-set gating."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from vision.dedup import DedupCandidate, first_duplicate
from vision.schemas import DetectionSet

NOW = datetime(2026, 9, 2, 18, 0, tzinfo=UTC)


def _candidate(**overrides: object) -> DedupCandidate:
    values: dict[str, object] = {
        "content_hash": "a" * 64,
        "dhash64": "aaaaaaaaaaaaaaaa",
        "phash64": "bbbbbbbbbbbbbbbb",
        "zone_id": "zone-1",
        "pose_bucket": "desk:0",
        "model_version": "v1",
        "captured_at": NOW - timedelta(minutes=5),
        "detection_set": DetectionSet.from_labels(["mug"]),
        "produced_event": False,
    }
    values.update(overrides)
    return DedupCandidate(**values)  # type: ignore[arg-type]


class TestDedup:
    def test_exact_hash_is_a_duplicate(self) -> None:
        incoming = _candidate(captured_at=NOW)
        result = first_duplicate(incoming, [_candidate()], now=NOW)
        assert result.duplicate is True

    def test_event_frames_bypass_dedup(self) -> None:
        incoming = _candidate(captured_at=NOW, produced_event=True)
        result = first_duplicate(incoming, [_candidate()], now=NOW, event_bypass=True)
        assert result.duplicate is False

    def test_different_detection_set_is_never_a_duplicate(self) -> None:
        incoming = _candidate(
            captured_at=NOW,
            content_hash="b" * 64,
            detection_set=DetectionSet.from_labels(["mug", "book"]),
        )
        result = first_duplicate(incoming, [_candidate()], now=NOW)
        assert result.duplicate is False

    def test_different_pose_bucket_is_out_of_scope(self) -> None:
        incoming = _candidate(captured_at=NOW, pose_bucket="shelf:90", content_hash="b" * 64)
        result = first_duplicate(incoming, [_candidate()], now=NOW)
        assert result.duplicate is False

    def test_outside_24h_window_is_ignored(self) -> None:
        old = _candidate(captured_at=NOW - timedelta(hours=30))
        incoming = _candidate(captured_at=NOW)
        result = first_duplicate(incoming, [old], now=NOW)
        assert result.duplicate is False
