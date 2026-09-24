"""Uses the configured workflow."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from vision.enums import IngestSkipReason, ObservationTrigger
from vision.keyframe import LastKeyframe, evaluate_keyframe
from vision.schemas import DetectionSet

NOW = datetime(2026, 9, 2, 18, 0, tzinfo=UTC)
INTERVAL = timedelta(seconds=10)


class TestKeyframeGate:
    def test_first_sample_persists(self) -> None:
        verdict = evaluate_keyframe(
            visual_enabled=True,
            trigger=ObservationTrigger.PASSIVE,
            is_event=False,
            now=NOW,
            min_persist_interval=INTERVAL,
            current_dhash="aaaaaaaaaaaaaaaa",
            current_phash="bbbbbbbbbbbbbbbb",
            current_detections=DetectionSet.from_labels(["mug"]),
            last=None,
        )
        assert verdict.persist is True

    def test_unchanged_frame_does_not_persist(self) -> None:
        last = LastKeyframe(
            persisted_at=NOW - timedelta(seconds=30),
            dhash64="aaaaaaaaaaaaaaaa",
            phash64="bbbbbbbbbbbbbbbb",
            detection_set=DetectionSet.from_labels(["mug"]),
        )
        verdict = evaluate_keyframe(
            visual_enabled=True,
            trigger=ObservationTrigger.PASSIVE,
            is_event=False,
            now=NOW,
            min_persist_interval=INTERVAL,
            current_dhash="aaaaaaaaaaaaaaaa",
            current_phash="bbbbbbbbbbbbbbbb",
            current_detections=DetectionSet.from_labels(["mug"]),
            last=last,
        )
        assert verdict.persist is False
        assert verdict.reason == IngestSkipReason.UNCHANGED.value

    def test_one_hertz_unchanged_samples_create_zero_rows(self) -> None:
        last = LastKeyframe(
            persisted_at=NOW - timedelta(seconds=1),
            dhash64="aaaaaaaaaaaaaaaa",
            phash64="bbbbbbbbbbbbbbbb",
            detection_set=DetectionSet.from_labels(["mug"]),
        )
        persisted = 0
        for offset in range(10):
            verdict = evaluate_keyframe(
                visual_enabled=True,
                trigger=ObservationTrigger.PASSIVE,
                is_event=False,
                now=NOW + timedelta(seconds=offset),
                min_persist_interval=INTERVAL,
                current_dhash="aaaaaaaaaaaaaaaa",
                current_phash="bbbbbbbbbbbbbbbb",
                current_detections=DetectionSet.from_labels(["mug"]),
                last=last,
            )
            if verdict.persist:
                persisted += 1
        assert persisted == 0

    def test_changed_detection_set_persists_after_interval(self) -> None:
        last = LastKeyframe(
            persisted_at=NOW - timedelta(seconds=30),
            dhash64="aaaaaaaaaaaaaaaa",
            phash64="bbbbbbbbbbbbbbbb",
            detection_set=DetectionSet.from_labels(["mug"]),
        )
        verdict = evaluate_keyframe(
            visual_enabled=True,
            trigger=ObservationTrigger.PASSIVE,
            is_event=False,
            now=NOW,
            min_persist_interval=INTERVAL,
            current_dhash="aaaaaaaaaaaaaaaa",
            current_phash="bbbbbbbbbbbbbbbb",
            current_detections=DetectionSet.from_labels(["mug", "book"]),
            last=last,
        )
        assert verdict.persist is True

    def test_query_persists_even_when_unchanged(self) -> None:
        last = LastKeyframe(
            persisted_at=NOW - timedelta(seconds=1),
            dhash64="aaaaaaaaaaaaaaaa",
            phash64="bbbbbbbbbbbbbbbb",
            detection_set=DetectionSet.from_labels(["mug"]),
        )
        verdict = evaluate_keyframe(
            visual_enabled=True,
            trigger=ObservationTrigger.QUERY,
            is_event=False,
            now=NOW,
            min_persist_interval=INTERVAL,
            current_dhash="aaaaaaaaaaaaaaaa",
            current_phash="bbbbbbbbbbbbbbbb",
            current_detections=DetectionSet.from_labels(["mug"]),
            last=last,
        )
        assert verdict.persist is True

    def test_min_interval_rate_limits_changed_passive_frames(self) -> None:
        last = LastKeyframe(
            persisted_at=NOW - timedelta(seconds=2),
            dhash64="1111111111111111",
            phash64="2222222222222222",
            detection_set=DetectionSet.from_labels(["mug"]),
        )
        verdict = evaluate_keyframe(
            visual_enabled=True,
            trigger=ObservationTrigger.PASSIVE,
            is_event=False,
            now=NOW,
            min_persist_interval=INTERVAL,
            current_dhash="aaaaaaaaaaaaaaaa",
            current_phash="bbbbbbbbbbbbbbbb",
            current_detections=DetectionSet.from_labels(["book"]),
            last=last,
        )
        assert verdict.persist is False
        assert verdict.reason == IngestSkipReason.RATE_LIMITED.value
