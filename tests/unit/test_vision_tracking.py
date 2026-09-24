"""Tracking: Hungarian + no-match spawns a new track instead of a forced pair."""

from __future__ import annotations

from datetime import UTC, datetime

from vision.enums import AssociationDecision
from vision.tracking import (
    TrackHypothesis,
    associate_detections,
    hypothesis_from_write,
    update_tracks,
)


def test_far_boxes_are_unmatched() -> None:
    track = TrackHypothesis(
        track_id="t0",
        label="mug",
        box=hypothesis_from_write("mug", box_xyxy=(0.0, 0.0, 0.2, 0.2)).box,
    )
    det = hypothesis_from_write("mug", box_xyxy=(0.7, 0.7, 0.95, 0.95))
    _tracks, result = associate_detections([track], [det], unmatched_cost=0.4)
    assert result.assignments[0].decision is AssociationDecision.NO_MATCH
    assert result.assignments[0].track_id is None


def test_overlap_matches_existing_track() -> None:
    box = (0.2, 0.2, 0.5, 0.5)
    track = TrackHypothesis(
        track_id="t0",
        label="mug",
        box=hypothesis_from_write("mug", box_xyxy=box).box,
    )
    det = hypothesis_from_write("mug", box_xyxy=(0.22, 0.22, 0.52, 0.52))
    _tracks, result = associate_detections([track], [det], unmatched_cost=0.7)
    assert result.assignments[0].track_id == "t0"


def test_update_spawns_and_confirms() -> None:
    now = datetime(2026, 9, 2, tzinfo=UTC)
    det = hypothesis_from_write("cup", box_xyxy=(0.1, 0.1, 0.3, 0.3))
    links = associate_detections([], [det])
    tracks, next_id = update_tracks([], [det], links, next_id=1, captured_at=now, confirm_after=1)
    assert next_id == 2
    assert tracks[0].detection_count == 1
    assert tracks[0].label == "cup"
