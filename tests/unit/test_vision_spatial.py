"""Qualitative spatial phrases: no metric units, egocentric needs a viewpoint."""

from __future__ import annotations

from vision.enums import FrameClass, PresenceKind, SpatialPredicate
from vision.geometry import Box
from vision.phrases import (
    all_template_strings,
    assert_no_metric_claim,
    contains_metric_claim,
    render_last_seen,
    render_presence,
    render_relation,
)
from vision.spatial import relations_from_detections
from vision.tracking import hypothesis_from_write


def test_left_of_from_boxes() -> None:
    left = hypothesis_from_write("mug", box_xyxy=(0.1, 0.2, 0.3, 0.5))
    right = hypothesis_from_write("keyboard", box_xyxy=(0.5, 0.2, 0.8, 0.5))
    facts = relations_from_detections([left, right], quality_ok=True)
    predicates = {(f.predicate, f.subject_index, f.object_index) for f in facts}
    assert (SpatialPredicate.LEFT_OF, 0, 1) in predicates
    assert all(
        f.frame_class is FrameClass.EGOCENTRIC or f.predicate is SpatialPredicate.IN_REGION
        for f in facts
    )


def test_quality_gate_blocks_relations() -> None:
    det = hypothesis_from_write("mug", box_xyxy=(0.1, 0.2, 0.3, 0.5))
    assert relations_from_detections([det], quality_ok=False) == []


def test_templates_have_no_metric_units() -> None:
    for phrase in all_template_strings():
        assert_no_metric_claim(phrase)
    phrase = render_relation(
        subject_label="mug",
        predicate=SpatialPredicate.LEFT_OF,
        object_label="keyboard",
        frame_class=FrameClass.EGOCENTRIC,
    )
    assert "From this viewpoint" in phrase
    assert "measured distance" in phrase
    assert not contains_metric_claim(phrase)


def test_metric_lint_catches_centimetres() -> None:
    assert contains_metric_claim("the mug is about 30 cm to the left")
    assert contains_metric_claim("about 2 meters away")


def test_nothing_detected_is_not_absent() -> None:
    detected = render_presence(PresenceKind.NOTHING_DETECTED, label="mug")
    absent = render_presence(PresenceKind.ABSENT, label="mug", region="the desk")
    assert detected != absent
    assert "not the same as it being gone" in detected


def test_last_seen_abstain() -> None:
    phrase = render_last_seen(label="mug", zone_name="desk", when_phrase="earlier", abstained=True)
    assert "cannot tell them apart" in phrase


def test_box_iou() -> None:
    a = Box(0.0, 0.0, 0.5, 0.5)
    b = Box(0.25, 0.25, 0.75, 0.75)
    assert 0.1 < a.iou(b) < 0.3
