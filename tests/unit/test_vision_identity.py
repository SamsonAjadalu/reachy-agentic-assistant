"""Uses the configured workflow."""

from __future__ import annotations

from vision.enums import AssociationDecision
from vision.identity import associate_entities, count_similar_instances
from vision.perception_mock import identical_mug_scene
from vision.person import is_person_label
from vision.tracking import hypothesis_from_write


def test_three_identical_mugs_abstain() -> None:
    detections = identical_mug_scene()
    counts = count_similar_instances(detections)
    assert counts == [3, 3, 3]
    links = associate_entities(detections, [])
    assert all(link.decision is AssociationDecision.ABSTAIN for link in links)
    assert all(link.entity_id is None for link in links)


def test_person_is_blocked() -> None:
    det = hypothesis_from_write("person", box_xyxy=(0.1, 0.1, 0.4, 0.8), embedding=[0.1] * 8)
    assert det.is_person
    assert det.embedding is None
    links = associate_entities([det], [])
    assert links[0].decision is AssociationDecision.BLOCKED_PERSON


def test_distinct_labels_spawn_new_entities() -> None:
    dets = [
        hypothesis_from_write("mug", embedding=[1.0, 0.0], box_xyxy=(0.1, 0.1, 0.3, 0.3)),
        hypothesis_from_write("keyboard", embedding=[0.0, 1.0], box_xyxy=(0.4, 0.4, 0.8, 0.7)),
    ]
    links = associate_entities(dets, [])
    assert [link.decision for link in links] == [
        AssociationDecision.NEW,
        AssociationDecision.NEW,
    ]


def test_person_label_helper() -> None:
    assert is_person_label("Human")
    assert not is_person_label("mug")
