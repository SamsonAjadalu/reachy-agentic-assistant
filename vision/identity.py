"""Uses the configured workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from database.base import new_uuid
from vision.enums import AssociationDecision, EntityStatus
from vision.hungarian import linear_sum_assignment
from vision.labels import labels_compatible, normalise_label
from vision.repositories import block_person_embedding, block_person_entity
from vision.tracking import DetectionHypothesis
from vision.vectors import cosine, running_mean

DEFAULT_COST_GATE = 0.55
DEFAULT_MARGIN = 0.12
DEFAULT_IDENTICAL_COSINE = 0.97


@dataclass
class EntityHypothesis:
    entity_id: str
    label: str
    prototype: list[float] | None
    status: str = EntityStatus.ACTIVE.value
    observation_count: int = 0
    is_person: bool = False


@dataclass(frozen=True)
class IdentityLink:
    detection_index: int
    entity_id: str | None
    decision: AssociationDecision
    match_score: float | None
    margin: float | None
    n_competing: int
    n_similar: int
    reason: str | None = None


def link_entities(
    entities: list[EntityHypothesis],
    detections: list[DetectionHypothesis],
    *,
    n_similar_by_label: dict[str, int] | None = None,
    cost_gate: float = DEFAULT_COST_GATE,
    margin: float = DEFAULT_MARGIN,
    identical_cosine: float = DEFAULT_IDENTICAL_COSINE,
    next_entity_serial: int = 1,
) -> tuple[list[EntityHypothesis], list[IdentityLink]]:
    """Associate detections with persistent entities.

    No-match creates a new entity. Identical-object ambiguity abstains.
    Person detections are refused.
    """
    similar = n_similar_by_label or {}
    links: list[IdentityLink] = []
    if not detections:
        return entities, links

    active = [e for e in entities if e.status == EntityStatus.ACTIVE.value and not e.is_person]
    cost = np.full((len(detections), max(len(active), 1)), np.inf, dtype=np.float64)
    if active:
        for d_index, detection in enumerate(detections):
            for e_index, entity in enumerate(active):
                if detection.is_person or entity.is_person:
                    cost[d_index, e_index] = float("inf")
                    continue
                if not labels_compatible(entity.label, detection.label):
                    cost[d_index, e_index] = float("inf")
                    continue
                score = cosine(entity.prototype, detection.embedding)
                if score is None:
                    cost[d_index, e_index] = 0.85
                    continue
                cost[d_index, e_index] = 1.0 - max(0.0, min(1.0, score))

        row_ind, col_ind = linear_sum_assignment(cost)
        matched = {int(r): int(c) for r, c in zip(row_ind, col_ind, strict=True)}
    else:
        matched = {}

    claimed_entities: set[int] = set(matched.values())
    for d_index, detection in enumerate(detections):
        n_similar = similar.get(normalise_label(detection.label), 1)
        if detection.is_person:
            links.append(
                IdentityLink(
                    detection_index=d_index,
                    entity_id=None,
                    decision=AssociationDecision.BLOCKED_PERSON,
                    match_score=None,
                    margin=None,
                    n_competing=0,
                    n_similar=0,
                    reason="is_person blocks entity creation and embeddings",
                )
            )
            continue
        if detection.embedding is not None:
            block_person_embedding(is_person=False)

        if n_similar >= 2:
            links.append(
                IdentityLink(
                    detection_index=d_index,
                    entity_id=None,
                    decision=AssociationDecision.ABSTAIN,
                    match_score=None,
                    margin=0.0,
                    n_competing=n_similar,
                    n_similar=n_similar,
                    reason="identical objects; identity abstained",
                )
            )
            continue

        if d_index in matched and active:
            e_index = matched[d_index]
            pair_cost = float(cost[d_index, e_index])
            row = cost[d_index]
            finite = sorted(float(v) for v in row if np.isfinite(v))
            second = finite[1] if len(finite) > 1 else None
            pair_margin = None if second is None else second - pair_cost
            n_competing = sum(1 for v in row if np.isfinite(v) and float(v) <= cost_gate)
            match_score = None if not np.isfinite(pair_cost) else 1.0 - pair_cost
            if pair_cost > cost_gate:
                entity, next_entity_serial = _new_entity(
                    detection, serial=next_entity_serial, entities=entities
                )
                links.append(
                    IdentityLink(
                        detection_index=d_index,
                        entity_id=entity.entity_id,
                        decision=AssociationDecision.NO_MATCH,
                        match_score=match_score,
                        margin=pair_margin,
                        n_competing=n_competing,
                        n_similar=n_similar,
                        reason="cost above no-match gate; new entity",
                    )
                )
                continue
            if pair_margin is not None and pair_margin < margin:
                links.append(
                    IdentityLink(
                        detection_index=d_index,
                        entity_id=None,
                        decision=AssociationDecision.ABSTAIN,
                        match_score=match_score,
                        margin=pair_margin,
                        n_competing=n_competing,
                        n_similar=n_similar,
                        reason="association margin too small",
                    )
                )
                continue
            entity = active[e_index]
            entity.prototype = running_mean(
                entity.prototype, detection.embedding or [], entity.observation_count
            )
            entity.observation_count += 1
            claimed_entities.add(e_index)
            links.append(
                IdentityLink(
                    detection_index=d_index,
                    entity_id=entity.entity_id,
                    decision=AssociationDecision.MATCHED,
                    match_score=match_score,
                    margin=pair_margin,
                    n_competing=n_competing,
                    n_similar=n_similar,
                )
            )
            continue

        entity, next_entity_serial = _new_entity(
            detection, serial=next_entity_serial, entities=entities
        )
        links.append(
            IdentityLink(
                detection_index=d_index,
                entity_id=entity.entity_id,
                decision=AssociationDecision.NEW,
                match_score=None,
                margin=None,
                n_competing=0,
                n_similar=n_similar,
            )
        )

    _ = claimed_entities
    return entities, links


def associate_entities(
    detections: list[DetectionHypothesis],
    entities: list[EntityHypothesis],
) -> list[IdentityLink]:
    from vision.tracking import count_similar_instances as similar_by_label

    _entities, links = link_entities(
        list(entities),
        detections,
        n_similar_by_label=similar_by_label(detections),
    )
    return links


def count_similar_instances(detections: list[DetectionHypothesis]) -> list[int]:
    from vision.tracking import count_similar_instances as similar_by_label

    by_label = similar_by_label(detections)
    return [by_label.get(item.normalised_label, 1) for item in detections]


def _new_entity(
    detection: DetectionHypothesis,
    *,
    serial: int,
    entities: list[EntityHypothesis],
) -> tuple[EntityHypothesis, int]:
    block_person_entity(is_person=detection.is_person)
    entity = EntityHypothesis(
        entity_id=new_uuid(),
        label=normalise_label(detection.label),
        prototype=None if detection.is_person else detection.embedding,
        observation_count=1,
        is_person=False,
    )
    entities.append(entity)
    return entity, serial + 1
