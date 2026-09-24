"""Faithfulness auditor: re-read the DB against every cited observation."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models.vision import ObjectDetection, VisualObservation, VisualSpatialRelation
from vision.phrases import METRIC_LENGTH, assert_no_metric_language


def audit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Pure-function checks on a tool payload. An invariant, not a score."""
    violations: list[str] = []
    phrase = str(payload.get("phrase") or "")
    if METRIC_LENGTH.search(phrase):
        violations.append("metric_length_in_phrase")
    try:
        assert_no_metric_language(phrase)
    except Exception:
        if "metric_length_in_phrase" not in violations:
            violations.append("metric_length_in_phrase")
    for key in ("distance_m", "depth_m", "position_xyz", "bearing_deg", "size_cm"):
        if key in payload:
            violations.append(f"forbidden_field:{key}")
    presence = payload.get("presence")
    if presence is not None and presence not in {
        "present",
        "nothing_detected",
        "nothing_present",
        "unknown",
        "abstained",
    }:
        violations.append("unknown_presence")
    return {
        "ok": not violations,
        "violations": violations,
        "overclaim_metric": int("metric_length_in_phrase" in violations),
        "overclaim_forbidden_field": int(any(v.startswith("forbidden_field") for v in violations)),
        "overclaim_presence": int("unknown_presence" in violations),
        "overclaim_citation": 0,
    }


async def audit_citations(session: AsyncSession, payload: dict[str, Any]) -> dict[str, Any]:
    report = audit_payload(payload)
    observation_id = payload.get("observation_id")
    if observation_id:
        row = await session.get(VisualObservation, observation_id)
        if row is None:
            report["violations"].append("missing_observation")
            report["overclaim_citation"] = 1
            report["ok"] = False
        else:
            if payload.get("world_frame_status") not in (None, "unknown", row.world_frame_status):
                report["violations"].append("world_frame_overclaim")
                report["ok"] = False
            cited_labels = {item.get("label") for item in payload.get("detections") or []}
            if cited_labels:
                actual = {
                    det.label
                    for det in (
                        await session.execute(
                            select(ObjectDetection).where(
                                ObjectDetection.observation_id == observation_id
                            )
                        )
                    ).scalars()
                }
                extra = {label for label in cited_labels if label and label not in actual}
                if extra:
                    report["violations"].append("uncited_label")
                    report["ok"] = False
            for rel in payload.get("relations") or []:
                predicate = rel.get("predicate")
                if predicate:
                    exists = (
                        await session.execute(
                            select(VisualSpatialRelation).where(
                                VisualSpatialRelation.observed_from_observation_id
                                == observation_id,
                                VisualSpatialRelation.predicate == predicate,
                            )
                        )
                    ).scalar_one_or_none()
                    if exists is None:
                        report["violations"].append("relation_not_in_db")
                        report["ok"] = False
    report["ok"] = not report["violations"]
    return report
