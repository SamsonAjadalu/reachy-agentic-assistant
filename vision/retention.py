"""Retention classes, byte budget, and reference-recomputing cleanup.

Phase A marks (database only). Phase B unlinks files. Phase C sweeps orphans.
Uses the configured workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Select, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from database.models.vision import (
    EvaluationSample,
    EvidenceAsset,
    ObjectDetection,
    ObjectEntity,
    VisualEvent,
    VisualObservation,
    VisualWatch,
)
from vision.enums import EvidenceState, RetentionClass, VisualWatchStatus
from vision.storage import (
    quarantine_orphan,
    relative_evidence_path,
    unlink_evidence_file,
    visual_evidence_root,
)


@dataclass
class RetentionReport:
    marked_expired: int = 0
    unlinked: int = 0
    orphans_quarantined: int = 0
    missing_marked: int = 0
    referenced_skipped: int = 0
    bytes_active: int = 0


@dataclass
class ReferenceSet:
    asset_ids: set[str] = field(default_factory=set)

    def __contains__(self, asset_id: str) -> bool:
        return asset_id in self.asset_ids


async def recompute_references(session: AsyncSession) -> ReferenceSet:
    """Live foreign keys plus evaluation RESTRICT rows. Not the stored hint."""
    ids: set[str] = set()

    async def _collect(stmt: Select[Any]) -> None:
        rows = (await session.execute(stmt)).all()
        for row in rows:
            for value in row:
                if value:
                    ids.add(value)

    await _collect(
        select(VisualObservation.full_frame_evidence_id, VisualObservation.thumbnail_evidence_id)
    )
    await _collect(select(ObjectDetection.mask_evidence_id, ObjectDetection.crop_evidence_id))
    await _collect(select(ObjectEntity.canonical_evidence_id))
    await _collect(select(VisualEvent.before_evidence_id, VisualEvent.after_evidence_id))
    await _collect(
        select(VisualWatch.baseline_evidence_id).where(
            VisualWatch.status == VisualWatchStatus.ACTIVE.value
        )
    )
    await _collect(select(EvaluationSample.evidence_id))
    return ReferenceSet(asset_ids=ids)


async def mark_expired(
    session: AsyncSession,
    settings: Settings,
    *,
    now: datetime | None = None,
) -> RetentionReport:
    stamp = now or datetime.now(UTC)
    refs = await recompute_references(session)
    report = RetentionReport()

    result = await session.execute(
        select(EvidenceAsset).where(
            EvidenceAsset.state == EvidenceState.ACTIVE.value,
            EvidenceAsset.retention_class.notin_(
                [RetentionClass.PINNED.value, RetentionClass.EVALUATION.value]
            ),
            EvidenceAsset.expires_at.is_not(None),
            EvidenceAsset.expires_at <= stamp,
        )
    )
    for asset in result.scalars():
        if asset.id in refs:
            report.referenced_skipped += 1
            continue
        asset.state = EvidenceState.EXPIRED.value
        report.marked_expired += 1

    budget = settings.visual_evidence_max_bytes
    if budget > 0:
        extra = await _mark_over_budget(session, refs, budget, stamp)
        report.marked_expired += extra

    await session.flush()
    report.bytes_active = await active_evidence_bytes(session)
    return report


async def unlink_expired(session: AsyncSession, settings: Settings, *, limit: int = 200) -> int:
    result = await session.execute(
        select(EvidenceAsset)
        .where(
            EvidenceAsset.state == EvidenceState.EXPIRED.value,
            EvidenceAsset.relative_path.is_not(None),
        )
        .limit(limit)
    )
    unlinked = 0
    for asset in result.scalars():
        if unlink_evidence_file(settings, asset.relative_path):
            unlinked += 1
    return unlinked


async def sweep_orphans(session: AsyncSession, settings: Settings) -> RetentionReport:
    """Files without rows are quarantined; rows without files become ``missing``."""
    report = RetentionReport()
    root = visual_evidence_root(settings)
    if not root.exists():
        return report

    known: set[str] = set()
    rows = (await session.execute(select(EvidenceAsset))).scalars().all()
    for asset in rows:
        known.add(asset.relative_path)
        path = root / asset.relative_path
        if asset.state == EvidenceState.ACTIVE.value and not path.is_file():
            asset.state = EvidenceState.MISSING.value
            report.missing_marked += 1

    for file_path in root.rglob("*.jpg"):
        if file_path.name.startswith(".tmp-"):
            continue
        try:
            relative = str(file_path.relative_to(root))
        except ValueError:
            continue
        if relative not in known:
            quarantine_orphan(settings, file_path)
            report.orphans_quarantined += 1

    await session.flush()
    return report


async def active_evidence_bytes(session: AsyncSession) -> int:
    value = (
        await session.execute(
            select(func.coalesce(func.sum(EvidenceAsset.byte_size), 0)).where(
                EvidenceAsset.state == EvidenceState.ACTIVE.value
            )
        )
    ).scalar_one()
    return int(value)


async def refresh_ref_count_hints(session: AsyncSession) -> None:
    refs = await recompute_references(session)
    assets = (await session.execute(select(EvidenceAsset))).scalars().all()
    for asset in assets:
        asset.reference_count = 1 if asset.id in refs else 0
    await session.flush()


def expected_relative_path(kind: str, content_hash: str) -> str:
    return relative_evidence_path(kind, content_hash)


async def _mark_over_budget(
    session: AsyncSession,
    refs: ReferenceSet,
    budget: int,
    now: datetime,
) -> int:
    total = await active_evidence_bytes(session)
    if total <= budget:
        return 0
    marked = 0
    order = (
        RetentionClass.EPHEMERAL.value,
        RetentionClass.THUMBNAIL.value,
        RetentionClass.EVIDENCE.value,
    )
    result = await session.execute(
        select(EvidenceAsset)
        .where(
            EvidenceAsset.state == EvidenceState.ACTIVE.value,
            EvidenceAsset.retention_class.in_(order),
            or_(
                EvidenceAsset.expires_at.is_(None),
                EvidenceAsset.expires_at > now,
            ),
        )
        .order_by(EvidenceAsset.created_at.asc())
    )
    for asset in result.scalars():
        if total <= budget:
            break
        if asset.id in refs:
            continue
        asset.state = EvidenceState.EXPIRED.value
        asset.expires_at = now
        total -= asset.byte_size
        marked += 1
    return marked


def iter_evidence_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*.jpg") if not path.name.startswith(".tmp-")]
