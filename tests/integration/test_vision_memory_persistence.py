"""End-to-end persistence: file then row, keyframe-only writes."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta

import pytest
from PIL import ExifTags, Image
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from database.models.vision import (
    EvidenceAsset,
    ObjectDetection,
    VisualObservation,
)
from shared.errors import ValidationError
from vision.enums import (
    WORLD_FRAME_STATUS_VALUE,
    FrameClass,
    ObservationTrigger,
    RetentionClass,
    SpatialPredicate,
)
from vision.repositories import (
    create_embedding,
    create_entity,
    ingest_frame,
)
from vision.retention import mark_expired, recompute_references, sweep_orphans
from vision.schemas import DetectionSet, FrameIngest, SpatialRelationWrite
from vision.storage import read_evidence


def _jpeg(colour: tuple[int, int, int] = (40, 90, 140), size: tuple[int, int] = (96, 72)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()


def _enabled(settings: Settings, **overrides: object) -> Settings:
    payload = {"visual_enabled": True, **overrides}
    return settings.model_copy(update=payload)


def _frame(
    image: bytes,
    captured_at: datetime,
    *,
    trigger: ObservationTrigger = ObservationTrigger.PASSIVE,
    **kwargs: object,
) -> FrameIngest:
    return FrameIngest(image_bytes=image, captured_at=captured_at, trigger=trigger, **kwargs)  # type: ignore[arg-type]


class TestVisionMemoryPersistence:
    async def test_ingest_writes_file_then_row(
        self, settings: Settings, session: AsyncSession
    ) -> None:
        enabled = _enabled(settings)
        enabled.ensure_directories()
        now = datetime(2026, 9, 2, 18, 0, tzinfo=UTC)
        result = await ingest_frame(session, enabled, _frame(_jpeg(), now))
        await session.commit()
        assert result.persisted is True
        assert result.observation_id is not None
        observation = await session.get(VisualObservation, result.observation_id)
        assert observation is not None
        assert observation.world_frame_status == WORLD_FRAME_STATUS_VALUE
        asset = await session.get(EvidenceAsset, result.thumbnail_id)
        assert asset is not None
        path = enabled.visual_evidence_path / asset.relative_path
        assert path.is_file()
        assert path.stat().st_mode & 0o777 == 0o600
        assert result.full_frame_id is None

    async def test_unchanged_second_frame_creates_no_row_or_file(
        self, settings: Settings, session: AsyncSession
    ) -> None:
        enabled = _enabled(settings)
        enabled.ensure_directories()
        image = _jpeg()
        t0 = datetime(2026, 9, 2, 18, 0, tzinfo=UTC)
        first = await ingest_frame(session, enabled, _frame(image, t0))
        await session.commit()
        files_after_first = list(enabled.visual_evidence_path.rglob("*.jpg"))
        second = await ingest_frame(session, enabled, _frame(image, t0 + timedelta(seconds=1)))
        await session.commit()
        assert second.persisted is False
        assert second.skip_reason == "unchanged"
        count = (
            await session.execute(select(func.count()).select_from(VisualObservation))
        ).scalar_one()
        assert int(count) == 1
        assert list(enabled.visual_evidence_path.rglob("*.jpg")) == files_after_first
        assert first.observation_id is not None

    async def test_one_hertz_unchanged_stream_is_one_observation(
        self, settings: Settings, session: AsyncSession
    ) -> None:
        enabled = _enabled(settings)
        enabled.ensure_directories()
        image = _jpeg((12, 24, 36))
        t0 = datetime(2026, 9, 2, 19, 0, tzinfo=UTC)
        for second in range(5):
            await ingest_frame(session, enabled, _frame(image, t0 + timedelta(seconds=second)))
        await session.commit()
        count = (
            await session.execute(select(func.count()).select_from(VisualObservation))
        ).scalar_one()
        assert int(count) == 1

    async def test_query_persists_a_keyframe(
        self, settings: Settings, session: AsyncSession
    ) -> None:
        enabled = _enabled(settings)
        enabled.ensure_directories()
        t0 = datetime(2026, 9, 2, 18, 0, tzinfo=UTC)
        await ingest_frame(session, enabled, _frame(_jpeg(), t0))
        await session.commit()
        result = await ingest_frame(
            session,
            enabled,
            _frame(
                _jpeg((200, 10, 10)), t0 + timedelta(seconds=1), trigger=ObservationTrigger.QUERY
            ),
        )
        await session.commit()
        assert result.persisted is True

    async def test_idempotent_replay_returns_same_observation(
        self, settings: Settings, session: AsyncSession
    ) -> None:
        enabled = _enabled(settings)
        enabled.ensure_directories()
        t0 = datetime(2026, 9, 2, 18, 0, tzinfo=UTC)
        frame = _frame(_jpeg((1, 2, 3)), t0, idempotency_key="look-1")
        first = await ingest_frame(session, enabled, frame)
        await session.commit()
        second = await ingest_frame(session, enabled, frame)
        assert second.observation_id == first.observation_id
        assert second.skip_reason == "idempotent_replay"

    async def test_exif_is_stripped_from_stored_jpeg(
        self, settings: Settings, session: AsyncSession
    ) -> None:
        enabled = _enabled(settings)
        enabled.ensure_directories()
        buffer = io.BytesIO()
        image = Image.new("RGB", (80, 80), (8, 16, 24))
        exif = image.getexif()
        exif[ExifTags.Base.Make] = "TestCamera"
        image.save(buffer, format="JPEG", exif=exif)
        result = await ingest_frame(
            session,
            enabled,
            _frame(buffer.getvalue(), datetime(2026, 9, 2, 18, 0, tzinfo=UTC)),
        )
        await session.commit()
        asset = await session.get(EvidenceAsset, result.thumbnail_id)
        assert asset is not None
        stored = Image.open(io.BytesIO(read_evidence(enabled, asset.relative_path)))
        assert not dict(stored.getexif())

    async def test_person_detection_cannot_create_entity_or_embedding(
        self, settings: Settings, session: AsyncSession
    ) -> None:
        now = datetime(2026, 9, 2, 18, 0, tzinfo=UTC)
        with pytest.raises(ValidationError, match="persistent entity"):
            await create_entity(session, display_label="someone", captured_at=now, is_person=True)
        entity = await create_entity(session, display_label="mug", captured_at=now)
        with pytest.raises(ValidationError, match="instance embeddings"):
            await create_embedding(
                session,
                owner_kind="entity",
                owner_id=entity.id,
                space="appearance",
                model_name="mock",
                model_version="0",
                vector=[0.1, 0.2],
                is_person=True,
            )

    async def test_egocentric_relation_requires_viewpoint(self, session: AsyncSession) -> None:
        now = datetime(2026, 9, 2, 18, 0, tzinfo=UTC)
        entity = await create_entity(session, display_label="mug", captured_at=now)
        with pytest.raises(Exception, match="observed_from"):
            SpatialRelationWrite(
                predicate=SpatialPredicate.LEFT_OF,
                frame_class=FrameClass.EGOCENTRIC,
                subject_entity_id=entity.id,
                captured_at=now,
                confidence=0.8,
            )

    async def test_detection_cascade_and_person_flags(
        self, settings: Settings, session: AsyncSession
    ) -> None:
        enabled = _enabled(settings)
        enabled.ensure_directories()
        now = datetime(2026, 9, 2, 18, 0, tzinfo=UTC)
        result = await ingest_frame(
            session,
            enabled,
            _frame(
                _jpeg((9, 9, 9)),
                now,
                detection_set=DetectionSet.from_labels(["person"], is_person=True),
            ),
        )
        await session.commit()
        detection = (
            await session.execute(
                select(ObjectDetection).where(
                    ObjectDetection.observation_id == result.observation_id
                )
            )
        ).scalar_one()
        assert detection.is_person is True
        assert detection.blocks_entity is True
        assert detection.blocks_embedding is True

    async def test_retention_mark_skips_referenced_assets(
        self, settings: Settings, session: AsyncSession
    ) -> None:
        enabled = _enabled(settings)
        enabled.ensure_directories()
        now = datetime(2026, 9, 2, 18, 0, tzinfo=UTC)
        result = await ingest_frame(session, enabled, _frame(_jpeg((3, 4, 5)), now))
        await session.commit()
        asset = await session.get(EvidenceAsset, result.thumbnail_id)
        assert asset is not None
        asset.expires_at = now - timedelta(minutes=1)
        asset.retention_class = RetentionClass.THUMBNAIL.value
        await session.flush()
        report = await mark_expired(session, enabled, now=now)
        assert report.referenced_skipped >= 1
        refs = await recompute_references(session)
        assert asset.id in refs

    async def test_orphan_files_are_quarantined(
        self, settings: Settings, session: AsyncSession
    ) -> None:
        enabled = _enabled(settings)
        enabled.ensure_directories()
        orphan = enabled.visual_evidence_path / "thumbnail" / "aa" / "bb" / ("c" * 64 + ".jpg")
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_bytes(_jpeg())
        report = await sweep_orphans(session, enabled)
        assert report.orphans_quarantined == 1
        assert not orphan.exists()
        quarantined = list(enabled.visual_quarantine_path.rglob("*.jpg"))
        assert len(quarantined) == 1
