"""Disk-full and byte-budget refusal for visual evidence."""

from __future__ import annotations

import errno
import io
from datetime import UTC, datetime
from unittest.mock import patch

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from vision.enums import ObservationTrigger
from vision.repositories import ingest_frame
from vision.schemas import FrameIngest
from vision.storage import DiskFullError, atomic_replace


def _jpeg(colour: tuple[int, int, int] = (50, 60, 70)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), colour).save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()


def _enabled(settings: Settings, **overrides: object) -> Settings:
    return settings.model_copy(update={"visual_enabled": True, **overrides})


class TestVisionDiskFull:
    async def test_enospc_during_replace_skips_without_a_row(
        self, settings: Settings, session: AsyncSession
    ) -> None:
        enabled = _enabled(settings)
        enabled.ensure_directories()
        frame = FrameIngest(
            image_bytes=_jpeg(),
            captured_at=datetime(2026, 9, 2, 18, 0, tzinfo=UTC),
            trigger=ObservationTrigger.PASSIVE,
        )

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise OSError(errno.ENOSPC, "No space left on device")

        with patch("vision.storage.os.replace", _boom):
            result = await ingest_frame(session, enabled, frame)
        assert result.persisted is False
        assert result.skip_reason == "disk_full"
        assert result.observation_id is None

    async def test_byte_budget_refuses_a_second_capture(
        self, settings: Settings, session: AsyncSession
    ) -> None:
        enabled = _enabled(settings, visual_evidence_max_bytes=10_000_000)
        enabled.ensure_directories()
        first = await ingest_frame(
            session,
            enabled,
            FrameIngest(
                image_bytes=_jpeg((10, 20, 30)),
                captured_at=datetime(2026, 9, 2, 18, 0, tzinfo=UTC),
            ),
        )
        await session.commit()
        assert first.persisted is True
        tight = _enabled(settings, visual_evidence_max_bytes=1)
        second = await ingest_frame(
            session,
            tight,
            FrameIngest(
                image_bytes=_jpeg((200, 10, 10)),
                captured_at=datetime(2026, 9, 2, 18, 0, 11, tzinfo=UTC),
                trigger=ObservationTrigger.QUERY,
            ),
        )
        assert second.persisted is False
        assert second.skip_reason == "budget_exceeded"

    def test_atomic_replace_maps_enospc(self, tmp_path: object) -> None:
        from pathlib import Path

        destination = Path(tmp_path) / "a" / "b" / "c.jpg"  # type: ignore[arg-type]

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise OSError(errno.ENOSPC, "No space left on device")

        with patch("vision.storage.os.replace", _boom):
            try:
                atomic_replace(destination, b"data")
            except DiskFullError:
                return
            raise AssertionError("expected DiskFullError")
