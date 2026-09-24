"""UTC timestamps and naive datetime refusal."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.config import AppEnv, Settings
from database.base import UtcDateTime
from vision.enums import WORLD_FRAME_STATUS_VALUE, ObservationTrigger
from vision.schemas import FrameIngest


class TestVisionTimestamps:
    def test_frame_ingest_rejects_naive_captured_at(self) -> None:
        with pytest.raises(PydanticValidationError, match="timezone-aware"):
            FrameIngest(
                image_bytes=b"not-an-image",
                captured_at=datetime(2026, 9, 2, 18, 0),
                trigger=ObservationTrigger.PASSIVE,
            )

    def test_frame_ingest_accepts_aware_utc(self) -> None:
        frame = FrameIngest(
            image_bytes=b"not-an-image",
            captured_at=datetime(2026, 9, 2, 18, 0, tzinfo=UTC),
        )
        assert frame.captured_at.tzinfo is not None
        assert frame.world_frame_status == WORLD_FRAME_STATUS_VALUE

    def test_world_frame_status_cannot_be_map(self) -> None:
        with pytest.raises(PydanticValidationError, match="always 'unknown'"):
            FrameIngest(
                image_bytes=b"x",
                captured_at=datetime(2026, 9, 2, 18, 0, tzinfo=UTC),
                world_frame_status="map",
            )

    def test_utc_datetime_bind_rejects_naive_values(self) -> None:
        codec = UtcDateTime()
        with pytest.raises(ValueError, match="naive datetime"):
            codec.process_bind_param(datetime(2026, 9, 2, 18, 0), None)

    def test_utc_datetime_round_trips_aware(self) -> None:
        codec = UtcDateTime()
        original = datetime(2026, 9, 2, 18, 0, tzinfo=UTC)
        stored = codec.process_bind_param(original, None)
        assert stored is not None
        assert stored.tzinfo is None
        restored = codec.process_result_value(stored, None)
        assert restored == original


class TestVisualSettings:
    def test_default_persist_interval_is_at_least_ten_seconds(self, tmp_path: object) -> None:
        from pathlib import Path

        settings = Settings(app_env=AppEnv.TEST, app_data_dir=Path(tmp_path))  # type: ignore[arg-type]
        assert settings.visual_min_capture_interval_seconds >= 10
        assert settings.visual_keep_full_frames is False
        assert settings.visual_enabled is False
        assert settings.visual_sampling_hz == 1.0

    def test_refuses_sub_ten_second_persist_interval(self, tmp_path: object) -> None:
        from pathlib import Path

        with pytest.raises(PydanticValidationError):
            Settings(
                app_env=AppEnv.TEST,
                app_data_dir=Path(tmp_path),  # type: ignore[arg-type]
                visual_min_capture_interval_seconds=1,
            )
