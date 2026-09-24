"""Visual memory settings: persistence floor vs ephemeral sampling."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.config import AppEnv, Settings


def _base(tmp_path: Path, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "app_env": AppEnv.TEST,
        "app_data_dir": tmp_path,
        "pa_api_token": "a" * 48,
    }
    values.update(overrides)
    return values


def test_default_visual_settings_are_safe(tmp_path: Path) -> None:
    settings = Settings(**_base(tmp_path))
    assert settings.visual_enabled is False
    assert settings.visual_keep_full_frames is False
    assert settings.visual_min_capture_interval_seconds == 10.0
    assert settings.visual_sampling_hz == 1.0
    assert settings.visual_evidence_path == tmp_path.resolve() / "visual" / "evidence"


def test_one_hertz_persistence_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PydanticValidationError):
        Settings(**_base(tmp_path, visual_min_capture_interval_seconds=1.0))


def test_ephemeral_one_hertz_sampling_is_allowed(tmp_path: Path) -> None:
    settings = Settings(**_base(tmp_path, visual_sampling_hz=1.0))
    assert settings.visual_sampling_hz == 1.0


def test_ensure_directories_creates_visual_tree(tmp_path: Path) -> None:
    settings = Settings(**_base(tmp_path))
    settings.ensure_directories()
    assert settings.visual_evidence_path.is_dir()
    assert settings.visual_quarantine_path.is_dir()
    assert settings.visual_root.stat().st_mode & 0o777 == 0o700
