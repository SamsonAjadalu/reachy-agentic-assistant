"""Sidecar settings: APP_DATA_DIR, token, mock."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.config import REPO_ROOT
from vision_sidecar.config import SidecarSettings

TOKEN = "a" * 48


def _base(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "app_env": "test",
        "vision_sidecar_token": TOKEN,
        "mock_mode": True,
    }
    defaults.update(overrides)
    return defaults


class TestSidecarDataDir:
    def test_rejects_a_data_dir_inside_the_repository(self) -> None:
        with pytest.raises(PydanticValidationError, match="inside the application directory"):
            SidecarSettings(**_base(app_data_dir=REPO_ROOT / "data"))

    def test_accepts_a_data_dir_outside_the_repository(self, tmp_path: Path) -> None:
        settings = SidecarSettings(**_base(app_data_dir=tmp_path))
        assert settings.app_data_dir == tmp_path.resolve()
        assert tmp_path.resolve() in settings.hf_cache_dir.parents

    def test_hf_cache_is_under_app_data_dir(self, tmp_path: Path) -> None:
        settings = SidecarSettings(**_base(app_data_dir=tmp_path))
        settings.ensure_directories()
        assert settings.hf_cache_dir.is_dir()
        assert settings.hf_cache_dir == tmp_path.resolve() / "vision" / "hf"


class TestSidecarToken:
    def test_prefers_sidecar_token_over_pa_token(self, tmp_path: Path) -> None:
        settings = SidecarSettings(
            **_base(app_data_dir=tmp_path, pa_api_token="b" * 48, vision_sidecar_token=TOKEN)
        )
        assert settings.auth_token == TOKEN

    def test_falls_back_to_pa_api_token(self, tmp_path: Path) -> None:
        settings = SidecarSettings(
            **_base(app_data_dir=tmp_path, vision_sidecar_token="", pa_api_token="b" * 48)
        )
        assert settings.auth_token == "b" * 48

    def test_production_refuses_placeholder_token(self, tmp_path: Path) -> None:
        with pytest.raises(PydanticValidationError, match="real secret"):
            SidecarSettings(
                **_base(
                    app_env="production",
                    app_data_dir=tmp_path,
                    mock_mode=False,
                    vision_sidecar_token="changeme",
                )
            )

    def test_production_refuses_mock_mode(self, tmp_path: Path) -> None:
        with pytest.raises(PydanticValidationError, match="MOCK_MODE"):
            SidecarSettings(**_base(app_env="production", app_data_dir=tmp_path, mock_mode=True))


class TestDevice:
    def test_auto_falls_back_to_cpu_without_cuda(self, tmp_path: Path) -> None:
        settings = SidecarSettings(**_base(app_data_dir=tmp_path, vision_device="auto"))
        assert settings.resolved_device(False) == "cpu"

    def test_cuda_request_can_fall_back(self, tmp_path: Path) -> None:
        settings = SidecarSettings(
            **_base(app_data_dir=tmp_path, vision_device="cuda:0", vision_cpu_fallback=True)
        )
        assert settings.resolved_device(False) == "cpu"
