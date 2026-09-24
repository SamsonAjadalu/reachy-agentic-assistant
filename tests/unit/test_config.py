"""Configuration validation and redaction."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.config import REPO_ROOT, AppEnv, Settings, looks_like_placeholder


def _base(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "app_env": AppEnv.TEST,
        "pa_api_token": "a" * 48,
    }
    defaults.update(overrides)
    return defaults


class TestDataDirectory:
    def test_rejects_a_data_dir_inside_the_repository(self, tmp_path: Path) -> None:
        with pytest.raises(PydanticValidationError, match="inside the git checkout"):
            Settings(**_base(app_data_dir=REPO_ROOT / "data"))

    def test_rejects_the_repository_root_itself(self) -> None:
        with pytest.raises(PydanticValidationError, match="inside the git checkout"):
            Settings(**_base(app_data_dir=REPO_ROOT))

    def test_accepts_a_data_dir_outside_the_repository(self, tmp_path: Path) -> None:
        settings = Settings(**_base(app_data_dir=tmp_path))
        assert settings.app_data_dir == tmp_path.resolve()

    def test_derived_paths_all_live_under_the_data_dir(self, tmp_path: Path) -> None:
        settings = Settings(**_base(app_data_dir=tmp_path))
        derived = [
            settings.database_path,
            settings.google_token_store_path,
            settings.wardrobe_image_root,
            settings.backup_root,
            settings.cache_dir,
            settings.task_output_dir,
            settings.document_index_path,
            settings.visual_root,
            settings.visual_evidence_path,
            Path(settings.scheduler_jobstore_url.removeprefix("sqlite:///")),
        ]
        for path in derived:
            assert path is not None
            assert tmp_path.resolve() in path.resolve().parents

    def test_no_derived_runtime_path_is_inside_the_repository(self, tmp_path: Path) -> None:
        settings = Settings(**_base(app_data_dir=tmp_path))
        for path in settings.runtime_directories():
            assert REPO_ROOT not in path.resolve().parents

    def test_ensure_directories_locks_down_the_secrets_directory(self, tmp_path: Path) -> None:
        settings = Settings(**_base(app_data_dir=tmp_path))
        settings.ensure_directories()
        assert settings.secrets_dir.is_dir()
        assert settings.secrets_dir.stat().st_mode & 0o777 == 0o700


class TestCsvParsing:
    def test_parses_comma_separated_networks(self, tmp_path: Path) -> None:
        settings = Settings(
            **_base(app_data_dir=tmp_path, pa_api_allowed_networks="10.0.0.0/8, 127.0.0.1/32")
        )
        assert settings.pa_api_allowed_networks == ["10.0.0.0/8", "127.0.0.1/32"]

    def test_parses_comma_separated_chat_ids_as_integers(self, tmp_path: Path) -> None:
        settings = Settings(**_base(app_data_dir=tmp_path, telegram_allowed_chat_ids="12,34"))
        assert settings.telegram_allowed_chat_ids == [12, 34]

    def test_rejects_an_invalid_cidr(self, tmp_path: Path) -> None:
        with pytest.raises(PydanticValidationError, match="Invalid CIDR"):
            Settings(**_base(app_data_dir=tmp_path, pa_api_allowed_networks="not-a-network"))

    def test_rejects_an_unknown_timezone(self, tmp_path: Path) -> None:
        with pytest.raises(PydanticValidationError, match="Unknown APP_TIMEZONE"):
            Settings(**_base(app_data_dir=tmp_path, app_timezone="Mars/Olympus"))


class TestProductionGuards:
    def test_refuses_a_placeholder_token_in_production(self, tmp_path: Path) -> None:
        with pytest.raises(PydanticValidationError, match="PA_API_TOKEN"):
            Settings(
                app_env=AppEnv.PRODUCTION,
                app_data_dir=tmp_path,
                pa_api_token="changeme-generate-a-real-token",
            )

    def test_refuses_a_short_token_in_production(self, tmp_path: Path) -> None:
        with pytest.raises(PydanticValidationError, match="PA_API_TOKEN"):
            Settings(app_env=AppEnv.PRODUCTION, app_data_dir=tmp_path, pa_api_token="short")

    def test_refuses_mock_mode_in_production(self, tmp_path: Path) -> None:
        with pytest.raises(PydanticValidationError, match="MOCK_MODE"):
            Settings(
                app_env=AppEnv.PRODUCTION,
                app_data_dir=tmp_path,
                pa_api_token="b" * 48,
                mock_mode=True,
            )

    def test_refuses_telegram_enabled_without_a_chat_allowlist(self, tmp_path: Path) -> None:
        with pytest.raises(PydanticValidationError, match="TELEGRAM_ALLOWED_CHAT_IDS"):
            Settings(
                app_env=AppEnv.PRODUCTION,
                app_data_dir=tmp_path,
                pa_api_token="b" * 48,
                telegram_enabled=True,
                telegram_bot_token="123456789:AAreal-looking-token-value-here-xx",
                telegram_allowed_chat_ids=[],
            )

    def test_allows_a_fully_configured_production_setup(self, tmp_path: Path) -> None:
        settings = Settings(
            app_env=AppEnv.PRODUCTION,
            app_data_dir=tmp_path,
            pa_api_token="c" * 48,
            telegram_enabled=True,
            telegram_bot_token="123456789:AAreal-looking-token-value-here-xx",
            telegram_allowed_chat_ids="42",
        )
        assert settings.is_production


class TestRedactedDiagnostics:
    def test_never_reveals_a_secret_value(self, tmp_path: Path) -> None:
        token = "super-secret-token-value-that-must-not-leak"
        settings = Settings(
            **_base(app_data_dir=tmp_path, pa_api_token=token, notion_token="secret_abc123xyz")
        )
        rendered = repr(settings.redacted_diagnostics())
        assert token not in rendered
        assert "secret_abc123xyz" not in rendered
        assert "<set: " in rendered

    def test_reports_the_secret_key_source(self, tmp_path: Path, secret_key_file) -> None:
        settings = Settings(**_base(app_data_dir=tmp_path, pa_secret_key_file=secret_key_file))
        assert settings.redacted_diagnostics()["pa_secret_key_source"].startswith("file:")

    def test_flags_placeholders_distinctly_from_real_values(self, tmp_path: Path) -> None:
        settings = Settings(**_base(app_data_dir=tmp_path, notion_token="your-token-here"))
        assert settings.redacted_diagnostics()["notion_token"] == "<placeholder>"


class TestProviderEnablement:
    def test_mock_mode_disables_every_real_provider(self, tmp_path: Path) -> None:
        settings = Settings(
            **_base(app_data_dir=tmp_path, mock_mode=True, google_enabled=True, notion_enabled=True)
        )
        assert not settings.provider_enabled("google")
        assert not settings.provider_enabled("notion")

    def test_enabled_provider_is_reported_when_not_mocking(self, tmp_path: Path) -> None:
        settings = Settings(**_base(app_data_dir=tmp_path, mock_mode=False, google_enabled=True))
        assert settings.provider_enabled("google")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("changeme", True),
        ("your-token-here", True),
        ("<paste here>", True),
        ("", True),
        ("xxx", True),
        ("Kx9fQ2mLp7Rv4TnW8sYb3Zc6Ad1Ge5Hj", False),
    ],
)
def test_placeholder_detection(value: str, expected: bool) -> None:
    assert looks_like_placeholder(value) is expected
