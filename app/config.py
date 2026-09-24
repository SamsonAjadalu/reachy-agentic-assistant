"""Application configuration.

Two rules shape this module:

1. Runtime data never lives inside the repository. Everything mutable hangs off
   ``APP_DATA_DIR``, which defaults outside the checkout.
2. The secret that decrypts the OAuth token store is never stored beside the
   token store. It comes from the environment, a systemd credential or the OS
   keyring, and is only ever held in memory. See ``security/keyring.py``.
"""

from __future__ import annotations

import ipaddress
import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BeforeValidator, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

if TYPE_CHECKING:
    from integrations.reachy.camera.settings import PiCameraSettings

REPO_ROOT = Path(__file__).resolve().parent.parent

# Values that mean "the user copied .env.example and did not fill this in".
PLACEHOLDER_MARKERS = (
    "changeme",
    "change-me",
    "your-",
    "your_",
    "replace-me",
    "replace_me",
    "xxx",
    "placeholder",
    "example",
    "<",
)

MIN_TOKEN_LENGTH = 32


def _split_csv(value: Any) -> Any:
    """Accept either a real list or a comma-separated environment string."""
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


# NoDecode stops pydantic-settings from JSON-parsing the raw environment value,
# which would reject the comma-separated form that is far friendlier in a .env
# file. The BeforeValidator then does the splitting.
CsvList = Annotated[list[str], NoDecode, BeforeValidator(_split_csv)]
CsvIntList = Annotated[list[int], NoDecode, BeforeValidator(_split_csv)]


class AppEnv(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class WeatherProvider(StrEnum):
    OPEN_METEO = "open_meteo"
    OPENWEATHERMAP = "openweathermap"
    MOCK = "mock"


def looks_like_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered:
        return True
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


class Settings(BaseSettings):
    """Runtime settings, loaded from the environment and an optional ``.env``."""

    model_config = SettingsConfigDict(
        env_file=os.environ.get("PA_ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ core
    app_env: AppEnv = AppEnv.DEVELOPMENT
    app_host: str = "127.0.0.1"
    app_port: int = 8080
    app_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    app_log_format: Literal["json", "console"] = "json"
    app_timezone: str = "America/Toronto"

    app_data_dir: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "APP_DATA_DIR",
                str(Path.home() / ".local" / "share" / "reachy-personal-assistant"),
            )
        ),
        description="Root for all mutable runtime state. Must be outside the git checkout.",
    )

    database_url: str = Field(
        default="",
        description="Async SQLAlchemy URL. Defaults to assistant.db inside APP_DATA_DIR.",
    )

    # -------------------------------------------------------------- security
    pa_api_token: SecretStr = SecretStr("")
    pa_api_allowed_networks: CsvList = Field(
        default_factory=lambda: ["127.0.0.0/8", "::1/128", "192.168.0.0/16", "100.64.0.0/10"],
        description="CIDRs permitted to reach the API. Includes the Tailscale CGNAT range.",
    )
    pa_rate_limit_per_minute: int = 240
    pa_max_request_bytes: int = 2 * 1024 * 1024

    pa_secret_key: SecretStr = Field(
        default=SecretStr(""),
        description=(
            "Fernet key protecting the OAuth token store. Supplied via the environment, "
            "a systemd LoadCredential file or the OS keyring; never written next to the "
            "encrypted store it protects."
        ),
    )
    pa_secret_key_file: Path | None = Field(
        default=None,
        description=(
            "Path to a file holding the Fernet key, for example "
            "$CREDENTIALS_DIRECTORY/pa_secret_key under systemd."
        ),
    )
    pa_secret_key_keyring: bool = Field(
        default=False,
        description="Read the Fernet key from the OS keyring instead of the environment.",
    )

    # --------------------------------------------------------------- runtime
    mock_mode: bool = Field(
        default=False,
        description="Force every external provider to its mock implementation.",
    )
    scheduler_enabled: bool = True
    worker_enabled: bool = True
    worker_concurrency: int = 2
    worker_lease_seconds: int = 300

    # -------------------------------------------------------------- telegram
    telegram_enabled: bool = False
    telegram_bot_token: SecretStr = SecretStr("")
    telegram_allowed_chat_ids: CsvIntList = Field(default_factory=list)
    telegram_approval_ttl_minutes: int = 60

    # ----------------------------------------------------------- reachy chat
    reachy_text_turn_enabled: bool = False
    reachy_text_turn_url: str = "http://reachy-mini.local:7861/api/v1/text-turn"
    reachy_text_turn_token: SecretStr = SecretStr("")
    reachy_text_turn_timeout_seconds: int = 90

    # ---------------------------------------------------------------- google
    google_enabled: bool = False
    google_client_id: str = ""
    google_client_secret: SecretStr = SecretStr("")
    google_redirect_uri: str = "http://localhost:8765/oauth2callback"
    google_token_store_path: Path | None = None

    # ---------------------------------------------------------------- notion
    notion_enabled: bool = False
    notion_token: SecretStr = SecretStr("")
    notion_allowed_page_ids: CsvList = Field(default_factory=list)
    notion_cache_ttl_seconds: int = 300

    # --------------------------------------------------------------- weather
    weather_provider: WeatherProvider = WeatherProvider.OPEN_METEO
    weather_api_key: SecretStr = SecretStr("")
    weather_default_location: str = "Toronto,CA"
    weather_default_latitude: float = 43.6532
    weather_default_longitude: float = -79.3832
    weather_cache_ttl_seconds: int = 900

    # ------------------------------------------------------- local resources
    wardrobe_image_root: Path | None = None
    document_index_roots: CsvList = Field(default_factory=list)
    document_max_bytes: int = 50 * 1024 * 1024
    document_follow_symlinks: bool = False
    document_index_interval_seconds: int = 3600
    gmail_attachment_max_bytes: int = 15 * 1024 * 1024
    backup_root: Path | None = None
    backup_retention_days: int = 30

    workstation_script_registry: Path = Field(
        default_factory=lambda: REPO_ROOT / "config" / "registered_scripts.yaml"
    )
    workstation_allowed_services: CsvList = Field(default_factory=list)
    workstation_service_scope: Literal["user", "system"] = "user"
    workstation_max_output_bytes: int = 64 * 1024
    workstation_max_concurrent_runs: int = 2
    workstation_default_timeout_seconds: int = 300

    briefing_config_path: Path = Field(
        default_factory=lambda: REPO_ROOT / "config" / "briefing.yaml"
    )

    # -------------------------------------------------------- visual memory
    visual_enabled: bool = Field(
        default=False,
        description="Persist keyframes and evidence. Off by default; does not start inference.",
    )
    visual_min_capture_interval_seconds: float = Field(
        default=10.0,
        ge=10.0,
        description=(
            "Minimum seconds between persisted observations. Floor is 10 s; 1 Hz row "
            "persistence is refused. Ephemeral sampling uses visual_sampling_hz instead."
        ),
    )
    visual_sampling_hz: float = Field(
        default=1.0,
        gt=0.0,
        le=2.0,
        description="Ephemeral change-detection rate. Unchanged frames are not persisted.",
    )
    visual_ephemeral_retention_minutes: int = Field(default=60, ge=1)
    visual_thumbnail_retention_days: int = Field(default=30, ge=1)
    visual_evidence_retention_days: int = Field(default=90, ge=1)
    visual_evidence_max_bytes: int = Field(
        default=2 * 1024 * 1024 * 1024,
        gt=0,
        description="Hard byte budget for active evidence files under APP_DATA_DIR/visual.",
    )
    visual_keep_full_frames: bool = Field(
        default=False,
        description="When false, only thumbnails and cited crops are written.",
    )
    visual_daily_observation_cap: int = Field(default=2000, ge=1)
    visual_scan_hourly_cap: int = Field(default=6, ge=1, le=30)
    visual_watch_poll_seconds: int = Field(default=60, ge=15, le=3600)
    visual_need_look_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    visual_retention_unlink_threshold: int = Field(default=10, ge=1)
    visual_default_queries: CsvList = Field(
        default_factory=lambda: ["mug", "cup", "bottle", "phone", "keys", "book", "remote"]
    )

    reachy_camera_enabled: bool = False
    reachy_camera_url: str = "http://reachy-mini.local:7861"
    reachy_vision_token: SecretStr = SecretStr("")
    reachy_camera_token: SecretStr = SecretStr("")
    reachy_camera_timeout_seconds: float = Field(default=10.0, gt=0.0)
    reachy_camera_connect_timeout_seconds: float = Field(default=5.0, gt=0.0)
    reachy_camera_max_body_bytes: int = Field(default=5 * 1024 * 1024, ge=1024)
    reachy_camera_stale_frame_ms: int = Field(default=1000, ge=0)
    reachy_camera_frame_path: str = "/api/v1/camera/frame"
    reachy_camera_frame_method: str = "POST"
    reachy_camera_status_path: str = "/api/v1/camera/status"
    reachy_camera_scan_path: str = "/api/v1/scan"
    reachy_camera_scan_status_path: str = "/api/v1/scan/{scan_id}"
    reachy_camera_scan_cancel_path: str = "/api/v1/scan/{scan_id}/cancel"
    reachy_camera_get_attempts: int = Field(default=2, ge=1)

    vision_sidecar_token: SecretStr = SecretStr("")
    vision_sidecar_host: str = "127.0.0.1"
    vision_sidecar_port: int = Field(default=8090, ge=1, le=65535)
    vision_sidecar_timeout_seconds: float = Field(default=30.0, gt=0.0)
    vision_allow_downloads: bool = False
    yolo_world_enabled: bool = False
    visual_sidecar_enabled: bool = False

    # ------------------------------------------------------------ validators
    @field_validator("app_data_dir", mode="after")
    @classmethod
    def _data_dir_outside_repo(cls, value: Path) -> Path:
        resolved = value.expanduser().resolve()
        if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
            raise ValueError(
                f"APP_DATA_DIR ({resolved}) is inside the git checkout ({REPO_ROOT}). "
                "Runtime data must live outside the repository."
            )
        return resolved

    @field_validator("pa_api_allowed_networks", mode="after")
    @classmethod
    def _validate_networks(cls, value: list[str]) -> list[str]:
        for entry in value:
            try:
                ipaddress.ip_network(entry, strict=False)
            except ValueError as exc:
                raise ValueError(f"Invalid CIDR in PA_API_ALLOWED_NETWORKS: {entry!r}") from exc
        return value

    @field_validator("app_timezone", mode="after")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Unknown APP_TIMEZONE: {value!r}") from exc
        return value

    @model_validator(mode="after")
    def _apply_derived_defaults(self) -> Settings:
        data_dir = self.app_data_dir
        if not self.database_url:
            self.database_url = f"sqlite+aiosqlite:///{data_dir / 'assistant.db'}"
        if self.google_token_store_path is None:
            self.google_token_store_path = data_dir / "secrets" / "google_tokens.enc"
        if self.wardrobe_image_root is None:
            self.wardrobe_image_root = data_dir / "wardrobe"
        if self.backup_root is None:
            self.backup_root = data_dir / "backups"
        return self

    @model_validator(mode="after")
    def _production_refuses_weak_secrets(self) -> Settings:
        if self.app_env is not AppEnv.PRODUCTION:
            return self

        problems: list[str] = []
        token = self.pa_api_token.get_secret_value()
        if len(token) < MIN_TOKEN_LENGTH or looks_like_placeholder(token):
            problems.append(
                "PA_API_TOKEN must be a real secret of at least "
                f"{MIN_TOKEN_LENGTH} characters (use scripts/generate_api_token.py)."
            )
        if self.mock_mode:
            problems.append("MOCK_MODE must be false in production.")
        if self.telegram_enabled:
            if looks_like_placeholder(self.telegram_bot_token.get_secret_value()):
                problems.append("TELEGRAM_BOT_TOKEN is unset or a placeholder.")
            if not self.telegram_allowed_chat_ids:
                problems.append("TELEGRAM_ALLOWED_CHAT_IDS must list at least one chat id.")
        if self.google_enabled:
            if looks_like_placeholder(self.google_client_id):
                problems.append("GOOGLE_CLIENT_ID is unset or a placeholder.")
            if looks_like_placeholder(self.google_client_secret.get_secret_value()):
                problems.append("GOOGLE_CLIENT_SECRET is unset or a placeholder.")
        if self.notion_enabled and looks_like_placeholder(self.notion_token.get_secret_value()):
            problems.append("NOTION_TOKEN is unset or a placeholder.")

        if problems:
            raise ValueError(
                "Refusing to start in production with an unsafe configuration:\n  - "
                + "\n  - ".join(problems)
            )
        return self

    # -------------------------------------------------------------- derived
    @property
    def is_production(self) -> bool:
        return self.app_env is AppEnv.PRODUCTION

    @property
    def is_test(self) -> bool:
        return self.app_env is AppEnv.TEST

    @property
    def secrets_dir(self) -> Path:
        return self.app_data_dir / "secrets"

    @property
    def cache_dir(self) -> Path:
        return self.app_data_dir / "cache"

    @property
    def task_output_dir(self) -> Path:
        return self.app_data_dir / "task_outputs"

    @property
    def script_log_dir(self) -> Path:
        return self.app_data_dir / "script_runs"

    @property
    def wardrobe_image_path(self) -> Path:
        return self.wardrobe_image_root or self.app_data_dir / "wardrobe"

    @property
    def document_index_path(self) -> Path:
        return self.app_data_dir / "documents" / "index.db"

    @property
    def visual_root(self) -> Path:
        return self.app_data_dir / "visual"

    @property
    def visual_evidence_path(self) -> Path:
        return self.visual_root / "evidence"

    @property
    def visual_quarantine_path(self) -> Path:
        return self.visual_root / "quarantine"

    @property
    def vision_sidecar_url(self) -> str:
        return f"http://{self.vision_sidecar_host}:{self.vision_sidecar_port}"

    @property
    def vision_sidecar_enabled(self) -> bool:
        return self.visual_sidecar_enabled

    def reachy_camera_bearer_token(self) -> str:
        return (
            self.reachy_vision_token.get_secret_value()
            or self.reachy_camera_token.get_secret_value()
            or self.reachy_text_turn_token.get_secret_value()
        )

    def pi_camera_settings(self) -> PiCameraSettings:
        from integrations.reachy.camera.settings import PiCameraSettings

        token = self.reachy_camera_bearer_token()
        method = self.reachy_camera_frame_method.upper()
        if method not in {"GET", "POST"}:
            method = "POST"
        return PiCameraSettings(
            enabled=self.reachy_camera_enabled and not self.mock_mode,
            base_url=self.reachy_camera_url.rstrip("/"),
            token=token,
            timeout_seconds=self.reachy_camera_timeout_seconds,
            connect_timeout_seconds=min(
                self.reachy_camera_connect_timeout_seconds, self.reachy_camera_timeout_seconds
            ),
            max_body_bytes=self.reachy_camera_max_body_bytes,
            stale_frame_ms=self.reachy_camera_stale_frame_ms,
            frame_path=self.reachy_camera_frame_path,
            frame_method=method,
            status_path=self.reachy_camera_status_path,
            scan_path=self.reachy_camera_scan_path,
            scan_status_path=self.reachy_camera_scan_status_path,
            scan_cancel_path=self.reachy_camera_scan_cancel_path,
            get_attempts=self.reachy_camera_get_attempts,
        )

    @property
    def scheduler_jobstore_url(self) -> str:
        """Synchronous URL: the APScheduler 3.x SQLAlchemy job store is sync-only."""
        return f"sqlite:///{self.app_data_dir / 'scheduler.db'}"

    @property
    def scheduler_lock_path(self) -> Path:
        return self.app_data_dir / "scheduler.lock"

    @property
    def sync_database_url(self) -> str:
        """Synchronous form of ``database_url``, used by Alembic and backups."""
        return self.database_url.replace("+aiosqlite", "").replace("sqlite+pysqlite", "sqlite")

    @property
    def database_path(self) -> Path | None:
        url = self.sync_database_url
        prefix = "sqlite:///"
        if not url.startswith(prefix):
            return None
        return Path(url[len(prefix) :])

    def provider_enabled(self, name: str) -> bool:
        """Whether a named integration should use its real provider."""
        if self.mock_mode:
            return False
        return bool(getattr(self, f"{name}_enabled", False))

    def runtime_directories(self) -> list[Path]:
        """Directories created at startup, in dependency order."""
        return [
            self.app_data_dir,
            self.secrets_dir,
            self.cache_dir,
            self.task_output_dir,
            self.app_data_dir / "documents",
            self.app_data_dir / "gmail" / "attachments",
            self.wardrobe_image_root or self.app_data_dir / "wardrobe",
            (self.wardrobe_image_root or self.app_data_dir / "wardrobe") / "items",
            (self.wardrobe_image_root or self.app_data_dir / "wardrobe") / "thumbnails",
            (self.wardrobe_image_root or self.app_data_dir / "wardrobe") / "outfits",
            self.backup_root or self.app_data_dir / "backups",
            self.visual_root,
            self.visual_evidence_path,
            self.visual_quarantine_path,
        ]

    def ensure_directories(self) -> None:
        """Create runtime directories with owner-only permissions."""
        for directory in self.runtime_directories():
            directory.mkdir(parents=True, exist_ok=True)
        # The secrets directory holds the encrypted token store; keep it 0700
        # even if the umask is permissive.
        self.secrets_dir.chmod(0o700)
        if self.visual_root.exists():
            self.visual_root.chmod(0o700)
        if self.visual_evidence_path.exists():
            self.visual_evidence_path.chmod(0o700)
        if self.visual_quarantine_path.exists():
            self.visual_quarantine_path.chmod(0o700)

    def redacted_diagnostics(self) -> dict[str, Any]:
        """Configuration snapshot safe to log, print or return over the API."""
        report: dict[str, Any] = {}
        for name, value in self.model_dump().items():
            if isinstance(getattr(self, name, None), SecretStr):
                secret = getattr(self, name).get_secret_value()
                report[name] = _describe_secret(secret)
            elif isinstance(value, Path):
                report[name] = str(value)
            elif isinstance(value, StrEnum):
                report[name] = value.value
            else:
                report[name] = value
        report["pa_secret_key_source"] = self.secret_key_source()
        report["repo_root"] = str(REPO_ROOT)
        return report

    def secret_key_source(self) -> str:
        if self.pa_secret_key_keyring:
            return "os_keyring"
        if self.pa_secret_key_file is not None:
            return f"file:{self.pa_secret_key_file}"
        if self.pa_secret_key.get_secret_value():
            return "environment"
        return "unset"


def _describe_secret(value: str) -> str:
    if not value:
        return "<unset>"
    if looks_like_placeholder(value):
        return "<placeholder>"
    return f"<set: {len(value)} chars>"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings. Used by tests and by ``scripts/check_configuration.py``."""
    get_settings.cache_clear()
