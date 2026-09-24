"""Sidecar process settings.

Lives in this package so the FastAPI service does not grow vision-specific
flags. ``APP_DATA_DIR`` is still required to sit separate from the application directory.
The bearer token is ``VISION_SIDECAR_TOKEN`` with fallback to ``PA_API_TOKEN``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config import MIN_TOKEN_LENGTH, REPO_ROOT, CsvList, looks_like_placeholder

AppEnvName = Literal["development", "test", "production"]
DeviceName = Literal["auto", "cpu", "cuda", "cuda:0"]
DetectionBackend = Literal["grounding_dino", "yolo_world"]
SegmentationBackend = Literal["sam2"]
DepthBackend = Literal["da3mono_large"]
EmbeddingBackend = Literal["dinov2"]


class SidecarSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.environ.get("PA_ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: AppEnvName = "development"
    app_data_dir: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "APP_DATA_DIR",
                str(Path.home() / ".local" / "share" / "reachy-personal-assistant"),
            )
        )
    )
    app_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    app_log_format: Literal["json", "console"] = "json"

    vision_sidecar_host: str = "127.0.0.1"
    vision_sidecar_port: int = 8090
    vision_sidecar_token: SecretStr = SecretStr("")
    pa_api_token: SecretStr = SecretStr("")
    vision_sidecar_allow_nonlocal: bool = False

    mock_mode: bool = False
    vision_mock_mode: bool | None = None

    vision_device: str = "auto"
    vision_cpu_fallback: bool = True
    vision_serial_gpu: bool = True
    vision_keep_loaded: bool = False
    vision_allow_downloads: bool = False
    vision_queue_max: int = 8
    vision_job_ttl_seconds: int = 600
    vision_max_finished_jobs: int = 256
    vision_max_request_bytes: int = 12 * 1024 * 1024
    vision_isolate_jobs: bool = True

    vision_load_timeout_seconds: float = 180.0
    vision_detect_timeout_seconds: float = 30.0
    vision_segment_timeout_seconds: float = 45.0
    vision_depth_timeout_seconds: float = 60.0
    vision_embed_timeout_seconds: float = 20.0

    detection_provider: DetectionBackend = "grounding_dino"
    segmentation_provider: SegmentationBackend = "sam2"
    depth_provider: DepthBackend = "da3mono_large"
    embedding_provider: EmbeddingBackend = "dinov2"

    grounding_dino_enabled: bool = True
    yolo_world_enabled: bool = False
    sam2_enabled: bool = True
    da3_enabled: bool = True
    dinov2_enabled: bool = True

    grounding_dino_model_id: str = "IDEA-Research/grounding-dino-tiny"
    yolo_world_weights: str = "yolov8s-worldv2.pt"
    sam2_model_id: str = "facebook/sam2.1-hiera-tiny"
    da3_model_id: str = "depth-anything/DA3MONO-LARGE"
    dinov2_model_id: str = "facebook/dinov2-small"

    vision_hf_cache_dir: Path | None = None
    pa_api_allowed_networks: CsvList = Field(default_factory=lambda: ["127.0.0.0/8", "::1/128"])
    pa_rate_limit_per_minute: int = 240

    @field_validator("app_data_dir", mode="after")
    @classmethod
    def _data_dir_outside_repo(cls, value: Path) -> Path:
        resolved = value.expanduser().resolve()
        if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
            raise ValueError(
                f"APP_DATA_DIR ({resolved}) is inside the application directory ({REPO_ROOT}). "
                "Runtime data must live outside the repository."
            )
        return resolved

    @field_validator("vision_sidecar_host", mode="after")
    @classmethod
    def _localhost_by_default(cls, value: str) -> str:
        return value.strip() or "127.0.0.1"

    @model_validator(mode="after")
    def _production_guards(self) -> Self:
        if self.app_env != "production":
            return self
        problems: list[str] = []
        token = self.auth_token
        if len(token) < MIN_TOKEN_LENGTH or looks_like_placeholder(token):
            problems.append(
                "VISION_SIDECAR_TOKEN or PA_API_TOKEN must be a real secret of at least "
                f"{MIN_TOKEN_LENGTH} characters."
            )
        if self.use_mock:
            problems.append("MOCK_MODE must be false in production.")
        host = self.vision_sidecar_host
        if host not in {"127.0.0.1", "localhost", "::1"} and not self.vision_sidecar_allow_nonlocal:
            problems.append(
                "VISION_SIDECAR_HOST must be localhost unless VISION_SIDECAR_ALLOW_NONLOCAL=true."
            )
        cache = self.hf_cache_dir
        if cache == REPO_ROOT or REPO_ROOT in cache.parents:
            problems.append("Hugging Face cache must live separate from the application directory.")
        if problems:
            raise ValueError(
                "Refusing to start the vision sidecar in production:\n  - "
                + "\n  - ".join(problems)
            )
        return self

    @property
    def use_mock(self) -> bool:
        if self.vision_mock_mode is not None:
            return self.vision_mock_mode
        return self.mock_mode

    @property
    def auth_token(self) -> str:
        sidecar = self.vision_sidecar_token.get_secret_value().strip()
        if sidecar:
            return sidecar
        return self.pa_api_token.get_secret_value().strip()

    @property
    def hf_cache_dir(self) -> Path:
        if self.vision_hf_cache_dir is not None:
            return self.vision_hf_cache_dir.expanduser().resolve()
        return self.app_data_dir / "vision" / "hf"

    @property
    def vision_root(self) -> Path:
        return self.app_data_dir / "vision"

    def timeout_for(self, kind: str) -> float:
        mapping = {
            "detect": self.vision_detect_timeout_seconds,
            "segment": self.vision_segment_timeout_seconds,
            "depth": self.vision_depth_timeout_seconds,
            "embed": self.vision_embed_timeout_seconds,
            "scene_embed": self.vision_embed_timeout_seconds,
            "load": self.vision_load_timeout_seconds,
        }
        return mapping.get(kind, self.vision_detect_timeout_seconds)

    def ensure_directories(self) -> None:
        for directory in (
            self.app_data_dir,
            self.vision_root,
            self.hf_cache_dir,
            self.vision_root / "tmp",
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def apply_hf_env(self) -> None:
        """Point Hugging Face caches at APP_DATA_DIR, under APP_DATA_DIR."""
        cache = str(self.hf_cache_dir)
        os.environ["HF_HOME"] = cache
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(self.hf_cache_dir / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(self.hf_cache_dir / "transformers")
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        # Uses the configured workflow.
        # entrypoint applies it; loaders also pass local_files_only.

    def resolved_device(self, cuda_available: bool) -> str:
        requested = self.vision_device.strip() or "auto"
        if requested == "auto":
            return "cuda:0" if cuda_available else "cpu"
        if requested.startswith("cuda") and not cuda_available:
            if self.vision_cpu_fallback:
                return "cpu"
            raise ValueError(
                "CUDA was requested but is not available, and CPU fallback is disabled."
            )
        return requested
