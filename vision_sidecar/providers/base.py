"""Shared helpers for real model wrappers.

A selected real provider either loads and infers or it fails. Returning mock
detections from these classes is a bug.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from vision_sidecar.config import SidecarSettings
from vision_sidecar.errors import GpuOomError, ProviderUnavailableError


class RealProvider:
    name: str = "real"
    model_name: str = ""
    model_version: str = "unloaded"
    licence: str = "unknown"

    def __init__(self, settings: SidecarSettings) -> None:
        self.settings = settings
        self._model: Any = None
        self._processor: Any = None
        self._device = "cpu"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
        self._processor = None
        if hasattr(self, "_torch"):
            self._torch = None
        self.model_version = "unloaded"

    def _cache_dir(self) -> str:
        path = self.settings.hf_cache_dir
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _require_torch(self) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise ProviderUnavailableError(
                "torch is not installed. Install the vision extra "
                "(see deployment/vision_3090/README.md).",
                integration=self.name,
            ) from exc
        return torch

    def _from_pretrained(self, loader: Callable[..., Any], model_id: str, **kwargs: Any) -> Any:
        cache = self._cache_dir()
        os.environ.setdefault("HF_HOME", cache)
        os.environ.setdefault("TRANSFORMERS_CACHE", cache)
        try:
            return loader(
                model_id,
                cache_dir=cache,
                local_files_only=not self.settings.vision_allow_downloads,
                **kwargs,
            )
        except Exception as exc:
            raise ProviderUnavailableError(
                f"{self.name} could not load {model_id}. "
                "Weights are not faked. Set VISION_ALLOW_DOWNLOADS=1 after verifying "
                f"the catalogue, and keep the cache at {cache}.",
                integration=self.name,
                details={"model_id": model_id, "error_type": type(exc).__name__},
            ) from exc

    def _maybe_oom(self, exc: BaseException) -> None:
        text = str(exc).lower()
        if "out of memory" in text or type(exc).__name__ in {
            "OutOfMemoryError",
            "CUDAOutOfMemoryError",
        }:
            self.unload()
            raise GpuOomError(
                f"{self.name} ran out of VRAM. The model was unloaded; "
                "the job was not replaced with mock output."
            ) from exc
