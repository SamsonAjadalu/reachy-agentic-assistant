"""Serial model residency on one GPU. Unload releases VRAM between providers."""

from __future__ import annotations

import asyncio
import gc
from typing import Any

from app.logging_config import get_logger
from vision_sidecar.protocols import LoadableProvider

logger = get_logger(__name__)


class ModelLifecycle:
    """One occupant at a time. Matches envelope A (~5 GB free beside ollama)."""

    def __init__(self, *, serial: bool = True, keep_loaded: bool = False) -> None:
        self.serial = serial
        self.keep_loaded = keep_loaded
        self._lock = asyncio.Lock()
        self._loaded: LoadableProvider | None = None
        self._last_name: str | None = None
        self.load_thrash = 0
        self.oom_count = 0
        self.device: str = "cpu"

    @property
    def loaded_name(self) -> str | None:
        return None if self._loaded is None else self._loaded.name

    async def run(
        self, provider: LoadableProvider, fn: Any, *, device: str, load_timeout: float
    ) -> Any:
        async with self._lock:
            await asyncio.wait_for(
                asyncio.to_thread(self._ensure_loaded, provider, device), load_timeout
            )
            try:
                return await asyncio.to_thread(fn)
            finally:
                if not self.keep_loaded:
                    await asyncio.to_thread(self._unload_current)

    def _ensure_loaded(self, provider: LoadableProvider, device: str) -> None:
        self.device = device
        if self._loaded is provider and provider.loaded:
            return
        if self._loaded is not None and self._loaded is not provider:
            logger.info(
                "Unloading vision provider to free VRAM",
                extra={"unloading": self._loaded.name, "loading": provider.name},
            )
            self._unload_current()
        if self._last_name is not None and self._last_name != provider.name:
            self.load_thrash += 1
        if not provider.loaded:
            logger.info(
                "Loading vision provider", extra={"provider": provider.name, "device": device}
            )
            provider.load(device)
        self._loaded = provider
        self._last_name = provider.name

    def _unload_current(self) -> None:
        if self._loaded is None:
            return
        try:
            self._loaded.unload()
        finally:
            self._loaded = None
            release_vram()

    def unload_all(self) -> None:
        self._unload_current()


def release_vram() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        return
