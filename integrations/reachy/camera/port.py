"""Port domain code uses for Pi camera access. No URLs leak through this surface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from integrations.reachy.camera.models import (
    CameraFrame,
    CameraStatus,
    ScanCancelResult,
    ScanOutcome,
    ScanStatus,
)


@runtime_checkable
class PiCameraPort(Protocol):
    """Uses the configured workflow."""

    adapter_version: str

    async def get_latest_frame(self, *, correlation_id: str | None = None) -> CameraFrame: ...

    async def get_status(self, *, correlation_id: str | None = None) -> CameraStatus: ...

    async def health_check(self, *, correlation_id: str | None = None) -> CameraStatus: ...

    async def request_scan(
        self,
        *,
        preset_id: str,
        reason: str | None = None,
        correlation_id: str | None = None,
    ) -> ScanOutcome: ...

    async def cancel_scan(
        self,
        *,
        scan_id: str,
        correlation_id: str | None = None,
    ) -> ScanCancelResult: ...

    async def get_scan_status(
        self,
        *,
        scan_id: str,
        correlation_id: str | None = None,
    ) -> ScanStatus: ...

    async def aclose(self) -> None: ...
