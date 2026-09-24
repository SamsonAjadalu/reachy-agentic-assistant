"""Versioned Reachy Pi camera HTTP adapter (v1).

Aligned with Pi visual-memory API at commit ``5b44bfc`` on port ``7861``.
"""

from integrations.reachy.camera.client import PiCameraClientV1
from integrations.reachy.camera.errors import (
    PiCameraAuthError,
    PiCameraError,
    PiCameraInvalidRequest,
    PiCameraMalformedResponse,
    PiCameraOversizedResponse,
    PiCameraStaleFrame,
    PiCameraTimeout,
    PiCameraUnavailable,
    PiCameraUnsupportedMedia,
)
from integrations.reachy.camera.models import (
    ADAPTER_CONTRACT,
    ADAPTER_CONTRACT_STATUS,
    ADAPTER_RELEASE,
    ADAPTER_VERSION,
    PROPOSED_PRESET_IDS,
    BodyYawSource,
    CameraFrame,
    CameraHealth,
    CameraPose,
    CameraStatus,
    ObservationTimestamps,
    ScanAccepted,
    ScanCancelResult,
    ScanOutcome,
    ScanRejected,
    ScanRejectReason,
    ScanStatus,
)
from integrations.reachy.camera.port import PiCameraPort
from integrations.reachy.camera.settings import PiCameraSettings

__all__ = [
    "ADAPTER_CONTRACT",
    "ADAPTER_CONTRACT_STATUS",
    "ADAPTER_RELEASE",
    "ADAPTER_VERSION",
    "PROPOSED_PRESET_IDS",
    "BodyYawSource",
    "CameraFrame",
    "CameraHealth",
    "CameraPose",
    "CameraStatus",
    "ObservationTimestamps",
    "PiCameraAuthError",
    "PiCameraClientV1",
    "PiCameraError",
    "PiCameraInvalidRequest",
    "PiCameraMalformedResponse",
    "PiCameraOversizedResponse",
    "PiCameraPort",
    "PiCameraSettings",
    "PiCameraStaleFrame",
    "PiCameraTimeout",
    "PiCameraUnavailable",
    "PiCameraUnsupportedMedia",
    "ScanAccepted",
    "ScanCancelResult",
    "ScanOutcome",
    "ScanRejectReason",
    "ScanRejected",
    "ScanStatus",
]
