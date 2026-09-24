"""Uses the configured workflow."""

from __future__ import annotations

from dataclasses import dataclass

from vision_sidecar.config import SidecarSettings
from vision_sidecar.errors import VisionDisabledError
from vision_sidecar.protocols import (
    DepthProvider,
    DetectionProvider,
    EmbeddingProvider,
    SegmentationProvider,
)
from vision_sidecar.providers.da3 import Da3MonoProvider
from vision_sidecar.providers.dinov2 import Dinov2Provider
from vision_sidecar.providers.grounding_dino import GroundingDinoProvider
from vision_sidecar.providers.mock import (
    MockDepthProvider,
    MockDetectionProvider,
    MockEmbeddingProvider,
    MockSegmentationProvider,
)
from vision_sidecar.providers.sam2 import Sam2Provider
from vision_sidecar.providers.yolo_world import YoloWorldProvider


@dataclass
class ProviderBundle:
    detection: DetectionProvider
    segmentation: SegmentationProvider
    depth: DepthProvider
    embedding: EmbeddingProvider
    mocked: bool


def build_providers(settings: SidecarSettings) -> ProviderBundle:
    if settings.use_mock:
        return ProviderBundle(
            detection=MockDetectionProvider(),
            segmentation=MockSegmentationProvider(),
            depth=MockDepthProvider(),
            embedding=MockEmbeddingProvider(),
            mocked=True,
        )
    return ProviderBundle(
        detection=_real_detection(settings),
        segmentation=_real_segmentation(settings),
        depth=_real_depth(settings),
        embedding=_real_embedding(settings),
        mocked=False,
    )


def _real_detection(settings: SidecarSettings) -> DetectionProvider:
    if settings.detection_provider == "yolo_world":
        if not settings.yolo_world_enabled:
            raise VisionDisabledError(
                "YOLO-World is disabled (GPL/AGPL licence blocker). Set YOLO_WORLD_ENABLED=true "
                "only after a recorded licence decision."
            )
        return YoloWorldProvider(settings)
    if not settings.grounding_dino_enabled:
        raise VisionDisabledError("GROUNDING_DINO_ENABLED is false.")
    return GroundingDinoProvider(settings)


def _real_segmentation(settings: SidecarSettings) -> SegmentationProvider:
    if not settings.sam2_enabled:
        raise VisionDisabledError("SAM2_ENABLED is false.")
    return Sam2Provider(settings)


def _real_depth(settings: SidecarSettings) -> DepthProvider:
    if not settings.da3_enabled:
        raise VisionDisabledError("DA3_ENABLED is false.")
    return Da3MonoProvider(settings)


def _real_embedding(settings: SidecarSettings) -> EmbeddingProvider:
    if not settings.dinov2_enabled:
        raise VisionDisabledError("DINOV2_ENABLED is false.")
    return Dinov2Provider(settings)
