"""Perception backend: in-process mock, or HTTP to the sidecar."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
from typing import Protocol

from PIL import Image

from app.config import Settings
from vision.geometry import Box
from vision.labels import is_person_label, normalise_label
from vision.tracking import DetectionHypothesis
from vision_sidecar.providers.mock import (
    MockDepthProvider,
    MockDetectionProvider,
    MockEmbeddingProvider,
    MockSegmentationProvider,
)
from vision_sidecar.types import DepthMap, Detection


@dataclass
class PerceptionOutput:
    detections: list[DetectionHypothesis]
    empty_reason: str | None = None
    mocked: bool = True
    model_name: str = "mock-detect"
    model_version: str = "mock-1"
    depth: DepthMap | None = None
    embeddings: list[list[float] | None] = field(default_factory=list)


class PerceptionBackend(Protocol):
    mocked: bool

    async def detect(
        self,
        image_bytes: bytes,
        queries: list[str],
        *,
        max_detections: int = 32,
        inject_labels: list[str] | None = None,
    ) -> PerceptionOutput: ...

    async def aclose(self) -> None: ...


class MockPerceptionBackend:
    """Same mock providers the sidecar uses. No GPU, no HTTP, no second LLM."""

    mocked = True

    def __init__(self) -> None:
        self._detect = MockDetectionProvider()
        self._segment = MockSegmentationProvider()
        self._depth = MockDepthProvider()
        self._embed = MockEmbeddingProvider()
        self._detect.load("cpu")
        self._embed.load("cpu")
        self._depth.load("cpu")
        _ = self._segment

    async def detect(
        self,
        image_bytes: bytes,
        queries: list[str],
        *,
        max_detections: int = 32,
        inject_labels: list[str] | None = None,
    ) -> PerceptionOutput:
        if inject_labels:
            from vision.perception_mock import perceive_bytes

            mock = perceive_bytes(image_bytes, inject_labels=inject_labels)
            return PerceptionOutput(
                detections=mock.detections[:max_detections],
                mocked=True,
                model_name=mock.model_name,
                model_version=mock.model_version,
            )
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        wanted = queries or ["object"]
        raw: list[Detection] = self._detect.detect(image, wanted, max_detections=max_detections)
        hypotheses: list[DetectionHypothesis] = []
        embeddings: list[list[float] | None] = []
        for item in raw:
            person = is_person_label(item.label)
            crop = _crop(image, item.box_xyxy)
            vector = None
            if not person:
                vector = self._embed.embed_images([crop])[0].values
            hypotheses.append(
                DetectionHypothesis(
                    label=normalise_label(item.label),
                    box=Box.from_xyxy(item.box_xyxy),
                    score=item.score,
                    is_person=person,
                    embedding=vector,
                    query=item.query,
                )
            )
            embeddings.append(vector)
        depth_payload = self._depth.depth(image, include_preview=False)
        depth = depth_payload["depth"]
        assert isinstance(depth, DepthMap)
        for hypothesis in hypotheses:
            hypothesis.depth_median = _box_depth(depth, hypothesis.box)
        empty_reason = None if hypotheses else "empty"
        return PerceptionOutput(
            detections=hypotheses,
            empty_reason=empty_reason,
            mocked=True,
            model_name=self._detect.model_name,
            model_version=self._detect.model_version,
            depth=depth,
            embeddings=embeddings,
        )

    async def aclose(self) -> None:
        return None


def _crop(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    x0, y0, x1, y1 = box
    left = max(0, int(x0 * width))
    top = max(0, int(y0 * height))
    right = min(width, max(left + 1, int(x1 * width)))
    bottom = min(height, max(top + 1, int(y1 * height)))
    return image.crop((left, top, right, bottom)).resize((64, 64))


def _box_depth(depth: DepthMap, box: Box) -> float:
    span = depth.relative_max - depth.relative_min
    # Ordinal only: interpolate by box centre. Not a metric depth.
    return round(depth.relative_min + span * ((box.cx + box.cy) / 2.0), 4)


def encode_b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


async def build_backend(settings: Settings) -> PerceptionBackend:
    if settings.mock_mode or not settings.visual_sidecar_enabled:
        return MockPerceptionBackend()
    from vision.sidecar_client import SidecarPerceptionBackend

    return SidecarPerceptionBackend(settings)
