"""Uses the configured workflow."""

from __future__ import annotations

import hashlib
import io
import time

from PIL import Image, ImageDraw

from vision_sidecar.errors import GpuOomError, ProviderUnavailableError, StageTimeoutError
from vision_sidecar.hashing import (
    DEFAULT_DIM,
    DUPLICATE_COSINE,
    controlled_pair,
    deterministic_embedding,
)
from vision_sidecar.types import (
    DepthMap,
    Detection,
    EmbeddingVector,
    FaultKind,
    Mask,
    ProviderMeta,
)


def _meta(name: str, model: str, device: str = "cpu") -> ProviderMeta:
    return ProviderMeta(
        provider_name=name,
        model_name=model,
        model_version="mock-1",
        licence="mock",
        device=device,
        mocked=True,
    )


def apply_fault(fault: FaultKind | None) -> None:
    if fault is None:
        return
    if fault is FaultKind.CAMERA_UNAVAILABLE:
        raise ProviderUnavailableError("Injected camera_unavailable.", integration="mock")
    if fault is FaultKind.DECODE_ERROR:
        raise ProviderUnavailableError("Injected decode_error.", integration="mock")
    if fault is FaultKind.TIMEOUT:
        time.sleep(0.05)
        raise StageTimeoutError("Injected timeout.")
    if fault is FaultKind.OOM:
        raise GpuOomError("Injected oom.")


class MockDetectionProvider:
    name = "mock-detect"
    model_name = "mock-detect"
    model_version = "mock-1"
    licence = "mock"

    def __init__(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self, device: str) -> None:
        self._loaded = True
        self.model_version = f"mock-1@{device}"

    def unload(self) -> None:
        self._loaded = False

    def detect(
        self, frame: Image.Image, queries: list[str], *, max_detections: int = 32
    ) -> list[Detection]:
        detections: list[Detection] = []
        width, height = frame.size
        seed = hashlib.blake2s(frame.tobytes()[:2048], digest_size=8, person=b"pa-vis01").digest()
        origin = int.from_bytes(seed[:2], "big") / 65535.0
        for index, query in enumerate(queries):
            labels = [query]
            if "mug" in query.lower():
                labels = [f"{query}#a", f"{query}#b"]
            for sub, label in enumerate(labels):
                slot = index * 2 + sub
                x1 = min(0.82, 0.05 + (origin * 0.2) + slot * 0.08)
                y1 = min(0.82, 0.10 + slot * 0.07)
                detections.append(
                    Detection(
                        label=label,
                        score=round(0.92 - slot * 0.03, 4),
                        box_xyxy=(
                            round(x1, 4),
                            round(y1, 4),
                            round(min(0.98, x1 + 0.14), 4),
                            round(min(0.98, y1 + 0.14), 4),
                        ),
                        query=query,
                    )
                )
                if len(detections) >= max_detections:
                    return detections
        _ = (width, height)
        return detections[:max_detections]


class MockSegmentationProvider:
    name = "mock-segment"
    model_name = "mock-segment"
    model_version = "mock-1"
    licence = "mock"

    def __init__(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self, device: str) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def segment(
        self,
        frame: Image.Image,
        *,
        boxes: list[tuple[float, float, float, float]],
        points: list[tuple[float, float, int]],
    ) -> list[Mask]:
        masks: list[Mask] = []
        for box in boxes:
            x1, y1, x2, y2 = box
            area = max(0.0, (x2 - x1) * (y2 - y1))
            masks.append(
                Mask(
                    score=0.91,
                    box_xyxy=box,
                    area_fraction=round(area, 4),
                    rle_counts=[int(area * frame.size[0] * frame.size[1]), 0],
                )
            )
        for x, y, _label in points:
            x1, y1 = max(0.0, x - 0.05), max(0.0, y - 0.05)
            x2, y2 = min(1.0, x + 0.05), min(1.0, y + 0.05)
            if x2 <= x1:
                x2 = min(1.0, x1 + 0.02)
            if y2 <= y1:
                y2 = min(1.0, y1 + 0.02)
            masks.append(
                Mask(
                    score=0.8,
                    box_xyxy=(x1, y1, x2, y2),
                    area_fraction=round((x2 - x1) * (y2 - y1), 4),
                )
            )
        return masks


class MockDepthProvider:
    name = "mock-depth"
    model_name = "mock-da3mono"
    model_version = "mock-1"
    licence = "mock"

    def __init__(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self, device: str) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def depth(
        self, frame: Image.Image, *, include_preview: bool = True, swap: bool = False
    ) -> dict[str, object]:
        width, height = frame.size
        # Near at the bottom, far at the top — unless the injected swap is on.
        preview = Image.new("L", (min(width, 64), min(height, 48)))
        draw = ImageDraw.Draw(preview)
        for y in range(preview.size[1]):
            t = y / max(1, preview.size[1] - 1)
            value = int(255 * (1.0 - t if swap else t))
            draw.line([(0, y), (preview.size[0], y)], fill=value)
        buf = io.BytesIO()
        preview.save(buf, format="PNG")
        import base64

        preview_b64 = base64.b64encode(buf.getvalue()).decode("ascii") if include_preview else None
        lo, hi = (0.9, 0.1) if swap else (0.1, 0.9)
        depth = DepthMap(
            width=width,
            height=height,
            relative_min=float(min(lo, hi)),
            relative_max=float(max(lo, hi)),
            relative_median=0.5,
            has_confidence=True,
            preview_png_b64=preview_b64,
            ordinal_ready=True,
            licence="mock",
            model_name=self.model_name,
            model_version=self.model_version,
        )
        return {"depth": depth, "meta": _meta(self.name, self.model_name)}


class MockEmbeddingProvider:
    name = "mock-embed"
    model_name = "mock-blake2s"
    model_version = "mock-1"
    licence = "mock"
    space = "mock-blake2s"
    dim = DEFAULT_DIM

    def __init__(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self, device: str) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def embed_images(self, frames: list[Image.Image]) -> list[EmbeddingVector]:
        if len(frames) == 2:
            left, right = controlled_pair(frames[0].tobytes()[:4096] + frames[1].tobytes()[:4096])
            return [self._vec(left), self._vec(right)]
        return [self._vec(deterministic_embedding(frame.tobytes()[:8192])) for frame in frames]

    def embed_texts(self, texts: list[str]) -> list[EmbeddingVector]:
        return [self._vec(deterministic_embedding(text.encode("utf-8"))) for text in texts]

    def embed_scene(self, frame: Image.Image) -> EmbeddingVector:
        return self._vec(deterministic_embedding(frame.tobytes()[:8192], personalize=b"scene"))

    def _vec(self, values: list[float]) -> EmbeddingVector:
        return EmbeddingVector(
            values=values,
            dim=self.dim,
            space=self.space,
            model_name=self.model_name,
            model_version=self.model_version,
        )


DUPLICATE_PAIR_COSINE = DUPLICATE_COSINE
