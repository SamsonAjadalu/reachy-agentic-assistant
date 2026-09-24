"""Perception capability protocols. Real and mock implementations share these."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from PIL import Image

from vision_sidecar.types import (
    Detection,
    EmbeddingVector,
    Mask,
)


@runtime_checkable
class LoadableProvider(Protocol):
    name: str
    model_name: str
    model_version: str
    licence: str

    @property
    def loaded(self) -> bool: ...

    def load(self, device: str) -> None: ...

    def unload(self) -> None: ...


@runtime_checkable
class DetectionProvider(LoadableProvider, Protocol):
    def detect(
        self, frame: Image.Image, queries: list[str], *, max_detections: int = 32
    ) -> list[Detection]: ...


@runtime_checkable
class SegmentationProvider(LoadableProvider, Protocol):
    def segment(
        self,
        frame: Image.Image,
        *,
        boxes: list[tuple[float, float, float, float]],
        points: list[tuple[float, float, int]],
    ) -> list[Mask]: ...


@runtime_checkable
class DepthProvider(LoadableProvider, Protocol):
    def depth(self, frame: Image.Image, *, include_preview: bool = True) -> dict[str, object]: ...


@runtime_checkable
class EmbeddingProvider(LoadableProvider, Protocol):
    space: str
    dim: int

    def embed_images(self, frames: list[Image.Image]) -> list[EmbeddingVector]: ...

    def embed_texts(self, texts: list[str]) -> list[EmbeddingVector]: ...

    def embed_scene(self, frame: Image.Image) -> EmbeddingVector: ...
