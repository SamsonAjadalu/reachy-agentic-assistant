"""Uses the configured workflow."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from vision.person import is_person_label
from vision.tracking import DetectionHypothesis, hypothesis_from_write


@dataclass(frozen=True)
class MockPerception:
    detections: list[DetectionHypothesis]
    scene_embedding: list[float]
    model_name: str = "mock-detect"
    model_version: str = "mock-1"


def perceive_bytes(
    image_bytes: bytes,
    *,
    queries: list[str] | None = None,
    inject_labels: list[str] | None = None,
) -> MockPerception:
    labels = [
        item.strip().lower() for item in (inject_labels or queries or ["object"]) if item.strip()
    ]
    if not labels:
        labels = ["object"]
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    seed = hashlib.blake2s(image.tobytes()[:4096], digest_size=8, person=b"pa-vis01").digest()
    origin = int.from_bytes(seed[:2], "big") / 65535.0
    detections: list[DetectionHypothesis] = []
    for index, label in enumerate(labels[:16]):
        x0 = min(0.7, 0.05 + (origin + index * 0.12) % 0.6)
        y0 = min(0.7, 0.08 + (origin * 0.5 + index * 0.09) % 0.55)
        x1 = min(0.98, x0 + 0.18)
        y1 = min(0.98, y0 + 0.2)
        person = is_person_label(label)
        embedding = None if person else _embedding(image_bytes, label)
        detections.append(
            hypothesis_from_write(
                label,
                score=0.55 + (index % 4) * 0.08,
                box_xyxy=(x0, y0, x1, y1),
                embedding=embedding,
                is_person=person,
                depth_median=0.3 + index * 0.12,
            )
        )
    return MockPerception(
        detections=detections,
        scene_embedding=_embedding(image_bytes, "scene"),
    )


def _embedding(image_bytes: bytes, label: str) -> list[float]:
    digest = hashlib.blake2s(
        image_bytes[:2048] + label.encode("utf-8"),
        digest_size=16,
        person=b"pa-vis01",
    ).digest()
    values = [((b / 255.0) * 2.0) - 1.0 for b in digest]
    norm = sum(v * v for v in values) ** 0.5 or 1.0
    return [v / norm for v in values]


def identical_mug_scene() -> list[DetectionHypothesis]:
    """Three indistinguishable mugs: identity must abstain."""
    shared = [0.1] * 16
    boxes = ((0.1, 0.2, 0.3, 0.5), (0.35, 0.2, 0.55, 0.5), (0.6, 0.2, 0.8, 0.5))
    return [
        hypothesis_from_write("mug", score=0.9, box_xyxy=box, embedding=list(shared))
        for box in boxes
    ]
