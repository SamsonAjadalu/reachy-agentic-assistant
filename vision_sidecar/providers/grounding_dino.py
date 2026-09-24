"""Grounding DINO (tiny) via Hugging Face transformers. Apache-2.0."""

from __future__ import annotations

from typing import Any

from PIL import Image

from vision_sidecar.catalog import CATALOG
from vision_sidecar.errors import ProviderUnavailableError
from vision_sidecar.providers.base import RealProvider
from vision_sidecar.types import Detection


class GroundingDinoProvider(RealProvider):
    name = "grounding_dino"
    licence = CATALOG["grounding_dino"].licence
    model_name = CATALOG["grounding_dino"].hf_id
    model_version = CATALOG["grounding_dino"].repo_sha[:12]

    def load(self, device: str) -> None:
        torch = self._require_torch()
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ProviderUnavailableError(
                "transformers is required for Grounding DINO.", integration=self.name
            ) from exc
        model_id = self.settings.grounding_dino_model_id
        self._processor = self._from_pretrained(AutoProcessor.from_pretrained, model_id)
        self._model = self._from_pretrained(
            AutoModelForZeroShotObjectDetection.from_pretrained, model_id
        )
        self._device = device
        self._model.to(device)
        self._model.eval()
        self._torch = torch
        self.model_version = CATALOG["grounding_dino"].repo_sha[:12]

    def detect(
        self, frame: Image.Image, queries: list[str], *, max_detections: int = 32
    ) -> list[Detection]:
        if self._model is None or self._processor is None:
            raise ProviderUnavailableError("Grounding DINO is not loaded.", integration=self.name)
        text = " . ".join(query.strip().rstrip(".") for query in queries if query.strip()) + "."
        inputs = self._processor(images=frame, text=text, return_tensors="pt")
        inputs = {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        torch = self._torch
        try:
            with torch.inference_mode():
                outputs = self._model(**inputs)
            target_sizes = [frame.size[::-1]]
            # transformers>=4.57 renamed box_threshold → threshold.
            processed = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.get("input_ids"),
                threshold=0.3,
                text_threshold=0.25,
                target_sizes=target_sizes,
            )
        except Exception as exc:
            self._maybe_oom(exc)
            raise
        if not processed:
            return []
        item: dict[str, Any] = processed[0]
        boxes = item.get("boxes")
        scores = item.get("scores")
        labels = item.get("labels") or item.get("text_labels") or []
        detections: list[Detection] = []
        width, height = frame.size
        count = 0 if boxes is None else len(boxes)
        for index in range(count):
            box = boxes[index]
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            nx1, ny1 = max(0.0, x1 / width), max(0.0, y1 / height)
            nx2, ny2 = min(1.0, x2 / width), min(1.0, y2 / height)
            if nx2 <= nx1 or ny2 <= ny1:
                continue
            label = str(labels[index]) if index < len(labels) else queries[0]
            query = next(
                (q for q in queries if q.lower() in label.lower() or label.lower() in q.lower()),
                queries[0],
            )
            detections.append(
                Detection(
                    label=label,
                    score=float(scores[index]) if scores is not None else 0.0,
                    box_xyxy=(nx1, ny1, nx2, ny2),
                    query=query,
                )
            )
            if len(detections) >= max_detections:
                break
        return detections
