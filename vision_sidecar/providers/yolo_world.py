"""YOLO-World wrapper. Licence is GPL-3.0 / AGPL — disabled by default."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from vision_sidecar.catalog import CATALOG
from vision_sidecar.errors import ProviderUnavailableError
from vision_sidecar.providers.base import RealProvider
from vision_sidecar.types import Detection


class YoloWorldProvider(RealProvider):
    name = "yolo_world"
    licence = CATALOG["yolo_world"].licence
    model_name = "yolov8s-worldv2"
    model_version = "upstream-unverified-weights"

    def load(self, device: str) -> None:
        self._require_torch()
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:
            raise ProviderUnavailableError(
                "ultralytics is required for YOLO-World, and the GPL/AGPL licence is not accepted "
                "by default. Leave YOLO_WORLD_ENABLED=false unless a licence decision is recorded.",
                integration=self.name,
            ) from exc
        weights = Path(self.settings.hf_cache_dir) / self.settings.yolo_world_weights
        if not weights.is_file() and not self.settings.vision_allow_downloads:
            raise ProviderUnavailableError(
                f"YOLO-World weights {weights} are not in the cache and downloads are disabled.",
                integration=self.name,
            )
        source = str(weights) if weights.is_file() else self.settings.yolo_world_weights
        self._model = YOLOWorld(source)
        self._device = device
        self.model_name = Path(source).name

    def detect(
        self, frame: Image.Image, queries: list[str], *, max_detections: int = 32
    ) -> list[Detection]:
        if self._model is None:
            raise ProviderUnavailableError("YOLO-World is not loaded.", integration=self.name)
        self._model.set_classes(list(queries))
        try:
            results = self._model.predict(frame, device=self._device, verbose=False)
        except Exception as exc:
            self._maybe_oom(exc)
            raise
        detections: list[Detection] = []
        if not results:
            return detections
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        width, height = frame.size
        xyxy = boxes.xyxy
        conf = boxes.conf
        cls = boxes.cls
        names_raw = getattr(result, "names", None)
        if isinstance(names_raw, dict):
            names = names_raw
        elif isinstance(names_raw, (list, tuple)):
            names = dict(enumerate(names_raw))
        else:
            names = dict(enumerate(queries))
        for index in range(len(xyxy)):
            x1, y1, x2, y2 = [float(v) for v in xyxy[index].tolist()]
            nx1, ny1 = max(0.0, x1 / width), max(0.0, y1 / height)
            nx2, ny2 = min(1.0, x2 / width), min(1.0, y2 / height)
            if nx2 <= nx1 or ny2 <= ny1:
                continue
            class_id = int(cls[index]) if cls is not None else 0
            label = str(names.get(class_id, queries[0]))
            query = queries[class_id] if class_id < len(queries) else queries[0]
            detections.append(
                Detection(
                    label=label,
                    score=float(conf[index]) if conf is not None else 0.0,
                    box_xyxy=(nx1, ny1, nx2, ny2),
                    query=query,
                )
            )
            if len(detections) >= max_detections:
                break
        return detections
