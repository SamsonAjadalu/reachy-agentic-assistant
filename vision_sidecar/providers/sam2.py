"""SAM 2.1 Hiera-Tiny via transformers. Apache-2.0."""

from __future__ import annotations

from PIL import Image

from vision_sidecar.catalog import CATALOG
from vision_sidecar.errors import ProviderUnavailableError
from vision_sidecar.providers.base import RealProvider
from vision_sidecar.types import Mask


class Sam2Provider(RealProvider):
    name = "sam2"
    licence = CATALOG["sam2"].licence
    model_name = CATALOG["sam2"].hf_id
    model_version = CATALOG["sam2"].repo_sha[:12]

    def load(self, device: str) -> None:
        torch = self._require_torch()
        try:
            from transformers import Sam2Model, Sam2Processor
        except ImportError as exc:
            raise ProviderUnavailableError(
                "transformers with Sam2Model is required for SAM 2.", integration=self.name
            ) from exc
        model_id = self.settings.sam2_model_id
        self._processor = self._from_pretrained(Sam2Processor.from_pretrained, model_id)
        self._model = self._from_pretrained(Sam2Model.from_pretrained, model_id)
        self._device = device
        self._model.to(device)
        self._model.eval()
        self._torch = torch

    def segment(
        self,
        frame: Image.Image,
        *,
        boxes: list[tuple[float, float, float, float]],
        points: list[tuple[float, float, int]],
    ) -> list[Mask]:
        if self._model is None or self._processor is None:
            raise ProviderUnavailableError("SAM 2 is not loaded.", integration=self.name)
        width, height = frame.size
        input_boxes = (
            [[[x1 * width, y1 * height, x2 * width, y2 * height] for x1, y1, x2, y2 in boxes]]
            if boxes
            else None
        )
        input_points = [[[x * width, y * height] for x, y, _label in points]] if points else None
        input_labels = [[label for _x, _y, label in points]] if points else None
        kwargs: dict[str, object] = {"images": frame, "return_tensors": "pt"}
        if input_boxes:
            kwargs["input_boxes"] = input_boxes
        if input_points:
            kwargs["input_points"] = input_points
            kwargs["input_labels"] = input_labels
        inputs = self._processor(**kwargs)
        inputs = {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        torch = self._torch
        try:
            with torch.inference_mode():
                outputs = self._model(**inputs)
            masks = self._processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"]
            )[0]
        except Exception as exc:
            self._maybe_oom(exc)
            raise
        results: list[Mask] = []
        for index, mask_tensor in enumerate(masks):
            array = mask_tensor.numpy()
            if array.ndim == 3:
                array = array[0]
            area = float(array.mean()) if array.size else 0.0
            box = boxes[index] if index < len(boxes) else (0.0, 0.0, 1.0, 1.0)
            results.append(Mask(score=0.9, box_xyxy=box, area_fraction=round(area, 4)))
        return results
