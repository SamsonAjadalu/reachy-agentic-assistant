"""DINOv2 embeddings via transformers. Apache-2.0. CPU fallback is reasonable."""

from __future__ import annotations

from PIL import Image

from vision_sidecar.catalog import CATALOG
from vision_sidecar.errors import ProviderUnavailableError
from vision_sidecar.hashing import l2_normalize
from vision_sidecar.providers.base import RealProvider
from vision_sidecar.types import EmbeddingVector


class Dinov2Provider(RealProvider):
    name = "dinov2"
    licence = CATALOG["dinov2"].licence
    model_name = CATALOG["dinov2"].hf_id
    model_version = CATALOG["dinov2"].repo_sha[:12]
    space = "dinov2"
    dim = 384

    def load(self, device: str) -> None:
        torch = self._require_torch()
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ProviderUnavailableError(
                "transformers is required for DINOv2.", integration=self.name
            ) from exc
        model_id = self.settings.dinov2_model_id
        self._processor = self._from_pretrained(AutoImageProcessor.from_pretrained, model_id)
        self._model = self._from_pretrained(AutoModel.from_pretrained, model_id)
        self._device = device
        self._model.to(device)
        self._model.eval()
        self._torch = torch
        hidden = int(getattr(self._model.config, "hidden_size", self.dim))
        self.dim = hidden
        self.model_name = model_id

    def embed_images(self, frames: list[Image.Image]) -> list[EmbeddingVector]:
        return [self._embed_one(frame) for frame in frames]

    def embed_texts(self, texts: list[str]) -> list[EmbeddingVector]:
        raise ProviderUnavailableError(
            "DINOv2 embeds image crops. Use image crops for visual embeddings.",
            integration=self.name,
        )

    def embed_scene(self, frame: Image.Image) -> EmbeddingVector:
        return self._embed_one(frame)

    def _embed_one(self, frame: Image.Image) -> EmbeddingVector:
        if self._model is None or self._processor is None:
            raise ProviderUnavailableError("DINOv2 is not loaded.", integration=self.name)
        inputs = self._processor(images=frame, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        torch = self._torch
        try:
            with torch.inference_mode():
                outputs = self._model(**inputs)
            token = outputs.last_hidden_state[:, 0]
            vector = token.squeeze(0).detach().float().cpu().tolist()
        except Exception as exc:
            self._maybe_oom(exc)
            raise
        values = l2_normalize([float(v) for v in vector])
        return EmbeddingVector(
            values=values,
            dim=len(values),
            space=self.space,
            model_name=self.model_name,
            model_version=self.model_version,
        )
