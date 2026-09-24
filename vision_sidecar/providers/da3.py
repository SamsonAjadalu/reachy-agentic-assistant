"""DA3MONO-LARGE relative depth. Apache-2.0. Metric depth is refused."""

from __future__ import annotations

import base64
import io
import statistics

from PIL import Image

from vision_sidecar.catalog import CATALOG
from vision_sidecar.errors import ProviderUnavailableError
from vision_sidecar.providers.base import RealProvider
from vision_sidecar.types import DepthMap, ProviderMeta


class Da3MonoProvider(RealProvider):
    name = "da3mono_large"
    licence = CATALOG["da3mono_large"].licence
    model_name = CATALOG["da3mono_large"].hf_id
    model_version = CATALOG["da3mono_large"].repo_sha[:12]

    def load(self, device: str) -> None:
        torch = self._require_torch()
        model_id = self.settings.da3_model_id
        cache = self._cache_dir()
        model = None
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError:
            DepthAnything3 = None  # type: ignore[misc, assignment]
        if DepthAnything3 is not None:
            try:
                model = DepthAnything3.from_pretrained(
                    model_id,
                    cache_dir=cache,
                    local_files_only=not self.settings.vision_allow_downloads,
                )
            except TypeError:
                model = self._from_pretrained(DepthAnything3.from_pretrained, model_id)
            except Exception as exc:
                raise ProviderUnavailableError(
                    f"DA3MONO-LARGE could not load {model_id}. Output is not faked.",
                    integration=self.name,
                ) from exc
        if model is None:
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ProviderUnavailableError(
                    "Neither depth_anything_3 nor transformers is available. "
                    "DA3MONO-LARGE will not be substituted with Depth Anything V2 "
                    "(Base/Large are CC-BY-NC; use the catalogued V2-Metric-Outdoor-Base weights).",
                    integration=self.name,
                ) from exc
            model = self._from_pretrained(
                AutoModel.from_pretrained, model_id, trust_remote_code=True
            )
        self._model = model
        self._device = device
        self._torch = torch
        if hasattr(self._model, "to"):
            self._model.to(device)

    def depth(self, frame: Image.Image, *, include_preview: bool = True) -> dict[str, object]:
        if self._model is None:
            raise ProviderUnavailableError("DA3MONO-LARGE is not loaded.", integration=self.name)
        try:
            prediction = self._infer(frame)
        except Exception as exc:
            self._maybe_oom(exc)
            raise
        depth_map = _as_list(prediction)
        if not depth_map:
            raise ProviderUnavailableError(
                "DA3MONO-LARGE returned no depth.", integration=self.name
            )
        flat = [float(v) for row in depth_map for v in row]
        preview = _preview_png(depth_map) if include_preview else None
        result = DepthMap(
            width=frame.size[0],
            height=frame.size[1],
            relative_min=min(flat),
            relative_max=max(flat),
            relative_median=float(statistics.median(flat)),
            has_confidence=_has_confidence(prediction),
            preview_png_b64=preview,
            ordinal_ready=True,
            licence=self.licence,
            model_name=self.model_name,
            model_version=self.model_version,
        )
        return {
            "depth": result,
            "meta": ProviderMeta(
                provider_name=self.name,
                model_name=self.model_name,
                model_version=self.model_version,
                licence=self.licence,
                device=self._device,
                mocked=False,
            ),
        }

    def _infer(self, frame: Image.Image) -> object:
        model = self._model
        if hasattr(model, "inference"):
            return model.inference([frame])
        if callable(model):
            return model(frame)
        raise ProviderUnavailableError(
            "DA3 API does not expose inference().", integration=self.name
        )


def _as_list(prediction: object) -> list[list[float]]:
    if hasattr(prediction, "depth") and not isinstance(prediction, dict):
        return _as_list(prediction.depth)
    if isinstance(prediction, dict):
        for key in ("depth", "relative_depth", "prediction"):
            if key in prediction:
                return _as_list(prediction[key])
    array = getattr(prediction, "cpu", lambda: prediction)()
    array = getattr(array, "numpy", lambda: array)()
    if hasattr(array, "tolist"):
        data = array.tolist()
        # Squeeze batch dims: [B,H,W] or [B,1,H,W] → [H,W]
        while (
            isinstance(data, list)
            and data
            and isinstance(data[0], list)
            and data[0]
            and isinstance(data[0][0], list)
            and not isinstance(data[0][0][0], list)
            and len(data) == 1
        ):
            data = data[0]
        if data and isinstance(data[0], list):
            if data and data[0] and isinstance(data[0][0], list):
                return data[0]
            return data
    return []


def _has_confidence(prediction: object) -> bool:
    if isinstance(prediction, dict):
        return any(key in prediction for key in ("confidence", "conf"))
    conf = getattr(prediction, "conf", None)
    return conf is not None


def _preview_png(depth_map: list[list[float]]) -> str:
    flat = [v for row in depth_map for v in row]
    lo, hi = min(flat), max(flat)
    span = (hi - lo) or 1.0
    height = min(len(depth_map), 64)
    width = min(len(depth_map[0]), 64) if depth_map else 1
    image = Image.new("L", (width, height))
    pixels = []
    for y in range(height):
        src_y = int(y * (len(depth_map) - 1) / max(1, height - 1))
        row = depth_map[src_y]
        for x in range(width):
            src_x = int(x * (len(row) - 1) / max(1, width - 1))
            pixels.append(int(255 * (row[src_x] - lo) / span))
    image.putdata(pixels)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
