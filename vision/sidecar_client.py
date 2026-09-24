"""HTTP client for the perception sidecar. Used only when MOCK_MODE is off."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import httpx
from PIL import Image

from app.config import Settings
from vision.backend import PerceptionOutput, encode_b64
from vision.geometry import Box
from vision.labels import is_person_label, normalise_label
from vision.tracking import DetectionHypothesis
from vision_sidecar.types import DepthMap, DepthResult, DetectResult, EmbedResult, SegmentResult


class SidecarPerceptionBackend:
    """Calls detect → optional segment → depth → crop embeddings on the sidecar.

    Uses the configured workflow.
    """

    mocked = False

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        token = (
            settings.vision_sidecar_token.get_secret_value()
            or settings.pa_api_token.get_secret_value()
        )
        self._token = token
        timeout = max(settings.vision_sidecar_timeout_seconds, 180.0)
        self._client = httpx.AsyncClient(
            base_url=settings.vision_sidecar_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    async def detect(
        self,
        image_bytes: bytes,
        queries: list[str],
        *,
        max_detections: int = 32,
        inject_labels: list[str] | None = None,
    ) -> PerceptionOutput:
        if inject_labels:
            raise RuntimeError(
                "inject_labels is mock-only and cannot be used with the live sidecar backend."
            )
        frame = {"image_b64": encode_b64(image_bytes)}
        detect_payload: dict[str, Any] = {
            "frame": frame,
            "queries": queries or ["object"],
            "max_detections": max_detections,
        }
        detect = DetectResult.model_validate(await self._post_result("/v1/detect", detect_payload))
        if detect.meta.mocked:
            raise RuntimeError(
                "Live sidecar returned mocked detections; refusing to treat them as real."
            )

        depth: DepthMap | None = None
        try:
            depth_result = DepthResult.model_validate(
                await self._post_result(
                    "/v1/depth",
                    {"frame": frame, "include_preview": False},
                )
            )
            if depth_result.meta.mocked:
                raise RuntimeError("Live sidecar returned mocked depth.")
            depth = depth_result.depth
        except (httpx.HTTPError, RuntimeError, ValueError):
            depth = None

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        hypotheses: list[DetectionHypothesis] = []
        embeddings: list[list[float] | None] = []
        for item in detect.detections:
            person = is_person_label(item.label)
            vector = None
            if not person:
                crop = _crop(image, item.box_xyxy)
                buf = BytesIO()
                crop.save(buf, format="JPEG", quality=90)
                embed = EmbedResult.model_validate(
                    await self._post_result(
                        "/v1/embed",
                        {"crops": [{"image_b64": encode_b64(buf.getvalue())}]},
                    )
                )
                if embed.meta.mocked:
                    raise RuntimeError("Live sidecar returned mocked embeddings.")
                if embed.vectors:
                    vector = embed.vectors[0].values
            depth_median = _box_depth(depth, Box.from_xyxy(item.box_xyxy)) if depth else None
            hypotheses.append(
                DetectionHypothesis(
                    label=normalise_label(item.label),
                    box=Box.from_xyxy(item.box_xyxy),
                    score=item.score,
                    is_person=person,
                    embedding=vector,
                    depth_median=depth_median,
                    query=item.query,
                )
            )
            embeddings.append(vector)

        if detect.detections:
            try:
                boxes = [
                    {"box_xyxy": item.box_xyxy, "label": item.label}
                    for item in detect.detections[:16]
                ]
                seg = SegmentResult.model_validate(
                    await self._post_result(
                        "/v1/segment",
                        {"frame": frame, "boxes": boxes},
                    )
                )
                if seg.meta.mocked:
                    raise RuntimeError("Live sidecar returned mocked masks.")
            except (httpx.HTTPError, RuntimeError, ValueError):
                pass

        return PerceptionOutput(
            detections=hypotheses,
            empty_reason=detect.empty_reason,
            mocked=False,
            model_name=detect.meta.model_name,
            model_version=detect.meta.model_version,
            depth=depth,
            embeddings=embeddings,
        )

    async def _post_result(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._client.post(path, json=payload)
        body = response.json()
        result = _unwrap_job_result(body)
        if response.status_code >= 400 or result is None:
            error = body.get("error") if isinstance(body, dict) else None
            message = ""
            if isinstance(error, dict):
                message = str(error.get("message") or error.get("code") or "")
            raise RuntimeError(f"Sidecar {path} failed ({response.status_code}): {message or body}")
        return result

    async def aclose(self) -> None:
        await self._client.aclose()


def _unwrap_job_result(body: Any) -> dict[str, Any] | None:
    if not isinstance(body, dict):
        return None
    data = body.get("data")
    if isinstance(data, dict):
        result = data.get("result")
        if isinstance(result, dict):
            return result
        if "detections" in data or "depth" in data or "vectors" in data or "masks" in data:
            return data
    if "detections" in body or "depth" in body or "vectors" in body or "masks" in body:
        return body
    return None


def _crop(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    x0, y0, x1, y1 = box
    left = max(0, int(x0 * width))
    top = max(0, int(y0 * height))
    right = min(width, max(left + 1, int(x1 * width)))
    bottom = min(height, max(top + 1, int(y1 * height)))
    return image.crop((left, top, right, bottom)).resize((224, 224))


def _box_depth(depth: DepthMap, box: Box) -> float:
    span = depth.relative_max - depth.relative_min
    return round(depth.relative_min + span * ((box.cx + box.cy) / 2.0), 4)


def parse_optional_depth(payload: dict[str, Any] | None) -> DepthMap | None:
    if not payload:
        return None
    return DepthMap.model_validate(payload)
