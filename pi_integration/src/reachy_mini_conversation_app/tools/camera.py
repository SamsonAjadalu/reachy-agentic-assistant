import base64
import asyncio
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.camera_runtime import CameraFrameError
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class Camera(Tool):
    """Take a picture with the camera and ask a question about it."""

    name = "camera"
    description = "Take a picture with the camera and ask a question about it."
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask about the picture",
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Take a picture with the camera and ask a question about it."""
        question = (kwargs.get("question") or "").strip()
        if not question:
            logger.warning("camera: empty question")
            return {"error": "question must be a non-empty string"}

        logger.info("Tool call: camera question=%s", question[:120])

        if not deps.camera_enabled:
            logger.error("Camera is disabled")
            return {"error": "Camera is disabled"}

        sampler = deps.camera_sampler
        if sampler is None:
            logger.error("No frame available from camera")
            return {"error": "No frame available", "code": "camera_unavailable"}

        try:
            snapshot = await asyncio.to_thread(
                sampler.get_snapshot,
                max_age_ms=1000,
                wait_timeout_s=1.0,
            )
        except CameraFrameError as exc:
            logger.error("No frame available from camera: %s", exc.code)
            return {
                "error": "No frame available",
                "code": exc.code,
                "retryable": exc.retryable,
            }

        return {
            "b64_im": base64.b64encode(snapshot.image_bytes).decode("utf-8"),
            "image_width": snapshot.image_width,
            "image_height": snapshot.image_height,
            "jpeg_bytes": len(snapshot.image_bytes),
            "frame_id": snapshot.frame_id,
            "capture_utc": snapshot.capture_utc.isoformat().replace("+00:00", "Z"),
            "frame_age_ms": snapshot.age_ms(),
        }
