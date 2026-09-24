"""Tests for the camera tool."""

import base64
from types import SimpleNamespace
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.camera import Camera
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


@pytest.mark.asyncio
async def test_camera_tool_returns_base64_of_sdk_jpeg() -> None:
    """The tool reads the shared immutable cache instead of pulling the SDK."""
    jpeg_bytes = b"\xff\xd8jpeg\xff\xd9"
    reachy_mini = MagicMock()
    sampler = MagicMock()
    sampler.get_snapshot.return_value = SimpleNamespace(
        image_bytes=jpeg_bytes,
        image_width=320,
        image_height=240,
        frame_id=7,
        capture_utc=datetime.now(timezone.utc),
        age_ms=lambda: 5,
    )

    deps = ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=True,
        camera_sampler=sampler,
    )

    result = await Camera()(deps, question="What color is this?")

    assert result["b64_im"] == base64.b64encode(jpeg_bytes).decode("utf-8")
    assert result["frame_id"] == 7
    reachy_mini.media.get_frame_jpeg.assert_not_called()


@pytest.mark.asyncio
async def test_camera_tool_reports_error_when_no_frame() -> None:
    """With no frame available the tool returns an error."""
    reachy_mini = MagicMock()

    deps = ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=True,
        camera_sampler=None,
    )

    result = await Camera()(deps, question="What color is this?")

    assert "error" in result


@pytest.mark.asyncio
async def test_camera_tool_reports_error_when_camera_disabled() -> None:
    """With the camera disabled the tool returns an error without reading a frame."""
    reachy_mini = MagicMock()
    deps = ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=False,
    )

    result = await Camera()(deps, question="What color is this?")

    assert "error" in result
    reachy_mini.media.get_frame_jpeg.assert_not_called()
