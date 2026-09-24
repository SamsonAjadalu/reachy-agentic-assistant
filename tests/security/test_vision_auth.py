"""Vision endpoints require the API token."""

from __future__ import annotations

from httpx import AsyncClient


class TestVisionAuth:
    async def test_look_rejects_anonymous(self, anonymous_client: AsyncClient) -> None:
        response = await anonymous_client.post("/api/v1/vision/look", json={"query": "mug"})
        assert response.status_code == 401

    async def test_camera_status_rejects_anonymous(self, anonymous_client: AsyncClient) -> None:
        response = await anonymous_client.get("/api/v1/vision/camera/status")
        assert response.status_code == 401

    async def test_watches_reject_anonymous(self, anonymous_client: AsyncClient) -> None:
        response = await anonymous_client.get("/api/v1/vision/watches")
        assert response.status_code == 401
