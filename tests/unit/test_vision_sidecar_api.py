"""Sidecar HTTP: health is public, jobs need the bearer token, mocks run."""

from __future__ import annotations

import base64
import io
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from vision_sidecar.app import create_app, lifespan
from vision_sidecar.config import SidecarSettings

TOKEN = "test-sidecar-token-0123456789abcdef"


def _png_b64() -> str:
    image = Image.new("RGB", (32, 24), color=(12, 64, 128))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
def sidecar_settings(tmp_path: Path) -> SidecarSettings:
    data = tmp_path / "pa-data"
    data.mkdir()
    return SidecarSettings(
        app_env="test",
        app_data_dir=data,
        mock_mode=True,
        vision_sidecar_token=TOKEN,
        pa_rate_limit_per_minute=0,
        vision_queue_max=8,
    )


@pytest.fixture
async def sidecar_client(sidecar_settings: SidecarSettings) -> AsyncIterator[AsyncClient]:
    app = create_app(sidecar_settings)
    async with lifespan(app):
        transport = ASGITransport(app=app, client=("127.0.0.1", 123))
        async with AsyncClient(
            transport=transport,
            base_url="http://127.0.0.1",
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as client:
            yield client


@pytest.fixture
async def anonymous_sidecar(sidecar_settings: SidecarSettings) -> AsyncIterator[AsyncClient]:
    app = create_app(sidecar_settings)
    async with lifespan(app):
        transport = ASGITransport(app=app, client=("127.0.0.1", 123))
        async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            yield client


class TestHealth:
    async def test_health_and_ready_are_public(self, anonymous_sidecar: AsyncClient) -> None:
        health = await anonymous_sidecar.get("/health")
        ready = await anonymous_sidecar.get("/ready")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert ready.status_code == 200
        assert ready.json()["ready"] is True

    async def test_jobs_require_token(self, anonymous_sidecar: AsyncClient) -> None:
        response = await anonymous_sidecar.post(
            "/v1/detect", json={"frame": {"image_b64": _png_b64()}, "queries": ["mug"]}
        )
        assert response.status_code == 401
        assert TOKEN not in response.text
        assert response.json()["error"]["code"] == "unauthorized"

    async def test_wrong_token_is_rejected(self, anonymous_sidecar: AsyncClient) -> None:
        response = await anonymous_sidecar.get(
            "/v1/metrics", headers={"Authorization": "Bearer totally-not-the-token"}
        )
        assert response.status_code == 401


class TestTypedJobs:
    async def test_detect_returns_mock_boxes(self, sidecar_client: AsyncClient) -> None:
        response = await sidecar_client.post(
            "/v1/detect",
            json={"frame": {"image_b64": _png_b64()}, "queries": ["mug"]},
            headers={"X-Correlation-ID": "corr-detect-1"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        data = body["data"]
        assert data["status"] == "succeeded"
        assert data["kind"] == "detect"
        assert data["correlation_id"] == "corr-detect-1"
        assert data["result"]["detections"]
        assert data["provider_name"] == "mock-detect"

    async def test_scene_embed_is_blake2s_unit_vector(self, sidecar_client: AsyncClient) -> None:
        response = await sidecar_client.post(
            "/v1/scene_embed", json={"frame": {"image_b64": _png_b64()}}
        )
        assert response.status_code == 200
        vector = response.json()["data"]["result"]["vectors"][0]
        assert vector["space"] == "mock-blake2s"
        assert vector["dim"] == len(vector["values"]) == 384
        norm = sum(v * v for v in vector["values"]) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-5)

    async def test_metrics_include_gpu_inventory(self, sidecar_client: AsyncClient) -> None:
        response = await sidecar_client.get("/v1/metrics")
        assert response.status_code == 200
        resources = response.json()["data"]["resources"]
        assert "gpu_count" in resources
        for gpu in resources["gpus"]:
            assert "3090" not in gpu["name"]

    async def test_idempotency_key(self, sidecar_client: AsyncClient) -> None:
        payload = {"frame": {"image_b64": _png_b64()}, "queries": ["keys"]}
        first = await sidecar_client.post(
            "/v1/jobs",
            json={"kind": "detect", "detect": payload, "wait_ms": 5000},
            headers={"Idempotency-Key": "idem-1"},
        )
        second = await sidecar_client.post(
            "/v1/jobs",
            json={"kind": "detect", "detect": payload, "wait_ms": 5000},
            headers={"Idempotency-Key": "idem-1"},
        )
        assert first.json()["data"]["job_id"] == second.json()["data"]["job_id"]
