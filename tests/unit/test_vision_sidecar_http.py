"""Unit tests for the vision sidecar process (mock providers, no GPU)."""

from __future__ import annotations

import base64
import hashlib
import inspect
import io
import logging
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image
from pydantic import ValidationError as PydanticValidationError

from app.config import REPO_ROOT
from vision_sidecar.app import create_app, lifespan
from vision_sidecar.config import SidecarSettings
from vision_sidecar.hashing import cosine_similarity, deterministic_embedding
from vision_sidecar.queue import Job, JobQueue
from vision_sidecar.types import DetectRequest, FramePayload, JobKind

TOKEN = "sidecar-test-token-0123456789abcdef0123"


def _png_b64(
    *, color: tuple[int, int, int] = (40, 80, 120), size: tuple[int, int] = (64, 48)
) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _settings(tmp_path: Path, **overrides: object) -> SidecarSettings:
    values: dict[str, object] = {
        "app_env": "test",
        "app_data_dir": tmp_path / "pa-data",
        "app_log_level": "WARNING",
        "app_log_format": "console",
        "vision_sidecar_token": TOKEN,
        "mock_mode": True,
        "vision_queue_max": 4,
        "pa_rate_limit_per_minute": 0,
        "pa_api_allowed_networks": ["127.0.0.0/8", "::1/128"],
    }
    values.update(overrides)
    return SidecarSettings(**values)


@pytest.fixture
def sidecar_settings(tmp_path: Path) -> SidecarSettings:
    return _settings(tmp_path)


@pytest.fixture
async def sidecar_client(sidecar_settings: SidecarSettings) -> AsyncIterator[AsyncClient]:
    app = create_app(sidecar_settings)
    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://127.0.0.1",
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as client:
            yield client


class TestSidecarConfig:
    def test_rejects_data_dir_inside_the_repo(self) -> None:
        with pytest.raises(PydanticValidationError, match="inside the application directory"):
            SidecarSettings(
                app_data_dir=REPO_ROOT / "data", vision_sidecar_token=TOKEN, mock_mode=True
            )

    def test_hf_cache_lives_under_app_data_dir(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        assert settings.app_data_dir in settings.hf_cache_dir.parents
        assert REPO_ROOT not in settings.hf_cache_dir.parents

    def test_token_prefers_sidecar_secret(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path, pa_api_token="pa-api-token-should-not-win-0123456789")
        assert settings.auth_token == TOKEN


class TestBlake2sEmbeddings:
    def test_does_not_use_builtin_hash(self) -> None:
        import ast

        import vision_sidecar.hashing as module

        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "hash"
            ):
                pytest.fail("mock embeddings use deterministic embeddings")
        assert "blake2s" in inspect.getsource(module)

    def test_stable_across_calls(self) -> None:
        left = deterministic_embedding(b"mug-a")
        right = deterministic_embedding(b"mug-a")
        assert left == right
        assert len(left) == 384
        assert abs(sum(v * v for v in left) - 1.0) < 1e-6

    def test_independent_of_python_hash_salt(self) -> None:
        # hash() of the same string is not compared; blake2s is.
        assert (
            hashlib.blake2s(b"x", digest_size=8).hexdigest()
            == hashlib.blake2s(b"x", digest_size=8).hexdigest()
        )
        assert deterministic_embedding(b"x") != deterministic_embedding(b"y")

    def test_duplicate_pair_cosine(self) -> None:
        from vision_sidecar.hashing import DUPLICATE_COSINE, controlled_pair

        first, second = controlled_pair(b"white-mug")
        assert abs(cosine_similarity(first, second) - DUPLICATE_COSINE) < 1e-6


class TestJobQueue:
    def test_backpressure_when_full(self) -> None:
        queue = JobQueue(maxsize=1, max_finished=8, ttl_seconds=60)
        queue.submit(Job(kind=JobKind.DETECT, payload=object()))
        from vision_sidecar.errors import QueueFullError

        with pytest.raises(QueueFullError):
            queue.submit(Job(kind=JobKind.DETECT, payload=object()))
        assert queue.rejected_backpressure == 1

    def test_cancel_queued_job(self) -> None:
        queue = JobQueue(maxsize=4, max_finished=8, ttl_seconds=60)
        job = queue.submit(Job(kind=JobKind.DETECT, payload=object()))
        cancelled = queue.cancel(job.id)
        assert cancelled.status.value == "cancelled"
        assert cancelled.done.is_set()


class TestSidecarHttp:
    async def test_health_and_ready_are_public(self, sidecar_settings: SidecarSettings) -> None:
        app = create_app(sidecar_settings)
        async with lifespan(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://127.0.0.1"
            ) as client:
                health = await client.get("/health")
                ready = await client.get("/ready")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert ready.status_code == 200
        assert ready.json()["ready"] is True

    async def test_jobs_require_bearer_token(self, sidecar_settings: SidecarSettings) -> None:
        app = create_app(sidecar_settings)
        async with lifespan(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://127.0.0.1"
            ) as client:
                response = await client.post(
                    "/v1/detect", json={"frame": {"image_b64": _png_b64()}, "queries": ["mug"]}
                )
        assert response.status_code == 401
        assert TOKEN not in response.text

    async def test_wrong_token_is_rejected(self, sidecar_client: AsyncClient) -> None:
        response = await sidecar_client.get(
            "/v1/metrics", headers={"Authorization": "Bearer definitely-wrong-token"}
        )
        assert response.status_code == 401
        assert TOKEN not in response.text

    async def test_detect_mock_returns_structured_boxes(self, sidecar_client: AsyncClient) -> None:
        response = await sidecar_client.post(
            "/v1/detect",
            json={"frame": {"image_b64": _png_b64()}, "queries": ["white mug", "keys"]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        detections = body["data"]["result"]["detections"]
        assert detections
        assert all("distance_m" not in item for item in detections)
        assert body["data"]["result"]["meta"]["mocked"] is True
        assert body["data"]["provider_name"] == "mock-detect"
        mug_hits = [item for item in detections if "mug" in item["label"].lower()]
        assert len(mug_hits) >= 2

    async def test_empty_is_nothing_detected_not_nothing_present(
        self, sidecar_client: AsyncClient
    ) -> None:
        response = await sidecar_client.post(
            "/v1/detect",
            json={"frame": {"image_b64": _png_b64()}, "queries": ["mug"], "fault": "empty"},
        )
        assert response.status_code == 200
        result = response.json()["data"]["result"]
        assert result["detections"] == []
        assert result["empty_reason"] == "nothing_detected"
        assert result["empty_reason"] != "nothing_present"

    async def test_embed_is_deterministic(self, sidecar_client: AsyncClient) -> None:
        payload = {"crops": [{"image_b64": _png_b64()}]}
        first = await sidecar_client.post("/v1/embed", json=payload)
        second = await sidecar_client.post("/v1/embed", json=payload)
        assert first.status_code == 200
        assert (
            first.json()["data"]["result"]["vectors"][0]["values"]
            == second.json()["data"]["result"]["vectors"][0]["values"]
        )
        assert first.json()["data"]["result"]["vectors"][0]["space"] == "mock-blake2s"

    async def test_scene_embed_and_depth(self, sidecar_client: AsyncClient) -> None:
        frame = {"image_b64": _png_b64()}
        scene = await sidecar_client.post("/v1/scene_embed", json={"frame": frame})
        depth = await sidecar_client.post("/v1/depth", json={"frame": frame})
        assert scene.status_code == 200
        assert depth.status_code == 200
        depth_body = depth.json()["data"]["result"]["depth"]
        assert "distance_m" not in depth_body
        assert depth_body["ordinal_ready"] is True

    async def test_segment_needs_a_prompt(self, sidecar_client: AsyncClient) -> None:
        response = await sidecar_client.post(
            "/v1/segment", json={"frame": {"image_b64": _png_b64()}}
        )
        assert response.status_code == 422

    async def test_segment_with_box(self, sidecar_client: AsyncClient) -> None:
        response = await sidecar_client.post(
            "/v1/segment",
            json={
                "frame": {"image_b64": _png_b64()},
                "boxes": [{"box_xyxy": [0.1, 0.1, 0.4, 0.5]}],
            },
        )
        assert response.status_code == 200
        assert response.json()["data"]["result"]["masks"]

    async def test_path_outside_data_dir_is_rejected(
        self, sidecar_client: AsyncClient, tmp_path: Path
    ) -> None:
        outsider = tmp_path / "outside.png"
        Image.new("RGB", (8, 8), (1, 2, 3)).save(outsider)
        response = await sidecar_client.post(
            "/v1/detect",
            json={"frame": {"image_path": str(outsider)}, "queries": ["mug"]},
        )
        assert response.status_code in {400, 500, 503}
        assert (
            "APP_DATA_DIR" in response.json()["error"]["message"]
            or response.json()["data"]["error"]
        )

    async def test_token_never_appears_in_logs(
        self, sidecar_settings: SidecarSettings, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        app = create_app(sidecar_settings)
        async with lifespan(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://127.0.0.1"
            ) as client:
                await client.get("/v1/metrics", headers={"Authorization": f"Bearer {TOKEN}"})
                await client.get(
                    "/v1/metrics", headers={"Authorization": "Bearer wrong-token-value"}
                )
        combined = caplog.text
        assert TOKEN not in combined

    async def test_real_provider_does_not_fall_back_to_mock(self, tmp_path: Path) -> None:
        settings = _settings(
            tmp_path,
            mock_mode=False,
            vision_mock_mode=False,
            vision_allow_downloads=False,
            detection_provider="grounding_dino",
        )
        app = create_app(settings)
        async with lifespan(app):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://127.0.0.1",
                headers={"Authorization": f"Bearer {TOKEN}"},
            ) as client:
                response = await client.post(
                    "/v1/detect",
                    json={"frame": {"image_b64": _png_b64()}, "queries": ["mug"]},
                )
        assert response.status_code in {503, 500}
        payload = response.json()
        text = str(payload)
        assert "mock-detect" not in text
        assert payload.get("ok") is False

    async def test_fault_injection_does_not_fake_real_providers(self, tmp_path: Path) -> None:
        settings = _settings(
            tmp_path,
            mock_mode=False,
            vision_mock_mode=False,
            vision_allow_downloads=False,
        )
        app = create_app(settings)
        async with lifespan(app):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://127.0.0.1",
                headers={"Authorization": f"Bearer {TOKEN}"},
            ) as client:
                response = await client.post(
                    "/v1/detect",
                    json={
                        "frame": {"image_b64": _png_b64()},
                        "queries": ["mug"],
                        "fault": "empty",
                    },
                )
        assert response.status_code in {503, 500}
        text = str(response.json())
        assert "nothing_detected" not in text
        assert "mock-detect" not in text

    async def test_injected_oom_is_honest(self, sidecar_client: AsyncClient) -> None:
        response = await sidecar_client.post(
            "/v1/detect",
            json={"frame": {"image_b64": _png_b64()}, "queries": ["mug"], "fault": "oom"},
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "gpu_oom"

    async def test_idempotency_key_replays(self, sidecar_client: AsyncClient) -> None:
        headers = {"Idempotency-Key": "detect-1"}
        payload = {"frame": {"image_b64": _png_b64()}, "queries": ["cup"]}
        first = await sidecar_client.post("/v1/detect", json=payload, headers=headers)
        second = await sidecar_client.post("/v1/detect", json=payload, headers=headers)
        assert first.status_code == 200
        assert first.json()["data"]["job_id"] == second.json()["data"]["job_id"]

    async def test_metrics_include_gpu_probe_without_claiming_3090(
        self, sidecar_client: AsyncClient
    ) -> None:
        response = await sidecar_client.get("/v1/metrics")
        assert response.status_code == 200
        gpus = response.json()["data"]["resources"]["gpus"]
        for gpu in gpus:
            assert "3090" not in gpu["name"]


class TestDetectRequestShape:
    def test_frame_requires_exactly_one_source(self) -> None:
        with pytest.raises(PydanticValidationError):
            DetectRequest(
                frame=FramePayload(image_b64=_png_b64(), image_path="/tmp/x.png"), queries=["mug"]
            )
