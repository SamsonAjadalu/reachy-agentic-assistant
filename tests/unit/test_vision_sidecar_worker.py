"""Crash isolation, lifecycle serialisation, and worker timeouts."""

from __future__ import annotations

import asyncio
from pathlib import Path

from PIL import Image

from vision_sidecar.config import SidecarSettings
from vision_sidecar.lifecycle import ModelLifecycle
from vision_sidecar.providers.mock import (
    MockDepthProvider,
    MockDetectionProvider,
    MockEmbeddingProvider,
    MockSegmentationProvider,
)
from vision_sidecar.providers.registry import ProviderBundle
from vision_sidecar.queue import Job, JobQueue
from vision_sidecar.types import DetectRequest, FramePayload, JobKind, JobStatus
from vision_sidecar.worker import VisionWorker

TOKEN = "sidecar-test-token-0123456789abcdef0123"


class BoomProvider(MockDetectionProvider):
    def detect(self, frame: Image.Image, queries: list[str], *, max_detections: int = 32):
        raise RuntimeError("simulated provider crash")


def _settings(tmp_path: Path) -> SidecarSettings:
    return SidecarSettings(
        app_env="test",
        app_data_dir=tmp_path / "pa-data",
        vision_sidecar_token=TOKEN,
        mock_mode=True,
        vision_detect_timeout_seconds=5.0,
        vision_load_timeout_seconds=5.0,
        vision_queue_max=8,
        pa_rate_limit_per_minute=0,
    )


def _png_frame() -> FramePayload:
    import base64
    import io

    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (9, 9, 9)).save(buf, format="PNG")
    return FramePayload(image_b64=base64.b64encode(buf.getvalue()).decode("ascii"))


class TestLifecycle:
    async def test_serial_unload_counts_as_thrash(self) -> None:
        slot = ModelLifecycle(serial=True, keep_loaded=False)
        detect = MockDetectionProvider()
        embed = MockEmbeddingProvider()
        await slot.run(
            detect,
            lambda: detect.detect(Image.new("RGB", (8, 8)), ["mug"]),
            device="cpu",
            load_timeout=5,
        )
        await slot.run(embed, lambda: embed.embed_texts(["mug"]), device="cpu", load_timeout=5)
        assert slot.load_thrash >= 1
        assert slot.loaded_name is None


class TestWorkerIsolation:
    async def test_a_crashing_job_does_not_kill_the_worker(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        settings.ensure_directories()
        queue = JobQueue(maxsize=8, max_finished=16, ttl_seconds=60)
        boom = BoomProvider()
        bundle = ProviderBundle(
            detection=boom,
            segmentation=MockSegmentationProvider(),
            depth=MockDepthProvider(),
            embedding=MockEmbeddingProvider(),
            mocked=True,
        )
        worker = VisionWorker(settings, queue, bundle, ModelLifecycle(keep_loaded=False))
        await worker.start()
        try:
            bad = queue.submit(
                Job(kind=JobKind.DETECT, payload=DetectRequest(frame=_png_frame(), queries=["mug"]))
            )
            await asyncio.wait_for(bad.done.wait(), timeout=5)
            assert bad.status is JobStatus.FAILED
            good_bundle = ProviderBundle(
                detection=MockDetectionProvider(),
                segmentation=MockSegmentationProvider(),
                depth=MockDepthProvider(),
                embedding=MockEmbeddingProvider(),
                mocked=True,
            )
            worker.providers = good_bundle
            ok = queue.submit(
                Job(
                    kind=JobKind.DETECT, payload=DetectRequest(frame=_png_frame(), queries=["keys"])
                )
            )
            await asyncio.wait_for(ok.done.wait(), timeout=5)
            assert ok.status is JobStatus.SUCCEEDED
            assert worker.jobs_succeeded >= 1
        finally:
            await worker.stop()
