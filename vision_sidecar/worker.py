"""Serial job worker: per-stage timeouts, cancellation, crash isolation."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from PIL import Image

from app.logging_config import correlation_id_var, get_logger
from shared.errors import AssistantError
from vision_sidecar.config import SidecarSettings
from vision_sidecar.errors import GpuOomError, JobCancelledError
from vision_sidecar.frames import decode_frame
from vision_sidecar.lifecycle import ModelLifecycle
from vision_sidecar.metrics import cuda_available
from vision_sidecar.providers.mock import apply_fault
from vision_sidecar.providers.registry import ProviderBundle
from vision_sidecar.queue import Job, JobQueue
from vision_sidecar.types import (
    DepthRequest,
    DetectRequest,
    EmbeddingVector,
    EmbedRequest,
    EmbedResult,
    FaultKind,
    JobKind,
    JobStatus,
    ProviderMeta,
    SceneEmbedRequest,
    SegmentRequest,
)

logger = get_logger(__name__)


class VisionWorker:
    def __init__(
        self,
        settings: SidecarSettings,
        queue: JobQueue,
        providers: ProviderBundle,
        lifecycle: ModelLifecycle,
    ) -> None:
        self.settings = settings
        self.queue = queue
        self.providers = providers
        self.lifecycle = lifecycle
        self._stopping = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self.jobs_failed = 0
        self.jobs_succeeded = 0
        self.crashes_isolated = 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        self._stopping.clear()
        self._task = asyncio.create_task(self._loop(), name="vision-sidecar-worker")

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.lifecycle.unload_all()

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                get_next = asyncio.create_task(self.queue.get_next())
                stop = asyncio.create_task(self._stopping.wait())
                done, pending = await asyncio.wait(
                    {get_next, stop}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                if stop in done:
                    get_next.cancel()
                    return
                job = get_next.result()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Vision worker failed to dequeue")
                await asyncio.sleep(0.2)
                continue
            try:
                await self._run_job(job)
            except asyncio.CancelledError:
                job.status = JobStatus.CANCELLED
                job.finished_at = time.time()
                job.done.set()
                raise
            except Exception:
                self.crashes_isolated += 1
                logger.exception("Vision job crashed; worker continues", extra={"job_id": job.id})
                _fail(job, "handler_error", "The vision worker isolated a crash and continued.")
                job.finished_at = time.time()
                job.done.set()
            finally:
                self.queue.task_done()

    async def _run_job(self, job: Job) -> None:
        token = correlation_id_var.set(job.correlation_id)
        started = time.perf_counter()
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        try:
            job.raise_if_cancelled()
            timeout = self.settings.timeout_for(job.kind.value)
            result = await asyncio.wait_for(self._execute(job), timeout=timeout)
            job.result = result
            job.status = JobStatus.SUCCEEDED
            self.jobs_succeeded += 1
        except TimeoutError:
            _fail(
                job, "stage_timeout", f"Exceeded {self.settings.timeout_for(job.kind.value):.0f}s."
            )
        except JobCancelledError:
            job.status = JobStatus.CANCELLED
            job.error = {"code": "job_cancelled", "message": "Cancelled."}
        except AssistantError as exc:
            _fail(job, exc.code, exc.message)
        except Exception as exc:
            self.crashes_isolated += 1
            logger.exception("Vision job raised", extra={"job_id": job.id})
            _fail(job, "handler_error", f"{type(exc).__name__}: {exc}")
        finally:
            job.latency_ms = round((time.perf_counter() - started) * 1000, 2)
            job.finished_at = time.time()
            job.done.set()
            correlation_id_var.reset(token)
            if job.status is JobStatus.FAILED:
                self.jobs_failed += 1

    async def _execute(self, job: Job) -> dict[str, Any]:
        device = self._device()
        kind = job.kind
        payload = job.payload
        fault = getattr(payload, "fault", None) if self.providers.mocked else None
        apply_fault(fault)
        if fault is FaultKind.EMPTY:
            meta = _mock_or_real_meta(self.providers, device)
            job.provider_name = meta.provider_name
            job.model_name = meta.model_name
            job.model_version = meta.model_version
            return {"detections": [], "empty_reason": "nothing_detected", "meta": meta.model_dump()}
        if fault is FaultKind.STALE_FRAME:
            job.degraded = True
            job.degraded_reason = "stale_frame"

        if kind is JobKind.DETECT:
            assert isinstance(payload, DetectRequest)
            return await self._detect(job, payload, device)
        if kind is JobKind.SEGMENT:
            assert isinstance(payload, SegmentRequest)
            return await self._segment(job, payload, device)
        if kind is JobKind.DEPTH:
            assert isinstance(payload, DepthRequest)
            return await self._depth(job, payload, device)
        if kind is JobKind.EMBED:
            assert isinstance(payload, EmbedRequest)
            return await self._embed(job, payload, device)
        assert isinstance(payload, SceneEmbedRequest)
        return await self._scene_embed(job, payload, device)

    async def _detect(self, job: Job, payload: DetectRequest, device: str) -> dict[str, Any]:
        image = self._image(payload.frame)
        provider = self.providers.detection
        detections = await self._call(
            provider,
            lambda: provider.detect(image, payload.queries, max_detections=payload.max_detections),
            device,
        )
        _stamp(job, provider, device)
        empty_reason = "nothing_detected" if not detections else None
        meta = _meta_of(provider, device, self.providers.mocked)
        return {
            "detections": [item.model_dump() for item in detections],
            "empty_reason": empty_reason,
            "meta": meta.model_dump(),
        }

    async def _segment(self, job: Job, payload: SegmentRequest, device: str) -> dict[str, Any]:
        image = self._image(payload.frame)
        provider = self.providers.segmentation
        boxes = [item.box_xyxy for item in payload.boxes]
        points = [(item.xy[0], item.xy[1], item.label) for item in payload.points]
        masks = await self._call(
            provider, lambda: provider.segment(image, boxes=boxes, points=points), device
        )
        _stamp(job, provider, device)
        return {
            "masks": [item.model_dump() for item in masks],
            "meta": _meta_of(provider, device, self.providers.mocked).model_dump(),
        }

    async def _depth(self, job: Job, payload: DepthRequest, device: str) -> dict[str, Any]:
        image = self._image(payload.frame)
        provider = self.providers.depth
        swap = payload.fault is FaultKind.DEPTH_SWAP

        def run() -> dict[str, Any]:
            if self.providers.mocked:
                from vision_sidecar.providers.mock import MockDepthProvider

                assert isinstance(provider, MockDepthProvider)
                raw = provider.depth(image, include_preview=payload.include_preview, swap=swap)
                depth = raw["depth"]
                meta = raw["meta"]
                return {"depth": depth.model_dump(), "meta": meta.model_dump()}  # type: ignore[union-attr]
            raw = provider.depth(image, include_preview=payload.include_preview)
            depth = raw["depth"]
            meta = raw["meta"]
            return {"depth": depth.model_dump(), "meta": meta.model_dump()}  # type: ignore[union-attr]

        result = await self._call(provider, run, device)
        _stamp(job, provider, device)
        return result

    async def _embed(self, job: Job, payload: EmbedRequest, device: str) -> dict[str, Any]:
        provider = self.providers.embedding
        frames = [self._image(crop) for crop in payload.crops]

        def run() -> list[EmbeddingVector]:
            vectors: list[EmbeddingVector] = []
            if frames:
                vectors.extend(provider.embed_images(frames))
            if payload.texts:
                vectors.extend(provider.embed_texts(payload.texts))
            return vectors

        vectors = await self._call(provider, run, device)
        _stamp(job, provider, device)
        return EmbedResult(
            vectors=vectors,
            meta=_meta_of(provider, device, self.providers.mocked),
        ).model_dump()

    async def _scene_embed(
        self, job: Job, payload: SceneEmbedRequest, device: str
    ) -> dict[str, Any]:
        image = self._image(payload.frame)
        provider = self.providers.embedding
        vector = await self._call(provider, lambda: provider.embed_scene(image), device)
        _stamp(job, provider, device)
        return EmbedResult(
            vectors=[vector],
            meta=_meta_of(provider, device, self.providers.mocked),
        ).model_dump()

    async def _call(self, provider: Any, fn: Any, device: str) -> Any:
        try:
            return await self.lifecycle.run(
                provider, fn, device=device, load_timeout=self.settings.vision_load_timeout_seconds
            )
        except GpuOomError:
            if device != "cpu" and self.settings.vision_cpu_fallback:
                logger.warning(
                    "GPU OOM; retrying provider on CPU", extra={"provider": provider.name}
                )
                return await self.lifecycle.run(
                    provider,
                    fn,
                    device="cpu",
                    load_timeout=self.settings.vision_load_timeout_seconds,
                )
            raise

    def _image(self, frame: Any) -> Image.Image:
        image, _raw, _digest = decode_frame(
            image_b64=frame.image_b64,
            image_path=frame.image_path,
            data_dir=self.settings.app_data_dir,
        )
        return image

    def _device(self) -> str:
        return self.settings.resolved_device(cuda_available())


def _fail(job: Job, code: str, message: str) -> None:
    job.status = JobStatus.FAILED
    job.error = {"code": code, "message": message}


def _stamp(job: Job, provider: Any, device: str) -> None:
    job.provider_name = provider.name
    job.model_name = provider.model_name
    job.model_version = provider.model_version
    if device == "cpu" and not str(getattr(provider, "model_version", "")).endswith("@cpu"):
        pass


def _meta_of(provider: Any, device: str, mocked: bool) -> ProviderMeta:
    return ProviderMeta(
        provider_name=provider.name,
        model_name=provider.model_name,
        model_version=provider.model_version,
        licence=provider.licence,
        device=device,
        mocked=mocked,
    )


def _mock_or_real_meta(bundle: ProviderBundle, device: str) -> ProviderMeta:
    return _meta_of(bundle.detection, device, bundle.mocked)
