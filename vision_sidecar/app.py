"""FastAPI app for the perception sidecar."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from starlette.responses import JSONResponse

from app.error_handlers import register_error_handlers
from app.logging_config import configure_logging, get_logger
from app.middleware.context import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from app.middleware.network import NetworkAllowlistMiddleware
from security.redaction import register_secret
from shared.errors import NotFoundError
from shared.timeutils import isoformat_utc, utcnow
from vision_sidecar import __version__
from vision_sidecar.auth import require_token
from vision_sidecar.config import SidecarSettings
from vision_sidecar.lifecycle import ModelLifecycle
from vision_sidecar.metrics import snapshot_as_dict, snapshot_resources
from vision_sidecar.providers.registry import build_providers
from vision_sidecar.queue import Job, JobQueue
from vision_sidecar.types import (
    DepthRequest,
    DetectRequest,
    EmbedRequest,
    JobKind,
    JobPublic,
    JobStatus,
    JobSubmitRequest,
    SceneEmbedRequest,
    SegmentRequest,
)
from vision_sidecar.worker import VisionWorker

logger = get_logger(__name__)

PUBLIC_PATHS = frozenset({"/health", "/ready"})


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: SidecarSettings = app.state.settings
    settings.ensure_directories()
    settings.apply_hf_env()
    register_secret(settings.auth_token)
    providers = build_providers(settings)
    queue = JobQueue(
        maxsize=settings.vision_queue_max,
        max_finished=settings.vision_max_finished_jobs,
        ttl_seconds=settings.vision_job_ttl_seconds,
    )
    lifecycle = ModelLifecycle(
        serial=settings.vision_serial_gpu, keep_loaded=settings.vision_keep_loaded
    )
    worker = VisionWorker(settings, queue, providers, lifecycle)
    app.state.queue = queue
    app.state.worker = worker
    app.state.providers = providers
    app.state.lifecycle = lifecycle
    await worker.start()
    logger.info(
        "Vision sidecar started",
        extra={
            "mock": providers.mocked,
            "queue_max": settings.vision_queue_max,
            "host": settings.vision_sidecar_host,
            "port": settings.vision_sidecar_port,
        },
    )
    try:
        yield
    finally:
        await worker.stop()
        logger.info("Vision sidecar stopped")


def create_app(settings: SidecarSettings | None = None) -> FastAPI:
    settings = settings or SidecarSettings()
    configure_logging(settings.app_log_level, settings.app_log_format)
    app = FastAPI(
        title="Reachy vision sidecar",
        version=__version__,
        description="Perception inference process. FastAPI submits typed jobs over localhost HTTP.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.settings = settings
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.pa_rate_limit_per_minute,
        exempt_paths=PUBLIC_PATHS,
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.vision_max_request_bytes)
    app.add_middleware(
        NetworkAllowlistMiddleware,
        allowed_networks=settings.pa_api_allowed_networks,
        exempt_paths=PUBLIC_PATHS,
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)
    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "timestamp": isoformat_utc(utcnow())}

    @app.get("/ready")
    async def ready(request: Request) -> dict[str, Any]:
        settings: SidecarSettings = request.app.state.settings
        worker: VisionWorker = request.app.state.worker
        queue: JobQueue = request.app.state.queue
        checks: dict[str, Any] = {
            "token_configured": bool(settings.auth_token),
            "queue": {"ok": True, "depth": queue.depth, "max": settings.vision_queue_max},
            "worker": {"ok": worker.running},
            "mock": request.app.state.providers.mocked,
            "loaded_provider": request.app.state.lifecycle.loaded_name,
        }
        ready_ok = bool(checks["token_configured"] and checks["worker"]["ok"])
        return {"ready": ready_ok, "checks": checks}

    @app.get("/v1/metrics")
    async def metrics(request: Request, _: str = Depends(require_token)) -> dict[str, Any]:
        settings: SidecarSettings = request.app.state.settings
        queue: JobQueue = request.app.state.queue
        worker: VisionWorker = request.app.state.worker
        lifecycle: ModelLifecycle = request.app.state.lifecycle
        resources = snapshot_as_dict(snapshot_resources())
        return {
            "ok": True,
            "data": {
                "queue_depth": queue.depth,
                "queue_max": settings.vision_queue_max,
                "rejected_backpressure": queue.rejected_backpressure,
                "accepted": queue.accepted,
                "jobs_succeeded": worker.jobs_succeeded,
                "jobs_failed": worker.jobs_failed,
                "crashes_isolated": worker.crashes_isolated,
                "load_thrash_count": lifecycle.load_thrash,
                "loaded_provider": lifecycle.loaded_name,
                "device": lifecycle.device,
                "mock": request.app.state.providers.mocked,
                "resources": resources,
            },
            "request_id": getattr(request.state, "request_id", None),
        }

    @app.get("/v1/providers")
    async def providers(request: Request, _: str = Depends(require_token)) -> dict[str, Any]:
        bundle = request.app.state.providers
        settings: SidecarSettings = request.app.state.settings
        return {
            "ok": True,
            "data": {
                "mocked": bundle.mocked,
                "detection": bundle.detection.name,
                "segmentation": bundle.segmentation.name,
                "depth": bundle.depth.name,
                "embedding": bundle.embedding.name,
                "detection_backend": settings.detection_provider,
                "allow_downloads": settings.vision_allow_downloads,
            },
            "request_id": getattr(request.state, "request_id", None),
        }

    @app.post("/v1/jobs", status_code=202)
    async def submit_job(
        body: JobSubmitRequest, request: Request, _: str = Depends(require_token)
    ) -> dict[str, Any]:
        job = await _enqueue(request, body.kind, _payload_for(body), body.wait_ms)
        status = 200 if job.status is JobStatus.SUCCEEDED else 202
        return _job_response(request, job, status)

    @app.get("/v1/jobs/{job_id}")
    async def get_job(
        job_id: str, request: Request, _: str = Depends(require_token)
    ) -> dict[str, Any]:
        job = request.app.state.queue.get(job_id)
        if job is None:
            raise NotFoundError(f"No vision job {job_id}.")
        return _job_response(request, job, 200)

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_job(
        job_id: str, request: Request, _: str = Depends(require_token)
    ) -> dict[str, Any]:
        job = request.app.state.queue.cancel(job_id)
        return _job_response(request, job, 200)

    @app.post("/v1/detect")
    async def detect(
        body: DetectRequest, request: Request, _: str = Depends(require_token)
    ) -> dict[str, Any]:
        wait_ms = _wait_header(
            request, default_ms=int(request.app.state.settings.vision_detect_timeout_seconds * 1000)
        )
        job = await _enqueue(request, JobKind.DETECT, body, wait_ms)
        return _job_response(request, job, 200 if job.status is JobStatus.SUCCEEDED else 202)

    @app.post("/v1/segment")
    async def segment(
        body: SegmentRequest, request: Request, _: str = Depends(require_token)
    ) -> dict[str, Any]:
        wait_ms = _wait_header(
            request,
            default_ms=int(request.app.state.settings.vision_segment_timeout_seconds * 1000),
        )
        job = await _enqueue(request, JobKind.SEGMENT, body, wait_ms)
        return _job_response(request, job, 200 if job.status is JobStatus.SUCCEEDED else 202)

    @app.post("/v1/depth")
    async def depth(
        body: DepthRequest, request: Request, _: str = Depends(require_token)
    ) -> dict[str, Any]:
        wait_ms = _wait_header(
            request, default_ms=int(request.app.state.settings.vision_depth_timeout_seconds * 1000)
        )
        job = await _enqueue(request, JobKind.DEPTH, body, wait_ms)
        return _job_response(request, job, 200 if job.status is JobStatus.SUCCEEDED else 202)

    @app.post("/v1/embed")
    async def embed(
        body: EmbedRequest, request: Request, _: str = Depends(require_token)
    ) -> dict[str, Any]:
        wait_ms = _wait_header(
            request, default_ms=int(request.app.state.settings.vision_embed_timeout_seconds * 1000)
        )
        job = await _enqueue(request, JobKind.EMBED, body, wait_ms)
        return _job_response(request, job, 200 if job.status is JobStatus.SUCCEEDED else 202)

    @app.post("/v1/scene_embed")
    async def scene_embed(
        body: SceneEmbedRequest, request: Request, _: str = Depends(require_token)
    ) -> dict[str, Any]:
        wait_ms = _wait_header(
            request, default_ms=int(request.app.state.settings.vision_embed_timeout_seconds * 1000)
        )
        job = await _enqueue(request, JobKind.SCENE_EMBED, body, wait_ms)
        return _job_response(request, job, 200 if job.status is JobStatus.SUCCEEDED else 202)

    @app.post("/v1/models/unload")
    async def unload(request: Request, _: str = Depends(require_token)) -> dict[str, Any]:
        request.app.state.lifecycle.unload_all()
        return {
            "ok": True,
            "data": {"loaded_provider": None},
            "request_id": getattr(request.state, "request_id", None),
        }


async def _enqueue(request: Request, kind: JobKind, payload: Any, wait_ms: int) -> Job:
    queue: JobQueue = request.app.state.queue
    job = Job(
        kind=kind,
        payload=payload,
        correlation_id=getattr(request.state, "correlation_id", None),
        idempotency_key=request.headers.get("Idempotency-Key"),
    )
    job = queue.submit(job)
    if wait_ms > 0 and not job.done.is_set():
        try:
            await asyncio.wait_for(job.done.wait(), timeout=wait_ms / 1000.0)
        except TimeoutError:
            pass
    return job


def _payload_for(body: JobSubmitRequest) -> Any:
    mapping = {
        JobKind.DETECT: body.detect,
        JobKind.SEGMENT: body.segment,
        JobKind.DEPTH: body.depth,
        JobKind.EMBED: body.embed,
        JobKind.SCENE_EMBED: body.scene_embed,
    }
    return mapping[body.kind]


def _wait_header(request: Request, *, default_ms: int) -> int:
    raw = request.headers.get("X-Wait-Ms")
    if raw is None:
        return default_ms
    try:
        return max(0, min(120_000, int(raw)))
    except ValueError:
        return default_ms


def _job_response(request: Request, job: Job, status: int) -> JSONResponse:
    public: JobPublic = job.public()
    payload = {
        "ok": job.status is not JobStatus.FAILED,
        "data": public.model_dump(),
        "request_id": getattr(request.state, "request_id", None),
    }
    if job.status is JobStatus.FAILED:
        payload["ok"] = False
        payload["error"] = job.error
        status = (
            503
            if (job.error or {}).get("code")
            in {"provider_unavailable", "gpu_oom", "vision_disabled"}
            else 500
        )
        if (job.error or {}).get("code") == "stage_timeout":
            status = 504
        if (job.error or {}).get("code") == "job_cancelled":
            status = 409
    return JSONResponse(status_code=status, content=payload)
