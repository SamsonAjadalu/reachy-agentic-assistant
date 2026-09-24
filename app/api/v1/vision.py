"""Visual spatial memory endpoints for HF tools and Telegram-driven watches."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.dependencies.auth import Principal, require_token
from app.dependencies.common import Pagination, get_session, idempotency_key, pagination
from app.schemas.vision import (
    CameraStatusOut,
    CompareOut,
    CompareRequest,
    LastSeenOut,
    LookNowRequest,
    ObserveOut,
    ScanRequest,
    SceneOut,
    WatchCreate,
    WatchList,
    WatchOut,
)
from vision.enums import ObservationTrigger
from vision.pipeline import camera_status, observe_now, queue_scan
from vision.retrieval import compare_scenes, describe_scene, find_last_seen, search_memory
from vision.schemas import load_json as load_watch_json
from vision.watches import cancel_watch, create_watch, list_watches

router = APIRouter(prefix="/api/v1/vision", tags=["vision"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
PrincipalDep = Annotated[Principal, Depends(require_token)]
PaginationDep = Annotated[Pagination, Depends(pagination)]
IdempotencyDep = Annotated[str | None, Depends(idempotency_key)]


def _observe_out(result: Any) -> ObserveOut:
    presence = result.presence.value if hasattr(result.presence, "value") else str(result.presence)
    return ObserveOut(
        presence=presence,
        phrase=result.phrase,
        deferred=result.deferred,
        task_id=result.task_id,
        poll_url=result.poll_url,
        observation_id=result.observation_id,
        snapshot_id=result.snapshot_id,
        request_id=result.request_id,
        detections=result.detections or [],
        relations=result.relations or [],
        need_look=result.need_look,
        need_look_terms=result.need_look_terms,
        scan_outcome=result.scan_outcome,
        evidence_id=result.evidence_id,
        world_frame_status=result.world_frame_status,
        may_move_head=result.may_move_head,
    )


@router.get("/camera/status", response_model=CameraStatusOut, summary="Camera health")
async def get_camera_status(_: PrincipalDep, settings: SettingsDep) -> CameraStatusOut:
    payload = await camera_status(settings)
    return CameraStatusOut(**payload)


@router.post("/look", response_model=ObserveOut, summary="Look now")
async def look_now(
    payload: LookNowRequest,
    request: Request,
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    key: IdempotencyDep,
) -> ObserveOut | JSONResponse:
    result = await observe_now(
        session,
        settings,
        query=payload.query,
        zone_name=payload.zone,
        trigger=ObservationTrigger.QUERY,
        allow_scan=payload.allow_scan,
        preset_id=payload.preset_id,
        correlation_id=getattr(request.state, "correlation_id", None),
        idempotency_key=key,
        inject_labels=payload.inject_labels or None,
    )
    body = _observe_out(result)
    if result.deferred:
        return JSONResponse(status_code=202, content=body.model_dump(mode="json"))
    return body


@router.get("/memory/last-seen", response_model=LastSeenOut, summary="Find last seen")
async def last_seen(
    _: PrincipalDep,
    session: SessionDep,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> LastSeenOut:
    payload = await find_last_seen(session, q, limit=limit)
    return LastSeenOut(**payload)


@router.get("/memory/search", response_model=LastSeenOut, summary="Search visual memory")
async def search(
    _: PrincipalDep,
    session: SessionDep,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    limit: Annotated[int, Query(ge=1, le=20)] = 8,
) -> LastSeenOut:
    payload = await search_memory(session, q, limit=limit)
    return LastSeenOut(**payload)


@router.get("/scenes/{snapshot_id}", response_model=SceneOut, summary="Describe a previous scene")
async def get_scene(snapshot_id: str, _: PrincipalDep, session: SessionDep) -> SceneOut:
    return SceneOut(**await describe_scene(session, snapshot_id))


@router.post("/scenes/compare", response_model=CompareOut, summary="Compare two scenes")
async def compare(payload: CompareRequest, _: PrincipalDep, session: SessionDep) -> CompareOut:
    compared = await compare_scenes(session, payload.left_snapshot_id, payload.right_snapshot_id)
    return CompareOut(**compared)


@router.post("/watches", response_model=WatchOut, status_code=201, summary="Create a visual watch")
async def create_visual_watch(
    payload: WatchCreate,
    _: PrincipalDep,
    session: SessionDep,
    key: IdempotencyDep,
) -> WatchOut:
    zone_id = None
    if payload.zone:
        from vision.repositories import get_or_create_zone

        zone = await get_or_create_zone(session, payload.zone)
        zone_id = zone.id
    watch = await create_watch(
        session,
        query={"label": payload.label, "event": payload.event, "zone": payload.zone},
        cooldown_seconds=payload.cooldown_seconds,
        quiet_hours=payload.quiet_hours,
        zone_id=zone_id,
        idempotency_key=key,
    )
    return _watch_out(watch)


@router.get("/watches", response_model=WatchList, summary="List visual watches")
async def list_visual_watches(
    _: PrincipalDep,
    session: SessionDep,
    page: PaginationDep,
    status: Annotated[str | None, Query()] = None,
) -> WatchList:
    rows, total = await list_watches(session, status=status, limit=page.limit, offset=page.offset)
    items = [_watch_out(row) for row in rows]
    phrase = f"{total} visual watch(es)." if total else "No visual watches."
    return WatchList(items=items, total=total, limit=page.limit, offset=page.offset, phrase=phrase)


@router.post("/watches/{watch_id}/cancel", response_model=WatchOut, summary="Cancel a visual watch")
async def cancel_visual_watch(watch_id: str, _: PrincipalDep, session: SessionDep) -> WatchOut:
    watch = await cancel_watch(session, watch_id)
    out = _watch_out(watch)
    out.phrase = "That visual watch is cancelled."
    return out


@router.post("/scans", summary="Request a preset scan (202 ticket; not approval-gated)")
async def request_scan(
    payload: ScanRequest,
    request: Request,
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
) -> JSONResponse:
    queued = await queue_scan(
        session,
        settings,
        preset_id=payload.preset_id,
        query=payload.query,
        zone_id=None,
        correlation_id=getattr(request.state, "correlation_id", None),
        prior_belief=0.5,
    )
    status = 202 if queued.get("deferred") else 409
    return JSONResponse(status_code=status, content=queued)


def _watch_out(watch: Any) -> WatchOut:
    query = load_watch_json(watch.query_json, default={}) or {}
    return WatchOut(
        id=watch.id,
        status=watch.status,
        trigger=watch.trigger,
        query=query if isinstance(query, dict) else {},
        next_check_at=watch.next_check_at,
        cooldown_seconds=watch.cooldown_seconds,
        last_fired_at=watch.last_fired_at,
    )
