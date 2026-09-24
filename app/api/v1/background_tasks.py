"""Background task tickets.

The endpoints a Reachy tool polls after an operation returned a ticket instead of
a result. Only genuinely long-running work produces one; fast reads answer inline.
"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.auth import Principal, require_token
from app.dependencies.common import Pagination, get_session, pagination
from app.schemas.tasks import BackgroundTaskEventOut, BackgroundTaskList, BackgroundTaskOut
from database.models import BackgroundTask, BackgroundTaskEvent
from shared.enums import BackgroundTaskStatus
from shared.errors import NotFoundError
from workers import queue as task_queue
from workers.registry import registered_tasks

router = APIRouter(prefix="/api/v1/background-tasks", tags=["background tasks"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
PrincipalDep = Annotated[Principal, Depends(require_token)]
PaginationDep = Annotated[Pagination, Depends(pagination)]

MAX_EVENTS_RETURNED = 50


def _to_out(task: BackgroundTask, events: list[BackgroundTaskEvent]) -> BackgroundTaskOut:
    return BackgroundTaskOut(
        id=task.id,
        task_type=task.task_type,
        status=BackgroundTaskStatus(task.status),
        progress=task.progress,
        progress_message=task.progress_message,
        created_at=task.created_at,
        available_at=task.available_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
        attempts=task.attempts,
        max_attempts=task.max_attempts,
        cancel_requested=task.cancel_requested,
        result=json.loads(task.result_json) if task.result_json else None,
        result_path=task.result_path,
        error_code=task.error_code,
        error_message=task.error_message,
        events=[
            BackgroundTaskEventOut(
                occurred_at=event.occurred_at,
                event_type=event.event_type,
                message=event.message,
            )
            for event in events
        ],
    )


@router.get("", response_model=BackgroundTaskList, summary="List background tasks")
async def list_background_tasks(
    _: PrincipalDep,
    session: SessionDep,
    page: PaginationDep,
    status: Annotated[BackgroundTaskStatus | None, Query()] = None,
    task_type: Annotated[str | None, Query(max_length=60)] = None,
) -> BackgroundTaskList:
    conditions = []
    if status is not None:
        conditions.append(BackgroundTask.status == status.value)
    if task_type is not None:
        conditions.append(BackgroundTask.task_type == task_type)

    total = (
        await session.scalar(select(func.count()).select_from(BackgroundTask).where(*conditions))
        or 0
    )
    rows = (
        await session.scalars(
            select(BackgroundTask)
            .where(*conditions)
            .order_by(BackgroundTask.created_at.desc())
            .limit(page.limit)
            .offset(page.offset)
        )
    ).all()
    return BackgroundTaskList(
        items=[_to_out(row, []) for row in rows],
        total=int(total),
        limit=page.limit,
        offset=page.offset,
    )


@router.get(
    "/types",
    summary="Registered task types",
    description="What kinds of deferred work this build knows how to run.",
)
async def list_task_types(_: PrincipalDep) -> list[dict[str, object]]:
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "default_timeout_seconds": spec.default_timeout_seconds,
            "max_attempts": spec.max_attempts,
            "retryable": spec.retryable,
        }
        for spec in registered_tasks()
    ]


@router.get(
    "/{task_id}",
    response_model=BackgroundTaskOut,
    summary="Poll a background task",
    description="Returns progress while running and the result once it finishes.",
)
async def get_background_task(
    task_id: str, _: PrincipalDep, session: SessionDep
) -> BackgroundTaskOut:
    task = await session.get(BackgroundTask, task_id)
    if task is None:
        raise NotFoundError(f"No background task with id {task_id}.")
    events = (
        await session.scalars(
            select(BackgroundTaskEvent)
            .where(BackgroundTaskEvent.task_id == task_id)
            .order_by(BackgroundTaskEvent.occurred_at.asc())
            .limit(MAX_EVENTS_RETURNED)
        )
    ).all()
    return _to_out(task, list(events))


@router.post(
    "/{task_id}/cancel",
    response_model=BackgroundTaskOut,
    summary="Request cancellation",
    description=(
        "A queued task is cancelled immediately. A running task is flagged, and its handler "
        "stops at the next progress checkpoint - work already committed externally is not "
        "undone."
    ),
)
async def cancel_background_task(
    task_id: str, _: PrincipalDep, session: SessionDep
) -> BackgroundTaskOut:
    task = await task_queue.request_cancel(session, task_id)
    return _to_out(task, [])
