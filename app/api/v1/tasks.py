"""Task endpoints. All synchronous local operations."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.dependencies.auth import Principal, require_token
from app.dependencies.common import Pagination, get_session, idempotency_key, pagination
from app.schemas.common import DeleteResponse
from app.schemas.tasks import TaskCreate, TaskList, TaskOut, TaskUpdate
from app.services import idempotency as idem
from app.services import tasks as service
from shared.enums import TaskPriority, TaskStatus

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
PrincipalDep = Annotated[Principal, Depends(require_token)]
PaginationDep = Annotated[Pagination, Depends(pagination)]
IdempotencyDep = Annotated[str | None, Depends(idempotency_key)]

IDEMPOTENCY_SCOPE = "tasks.create"


@router.post("", response_model=TaskOut, status_code=201, summary="Create a task")
async def create_task(
    payload: TaskCreate,
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    key: IdempotencyDep,
) -> TaskOut:
    replay = await idem.lookup(session, IDEMPOTENCY_SCOPE, key, payload)
    if replay is not None:
        return TaskOut.model_validate(replay)

    task = await service.create_task(session, payload, settings=settings, idempotency_key=key)
    result = await service.to_out(session, task, settings.app_timezone)
    await idem.remember(
        session,
        IDEMPOTENCY_SCOPE,
        key,
        payload,
        result.model_dump(mode="json"),
        resource_id=task.id,
    )
    return result


@router.get("", response_model=TaskList, summary="List tasks")
async def list_tasks(
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    page: PaginationDep,
    status: Annotated[TaskStatus | None, Query()] = None,
    priority: Annotated[TaskPriority | None, Query()] = None,
    tag: Annotated[str | None, Query(max_length=40)] = None,
    due_before: Annotated[datetime | None, Query()] = None,
    overdue_only: Annotated[bool, Query()] = False,
) -> TaskList:
    rows, total = await service.list_tasks(
        session,
        status=status,
        priority=priority,
        tag=tag,
        due_before=due_before,
        overdue_only=overdue_only,
        limit=page.limit,
        offset=page.offset,
    )
    return TaskList(
        items=[await service.to_out(session, row, settings.app_timezone) for row in rows],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/{task_id}", response_model=TaskOut, summary="Get one task")
async def get_task(
    task_id: str, _: PrincipalDep, session: SessionDep, settings: SettingsDep
) -> TaskOut:
    task = await service.get_task(session, task_id)
    return await service.to_out(session, task, settings.app_timezone)


@router.patch("/{task_id}", response_model=TaskOut, summary="Edit a task")
async def update_task(
    task_id: str,
    payload: TaskUpdate,
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
) -> TaskOut:
    task = await service.update_task(session, task_id, payload)
    return await service.to_out(session, task, settings.app_timezone)


@router.post(
    "/{task_id}/complete",
    response_model=TaskOut,
    summary="Complete a task",
    description=(
        "A recurring task rolls forward to its next occurrence and stays open, so its "
        "identity in conversation and its history remain stable."
    ),
)
async def complete_task(
    task_id: str, _: PrincipalDep, session: SessionDep, settings: SettingsDep
) -> TaskOut:
    task = await service.complete_task(session, task_id)
    return await service.to_out(session, task, settings.app_timezone)


@router.post("/{task_id}/cancel", response_model=TaskOut, summary="Cancel a task")
async def cancel_task(
    task_id: str, _: PrincipalDep, session: SessionDep, settings: SettingsDep
) -> TaskOut:
    task = await service.cancel_task(session, task_id)
    return await service.to_out(session, task, settings.app_timezone)


@router.delete("/{task_id}", response_model=DeleteResponse, summary="Delete a task")
async def delete_task(task_id: str, _: PrincipalDep, session: SessionDep) -> DeleteResponse:
    await service.delete_task(session, task_id)
    return DeleteResponse(id=task_id)
