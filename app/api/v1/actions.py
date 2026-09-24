"""Action status lookup for Reachy polling."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.auth import Principal, require_token
from app.dependencies.common import get_session
from app.schemas.action_status import ActionStatusOut
from app.services import action_status as service

router = APIRouter(prefix="/api/v1/actions", tags=["actions"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
PrincipalDep = Annotated[Principal, Depends(require_token)]


@router.get(
    "/{action_id}/status",
    response_model=ActionStatusOut,
    summary="Get safe status for one pending action",
)
async def get_action_status(
    action_id: str, _: PrincipalDep, session: SessionDep
) -> ActionStatusOut:
    return await service.get_status_by_action_id(session, action_id)
