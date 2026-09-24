"""Google Drive endpoints (read-only).

There is no write path here at all. Read access is enough for every use the
assistant has, and a Drive write scope is not worth holding for a capability
nobody asked for.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.config import Settings, get_settings
from app.dependencies.auth import Principal, require_token
from app.schemas.google import DriveContentResponse, DriveListResponse
from integrations.registry import get_drive

router = APIRouter(prefix="/api/v1/drive", tags=["drive"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
PrincipalDep = Annotated[Principal, Depends(require_token)]


@router.get("/search", response_model=DriveListResponse, summary="Find files by name")
async def search_files(
    _: PrincipalDep,
    settings: SettingsDep,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    folder_id: Annotated[str | None, Query(max_length=100)] = None,
) -> DriveListResponse:
    items = await get_drive(settings).search(q, limit=limit, folder_id=folder_id)
    return DriveListResponse(items=items, total=len(items))


@router.get("/files/{file_id}", summary="File details")
async def get_file(file_id: str, _: PrincipalDep, settings: SettingsDep) -> dict[str, object]:
    record = await get_drive(settings).get_metadata(file_id)
    return {"file": record.model_dump(mode="json")}


@router.get(
    "/files/{file_id}/content",
    response_model=DriveContentResponse,
    summary="Read a file as text",
    description=(
        "Exports Google Docs, Sheets and Slides to text and reads plain-text files "
        "directly. Binary formats are refused rather than downloaded."
    ),
)
async def get_content(
    file_id: str,
    _: PrincipalDep,
    settings: SettingsDep,
    max_chars: Annotated[int, Query(ge=100, le=100000)] = 20000,
) -> DriveContentResponse:
    content = await get_drive(settings).export_text(file_id, max_chars=max_chars)
    return DriveContentResponse(file=content)
