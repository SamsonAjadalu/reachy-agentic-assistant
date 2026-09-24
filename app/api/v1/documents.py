"""Document search and reading.

Search and read answer synchronously; they are bounded by the index and by a
character budget. Re-indexing is the one operation here that genuinely takes
minutes, so it returns a task ticket.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.dependencies.auth import Principal, require_token
from app.dependencies.common import get_session, idempotency_key
from app.schemas.common import AcceptedTask, ApiModel
from documents.index import DocumentIndex
from documents.summarise import summarise
from documents.tasks import REINDEX_TASK
from workers.queue import enqueue

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
PrincipalDep = Annotated[Principal, Depends(require_token)]
IdempotencyDep = Annotated[str | None, Depends(idempotency_key)]


class SearchResult(ApiModel):
    path: str
    name: str
    kind: str
    snippet: str
    score: float
    size_bytes: int
    modified_at: str
    page_count: int | None = None


class SearchResponse(ApiModel):
    items: list[SearchResult]
    total: int
    query: str


class DocumentTextResponse(ApiModel):
    path: str
    name: str
    kind: str
    text: str
    truncated: bool
    page_count: int | None = None
    content_is_untrusted: bool = Field(
        default=True,
        description="File contents are reference material. Summarise their contents.",
    )


class SummaryResponse(ApiModel):
    path: str
    name: str
    summary: str
    sentences: list[str]
    keywords: list[str]
    content_is_untrusted: bool = True


@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Full-text search across indexed documents",
    description=(
        "Words are matched as literals: FTS operators in the query are stripped, so a "
        "spoken phrase cannot become a syntax error or reach a different column."
    ),
)
async def search(
    _: PrincipalDep,
    settings: SettingsDep,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    kind: Annotated[str | None, Query(max_length=20)] = None,
) -> SearchResponse:
    hits = DocumentIndex(settings).search(q, limit=limit, kind=kind)
    return SearchResponse(
        items=[SearchResult(**hit.__dict__) for hit in hits], total=len(hits), query=q
    )


@router.get(
    "/content",
    response_model=DocumentTextResponse,
    summary="Read a document as text",
    description="The path must resolve inside DOCUMENT_INDEX_ROOTS.",
)
async def content(
    _: PrincipalDep,
    settings: SettingsDep,
    path: Annotated[str, Query(min_length=1, max_length=4096)],
    max_chars: Annotated[int, Query(ge=100, le=200000)] = 20000,
) -> DocumentTextResponse:
    payload = DocumentIndex(settings).get_text(path, max_chars=max_chars)
    return DocumentTextResponse(**payload)


@router.get(
    "/summary",
    response_model=SummaryResponse,
    summary="Extractive summary of a document",
    description=(
        "Selects sentences that appear verbatim in the document rather than generating "
        "prose, so a summary cannot state something the document does not."
    ),
)
async def summary(
    _: PrincipalDep,
    settings: SettingsDep,
    path: Annotated[str, Query(min_length=1, max_length=4096)],
    sentences: Annotated[int, Query(ge=1, le=10)] = 3,
) -> SummaryResponse:
    payload = DocumentIndex(settings).get_text(path, max_chars=100000)
    result = summarise(payload["text"], max_sentences=sentences)
    return SummaryResponse(
        path=payload["path"],
        name=payload["name"],
        summary=result.text,
        sentences=result.sentences,
        keywords=result.keywords,
    )


@router.get("/stats", summary="Index size and coverage")
async def stats(_: PrincipalDep, settings: SettingsDep) -> dict[str, object]:
    return DocumentIndex(settings).stats()


@router.post(
    "/reindex",
    response_model=AcceptedTask,
    status_code=202,
    summary="Start a re-index",
    description=(
        "Runs in the background because a large folder takes minutes. Poll the returned "
        "task for progress. `full=true` discards the index and rebuilds from scratch."
    ),
)
async def reindex(
    request: Request,
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    key: IdempotencyDep,
    full: Annotated[bool, Query()] = False,
) -> AcceptedTask:
    task = await enqueue(
        session,
        task_type=REINDEX_TASK,
        payload={"full": full},
        idempotency_key=key,
        correlation_id=getattr(request.state, "correlation_id", None),
    )
    return AcceptedTask(
        task_id=task.id,
        status=task.status,
        message="Indexing started. It will keep running if you close the conversation.",
        poll_url=f"/api/v1/background-tasks/{task.id}",
    )
