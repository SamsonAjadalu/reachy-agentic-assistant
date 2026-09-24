"""Shared FastAPI dependencies."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import Depends, Header, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from database.session import get_sessionmaker

MAX_PAGE_SIZE = 200


async def get_session() -> AsyncIterator[AsyncSession]:
    """Request-scoped session.

    Commits on a clean return so endpoints do not each repeat the boilerplate,
    and rolls back on any exception so a partially applied write cannot escape.
    """
    factory = get_sessionmaker()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except BaseException:
            await session.rollback()
            raise


@dataclass(frozen=True)
class Pagination:
    limit: int
    offset: int


def pagination(
    limit: int = Query(default=50, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
) -> Pagination:
    return Pagination(limit=limit, offset=offset)


def get_request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def get_correlation_id(request: Request) -> str | None:
    return getattr(request.state, "correlation_id", None)


def idempotency_key(
    key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
        description=(
            "Client-generated key. Replaying the same key with the same body returns the "
            "original result instead of performing the action twice."
        ),
        max_length=200,
    ),
) -> str | None:
    return key


SessionDep = Depends(get_session)
SettingsDep = Depends(get_settings)
PaginationDep = Depends(pagination)
IdempotencyKeyDep = Depends(idempotency_key)

__all__ = [
    "MAX_PAGE_SIZE",
    "IdempotencyKeyDep",
    "Pagination",
    "PaginationDep",
    "SessionDep",
    "Settings",
    "SettingsDep",
    "get_correlation_id",
    "get_request_id",
    "get_session",
    "idempotency_key",
    "pagination",
]
