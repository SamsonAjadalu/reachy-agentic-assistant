"""Idempotent write handling.

A Reachy tool call can be retried by the robot when a voice turn times out, and
a dropped response looks identical to a failure from the client's side. Storing
the request fingerprint against the key lets a genuine retry replay the original
result, while the same key sent with different content is rejected rather than
quietly performing a second, different action.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import IdempotencyKey
from shared.contracts import payload_hash
from shared.errors import IdempotencyConflictError
from shared.timeutils import utcnow

DEFAULT_TTL = timedelta(hours=24)


async def lookup(
    session: AsyncSession, scope: str, key: str | None, request_payload: Any
) -> dict[str, Any] | None:
    """Return the stored response for a replay, or ``None`` for a first attempt.

    Raises ``IdempotencyConflictError`` if the key was used with different content.
    """
    if not key:
        return None

    record = await session.scalar(
        select(IdempotencyKey).where(IdempotencyKey.scope == scope, IdempotencyKey.key == key)
    )
    if record is None:
        return None

    if record.expires_at <= utcnow():
        await session.delete(record)
        await session.flush()
        return None

    if record.request_hash != payload_hash(request_payload):
        raise IdempotencyConflictError(
            f"Idempotency-Key {key!r} was already used for a different request. "
            "Use a new key for a new action."
        )

    return json.loads(record.response_json) if record.response_json else {}


async def remember(
    session: AsyncSession,
    scope: str,
    key: str | None,
    request_payload: Any,
    response_payload: Any,
    *,
    resource_id: str | None = None,
    ttl: timedelta = DEFAULT_TTL,
) -> None:
    if not key:
        return
    session.add(
        IdempotencyKey(
            scope=scope,
            key=key,
            request_hash=payload_hash(request_payload),
            response_json=json.dumps(response_payload, default=str),
            resource_id=resource_id,
            expires_at=utcnow() + ttl,
        )
    )
    await session.flush()


async def purge_expired(session: AsyncSession) -> int:
    result = await session.execute(
        delete(IdempotencyKey).where(IdempotencyKey.expires_at <= utcnow())
    )
    return int(result.rowcount or 0)  # type: ignore[attr-defined]
