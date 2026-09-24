"""Contact endpoints.

Two sources are merged: contacts stored locally in this database and, when
authorised, Google contacts. Local entries win on a name collision, because a
Uses the configured workflow.
"""

from __future__ import annotations

import contextlib
import json
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.dependencies.auth import Principal, require_token
from app.dependencies.common import get_session
from app.schemas.google import ContactListResponse
from database.models import Contact, ContactAlias
from integrations.google.models import ContactRecord
from integrations.registry import get_contacts

router = APIRouter(prefix="/api/v1/contacts", tags=["contacts"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
PrincipalDep = Annotated[Principal, Depends(require_token)]


@router.get(
    "/search",
    response_model=ContactListResponse,
    summary="Find a contact by name or alias",
    description=(
        "Searches local contacts and their spoken aliases first, then Google. "
        "Aliases exist so 'my supervisor' resolves without the model guessing."
    ),
)
async def search_contacts(
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    q: Annotated[str, Query(min_length=1, max_length=100)],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    include_google: Annotated[bool, Query()] = True,
) -> ContactListResponse:
    local = await _search_local(session, q, limit)
    results = list(local)

    if include_google and len(results) < limit:
        seen = {name.display_name.lower() for name in results}
        for record in await get_contacts(settings).search(q, limit=limit - len(results)):
            if record.display_name.lower() not in seen:
                results.append(record)

    return ContactListResponse(items=results[:limit], total=len(results[:limit]))


@router.get("", response_model=ContactListResponse, summary="List known contacts")
async def list_contacts(
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    source: Annotated[str, Query(pattern="^(all|local|google)$")] = "all",
) -> ContactListResponse:
    results: list[ContactRecord] = []
    if source in {"all", "local"}:
        rows = (
            await session.scalars(select(Contact).order_by(Contact.display_name).limit(limit))
        ).all()
        results.extend(_to_record(row) for row in rows)
    if source in {"all", "google"} and len(results) < limit:
        results.extend(await get_contacts(settings).list_contacts(limit=limit - len(results)))
    return ContactListResponse(items=results[:limit], total=len(results[:limit]))


async def _search_local(session: AsyncSession, query: str, limit: int) -> list[ContactRecord]:
    needle = query.strip().lower()
    pattern = f"%{needle}%"
    alias_matches = select(ContactAlias.contact_id).where(
        ContactAlias.normalised_alias.like(pattern)
    )
    rows = (
        await session.scalars(
            select(Contact)
            .where(
                or_(
                    Contact.normalised_name.like(pattern),
                    Contact.primary_email.ilike(pattern),
                    Contact.id.in_(alias_matches),
                )
            )
            .order_by(Contact.is_vip.desc(), Contact.display_name)
            .limit(limit)
        )
    ).all()
    return [_to_record(row) for row in rows]


def _to_record(row: Contact) -> ContactRecord:
    emails = [row.primary_email] if row.primary_email else []
    if row.secondary_emails:
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            emails.extend(str(entry) for entry in json.loads(row.secondary_emails))
    return ContactRecord(
        id=row.id,
        display_name=row.display_name,
        emails=emails,
        phones=[row.phone] if row.phone else [],
        organisation=row.organisation,
    )
