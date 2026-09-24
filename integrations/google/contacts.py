"""Google People API provider.

Contacts are read-only here. The assistant uses them to turn "email Nathan" into
an address, and that resolution is shown in the approval preview so an ambiguous
or wrong match is caught by the owner rather than discovered afterwards.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from integrations.google.client import PEOPLE_BASE, GoogleClient
from integrations.google.models import ContactRecord
from shared.errors import IntegrationError

CONTACTS_READ_SCOPE = "https://www.googleapis.com/auth/contacts.readonly"
PERSON_FIELDS = "names,emailAddresses,phoneNumbers,organizations,photos"
MAX_RESULTS = 100


class ContactsService:
    name = "contacts"

    def __init__(self, client: GoogleClient | None = None, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._client = client or GoogleClient(self._settings)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search(self, query: str, *, limit: int = 10) -> list[ContactRecord]:
        self._client.tokens.require_scope(CONTACTS_READ_SCOPE)
        cleaned = query.strip()
        if not cleaned:
            return []

        body = await self._client.request(
            "GET",
            f"{PEOPLE_BASE}/people:searchContacts",
            params={
                "query": cleaned[:100],
                "pageSize": max(1, min(limit, 30)),
                "readMask": PERSON_FIELDS,
            },
        )
        results = [
            _to_contact(entry["person"])
            for entry in body.get("results") or []
            if entry.get("person")
        ]
        return results[:limit]

    async def list_contacts(self, *, limit: int = 50) -> list[ContactRecord]:
        self._client.tokens.require_scope(CONTACTS_READ_SCOPE)
        limit = max(1, min(limit, MAX_RESULTS))
        contacts: list[ContactRecord] = []
        async for person in self._client.paginate(
            f"{PEOPLE_BASE}/people/me/connections",
            params={"personFields": PERSON_FIELDS, "sortOrder": "LAST_MODIFIED_DESCENDING"},
            items_key="connections",
            limit=limit,
        ):
            contacts.append(_to_contact(person))
        return contacts

    async def health(self) -> dict[str, Any]:
        if not self._client.tokens.is_authorised():
            return {"ok": False, "detail": "not authorised"}
        try:
            body = await self._client.request(
                "GET",
                f"{PEOPLE_BASE}/people/me/connections",
                params={"personFields": "names", "pageSize": 1},
            )
        except IntegrationError as exc:
            return {"ok": False, "detail": str(exc)}
        return {"ok": True, "total_contacts": body.get("totalPeople")}


def _to_contact(person: dict[str, Any]) -> ContactRecord:
    names = person.get("names") or [{}]
    primary = names[0]
    organisations = person.get("organizations") or []
    photos = [photo for photo in person.get("photos") or [] if not photo.get("default")]

    return ContactRecord(
        id=str(person.get("resourceName", "")),
        display_name=str(primary.get("displayName") or "Unknown"),
        given_name=primary.get("givenName"),
        family_name=primary.get("familyName"),
        emails=[
            str(item["value"]) for item in person.get("emailAddresses") or [] if item.get("value")
        ],
        phones=[
            str(item["value"]) for item in person.get("phoneNumbers") or [] if item.get("value")
        ],
        organisation=(organisations[0].get("name") if organisations else None),
        photo_url=(photos[0].get("url") if photos else None),
    )
