"""Google Drive provider (read-only).

Only text is ever pulled out of Drive, and only up to a bounded size: the point
is to answer "what does that document say", not to mirror a drive onto the
workstation. Binary files are described rather than downloaded.

Search terms are escaped before they enter a Drive query. Drive's query language
is string-concatenated by design, so an unescaped apostrophe in a filename is
enough to change the meaning of the query.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.config import Settings, get_settings
from integrations.google.client import DRIVE_BASE, GoogleClient
from integrations.google.models import DriveFileContent, DriveFileRecord
from shared.errors import IntegrationError, ValidationError
from shared.timeutils import ensure_utc

DRIVE_READ_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
FOLDER_MIME = "application/vnd.google-apps.folder"
MAX_RESULTS = 50
MAX_EXPORT_CHARS = 100_000

FILE_FIELDS = "id,name,mimeType,size,modifiedTime,owners(displayName),webViewLink"

# Google Workspace files have no bytes to download; they must be exported.
EXPORT_TYPES = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}
PLAIN_TEXT_TYPES = {"text/plain", "text/markdown", "text/csv", "application/json"}


class DriveService:
    name = "drive"

    def __init__(self, client: GoogleClient | None = None, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._client = client or GoogleClient(self._settings)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search(
        self, query: str, *, limit: int = 10, folder_id: str | None = None
    ) -> list[DriveFileRecord]:
        self._client.tokens.require_scope(DRIVE_READ_SCOPE)
        limit = max(1, min(limit, MAX_RESULTS))

        clauses = ["trashed = false"]
        if query.strip():
            clauses.append(f"name contains '{escape_query_term(query)}'")
        if folder_id:
            clauses.append(f"'{escape_query_term(folder_id)}' in parents")

        files: list[DriveFileRecord] = []
        async for item in self._client.paginate(
            f"{DRIVE_BASE}/files",
            params={
                "q": " and ".join(clauses),
                "fields": f"nextPageToken,files({FILE_FIELDS})",
                "orderBy": "modifiedTime desc",
            },
            items_key="files",
            limit=limit,
        ):
            files.append(_to_file(item))
        return files

    async def get_metadata(self, file_id: str) -> DriveFileRecord:
        self._client.tokens.require_scope(DRIVE_READ_SCOPE)
        payload = await self._client.request(
            "GET", f"{DRIVE_BASE}/files/{file_id}", params={"fields": FILE_FIELDS}
        )
        return _to_file(payload)

    async def export_text(self, file_id: str, *, max_chars: int = 20000) -> DriveFileContent:
        self._client.tokens.require_scope(DRIVE_READ_SCOPE)
        max_chars = max(100, min(max_chars, MAX_EXPORT_CHARS))
        metadata = await self.get_metadata(file_id)

        if metadata.is_folder:
            raise ValidationError("A folder has no text to read.")

        export_type = EXPORT_TYPES.get(metadata.mime_type)
        if export_type:
            raw = await self._client.request(
                "GET",
                f"{DRIVE_BASE}/files/{file_id}/export",
                params={"mimeType": export_type},
                expect_json=False,
            )
            text = raw if isinstance(raw, str) else await self._download(file_id)
        elif metadata.mime_type in PLAIN_TEXT_TYPES:
            text = await self._download(file_id)
        else:
            raise ValidationError(
                f"{metadata.name} is a {metadata.mime_type} file, which has no text form. "
                "Ask for its details instead."
            )

        truncated = len(text) > max_chars
        return DriveFileContent(
            id=metadata.id,
            name=metadata.name,
            mime_type=metadata.mime_type,
            text=text[:max_chars],
            truncated=truncated,
        )

    async def _download(self, file_id: str) -> str:
        raw = await self._client.request(
            "GET",
            f"{DRIVE_BASE}/files/{file_id}",
            params={"alt": "media"},
            expect_json=False,
        )
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw or "")

    async def health(self) -> dict[str, Any]:
        if not self._client.tokens.is_authorised():
            return {"ok": False, "detail": "not authorised"}
        try:
            body = await self._client.request(
                "GET", f"{DRIVE_BASE}/about", params={"fields": "user(emailAddress),storageQuota"}
            )
        except IntegrationError as exc:
            return {"ok": False, "detail": str(exc)}
        return {"ok": True, "account": (body.get("user") or {}).get("emailAddress")}


def escape_query_term(value: str) -> str:
    """Escape a value for interpolation into a Drive query string."""
    return value.replace("\\", "\\\\").replace("'", "\\'")[:200]


def _to_file(payload: dict[str, Any]) -> DriveFileRecord:
    owners = payload.get("owners") or []
    size = payload.get("size")
    modified = payload.get("modifiedTime")
    mime_type = str(payload.get("mimeType", ""))

    return DriveFileRecord(
        id=str(payload.get("id", "")),
        name=str(payload.get("name", "")),
        mime_type=mime_type,
        size_bytes=int(size) if size is not None else None,
        modified_at=_parse(modified) if modified else None,
        owner=(owners[0].get("displayName") if owners else None),
        web_view_link=payload.get("webViewLink"),
        is_folder=mime_type == FOLDER_MIME,
    )


def _parse(value: str) -> datetime | None:
    try:
        return ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None
