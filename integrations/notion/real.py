"""Notion provider.

Two boundaries define this integration.

The first is Notion's own: an internal integration can only see pages that have
been explicitly shared with it. The second is ours: ``NOTION_ALLOWED_PAGE_IDS``
narrows access further, to the specific pages and databases the owner nominated.
Both are enforced before a request is made, so a page id that arrives from a
conversation - which is to say, from a transcription of speech - cannot reach
anything that was not already intended.

Page ids are normalised to bare hex before comparison, because Notion prints
them with hyphens in some places and without in others, and an allowlist that
matches only one spelling is an allowlist with a hole in it.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.logging_config import get_logger
from integrations.notion.models import (
    NotionBlockText,
    NotionDatabaseRow,
    NotionPageContent,
    NotionPageSummary,
)
from security.redaction import register_secret
from shared.errors import (
    IntegrationError,
    PermissionDeniedError,
    ProviderRateLimitError,
    ValidationError,
)
from shared.timeutils import utcnow

logger = get_logger(__name__)

API_ROOT = "https://api.notion.com/v1"
# Pinned deliberately: Notion breaks response shapes between versions and the
# header is the only thing keeping this parser valid.
NOTION_VERSION = "2022-06-28"

MAX_ATTEMPTS = 3
MAX_PAGES = 10
MAX_BLOCKS = 300
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

_HEX_ID = re.compile(r"^[0-9a-f]{32}$")


def normalise_id(value: str) -> str:
    """Reduce any Notion id or URL to bare lowercase hex."""
    candidate = value.strip().lower()
    if "notion.so" in candidate or candidate.startswith("http"):
        candidate = candidate.rstrip("/").rsplit("/", 1)[-1]
        candidate = candidate.rsplit("-", 1)[-1]
    candidate = candidate.replace("-", "")
    if not _HEX_ID.match(candidate):
        raise ValidationError(f"{value!r} is not a Notion page id.")
    return candidate


class NotionClient:
    """Authenticated access to the Notion API, scoped by the allowlist."""

    name = "notion"

    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None):
        self._settings = settings or get_settings()
        self._token = self._settings.notion_token.get_secret_value()
        if self._token:
            register_secret(self._token)
        self._allowed = {
            normalise_id(entry) for entry in self._settings.notion_allowed_page_ids if entry.strip()
        }
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
        self._cache: dict[str, tuple[float, Any]] = {}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> NotionClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    @property
    def allowed_ids(self) -> set[str]:
        return set(self._allowed)

    def require_allowed(self, page_id: str) -> str:
        """Check an id against the allowlist before it is used in a URL."""
        normalised = normalise_id(page_id)
        if not self._allowed:
            raise PermissionDeniedError(
                "No Notion pages are allowlisted. Add page ids to NOTION_ALLOWED_PAGE_IDS "
                "before the assistant can read from Notion."
            )
        if normalised not in self._allowed:
            logger.warning("Blocked a Notion request outside the allowlist")
            raise PermissionDeniedError(
                "That Notion page is not in the allowlist configured on this machine."
            )
        return normalised

    # ------------------------------------------------------------- transport
    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._token:
            raise IntegrationError("NOTION_TOKEN is not configured.", integration="notion")

        headers = {
            "Authorization": f"Bearer {self._token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = await self._client.request(
                    method, f"{API_ROOT}{path}", json=json_body, params=params, headers=headers
                )
            except httpx.HTTPError as exc:
                if attempt == MAX_ATTEMPTS:
                    raise IntegrationError(
                        f"Could not reach Notion ({type(exc).__name__}).", integration="notion"
                    ) from exc
                await asyncio.sleep(min(2**attempt, 8))
                continue

            if response.status_code in RETRYABLE_STATUS and attempt < MAX_ATTEMPTS:
                await asyncio.sleep(_retry_after(response) or min(2**attempt, 8))
                continue

            return self._finish(response)

        raise IntegrationError("Notion request failed after retries.", integration="notion")

    def _finish(self, response: httpx.Response) -> dict[str, Any]:
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            raise ProviderRateLimitError(
                "Notion is rate limiting this integration; try again shortly.",
                details={"integration": "notion"},
            )
        if response.status_code in {httpx.codes.NOT_FOUND, httpx.codes.FORBIDDEN}:
            # Notion returns 404 for a page that exists but was never shared, so
            # the two cases are genuinely indistinguishable from here.
            raise PermissionDeniedError(
                "Notion returned no access to that page. Share it with the integration "
                "in Notion, then add its id to NOTION_ALLOWED_PAGE_IDS."
            )
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise IntegrationError(
                f"Notion rejected the request: HTTP {response.status_code}.",
                integration="notion",
            )
        body: dict[str, Any] = response.json()
        return body

    # ----------------------------------------------------------------- cache
    def _cached(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() > expires_at:
            del self._cache[key]
            return None
        return value

    def _store(self, key: str, value: Any) -> None:
        ttl = self._settings.notion_cache_ttl_seconds
        if ttl > 0:
            self._cache[key] = (time.monotonic() + ttl, value)

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------ read
    async def get_page(self, page_id: str) -> NotionPageSummary:
        normalised = self.require_allowed(page_id)
        cache_key = f"page:{normalised}"
        cached = self._cached(cache_key)
        if cached is not None:
            return NotionPageSummary.model_validate(cached)

        payload = await self.request("GET", f"/pages/{normalised}")
        summary = _to_summary(payload)
        self._store(cache_key, summary.model_dump(mode="json"))
        return summary

    async def get_page_content(self, page_id: str, *, max_chars: int = 8000) -> NotionPageContent:
        """Flatten a page's blocks into readable text."""
        normalised = self.require_allowed(page_id)
        summary = await self.get_page(normalised)

        blocks: list[NotionBlockText] = []
        cursor: str | None = None
        for _ in range(MAX_PAGES):
            params: dict[str, Any] = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            payload = await self.request("GET", f"/blocks/{normalised}/children", params=params)
            for block in payload.get("results") or []:
                text = _block_text(block)
                if text:
                    blocks.append(
                        NotionBlockText(id=str(block.get("id", "")), kind=block["type"], text=text)
                    )
                if len(blocks) >= MAX_BLOCKS:
                    break
            cursor = payload.get("next_cursor") if payload.get("has_more") else None
            if not cursor or len(blocks) >= MAX_BLOCKS:
                break

        joined = "\n".join(block.text for block in blocks)
        return NotionPageContent(
            page=summary,
            blocks=blocks,
            text=joined[:max_chars],
            truncated=len(joined) > max_chars or len(blocks) >= MAX_BLOCKS,
        )

    async def query_database(
        self,
        database_id: str,
        *,
        limit: int = 25,
        filter_payload: dict[str, Any] | None = None,
    ) -> list[NotionDatabaseRow]:
        normalised = self.require_allowed(database_id)
        limit = max(1, min(limit, 100))

        body: dict[str, Any] = {"page_size": limit}
        if filter_payload:
            body["filter"] = filter_payload

        payload = await self.request("POST", f"/databases/{normalised}/query", json_body=body)
        return [_to_row(entry) for entry in (payload.get("results") or [])][:limit]

    async def search(self, query: str, *, limit: int = 10) -> list[NotionPageSummary]:
        """Search, then discard anything outside the allowlist.

        Notion's search covers everything shared with the integration, which may
        be wider than what this machine is configured to read.
        """
        payload = await self.request(
            "POST",
            "/search",
            json_body={"query": query[:100], "page_size": min(limit * 2, 50)},
        )
        results: list[NotionPageSummary] = []
        for entry in payload.get("results") or []:
            try:
                normalised = normalise_id(str(entry.get("id", "")))
            except ValidationError:
                continue
            if normalised in self._allowed:
                results.append(_to_summary(entry))
        return results[:limit]

    # ----------------------------------------------------------------- write
    async def append_block(self, page_id: str, text: str) -> dict[str, Any]:
        """Append a paragraph. Only ever reached through an approved action."""
        normalised = self.require_allowed(page_id)
        if not text.strip():
            raise ValidationError("There is nothing to append.")

        result = await self.request(
            "PATCH",
            f"/blocks/{normalised}/children",
            json_body={
                "children": [
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": text[:1900]}}]
                        },
                    }
                ]
            },
        )
        self.clear_cache()
        logger.info("Appended a block to a Notion page")
        return {
            "page_id": normalised,
            "block_id": (result.get("results") or [{}])[0].get("id"),
            "appended_at": utcnow().isoformat(),
        }

    async def create_database_row(
        self, database_id: str, properties: dict[str, Any]
    ) -> NotionDatabaseRow:
        normalised = self.require_allowed(database_id)
        payload = await self.request(
            "POST",
            "/pages",
            json_body={"parent": {"database_id": normalised}, "properties": properties},
        )
        self.clear_cache()
        return _to_row(payload)

    async def health(self) -> dict[str, Any]:
        if not self._token:
            return {"ok": False, "detail": "token not configured"}
        if not self._allowed:
            return {"ok": False, "detail": "no allowlisted pages"}
        try:
            me = await self.request("GET", "/users/me")
        except (IntegrationError, PermissionDeniedError) as exc:
            return {"ok": False, "detail": str(exc)}
        return {
            "ok": True,
            "integration_name": (me.get("bot") or {}).get("workspace_name") or me.get("name"),
            "allowlisted_pages": len(self._allowed),
        }


def _retry_after(response: httpx.Response) -> float | None:
    header = response.headers.get("retry-after")
    return float(header) if header and header.replace(".", "", 1).isdigit() else None


def _rich_text(items: list[dict[str, Any]] | None) -> str:
    return "".join(str(item.get("plain_text", "")) for item in items or [])


def _to_summary(payload: dict[str, Any]) -> NotionPageSummary:
    return NotionPageSummary(
        id=normalise_id(str(payload.get("id", "0" * 32))),
        title=_page_title(payload),
        url=payload.get("url"),
        last_edited_at=payload.get("last_edited_time"),
        parent_kind=str((payload.get("parent") or {}).get("type") or "unknown"),
        archived=bool(payload.get("archived")),
    )


def _page_title(payload: dict[str, Any]) -> str:
    properties = payload.get("properties") or {}
    for value in properties.values():
        if value.get("type") == "title":
            title = _rich_text(value.get("title"))
            if title:
                return title[:300]
    return "(untitled)"


def _to_row(payload: dict[str, Any]) -> NotionDatabaseRow:
    """Flatten Notion's property shapes into plain values.

    Only the property types the assistant can speak aloud are handled; anything
    else is skipped rather than guessed at.
    """
    values: dict[str, Any] = {}
    for name, prop in (payload.get("properties") or {}).items():
        kind = prop.get("type")
        if kind == "title":
            values[name] = _rich_text(prop.get("title"))
        elif kind == "rich_text":
            values[name] = _rich_text(prop.get("rich_text"))
        elif kind in {"number", "checkbox", "url", "email", "phone_number"}:
            values[name] = prop.get(kind)
        elif kind == "select":
            values[name] = (prop.get("select") or {}).get("name")
        elif kind == "multi_select":
            values[name] = [item.get("name") for item in prop.get("multi_select") or []]
        elif kind == "date":
            values[name] = (prop.get("date") or {}).get("start")
        elif kind == "status":
            values[name] = (prop.get("status") or {}).get("name")
        elif kind == "people":
            values[name] = [item.get("name") for item in prop.get("people") or []]

    return NotionDatabaseRow(
        id=normalise_id(str(payload.get("id", "0" * 32))),
        title=_page_title(payload),
        url=payload.get("url"),
        properties=values,
        last_edited_at=payload.get("last_edited_time"),
    )


def _block_text(block: dict[str, Any]) -> str:
    kind = str(block.get("type", ""))
    content = block.get(kind) or {}
    text = _rich_text(content.get("rich_text"))
    if not text:
        return ""
    if kind == "to_do":
        marker = "[x]" if content.get("checked") else "[ ]"
        return f"{marker} {text}"
    if kind in {"bulleted_list_item", "numbered_list_item"}:
        return f"- {text}"
    if kind.startswith("heading_"):
        return f"{'#' * int(kind[-1])} {text}"
    return text
