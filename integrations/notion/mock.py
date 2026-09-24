"""In-memory Notion stand-in.

Enforces the same allowlist rules as the real client, so a test that proves a
page outside the allowlist is refused proves it about the real path too.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from integrations.notion.models import (
    NotionBlockText,
    NotionDatabaseRow,
    NotionPageContent,
    NotionPageSummary,
)
from integrations.notion.real import normalise_id
from shared.errors import PermissionDeniedError, ValidationError
from shared.timeutils import utcnow

RESEARCH_PAGE = "a" * 32
TASKS_DATABASE = "b" * 32
PRIVATE_PAGE = "c" * 32


class MockNotionClient:
    name = "notion"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        configured = {
            normalise_id(entry) for entry in self._settings.notion_allowed_page_ids if entry.strip()
        }
        # Without configuration the mock still has a usable workspace, otherwise
        # every mock-mode demo would fail the allowlist check.
        self._allowed = configured or {RESEARCH_PAGE, TASKS_DATABASE}
        self.appended: list[dict[str, Any]] = []
        self.created_rows: list[NotionDatabaseRow] = []

        self._pages = {
            RESEARCH_PAGE: NotionPageSummary(
                id=RESEARCH_PAGE,
                title="Manipulation research log",
                url="https://notion.so/manipulation-research-log",
                last_edited_at="2026-03-09T18:20:00.000Z",
                parent_kind="workspace",
            ),
            PRIVATE_PAGE: NotionPageSummary(
                id=PRIVATE_PAGE,
                title="Personal finances",
                url="https://notion.so/personal-finances",
                parent_kind="workspace",
            ),
        }
        self._blocks = {
            RESEARCH_PAGE: [
                NotionBlockText(id="blk-1", kind="heading_2", text="## Week 11"),
                NotionBlockText(
                    id="blk-2",
                    kind="paragraph",
                    text="Tactile ablation reduces grasp success by eleven points.",
                ),
                NotionBlockText(
                    id="blk-3", kind="to_do", text="[ ] Add confidence intervals to table 3"
                ),
            ]
        }
        self._rows = [
            NotionDatabaseRow(
                id="d" * 32,
                title="Rerun ablation with tactile disabled",
                properties={"Status": "In progress", "Priority": "High"},
                last_edited_at="2026-03-09T12:00:00.000Z",
            ),
            NotionDatabaseRow(
                id="e" * 32,
                title="Draft related work section",
                properties={"Status": "Not started", "Priority": "Medium"},
            ),
        ]

    @property
    def allowed_ids(self) -> set[str]:
        return set(self._allowed)

    def require_allowed(self, page_id: str) -> str:
        normalised = normalise_id(page_id)
        if normalised not in self._allowed:
            raise PermissionDeniedError(
                "That Notion page is not in the allowlist configured on this machine."
            )
        return normalised

    def clear_cache(self) -> None:
        return None

    async def get_page(self, page_id: str) -> NotionPageSummary:
        normalised = self.require_allowed(page_id)
        page = self._pages.get(normalised)
        if page is None:
            raise PermissionDeniedError("Notion returned no access to that page.")
        return page

    async def get_page_content(self, page_id: str, *, max_chars: int = 8000) -> NotionPageContent:
        page = await self.get_page(page_id)
        blocks = self._blocks.get(page.id, [])
        text = "\n".join(block.text for block in blocks)
        return NotionPageContent(
            page=page,
            blocks=blocks,
            text=text[:max_chars],
            truncated=len(text) > max_chars,
        )

    async def query_database(
        self,
        database_id: str,
        *,
        limit: int = 25,
        filter_payload: dict[str, Any] | None = None,
    ) -> list[NotionDatabaseRow]:
        self.require_allowed(database_id)
        return (self._rows + self.created_rows)[:limit]

    async def search(self, query: str, *, limit: int = 10) -> list[NotionPageSummary]:
        needle = query.strip().lower()
        return [
            page
            for page in self._pages.values()
            if page.id in self._allowed and needle in page.title.lower()
        ][:limit]

    async def append_block(self, page_id: str, text: str) -> dict[str, Any]:
        normalised = self.require_allowed(page_id)
        if not text.strip():
            raise ValidationError("There is nothing to append.")
        record = {
            "page_id": normalised,
            "block_id": f"mock-block-{len(self.appended) + 1}",
            "text": text,
            "appended_at": utcnow().isoformat(),
        }
        self.appended.append(record)
        self._blocks.setdefault(normalised, []).append(
            NotionBlockText(id=str(record["block_id"]), kind="paragraph", text=text)
        )
        return record

    async def create_database_row(
        self, database_id: str, properties: dict[str, Any]
    ) -> NotionDatabaseRow:
        self.require_allowed(database_id)
        title = ""
        for value in properties.values():
            if value.get("type") == "title" or "title" in value:
                entries = value.get("title") or []
                title = "".join(entry.get("text", {}).get("content", "") for entry in entries)
                break
        row = NotionDatabaseRow(
            id=f"{len(self.created_rows) + 1:032x}",
            title=title or "(untitled)",
            properties={key: value for key, value in properties.items() if key != "title"},
        )
        self.created_rows.append(row)
        return row

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "mock": True, "allowlisted_pages": len(self._allowed)}
