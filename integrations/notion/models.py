"""Normalised Notion shapes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NotionPageSummary(BaseModel):
    id: str
    title: str
    url: str | None = None
    last_edited_at: str | None = None
    parent_kind: str = "unknown"
    archived: bool = False


class NotionBlockText(BaseModel):
    id: str
    kind: str
    text: str


class NotionPageContent(BaseModel):
    page: NotionPageSummary
    blocks: list[NotionBlockText] = Field(default_factory=list)
    text: str = ""
    truncated: bool = False
    content_is_untrusted: bool = Field(
        default=True,
        description="Page text may have been written by anyone with edit access.",
    )


class NotionDatabaseRow(BaseModel):
    id: str
    title: str
    url: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    last_edited_at: str | None = None
