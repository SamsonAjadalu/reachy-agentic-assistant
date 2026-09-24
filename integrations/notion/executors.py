"""Approval executors for Notion writes."""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from approvals.state_machine import has_executor, register_executor
from integrations.registry import get_notion

APPEND_BLOCK = "notion.append_block"
CREATE_ROW = "notion.create_row"


async def append_block(
    session: AsyncSession, payload: dict[str, Any], settings: Settings
) -> dict[str, Any]:
    return await get_notion(settings).append_block(str(payload["page_id"]), str(payload["text"]))


async def create_row(
    session: AsyncSession, payload: dict[str, Any], settings: Settings
) -> dict[str, Any]:
    row = await get_notion(settings).create_database_row(
        str(payload["database_id"]), dict(payload["properties"])
    )
    return {"row_id": row.id, "title": row.title}


def register_notion_executors() -> None:
    handlers = {APPEND_BLOCK: append_block, CREATE_ROW: create_row}
    for action_type, handler in handlers.items():
        if not has_executor(action_type):
            register_executor(action_type)(handler)


def append_preview(page_title: str, text: str) -> str:
    body = text if len(text) <= 1500 else text[:1500] + "\n[truncated in preview]"
    return f"Append to '{page_title}':\n\n{body}"
