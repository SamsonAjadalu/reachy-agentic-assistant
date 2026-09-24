"""Background handlers for document work.

Re-indexing is the clearest case in the system for deferring work: it can take
minutes over a large folder, so it returns a ticket and reports progress rather
than holding a voice turn open.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from documents.index import DocumentIndex
from workers.registry import TaskContext, has_handler, register_task

REINDEX_TASK = "documents.reindex"


async def reindex_documents(context: TaskContext) -> dict[str, Any]:
    """Rebuild or refresh the document index."""
    index = DocumentIndex(context.settings)
    full = bool(context.payload.get("full"))
    loop = asyncio.get_running_loop()
    last_reported = -1

    def report(position: int, total: int) -> None:
        """Called from the scan thread, so the update is bounced onto the loop.

        Percentages are reported in steps of five to avoid one database write
        per file.
        """
        nonlocal last_reported
        if total <= 0:
            return
        percent = int(position / total * 100)
        if percent == last_reported or percent % 5:
            return
        last_reported = percent

        async def push() -> None:
            await context.report_progress(percent, f"{position} of {total} files")

        future = asyncio.run_coroutine_threadsafe(push(), loop)
        with contextlib.suppress(Exception):
            future.result(timeout=5)

    stats = await index.reindex_async(full=full, progress=report)
    return stats.as_dict()


def register_document_tasks() -> None:
    if not has_handler(REINDEX_TASK):
        register_task(
            REINDEX_TASK,
            timeout_seconds=3600,
            max_attempts=2,
            description="Scan the configured document roots and update the full-text index.",
        )(reindex_documents)
