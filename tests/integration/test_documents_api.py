"""Document endpoints end to end."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from database.models import BackgroundTask
from documents.index import DocumentIndex
from documents.tasks import REINDEX_TASK, register_document_tasks
from shared.enums import BackgroundTaskStatus
from tests.fixtures.make_documents import build_all
from workers.registry import clear_registry
from workers.runner import WorkerRunner


async def _wait_for_success(
    session: AsyncSession, task_id: str, deadline_seconds: float = 30.0
) -> BackgroundTask:
    deadline = asyncio.get_running_loop().time() + deadline_seconds
    while asyncio.get_running_loop().time() < deadline:
        session.expunge_all()
        task = await session.get(BackgroundTask, task_id)
        if task is not None and task.status == BackgroundTaskStatus.SUCCEEDED.value:
            return task
        await asyncio.sleep(0.05)
    raise AssertionError(f"Task {task_id} did not finish within {deadline_seconds}s")


@pytest.fixture(autouse=True)
def corpus(settings: Settings, tmp_path: Path) -> Path:
    directory = build_all(tmp_path / "docs")
    settings.document_index_roots = [str(directory)]
    DocumentIndex(settings).reindex()
    return directory


class TestSearch:
    async def test_a_search_finds_a_document(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/documents/search?q=deformable")).json()
        assert body["total"] >= 1
        assert body["items"][0]["snippet"]

    async def test_results_can_be_narrowed_by_format(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/documents/search?q=tactile&kind=pdf")).json()
        assert all(item["kind"] == "pdf" for item in body["items"])

    async def test_an_operator_laden_query_is_answered_not_rejected(
        self, client: AsyncClient
    ) -> None:
        response = await client.get('/api/v1/documents/search?q=tactile" OR body:"secret')
        assert response.status_code == 200

    async def test_an_empty_query_is_refused(self, client: AsyncClient) -> None:
        assert (await client.get("/api/v1/documents/search?q=")).status_code == 422


class TestReading:
    async def test_a_document_reads_back_flagged_untrusted(
        self, client: AsyncClient, corpus: Path
    ) -> None:
        body = (
            await client.get(
                "/api/v1/documents/content",
                params={"path": str(corpus / "research_notes.txt")},
            )
        ).json()
        assert "Tactile feedback" in body["text"]
        assert body["content_is_untrusted"] is True

    async def test_a_path_outside_the_roots_is_refused(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/documents/content", params={"path": "/etc/passwd"})
        assert response.status_code == 400

    async def test_the_read_budget_is_honoured(self, client: AsyncClient, corpus: Path) -> None:
        body = (
            await client.get(
                "/api/v1/documents/content",
                params={"path": str(corpus / "research_notes.txt"), "max_chars": 120},
            )
        ).json()
        assert len(body["text"]) <= 120
        assert body["truncated"] is True


class TestSummary:
    async def test_a_summary_quotes_the_document(self, client: AsyncClient, corpus: Path) -> None:
        path = str(corpus / "research_notes.txt")
        body = (
            await client.get("/api/v1/documents/summary", params={"path": path, "sentences": 2})
        ).json()
        source = (await client.get("/api/v1/documents/content", params={"path": path})).json()[
            "text"
        ]

        assert len(body["sentences"]) == 2
        for sentence in body["sentences"]:
            assert sentence in source


class TestStats:
    async def test_stats_report_coverage(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/documents/stats")).json()
        assert body["documents"] == 10
        assert body["roots"]


class TestReindex:
    @pytest.fixture(autouse=True)
    def _handlers(self):
        clear_registry()
        register_document_tasks()
        yield
        clear_registry()

    async def test_reindexing_returns_a_ticket_immediately(self, client: AsyncClient) -> None:
        """A large folder takes minutes; a voice turn cannot wait for it."""
        response = await client.post("/api/v1/documents/reindex")

        assert response.status_code == 202
        assert response.json()["poll_url"].startswith("/api/v1/background-tasks/")

    async def test_the_ticket_is_a_real_queued_task(
        self, client: AsyncClient, session: AsyncSession
    ) -> None:
        task_id = (await client.post("/api/v1/documents/reindex")).json()["task_id"]
        task = await session.get(BackgroundTask, task_id)
        assert task is not None
        assert task.task_type == REINDEX_TASK

    async def test_the_worker_runs_the_reindex_to_completion(
        self, client: AsyncClient, session: AsyncSession, settings: Settings
    ) -> None:
        task_id = (await client.post("/api/v1/documents/reindex?full=true")).json()["task_id"]

        runner = WorkerRunner(settings, worker_id="test-worker")
        await runner.start()
        try:
            task = await _wait_for_success(session, task_id)
        finally:
            await runner.stop()

        assert json.loads(task.result_json)["indexed"] == 10
        assert task.progress == 100

    async def test_a_repeated_request_reuses_the_ticket(self, client: AsyncClient) -> None:
        headers = {"Idempotency-Key": "reindex-once"}
        first = (await client.post("/api/v1/documents/reindex", headers=headers)).json()
        second = (await client.post("/api/v1/documents/reindex", headers=headers)).json()
        assert first["task_id"] == second["task_id"]


class TestAuthentication:
    async def test_documents_need_a_token(self, anonymous_client: AsyncClient) -> None:
        assert (await anonymous_client.get("/api/v1/documents/search?q=tactile")).status_code == 401
