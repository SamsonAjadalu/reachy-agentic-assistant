"""Vision API: nine HF tools, fake Pi look_now, watches, scans."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from database.session import session_scope
from notifications.dispatcher import set_channel_override
from notifications.mock import MockNotificationChannel
from vision.watches import evaluate_due_watches


@pytest.fixture(autouse=True)
def _mock_telegram() -> MockNotificationChannel:
    channel = MockNotificationChannel()
    set_channel_override(channel)
    yield channel
    set_channel_override(None)


class TestVisionTools:
    async def test_camera_status(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/vision/camera/status")
        assert response.status_code == 200
        body = response.json()
        assert body["camera_health"] == "ok"
        assert body["mocked"] is True
        assert body["may_move_head"] is False

    async def test_look_now_persists(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/vision/look",
            json={"query": "mug", "zone": "desk", "inject_labels": ["mug", "keyboard"]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["world_frame_status"] == "unknown"
        assert "cm" not in body["phrase"]
        assert body["may_move_head"] is False
        assert body["observation_id"]

    async def test_identical_mugs_abstain(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/vision/look",
            json={"query": "mug", "inject_labels": ["mug", "mug", "mug"]},
        )
        body = response.json()
        assert body["presence"] == "abstained"

    async def test_person_look_has_no_entity_link(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/vision/look", json={"inject_labels": ["person"]})
        assert response.status_code == 200
        for item in response.json().get("detections") or []:
            if item.get("label") == "person":
                assert item.get("decision") in {"blocked_person", None}

    async def test_last_seen_and_search(self, client: AsyncClient) -> None:
        await client.post(
            "/api/v1/vision/look", json={"query": "keyboard", "inject_labels": ["keyboard"]}
        )
        seen = await client.get("/api/v1/vision/memory/last-seen", params={"q": "keyboard"})
        assert seen.status_code == 200
        assert seen.json()["may_move_head"] is False
        search = await client.get("/api/v1/vision/memory/search", params={"q": "keyboard"})
        assert search.status_code == 200

    async def test_watch_crud_and_evaluate(self, client: AsyncClient, settings) -> None:
        created = await client.post(
            "/api/v1/vision/watches",
            json={"label": "mug", "event": "appear", "zone": "desk"},
            headers={"Idempotency-Key": "watch-mug-1"},
        )
        assert created.status_code == 201
        watch_id = created.json()["id"]
        listed = await client.get("/api/v1/vision/watches")
        assert listed.json()["total"] >= 1
        async with session_scope() as session:
            summary = await evaluate_due_watches(session, settings)
        assert summary["due"] >= 1
        cancelled = await client.post(f"/api/v1/vision/watches/{watch_id}/cancel")
        assert cancelled.json()["status"] == "cancelled"

    async def test_scan_returns_ticket_or_block(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/vision/scans", json={"preset_id": "REGION_SWEEP", "query": "mug"}
        )
        assert response.status_code in {202, 409}
        body = response.json()
        assert "joint" not in str(body).lower()
        if response.status_code == 202:
            assert body["task_id"]
            assert body["poll_url"].startswith("/api/v1/background-tasks/")

    async def test_look_now_scan_ticket(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/vision/look",
            json={
                "query": "mug",
                "allow_scan": True,
                "preset_id": "CLOSE_LOOK",
                "inject_labels": ["mug", "mug"],
            },
        )
        assert response.status_code in {200, 202}
        if response.status_code == 202:
            assert response.json()["deferred"] is True
            assert response.json()["may_move_head"] is True
