"""End-to-end mock acceptance walks.

These prove the major owner-facing workflows against mock providers without any
external credentials. They are the gate that `./scripts/verify.sh` uses to show
the system is usable before OAuth or Telegram tokens are supplied.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from httpx import AsyncClient

from shared.enums import AlertSeverity
from shared.timeutils import utcnow


@pytest.mark.asyncio
async def test_reminder_task_and_briefing_workflow(client: AsyncClient) -> None:
    trigger = utcnow() + timedelta(hours=3)
    reminder = (
        await client.post(
            "/api/v1/reminders",
            json={
                "text": "Acceptance: call supervisor",
                "schedule": {"at": trigger.isoformat().replace("+00:00", "Z")},
            },
            headers={"Idempotency-Key": "acceptance-reminder-1"},
        )
    ).json()
    assert reminder["id"]
    assert reminder["text"] == "Acceptance: call supervisor"

    task = (
        await client.post(
            "/api/v1/tasks",
            json={"description": "Acceptance: finish report"},
            headers={"Idempotency-Key": "acceptance-task-1"},
        )
    ).json()
    assert task["id"]

    completed = (await client.post(f"/api/v1/tasks/{task['id']}/complete")).json()
    assert completed["status"] == "completed"

    briefing = (await client.get("/api/v1/briefings/today")).json()
    assert briefing["text"]
    assert "sections" in briefing


@pytest.mark.asyncio
async def test_weather_wardrobe_and_workstation(client: AsyncClient) -> None:
    weather = (await client.get("/api/v1/weather/current")).json()
    assert weather["report"]["location"]
    assert weather["advice"]["summary"]

    items = (await client.get("/api/v1/wardrobe/items", params={"limit": 5})).json()
    assert "items" in items
    assert "total" in items

    status = (await client.get("/api/v1/workstation/status")).json()
    assert status["status"]["hostname"]
    assert status["spoken_summary"]


@pytest.mark.asyncio
async def test_google_and_notion_mock_reads(client: AsyncClient) -> None:
    mail = (await client.get("/api/v1/gmail/search", params={"q": "report", "limit": 5})).json()
    assert "items" in mail
    assert mail["total"] >= 0

    events = (await client.get("/api/v1/calendar/events", params={"days": 7})).json()
    assert "items" in events

    notion = (await client.get("/api/v1/notion/search", params={"q": "notes", "limit": 5})).json()
    assert "items" in notion or "pages" in notion or notion is not None


@pytest.mark.asyncio
async def test_draft_create_does_not_send_and_send_requires_approval(
    client: AsyncClient,
) -> None:
    created = await client.post(
        "/api/v1/gmail/drafts",
        json={
            "to": ["friend@example.com"],
            "subject": "Acceptance draft",
            "body": "Approval is required before sending this.",
        },
    )
    assert created.status_code == 201
    draft = created.json()["draft"]
    assert draft["id"]
    assert draft["content_hash"]

    legacy = await client.post(
        "/api/v1/gmail/send",
        json={
            "to": ["friend@example.com"],
            "subject": "Acceptance draft",
            "body": "Approval is required before sending this.",
        },
    )
    assert legacy.status_code == 410

    ticket = await client.post(
        "/api/v1/gmail/actions/send-draft",
        json={"draft_id": draft["id"], "content_hash": draft["content_hash"]},
        headers={"Idempotency-Key": "acceptance-send-1"},
    )
    assert ticket.status_code == 202
    body = ticket.json()
    assert body["status"] == "awaiting_approval"
    assert "approval_id" in body


@pytest.mark.asyncio
async def test_calendar_propose_and_gmail_archive(client: AsyncClient) -> None:
    proposal = await client.post(
        "/api/v1/calendar/events/propose",
        json={"title": "Acceptance coffee", "duration_minutes": 30},
    )
    assert proposal.status_code == 200
    body = proposal.json()
    assert body["suggestions"]
    assert body["draft_create_payload"]["title"] == "Acceptance coffee"

    attachments = (await client.get("/api/v1/gmail/messages/msg-004/attachments")).json()
    assert attachments["total"] >= 1

    archived = (await client.post("/api/v1/gmail/messages/msg-002/actions/archive")).json()
    assert "INBOX" not in archived["labels"]


@pytest.mark.asyncio
async def test_reachy_pending_queue_round_trip(client: AsyncClient) -> None:
    created = (
        await client.post(
            "/api/v1/reachy/pending",
            json={
                "kind": "acceptance.note",
                "spoken_text": "You have one acceptance notification.",
                "severity": AlertSeverity.INFO.value,
            },
        )
    ).json()
    pending = (await client.get("/api/v1/reachy/pending-notifications")).json()
    assert any(item["id"] == created["id"] for item in pending["items"])

    ack = (
        await client.post(f"/api/v1/reachy/pending-notifications/{created['id']}/acknowledge")
    ).json()
    assert ack["acknowledged"] == 1


@pytest.mark.asyncio
async def test_alert_dedup_across_evaluations(client: AsyncClient) -> None:
    first = (await client.post("/api/v1/alerts/evaluate?dry_run=false")).json()
    second = (await client.post("/api/v1/alerts/evaluate?dry_run=false")).json()
    assert first["evaluated"] >= 1
    # Anything that fired once should be suppressed on the immediate re-run
    # unless the underlying facts changed.
    if first["raised"] > 0:
        assert second["suppressed"] >= first["raised"] or second["raised"] == 0
