"""Briefings, alerts, cooldown ledger and the Reachy pending-notification queue."""

from __future__ import annotations

from datetime import timedelta

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import AlertDeduplication, Task
from notifications.dispatcher import clear_dedup, should_notify
from proactive import alerts as alert_service
from proactive.briefing import assemble, save_preferences
from shared.enums import AlertSeverity, TaskStatus
from shared.timeutils import utcnow


class TestBriefings:
    async def test_assemble_returns_every_default_section(
        self, client: AsyncClient, session: AsyncSession, settings
    ) -> None:
        result = await assemble(session, settings)
        names = {section.name for section in result.sections}
        assert {"calendar", "tasks", "reminders", "weather"} <= names
        assert result.text
        assert result.local_date

    async def test_today_endpoint(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/briefings/today")).json()
        assert "sections" in body
        assert "text" in body
        assert "task_id" not in body  # synchronous read, not a background ticket

    async def test_preferences_round_trip(self, client: AsyncClient) -> None:
        updated = (
            await client.patch(
                "/api/v1/briefings/preferences",
                json={
                    "send_hour": 6,
                    "send_minute": 30,
                    "sections": ["calendar", "weather"],
                    "quiet_hours_start": 22,
                    "quiet_hours_end": 7,
                },
            )
        ).json()
        assert updated["send_hour"] == 6
        assert updated["send_minute"] == 30
        assert updated["sections"] == ["calendar", "weather"]
        assert updated["quiet_hours_start"] == 22

        current = (await client.get("/api/v1/briefings/preferences")).json()
        assert current["send_hour"] == 6
        assert current["sections"] == ["calendar", "weather"]

    async def test_generate_alias_delivers(
        self, client: AsyncClient, session: AsyncSession, settings
    ) -> None:
        response = await client.post("/api/v1/briefings/generate")
        assert response.status_code == 200
        body = response.json()
        assert "briefing" in body
        assert body["sent"] is True or body.get("reason") in {
            "briefings_disabled",
            "quiet_hours",
        }

    async def test_quiet_hours_wrap_midnight(self, session: AsyncSession, settings) -> None:
        from datetime import datetime

        from shared.timeutils import from_local

        preferences = await save_preferences(
            session,
            {
                "quiet_hours_start": 22,
                "quiet_hours_end": 7,
            },
            settings,
        )
        await session.commit()
        # 23:00 local is quiet; 10:00 local is not.
        late = from_local(datetime(2026, 8, 2, 23, 0, 0), preferences.timezone)
        morning = from_local(datetime(2026, 8, 2, 10, 0, 0), preferences.timezone)
        assert preferences.in_quiet_hours(late) is True
        assert preferences.in_quiet_hours(morning) is False


class TestAlerts:
    async def test_rules_are_listed(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/alerts/rules")).json()
        assert "disk_space" in body["rules"]
        assert "overdue_tasks" in body["rules"]

    async def test_evaluate_dry_run(self, client: AsyncClient) -> None:
        body = (await client.post("/api/v1/alerts/evaluate?dry_run=true")).json()
        assert body["evaluated"] >= 1
        assert "alerts" in body

    async def test_test_alias(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/alerts/test?dry_run=true")
        assert response.status_code == 200

    async def test_dedup_suppresses_repeat_within_cooldown(self, session: AsyncSession) -> None:
        key = "test:dedup-key"
        assert await should_notify(session, key, "test.kind", cooldown_seconds=3600) is True
        assert await should_notify(session, key, "test.kind", cooldown_seconds=3600) is False
        await clear_dedup(session, key)
        assert await should_notify(session, key, "test.kind", cooldown_seconds=3600) is True

    async def test_overdue_tasks_rule_fires_above_threshold(
        self, session: AsyncSession, settings
    ) -> None:
        now = utcnow()
        for index in range(3):
            session.add(
                Task(
                    description=f"Overdue task {index}",
                    status=TaskStatus.OPEN.value,
                    due_at=now - timedelta(hours=index + 1),
                )
            )
        await session.commit()

        result = await alert_service.evaluate(
            session, settings, rules=["overdue_tasks"], dry_run=True
        )
        kinds = {entry["kind"] for entry in result["alerts"]}
        assert "tasks.overdue" in kinds

        # Second evaluation within the cooldown is suppressed when not dry-run.
        first = await alert_service.evaluate(
            session, settings, rules=["overdue_tasks"], dry_run=False
        )
        second = await alert_service.evaluate(
            session, settings, rules=["overdue_tasks"], dry_run=False
        )
        await session.commit()
        assert first["raised"] >= 1
        assert second["suppressed"] >= 1
        assert second["raised"] == 0

    async def test_cooldowns_endpoint(self, client: AsyncClient, session: AsyncSession) -> None:
        now = utcnow()
        session.add(
            AlertDeduplication(
                dedup_key="disk:/",
                alert_kind="workstation.disk_space",
                first_seen_at=now,
                last_notified_at=now,
                cooldown_seconds=3600,
                notify_count=1,
            )
        )
        await session.commit()
        body = (await client.get("/api/v1/alerts/cooldowns")).json()
        assert body["total"] >= 1
        assert any(item["dedup_key"] == "disk:/" for item in body["cooldowns"])


class TestReachyPending:
    async def test_queue_collect_acknowledge(
        self, client: AsyncClient, session: AsyncSession
    ) -> None:
        created = (
            await client.post(
                "/api/v1/reachy/pending",
                json={
                    "kind": "reminder.fired",
                    "spoken_text": "Remember to call Nathan.",
                    "severity": AlertSeverity.INFO.value,
                },
            )
        ).json()
        assert created["id"]

        pending = (await client.get("/api/v1/reachy/pending-notifications")).json()
        assert pending["total"] >= 1
        assert any(item["id"] == created["id"] for item in pending["items"])
        # Uses the configured workflow.
        again = (await client.get("/api/v1/reachy/pending")).json()
        assert any(item["id"] == created["id"] for item in again["items"])

        ack = (
            await client.post(f"/api/v1/reachy/pending-notifications/{created['id']}/acknowledge")
        ).json()
        assert ack["acknowledged"] == 1

        empty = (await client.get("/api/v1/reachy/pending")).json()
        assert all(item["id"] != created["id"] for item in empty["items"])

    async def test_greeting_includes_pending(self, client: AsyncClient) -> None:
        await client.post(
            "/api/v1/reachy/pending",
            json={"kind": "test", "spoken_text": "You have mail from Alice."},
        )
        body = (await client.get("/api/v1/reachy/greeting")).json()
        assert body["pending_count"] >= 1
        assert "Alice" in body["spoken_text"] or body["pending_ids"]
