"""System endpoints and database behaviour."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from database.models import Contact, ContactAlias, OwnerProfile, UserPreference
from database.session import check_integrity


class TestHealthAndReady:
    async def test_health_reports_ok(self, anonymous_client: AsyncClient) -> None:
        body = (await anonymous_client.get("/health")).json()
        assert body["status"] == "ok"
        assert body["timestamp"].endswith("Z")

    async def test_ready_checks_the_database(self, anonymous_client: AsyncClient) -> None:
        body = (await anonymous_client.get("/ready")).json()
        assert body["ready"] is True
        assert body["checks"]["database"]["ok"] is True


class TestPing:
    async def test_returns_both_utc_and_local_time(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/ping")).json()
        assert body["pong"] is True
        assert body["timezone"] == "America/Toronto"
        assert body["server_time_utc"].endswith("Z")
        assert body["server_time_utc"] != body["server_time_local"]

    async def test_answers_synchronously_without_a_task_ticket(self, client: AsyncClient) -> None:
        # Fast reads must not be deferred; only long-running work returns a ticket.
        body = (await client.get("/api/v1/ping")).json()
        assert "task_id" not in body


class TestStatus:
    async def test_reports_environment_and_database_health(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/status")).json()
        assert body["environment"] == "test"
        assert body["mock_mode"] is True
        assert body["database"]["ok"] is True
        assert body["database"]["foreign_keys_enabled"] is True

    async def test_reports_worker_queue_depth(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/status")).json()
        assert "queue_depth" in body["worker"]


class TestIntegrationsInventory:
    async def test_every_integration_is_listed(self, client: AsyncClient) -> None:
        names = {entry["name"] for entry in (await client.get("/api/v1/integrations")).json()}
        assert {
            "telegram",
            "google",
            "gmail",
            "calendar",
            "contacts",
            "drive",
            "notion",
            "weather",
            "workstation",
            "documents",
            "wardrobe",
        } <= names

    async def test_mock_mode_marks_everything_as_mocked(self, client: AsyncClient) -> None:
        entries = (await client.get("/api/v1/integrations")).json()
        assert {entry["mode"] for entry in entries} == {"mock"}


class TestDatabaseBehaviour:
    async def test_wal_and_foreign_keys_are_enabled(self, engine: AsyncEngine) -> None:
        result = await check_integrity(engine)
        assert result["journal_mode"] == "wal"
        assert result["foreign_keys_enabled"] is True
        assert result["ok"] is True

    async def test_foreign_keys_are_actually_enforced(self, session: AsyncSession) -> None:
        session.add(UserPreference(owner_id="does-not-exist", key="k", value="v"))
        with pytest.raises(IntegrityError):
            await session.flush()

    async def test_cascade_delete_removes_children(self, session: AsyncSession) -> None:
        owner = OwnerProfile(display_name="Owner", timezone="America/Toronto")
        session.add(owner)
        await session.flush()
        session.add(UserPreference(owner_id=owner.id, key="theme", value="dark"))
        await session.commit()

        await session.delete(owner)
        await session.commit()

        remaining = await session.scalar(text("SELECT COUNT(*) FROM user_preferences"))
        assert remaining == 0

    async def test_alias_uniqueness_is_enforced(self, session: AsyncSession) -> None:
        contact = Contact(display_name="Nathan", normalised_name="nathan")
        session.add(contact)
        await session.flush()
        session.add(
            ContactAlias(
                contact_id=contact.id, alias="my supervisor", normalised_alias="my supervisor"
            )
        )
        await session.commit()

        session.add(
            ContactAlias(
                contact_id=contact.id, alias="My Supervisor", normalised_alias="my supervisor"
            )
        )
        with pytest.raises(IntegrityError):
            await session.commit()

    async def test_naive_datetimes_are_refused_at_the_column(self, session: AsyncSession) -> None:
        from datetime import datetime

        from database.models import ActionAuditLog

        session.add(
            ActionAuditLog(
                occurred_at=datetime(2026, 1, 1, 12, 0),
                action_type="test",
                risk_level="read",
                outcome="ok",
            )
        )
        with pytest.raises(Exception, match="naive datetime"):
            await session.flush()

    async def test_datetimes_round_trip_as_aware_utc(self, session: AsyncSession) -> None:
        from datetime import UTC

        from database.models import ActionAuditLog
        from shared.timeutils import utcnow

        moment = utcnow()
        entry = ActionAuditLog(
            occurred_at=moment, action_type="test", risk_level="read", outcome="ok"
        )
        session.add(entry)
        await session.commit()
        session.expunge_all()

        loaded = await session.get(ActionAuditLog, entry.id)
        assert loaded is not None
        assert loaded.occurred_at.tzinfo is not None
        assert loaded.occurred_at.astimezone(UTC) == moment.replace(microsecond=moment.microsecond)
