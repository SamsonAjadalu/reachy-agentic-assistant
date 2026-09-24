"""Reminder endpoints end to end."""

from __future__ import annotations

from datetime import timedelta

from httpx import AsyncClient

from shared.timeutils import isoformat_utc, utcnow


def future(**kwargs: int) -> str:
    return isoformat_utc(utcnow() + timedelta(**kwargs))


class TestCreate:
    async def test_creates_from_an_absolute_time(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/reminders",
            json={"text": "Call the dentist", "schedule": {"at": future(hours=2)}},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["text"] == "Call the dentist"
        assert body["status"] == "active"
        assert body["trigger_at"].endswith("Z")
        assert body["trigger_at_local"] != body["trigger_at"]

    async def test_creates_from_a_structured_delay(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/reminders",
            json={"text": "Take the bread out", "schedule": {"delay": {"minutes": 25}}},
        )
        assert response.status_code == 201
        trigger = response.json()["trigger_at"]
        assert trigger > isoformat_utc(utcnow() + timedelta(minutes=20))

    async def test_rejects_a_past_time(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/reminders",
            json={
                "text": "Too late",
                "schedule": {"at": isoformat_utc(utcnow() - timedelta(hours=1))},
            },
        )
        assert response.status_code == 422
        assert "past" in response.json()["error"]["message"]

    async def test_rejects_a_naive_timestamp(self, client: AsyncClient) -> None:
        """Guessing the zone is how a reminder ends up firing hours early."""
        response = await client.post(
            "/api/v1/reminders",
            json={"text": "Ambiguous", "schedule": {"at": "2030-01-01T09:00:00"}},
        )
        assert response.status_code == 422

    async def test_rejects_both_at_and_delay(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/reminders",
            json={
                "text": "Confused",
                "schedule": {"at": future(hours=1), "delay": {"minutes": 5}},
            },
        )
        assert response.status_code == 422

    async def test_rejects_neither_at_nor_delay(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/reminders", json={"text": "Nothing", "schedule": {}})
        assert response.status_code == 422

    async def test_rejects_an_unknown_field(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/reminders",
            json={"text": "Typo", "schedule": {"at": future(hours=1)}, "reccurence": {}},
        )
        assert response.status_code == 422

    async def test_creates_a_recurring_reminder(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/reminders",
            json={
                "text": "Weekly review",
                "schedule": {"delay": {"minutes": 1}},
                "recurrence": {
                    "frequency": "weekly",
                    "by_weekday": [4],
                    "at_hour": 16,
                    "at_minute": 30,
                },
            },
        )
        assert response.status_code == 201
        recurrence = response.json()["recurrence"]
        assert recurrence["description"] == "every week on Friday at 16:30 America/Toronto"

    async def test_links_to_an_existing_task(self, client: AsyncClient) -> None:
        task = (await client.post("/api/v1/tasks", json={"description": "Submit paper"})).json()
        response = await client.post(
            "/api/v1/reminders",
            json={
                "text": "Submit the paper",
                "schedule": {"at": future(hours=3)},
                "task_id": task["id"],
            },
        )
        assert response.status_code == 201
        assert response.json()["task_id"] == task["id"]

    async def test_rejects_an_unknown_task_id(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/reminders",
            json={
                "text": "Orphan",
                "schedule": {"at": future(hours=1)},
                "task_id": "no-such-task",
            },
        )
        assert response.status_code == 404


class TestIdempotency:
    async def test_the_same_key_and_body_returns_the_original_reminder(
        self, client: AsyncClient
    ) -> None:
        payload = {"text": "Only once", "schedule": {"at": future(hours=5)}}
        headers = {"Idempotency-Key": "voice-turn-abc"}

        first = await client.post("/api/v1/reminders", json=payload, headers=headers)
        second = await client.post("/api/v1/reminders", json=payload, headers=headers)

        assert first.status_code == 201
        assert first.json()["id"] == second.json()["id"]

        listed = (await client.get("/api/v1/reminders")).json()
        assert listed["total"] == 1

    async def test_the_same_key_with_a_different_body_is_a_conflict(
        self, client: AsyncClient
    ) -> None:
        headers = {"Idempotency-Key": "voice-turn-xyz"}
        await client.post(
            "/api/v1/reminders",
            json={"text": "First", "schedule": {"at": future(hours=5)}},
            headers=headers,
        )
        response = await client.post(
            "/api/v1/reminders",
            json={"text": "Different", "schedule": {"at": future(hours=6)}},
            headers=headers,
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "idempotency_conflict"

    async def test_different_keys_create_separate_reminders(self, client: AsyncClient) -> None:
        payload = {"text": "Twice on purpose", "schedule": {"at": future(hours=5)}}
        await client.post("/api/v1/reminders", json=payload, headers={"Idempotency-Key": "a"})
        await client.post("/api/v1/reminders", json=payload, headers={"Idempotency-Key": "b"})
        assert (await client.get("/api/v1/reminders")).json()["total"] == 2


class TestLifecycle:
    async def _create(self, client: AsyncClient, text: str = "Test") -> dict:
        return (
            await client.post(
                "/api/v1/reminders", json={"text": text, "schedule": {"at": future(hours=4)}}
            )
        ).json()

    async def test_snooze_pushes_the_trigger_out_and_counts(self, client: AsyncClient) -> None:
        reminder = await self._create(client)
        response = await client.post(
            f"/api/v1/reminders/{reminder['id']}/snooze", json={"delay": {"minutes": 15}}
        )
        body = response.json()
        assert body["status"] == "snoozed"
        assert body["snooze_count"] == 1
        assert body["trigger_at"] != reminder["trigger_at"]

    async def test_the_original_trigger_time_is_preserved_across_snoozes(
        self, client: AsyncClient
    ) -> None:
        reminder = await self._create(client)
        await client.post(
            f"/api/v1/reminders/{reminder['id']}/snooze", json={"delay": {"minutes": 5}}
        )
        body = (await client.get(f"/api/v1/reminders/{reminder['id']}")).json()
        assert body["original_trigger_at"] == reminder["trigger_at"]

    async def test_complete_marks_it_done(self, client: AsyncClient) -> None:
        reminder = await self._create(client)
        body = (await client.post(f"/api/v1/reminders/{reminder['id']}/complete")).json()
        assert body["status"] == "completed"
        assert body["completed_at"] is not None

    async def test_delete_cancels_rather_than_erasing(self, client: AsyncClient) -> None:
        reminder = await self._create(client)
        assert (await client.delete(f"/api/v1/reminders/{reminder['id']}")).status_code == 200
        body = (await client.get(f"/api/v1/reminders/{reminder['id']}")).json()
        assert body["status"] == "cancelled"

    async def test_a_completed_reminder_cannot_be_edited(self, client: AsyncClient) -> None:
        reminder = await self._create(client)
        await client.post(f"/api/v1/reminders/{reminder['id']}/complete")
        response = await client.patch(
            f"/api/v1/reminders/{reminder['id']}", json={"text": "Changed"}
        )
        assert response.status_code == 409

    async def test_a_cancelled_reminder_cannot_be_snoozed(self, client: AsyncClient) -> None:
        reminder = await self._create(client)
        await client.delete(f"/api/v1/reminders/{reminder['id']}")
        response = await client.post(
            f"/api/v1/reminders/{reminder['id']}/snooze", json={"delay": {"minutes": 5}}
        )
        assert response.status_code == 409

    async def test_patch_requires_at_least_one_field(self, client: AsyncClient) -> None:
        reminder = await self._create(client)
        assert (
            await client.patch(f"/api/v1/reminders/{reminder['id']}", json={})
        ).status_code == 422

    async def test_unknown_reminder_returns_404(self, client: AsyncClient) -> None:
        assert (await client.get("/api/v1/reminders/nope")).status_code == 404


class TestListing:
    async def test_filters_by_status(self, client: AsyncClient) -> None:
        first = (
            await client.post(
                "/api/v1/reminders", json={"text": "A", "schedule": {"at": future(hours=1)}}
            )
        ).json()
        await client.post(
            "/api/v1/reminders", json={"text": "B", "schedule": {"at": future(hours=2)}}
        )
        await client.post(f"/api/v1/reminders/{first['id']}/complete")

        active = (await client.get("/api/v1/reminders", params={"status": "active"})).json()
        assert active["total"] == 1
        assert active["items"][0]["text"] == "B"

    async def test_orders_by_soonest_first(self, client: AsyncClient) -> None:
        await client.post(
            "/api/v1/reminders", json={"text": "Later", "schedule": {"at": future(hours=8)}}
        )
        await client.post(
            "/api/v1/reminders", json={"text": "Sooner", "schedule": {"at": future(hours=2)}}
        )
        items = (await client.get("/api/v1/reminders")).json()["items"]
        assert [item["text"] for item in items] == ["Sooner", "Later"]

    async def test_pagination_reports_the_full_total(self, client: AsyncClient) -> None:
        for index in range(5):
            await client.post(
                "/api/v1/reminders",
                json={"text": f"R{index}", "schedule": {"at": future(hours=index + 1)}},
            )
        page = (await client.get("/api/v1/reminders", params={"limit": 2, "offset": 2})).json()
        assert page["total"] == 5
        assert len(page["items"]) == 2

    async def test_upcoming_only_excludes_finished_reminders(self, client: AsyncClient) -> None:
        reminder = (
            await client.post(
                "/api/v1/reminders", json={"text": "Done", "schedule": {"at": future(hours=1)}}
            )
        ).json()
        await client.post(f"/api/v1/reminders/{reminder['id']}/complete")
        assert (await client.get("/api/v1/reminders", params={"upcoming_only": True})).json()[
            "total"
        ] == 0
