"""Task endpoints end to end."""

from __future__ import annotations

from datetime import timedelta

from httpx import AsyncClient

from shared.timeutils import isoformat_utc, utcnow


class TestCreate:
    async def test_creates_with_defaults(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/tasks", json={"description": "Write the report"})
        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "open"
        assert body["priority"] == "normal"
        assert body["due_at"] is None
        assert body["is_overdue"] is False

    async def test_stores_tags_lowercased_and_deduplicated(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/tasks",
            json={"description": "Tagged", "tags": ["Work", "work", "URGENT", " work "]},
        )
        assert sorted(response.json()["tags"]) == ["urgent", "work"]

    async def test_rejects_more_than_twenty_tags(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/tasks",
            json={"description": "Too many", "tags": [f"t{i}" for i in range(21)]},
        )
        assert response.status_code == 422

    async def test_a_past_due_date_is_reported_as_overdue(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/tasks",
            json={
                "description": "Late",
                "due_at": isoformat_utc(utcnow() - timedelta(days=1)),
            },
        )
        assert response.json()["is_overdue"] is True

    async def test_a_recurring_task_gets_its_first_due_date(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/tasks",
            json={
                "description": "Water the plants",
                "recurrence": {"frequency": "daily", "at_hour": 8},
            },
        )
        body = response.json()
        assert body["due_at"] is not None
        assert body["recurrence"]["description"] == "every day at 08:00 America/Toronto"

    async def test_idempotent_creation(self, client: AsyncClient) -> None:
        headers = {"Idempotency-Key": "task-key-1"}
        payload = {"description": "Only one"}
        first = await client.post("/api/v1/tasks", json=payload, headers=headers)
        second = await client.post("/api/v1/tasks", json=payload, headers=headers)
        assert first.json()["id"] == second.json()["id"]
        assert (await client.get("/api/v1/tasks")).json()["total"] == 1


class TestLifecycle:
    async def test_completing_a_plain_task_closes_it(self, client: AsyncClient) -> None:
        task = (await client.post("/api/v1/tasks", json={"description": "One off"})).json()
        body = (await client.post(f"/api/v1/tasks/{task['id']}/complete")).json()
        assert body["status"] == "completed"
        assert body["completed_at"] is not None

    async def test_completing_a_recurring_task_rolls_it_forward(self, client: AsyncClient) -> None:
        """The row stays open so the task keeps one stable identity in conversation."""
        task = (
            await client.post(
                "/api/v1/tasks",
                json={
                    "description": "Weekly review",
                    "recurrence": {"frequency": "weekly", "at_hour": 17},
                },
            )
        ).json()

        body = (await client.post(f"/api/v1/tasks/{task['id']}/complete")).json()
        assert body["status"] == "open"
        assert body["due_at"] > task["due_at"]
        assert body["id"] == task["id"]
        assert body["recurrence"]["occurrences_fired"] == 1

    async def test_a_recurrence_stops_at_max_occurrences(self, client: AsyncClient) -> None:
        task = (
            await client.post(
                "/api/v1/tasks",
                json={
                    "description": "Three times only",
                    "recurrence": {"frequency": "daily", "at_hour": 9, "max_occurrences": 2},
                },
            )
        ).json()

        first = (await client.post(f"/api/v1/tasks/{task['id']}/complete")).json()
        assert first["status"] == "open"
        second = (await client.post(f"/api/v1/tasks/{task['id']}/complete")).json()
        assert second["status"] == "completed"

    async def test_cancel_sets_the_cancelled_timestamp(self, client: AsyncClient) -> None:
        task = (await client.post("/api/v1/tasks", json={"description": "Forget it"})).json()
        body = (await client.post(f"/api/v1/tasks/{task['id']}/cancel")).json()
        assert body["status"] == "cancelled"
        assert body["cancelled_at"] is not None

    async def test_reopening_a_completed_task_clears_its_completion(
        self, client: AsyncClient
    ) -> None:
        task = (await client.post("/api/v1/tasks", json={"description": "Redo"})).json()
        await client.post(f"/api/v1/tasks/{task['id']}/complete")
        body = (await client.patch(f"/api/v1/tasks/{task['id']}", json={"status": "open"})).json()
        assert body["status"] == "open"
        assert body["completed_at"] is None

    async def test_editing_a_completed_task_without_reopening_is_a_conflict(
        self, client: AsyncClient
    ) -> None:
        task = (await client.post("/api/v1/tasks", json={"description": "Done"})).json()
        await client.post(f"/api/v1/tasks/{task['id']}/complete")
        response = await client.patch(
            f"/api/v1/tasks/{task['id']}", json={"description": "Changed"}
        )
        assert response.status_code == 409

    async def test_clear_due_at_removes_the_date(self, client: AsyncClient) -> None:
        task = (
            await client.post(
                "/api/v1/tasks",
                json={
                    "description": "Dated",
                    "due_at": isoformat_utc(utcnow() + timedelta(days=2)),
                },
            )
        ).json()
        body = (
            await client.patch(f"/api/v1/tasks/{task['id']}", json={"clear_due_at": True})
        ).json()
        assert body["due_at"] is None

    async def test_due_at_and_clear_due_at_together_is_rejected(self, client: AsyncClient) -> None:
        task = (await client.post("/api/v1/tasks", json={"description": "Ambiguous"})).json()
        response = await client.patch(
            f"/api/v1/tasks/{task['id']}",
            json={"clear_due_at": True, "due_at": isoformat_utc(utcnow() + timedelta(days=1))},
        )
        assert response.status_code == 422

    async def test_deleting_a_task_detaches_its_reminders(self, client: AsyncClient) -> None:
        task = (await client.post("/api/v1/tasks", json={"description": "Linked"})).json()
        reminder = (
            await client.post(
                "/api/v1/reminders",
                json={
                    "text": "Do the linked task",
                    "schedule": {"delay": {"hours": 3}},
                    "task_id": task["id"],
                },
            )
        ).json()

        assert (await client.delete(f"/api/v1/tasks/{task['id']}")).status_code == 200
        assert (await client.get(f"/api/v1/tasks/{task['id']}")).status_code == 404
        assert (await client.get(f"/api/v1/reminders/{reminder['id']}")).json()["task_id"] is None


class TestListing:
    async def _seed(self, client: AsyncClient) -> None:
        await client.post(
            "/api/v1/tasks",
            json={
                "description": "Overdue urgent",
                "priority": "urgent",
                "tags": ["work"],
                "due_at": isoformat_utc(utcnow() - timedelta(days=1)),
            },
        )
        await client.post(
            "/api/v1/tasks",
            json={
                "description": "Future normal",
                "tags": ["home"],
                "due_at": isoformat_utc(utcnow() + timedelta(days=3)),
            },
        )
        await client.post("/api/v1/tasks", json={"description": "No due date", "tags": ["work"]})

    async def test_filters_by_tag_without_matching_substrings(self, client: AsyncClient) -> None:
        await client.post("/api/v1/tasks", json={"description": "A", "tags": ["work"]})
        await client.post("/api/v1/tasks", json={"description": "B", "tags": ["homework"]})
        result = (await client.get("/api/v1/tasks", params={"tag": "work"})).json()
        assert result["total"] == 1
        assert result["items"][0]["description"] == "A"

    async def test_filters_by_priority(self, client: AsyncClient) -> None:
        await self._seed(client)
        result = (await client.get("/api/v1/tasks", params={"priority": "urgent"})).json()
        assert result["total"] == 1

    async def test_overdue_only_excludes_future_and_closed_tasks(self, client: AsyncClient) -> None:
        await self._seed(client)
        result = (await client.get("/api/v1/tasks", params={"overdue_only": True})).json()
        assert result["total"] == 1
        assert result["items"][0]["description"] == "Overdue urgent"

    async def test_undated_tasks_sort_after_dated_ones(self, client: AsyncClient) -> None:
        await self._seed(client)
        descriptions = [
            item["description"] for item in (await client.get("/api/v1/tasks")).json()["items"]
        ]
        assert descriptions.index("No due date") == len(descriptions) - 1

    async def test_unknown_task_returns_404(self, client: AsyncClient) -> None:
        assert (await client.get("/api/v1/tasks/missing")).status_code == 404
