"""Workstation endpoints end to end."""

from __future__ import annotations

import asyncio
import stat
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.services import workstation as service
from approvals.state_machine import clear_executors, resolve_by_token
from database.models import Approval, RegisteredScript, ScriptRun
from shared.enums import ScriptRunStatus
from workers.registry import clear_registry
from workstation.executors import register_workstation_executors
from workstation.tasks import RUN_SCRIPT_TASK, register_workstation_tasks


@pytest.fixture
def scripts(tmp_path: Path, settings: Settings) -> Path:
    echo = tmp_path / "echo_argv.py"
    echo.write_text(
        "#!/usr/bin/env python3\nimport sys\nprint('\\n'.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    echo.chmod(echo.stat().st_mode | stat.S_IXUSR)

    registry = tmp_path / "scripts.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "scripts": [
                    {
                        "name": "echo",
                        "description": "Print the arguments it was given.",
                        "executable": str(echo),
                        "requires_approval": False,
                        "risk_level": "read",
                        "timeout_seconds": 20,
                        "parameters": [{"name": "message", "type": "string", "required": True}],
                    },
                    {
                        "name": "gated",
                        "description": "Something worth confirming.",
                        "executable": str(echo),
                        "requires_approval": True,
                        "risk_level": "external_write",
                        "timeout_seconds": 20,
                        "parameters": [{"name": "message", "type": "string", "required": True}],
                    },
                    {"name": "broken", "description": "", "executable": "relative-path"},
                ]
            }
        ),
        encoding="utf-8",
    )
    settings.workstation_script_registry = registry
    service._cached_registry.cache_clear()
    return registry


@pytest.fixture(autouse=True)
def handlers() -> AsyncIterator[None]:
    clear_executors()
    clear_registry()
    register_workstation_executors()
    register_workstation_tasks()
    yield
    clear_executors()
    clear_registry()


async def wait_for(session: AsyncSession, run_id: str, *, seconds: float = 20) -> ScriptRun:
    """Poll the run record until it reaches a terminal state."""
    terminal = {
        ScriptRunStatus.SUCCEEDED.value,
        ScriptRunStatus.FAILED.value,
        ScriptRunStatus.TIMED_OUT.value,
        ScriptRunStatus.STOPPED.value,
    }
    deadline = asyncio.get_running_loop().time() + seconds
    while asyncio.get_running_loop().time() < deadline:
        session.expunge_all()
        run = await session.get(ScriptRun, run_id)
        if run is not None and run.status in terminal:
            return run
        await asyncio.sleep(0.1)
    pytest.fail(f"Run {run_id} remains active.")


class TestStatus:
    async def test_the_machine_reports_its_load(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/workstation/status")).json()
        assert body["status"]["cpu_count"] >= 1
        assert body["status"]["memory_percent"] > 0

    async def test_a_spoken_summary_is_provided(self, client: AsyncClient) -> None:
        """Reachy reads this out; it should be a sentence, not a table."""
        body = (await client.get("/api/v1/workstation/status")).json()
        assert body["spoken_summary"].endswith(".")
        assert "percent" in body["spoken_summary"]

    async def test_the_data_directory_disk_is_included(
        self, client: AsyncClient, settings: Settings
    ) -> None:
        body = (await client.get("/api/v1/workstation/status")).json()
        assert body["status"]["disks"]

    async def test_a_missing_gpu_is_a_note_not_an_error(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/workstation/status")).json()
        assert body["status"]["gpus"] or body["status"]["gpu_note"]


class TestServices:
    async def test_an_unlisted_service_cannot_be_queried(
        self, client: AsyncClient, settings: Settings
    ) -> None:
        settings.workstation_allowed_services = ["reachy-assistant.service"]
        response = await client.get("/api/v1/workstation/services/sshd.service")
        assert response.status_code == 422

    async def test_an_empty_allowlist_exposes_nothing(
        self, client: AsyncClient, settings: Settings
    ) -> None:
        settings.workstation_allowed_services = []
        assert (await client.get("/api/v1/workstation/services")).json() == []


class TestScriptListing:
    async def test_registered_scripts_are_listed(self, client: AsyncClient, scripts: Path) -> None:
        body = (await client.get("/api/v1/workstation/scripts")).json()
        assert {script["name"] for script in body["scripts"]} == {"echo", "gated"}

    async def test_a_rejected_entry_is_reported_not_hidden(
        self, client: AsyncClient, scripts: Path
    ) -> None:
        """Silent omission would leave the owner wondering why nothing happens."""
        body = (await client.get("/api/v1/workstation/scripts")).json()
        assert "broken" in body["rejected"]

    async def test_parameters_are_described_for_the_caller(
        self, client: AsyncClient, scripts: Path
    ) -> None:
        body = (await client.get("/api/v1/workstation/scripts")).json()
        echo = next(script for script in body["scripts"] if script["name"] == "echo")
        assert echo["parameters"][0] == {
            "name": "message",
            "type": "string",
            "required": True,
            "choices": [],
            "flag": None,
            "description": "",
        }

    async def test_reloading_mirrors_the_file_into_the_database(
        self, client: AsyncClient, session: AsyncSession, scripts: Path
    ) -> None:
        body = (await client.post("/api/v1/workstation/scripts/reload")).json()
        assert body == {"loaded": 2, "failed": 1, "retired": 0}

        rows = list(await session.scalars(select(RegisteredScript)))
        assert {row.name for row in rows} == {"echo", "gated", "broken"}

    async def test_a_removed_script_is_retired_on_reload(
        self, client: AsyncClient, session: AsyncSession, scripts: Path
    ) -> None:
        await client.post("/api/v1/workstation/scripts/reload")

        await asyncio.to_thread(
            scripts.write_text, yaml.safe_dump({"scripts": []}), encoding="utf-8"
        )
        service._cached_registry.cache_clear()
        body = (await client.post("/api/v1/workstation/scripts/reload")).json()

        # Only the two that were live get retired; 'broken' was already disabled.
        assert body["retired"] == 2
        session.expunge_all()
        rows = list(await session.scalars(select(RegisteredScript)))
        assert all(not row.enabled for row in rows)


class TestRunning:
    async def test_an_unregistered_script_cannot_be_run(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/workstation/scripts/rm/run", json={"arguments": {}})
        assert response.status_code == 422

    async def test_an_undeclared_argument_is_refused(
        self, client: AsyncClient, scripts: Path
    ) -> None:
        response = await client.post(
            "/api/v1/workstation/scripts/echo/run",
            json={"arguments": {"message": "hi", "sneaky": "--dangerous"}},
        )
        assert response.status_code == 422

    async def test_an_ungated_script_returns_a_run_id_immediately(
        self, client: AsyncClient, scripts: Path
    ) -> None:
        response = await client.post(
            "/api/v1/workstation/scripts/echo/run",
            json={"arguments": {"message": "hello"}},
        )

        assert response.status_code == 202
        body = response.json()
        assert body["poll_url"] == f"/api/v1/workstation/runs/{body['run_id']}"

    async def test_the_exact_argv_is_recorded(self, client: AsyncClient, scripts: Path) -> None:
        """What ran has to be inspectable afterwards, not inferred."""
        created = (
            await client.post(
                "/api/v1/workstation/scripts/echo/run",
                json={"arguments": {"message": "; rm -rf /"}},
            )
        ).json()

        run = (await client.get(f"/api/v1/workstation/runs/{created['run_id']}")).json()
        assert run["argv"][-1] == "; rm -rf /"

    async def test_the_worker_runs_it_to_completion(
        self, client: AsyncClient, session: AsyncSession, settings: Settings, scripts: Path
    ) -> None:
        from workers.runner import WorkerRunner

        created = (
            await client.post(
                "/api/v1/workstation/scripts/echo/run",
                json={"arguments": {"message": "hello from the worker"}},
            )
        ).json()

        worker = WorkerRunner(settings)
        await worker.start()
        try:
            run = await wait_for(session, created["run_id"])
        finally:
            await worker.stop()

        assert run.status == ScriptRunStatus.SUCCEEDED.value
        assert run.exit_code == 0
        assert "hello from the worker" in (run.stdout_tail or "")

    async def test_the_captured_log_is_readable(
        self, client: AsyncClient, session: AsyncSession, settings: Settings, scripts: Path
    ) -> None:
        from workers.runner import WorkerRunner

        created = (
            await client.post(
                "/api/v1/workstation/scripts/echo/run",
                json={"arguments": {"message": "logged output"}},
            )
        ).json()

        worker = WorkerRunner(settings)
        await worker.start()
        try:
            await wait_for(session, created["run_id"])
        finally:
            await worker.stop()

        body = (await client.get(f"/api/v1/workstation/runs/{created['run_id']}/log")).json()
        assert "logged output" in "\n".join(body["lines"])
        assert body["content_is_untrusted"] is True

    async def test_a_run_that_is_not_running_cannot_be_stopped(
        self, client: AsyncClient, session: AsyncSession, settings: Settings, scripts: Path
    ) -> None:
        from workers.runner import WorkerRunner

        created = (
            await client.post(
                "/api/v1/workstation/scripts/echo/run", json={"arguments": {"message": "x"}}
            )
        ).json()

        worker = WorkerRunner(settings)
        await worker.start()
        try:
            await wait_for(session, created["run_id"])
        finally:
            await worker.stop()

        response = await client.post(f"/api/v1/workstation/runs/{created['run_id']}/stop")
        assert response.status_code == 422


class TestApprovalGating:
    async def test_a_gated_script_only_creates_a_ticket(
        self, client: AsyncClient, session: AsyncSession, scripts: Path
    ) -> None:
        response = await client.post(
            "/api/v1/workstation/scripts/gated/run",
            json={"arguments": {"message": "careful"}},
        )

        assert response.status_code == 202
        assert response.json()["approval_id"]
        assert (await session.scalars(select(ScriptRun))).first() is None

    async def test_the_prompt_shows_the_real_command(
        self, client: AsyncClient, scripts: Path
    ) -> None:
        """A confirmation that hides what will run is not a confirmation."""
        body = (
            await client.post(
                "/api/v1/workstation/scripts/gated/run",
                json={"arguments": {"message": "careful"}},
            )
        ).json()

        assert "echo_argv.py" in body["preview"]
        assert "careful" in body["preview"]

    async def test_approving_queues_the_run(
        self,
        client: AsyncClient,
        session: AsyncSession,
        settings: Settings,
        scripts: Path,
    ) -> None:
        ticket = (
            await client.post(
                "/api/v1/workstation/scripts/gated/run",
                json={"arguments": {"message": "approved"}},
            )
        ).json()
        approval = await session.get(Approval, ticket["approval_id"])
        assert approval is not None

        await resolve_by_token(
            session, approval.callback_token, approve=True, chat_id=None, settings=settings
        )
        await session.commit()

        runs = list(await session.scalars(select(ScriptRun)))
        assert len(runs) == 1
        assert runs[0].script_name == "gated"

    async def test_approving_twice_queues_one_run(
        self,
        client: AsyncClient,
        session: AsyncSession,
        settings: Settings,
        scripts: Path,
    ) -> None:
        ticket = (
            await client.post(
                "/api/v1/workstation/scripts/gated/run",
                json={"arguments": {"message": "once"}},
            )
        ).json()
        approval = await session.get(Approval, ticket["approval_id"])
        assert approval is not None

        for _ in range(3):
            await resolve_by_token(
                session,
                approval.callback_token,
                approve=True,
                chat_id=None,
                settings=settings,
            )
        await session.commit()

        assert len(list(await session.scalars(select(ScriptRun)))) == 1

    async def test_declining_runs_nothing(
        self,
        client: AsyncClient,
        session: AsyncSession,
        settings: Settings,
        scripts: Path,
    ) -> None:
        ticket = (
            await client.post(
                "/api/v1/workstation/scripts/gated/run",
                json={"arguments": {"message": "no thanks"}},
            )
        ).json()
        approval = await session.get(Approval, ticket["approval_id"])
        assert approval is not None

        await resolve_by_token(
            session, approval.callback_token, approve=False, chat_id=None, settings=settings
        )
        await session.commit()

        assert (await session.scalars(select(ScriptRun))).first() is None


class TestRecovery:
    async def test_a_run_orphaned_by_a_restart_is_closed_out(self, session: AsyncSession) -> None:
        """A record still claiming to run after a restart is a lie."""
        from workstation.runner import reconcile_orphans

        session.add(
            ScriptRun(
                script_name="echo",
                arguments_json="{}",
                status=ScriptRunStatus.RUNNING.value,
            )
        )
        await session.flush()

        assert await reconcile_orphans(session) == 1

        run = (await session.scalars(select(ScriptRun))).one()
        assert run.status == ScriptRunStatus.FAILED.value
        assert "restarted" in (run.error or "")


class TestAuthentication:
    async def test_status_needs_a_token(self, anonymous_client: AsyncClient) -> None:
        assert (await anonymous_client.get("/api/v1/workstation/status")).status_code == 401

    async def test_running_needs_a_token(self, anonymous_client: AsyncClient) -> None:
        response = await anonymous_client.post(
            "/api/v1/workstation/scripts/echo/run", json={"arguments": {}}
        )
        assert response.status_code == 401


class TestTaskRegistration:
    def test_the_run_handler_is_not_retried(self) -> None:
        """Half of a script's effect applied twice is worse than none."""
        from workers.registry import get_handler

        spec = get_handler(RUN_SCRIPT_TASK)
        assert spec.retryable is False
        assert spec.max_attempts == 1


def _names(payload: dict[str, Any]) -> set[str]:
    return {script["name"] for script in payload["scripts"]}
