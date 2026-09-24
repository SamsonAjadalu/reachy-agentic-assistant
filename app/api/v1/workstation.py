"""Workstation endpoints.

Status and service checks are reads and answer immediately. Running a script
does not: it returns a run id, and the process continues in the background,
because a registered script may legitimately take half an hour.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.dependencies.auth import Principal, require_token
from app.dependencies.common import get_session, idempotency_key
from app.schemas.common import ApiModel
from app.schemas.google import ApprovalTicket
from app.services import workstation as service
from approvals.state_machine import request_approval
from database.models import ScriptRun
from shared.enums import ScriptRunStatus
from shared.errors import ValidationError
from workstation import runner
from workstation.executors import RUN_SCRIPT
from workstation.status import (
    ServiceStatus,
    WorkstationStatus,
    all_service_statuses,
    collect_status,
    service_status,
    summarise,
)

router = APIRouter(prefix="/api/v1/workstation", tags=["workstation"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
PrincipalDep = Annotated[Principal, Depends(require_token)]
IdempotencyDep = Annotated[str | None, Depends(idempotency_key)]


class StatusResponse(ApiModel):
    status: WorkstationStatus
    spoken_summary: str


class ParameterOut(ApiModel):
    name: str
    type: str
    required: bool
    choices: list[str] = Field(default_factory=list)
    flag: str | None = None
    description: str = ""


class ScriptOut(ApiModel):
    name: str
    description: str
    parameters: list[ParameterOut] = Field(default_factory=list)
    risk_level: str
    requires_approval: bool
    timeout_seconds: int
    enabled: bool


class ScriptListResponse(ApiModel):
    scripts: list[ScriptOut]
    total: int
    rejected: dict[str, str] = Field(default_factory=dict)
    registry_path: str


class RunRequest(ApiModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


class RunOut(ApiModel):
    id: str
    script_name: str
    status: str
    argv: list[str] = Field(default_factory=list)
    arguments: dict[str, Any] = Field(default_factory=dict)
    exit_code: int | None = None
    error: str | None = None
    stdout_tail: str | None = None
    stderr_tail: str | None = None
    started_at: Any = None
    finished_at: Any = None
    output_is_untrusted: bool = True


class RunListResponse(ApiModel):
    runs: list[RunOut]
    total: int


class RunAccepted(ApiModel):
    run_id: str
    script: str
    status: str
    poll_url: str


# ------------------------------------------------------------------- status
@router.get("/status", response_model=StatusResponse, summary="How this machine is doing")
async def status(_: PrincipalDep, settings: SettingsDep) -> StatusResponse:
    snapshot = await collect_status(settings)
    return StatusResponse(status=snapshot, spoken_summary=summarise(snapshot))


@router.get(
    "/services",
    response_model=list[ServiceStatus],
    summary="Allowlisted service states",
    description="Only units listed in WORKSTATION_ALLOWED_SERVICES are visible, and only read.",
)
async def services(_: PrincipalDep, settings: SettingsDep) -> list[ServiceStatus]:
    return await all_service_statuses(settings)


@router.get("/services/{name}", response_model=ServiceStatus, summary="One service")
async def one_service(name: str, _: PrincipalDep, settings: SettingsDep) -> ServiceStatus:
    return await service_status(name, settings)


# ------------------------------------------------------------------ scripts
@router.get(
    "/scripts",
    response_model=ScriptListResponse,
    summary="Registered scripts",
    description=(
        "The registry is a YAML file on disk. Entries that failed validation are reported "
        "under 'rejected' rather than silently omitted."
    ),
)
async def list_scripts(_: PrincipalDep, settings: SettingsDep) -> ScriptListResponse:
    registry = service.get_registry(settings)
    scripts = [
        ScriptOut(
            name=script.name,
            description=script.description,
            parameters=[
                ParameterOut(
                    name=parameter.name,
                    type=parameter.type,
                    required=parameter.required,
                    choices=list(parameter.choices),
                    flag=parameter.flag,
                    description=parameter.description,
                )
                for parameter in script.parameters
            ],
            risk_level=script.risk_level.value,
            requires_approval=script.requires_approval,
            timeout_seconds=script.timeout_seconds,
            enabled=script.enabled,
        )
        for script in sorted(registry.scripts.values(), key=lambda item: item.name)
    ]
    return ScriptListResponse(
        scripts=scripts,
        total=len(scripts),
        rejected=dict(registry.errors),
        registry_path=str(registry.path),
    )


@router.post("/scripts/reload", summary="Re-read the registry file")
async def reload_scripts(
    _: PrincipalDep, session: SessionDep, settings: SettingsDep
) -> dict[str, int]:
    service._cached_registry.cache_clear()
    result = await service.sync_registry(session, settings)
    await session.commit()
    return result


@router.post(
    "/scripts/{name}/run",
    status_code=202,
    summary="Run a registered script",
    description=(
        "Returns an approval ticket when the script requires approval, otherwise a run id. "
        "Arguments are validated against the registry and passed as argv; no shell is used."
    ),
)
async def run_script(
    name: str,
    payload: RunRequest,
    request: Request,
    _: PrincipalDep,
    session: SessionDep,
    settings: SettingsDep,
    key: IdempotencyDep,
) -> ApprovalTicket | RunAccepted:
    registry = service.get_registry(settings)
    script = registry.get(name)

    # Building argv now means a bad argument is refused before an approval is
    # raised, and the owner sees the real command in the prompt.
    argv = script.build_argv(dict(payload.arguments))

    if script.requires_approval:
        action, approval = await request_approval(
            session,
            action_type=RUN_SCRIPT,
            summary=f"Run the '{script.name}' script",
            payload={"script": script.name, "arguments": payload.arguments},
            preview_text=(
                f"{script.description or script.name}\n\n"
                f"Command: {' '.join(argv)}\n"
                f"Timeout: {script.timeout_seconds} seconds"
            ),
            risk_level=script.risk_level,
            settings=settings,
            idempotency_key=key,
            correlation_id=getattr(request.state, "correlation_id", None),
        )
        await session.commit()
        return ApprovalTicket(
            approval_id=approval.id,
            action_id=action.id,
            summary=action.summary,
            preview=action.preview_text,
            expires_at=approval.expires_at,
        )

    run = await service.queue_run(
        session,
        script,
        dict(payload.arguments),
        settings=settings,
        correlation_id=getattr(request.state, "correlation_id", None),
    )
    await session.commit()
    return RunAccepted(
        run_id=run.id,
        script=script.name,
        status=run.status,
        poll_url=f"/api/v1/workstation/runs/{run.id}",
    )


# --------------------------------------------------------------------- runs
@router.get("/runs", response_model=RunListResponse, summary="Script run history")
async def list_runs(
    _: PrincipalDep,
    session: SessionDep,
    script: Annotated[str | None, Query(max_length=80)] = None,
    status_filter: Annotated[str | None, Query(alias="status", max_length=30)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RunListResponse:
    runs, total = await service.list_runs(
        session, script_name=script, status=status_filter, limit=limit, offset=offset
    )
    return RunListResponse(runs=[_run_out(run) for run in runs], total=total)


@router.get("/runs/{run_id}", response_model=RunOut, summary="One run")
async def get_run(run_id: str, _: PrincipalDep, session: SessionDep) -> RunOut:
    return _run_out(await service.get_run(session, run_id))


@router.get(
    "/runs/{run_id}/log",
    summary="A run's captured output",
    description="Bounded by WORKSTATION_MAX_OUTPUT_BYTES at capture time. Output is untrusted.",
)
async def run_log(
    run_id: str,
    _: PrincipalDep,
    session: SessionDep,
    lines: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> dict[str, Any]:
    run = await service.get_run(session, run_id)
    return {
        "run_id": run.id,
        "script": run.script_name,
        "lines": runner.read_log(run, max_lines=lines),
        "content_is_untrusted": True,
    }


@router.post("/runs/{run_id}/stop", response_model=RunOut, summary="Stop a running script")
async def stop_run(run_id: str, _: PrincipalDep, session: SessionDep) -> RunOut:
    run = await service.get_run(session, run_id)
    if run.status not in {ScriptRunStatus.RUNNING.value, ScriptRunStatus.STARTING.value}:
        raise ValidationError(f"Run {run_id} is {run.status}; there is nothing to stop.")

    stopped = await runner.stop_run(run_id)
    if not stopped:
        # The process is gone but the record still says running, which happens
        # if the service restarted between the two.
        run = await service.mark_stopped(session, run)
        run.error = "The process was no longer running; the record has been closed."
    await session.commit()
    return _run_out(await service.get_run(session, run_id))


def _run_out(run: ScriptRun) -> RunOut:
    return RunOut(
        id=run.id,
        script_name=run.script_name,
        status=run.status,
        argv=json.loads(run.resolved_argv_json or "[]"),
        arguments=json.loads(run.arguments_json or "{}"),
        exit_code=run.exit_code,
        error=run.error,
        stdout_tail=run.stdout_tail,
        stderr_tail=run.stderr_tail,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )
