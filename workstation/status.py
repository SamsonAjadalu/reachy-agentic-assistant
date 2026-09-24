"""What this machine is doing.

Reads are cheap and local, so they answer synchronously. GPU figures come from
nvidia-smi when it is present; its absence is reported as "no GPU visible"
rather than as an error, because the same code runs on machines without one.
"""

from __future__ import annotations

import asyncio
import shutil
from datetime import UTC, datetime
from typing import Any

import psutil
from pydantic import BaseModel, Field

from app.config import Settings
from app.logging_config import get_logger
from shared.errors import ValidationError
from shared.timeutils import utcnow

logger = get_logger(__name__)

NVIDIA_SMI_TIMEOUT = 5
NVIDIA_QUERY = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu"


class DiskUsage(BaseModel):
    mount: str
    total_gb: float
    used_gb: float
    free_gb: float
    percent_used: float


class GpuStatus(BaseModel):
    index: int
    name: str
    utilisation_percent: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float | None = None


class WorkstationStatus(BaseModel):
    hostname: str
    checked_at: datetime
    uptime_seconds: int
    cpu_percent: float
    cpu_count: int
    load_average: list[float] = Field(default_factory=list)
    memory_total_gb: float
    memory_used_gb: float
    memory_percent: float
    swap_percent: float
    disks: list[DiskUsage] = Field(default_factory=list)
    gpus: list[GpuStatus] = Field(default_factory=list)
    gpu_note: str | None = None
    warnings: list[str] = Field(default_factory=list)


def _gb(value: float) -> float:
    return round(value / (1024**3), 2)


async def collect_status(settings: Settings) -> WorkstationStatus:
    """Snapshot CPU, memory, disk and GPU."""
    # psutil's blocking sample would stall the event loop for the interval, so
    # the whole snapshot runs on a thread.
    snapshot = await asyncio.to_thread(_collect_sync, settings)
    gpus, note = await _collect_gpus()
    snapshot.gpus = gpus
    snapshot.gpu_note = note
    snapshot.warnings.extend(_warn_about_gpus(gpus))
    return snapshot


def _collect_sync(settings: Settings) -> WorkstationStatus:
    import socket

    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    boot = datetime.fromtimestamp(psutil.boot_time(), tz=UTC)

    disks: list[DiskUsage] = []
    for mount in _mounts_of_interest(settings):
        try:
            usage = psutil.disk_usage(mount)
        except OSError:
            continue
        disks.append(
            DiskUsage(
                mount=mount,
                total_gb=_gb(usage.total),
                used_gb=_gb(usage.used),
                free_gb=_gb(usage.free),
                percent_used=usage.percent,
            )
        )

    warnings: list[str] = []
    for disk in disks:
        if disk.percent_used >= 90:
            warnings.append(f"{disk.mount} is {disk.percent_used:.0f}% full.")
    if virtual.percent >= 90:
        warnings.append(f"Memory is {virtual.percent:.0f}% used.")

    return WorkstationStatus(
        hostname=socket.gethostname(),
        checked_at=utcnow(),
        uptime_seconds=int((utcnow() - boot).total_seconds()),
        cpu_percent=psutil.cpu_percent(interval=0.2),
        cpu_count=psutil.cpu_count(logical=True) or 0,
        load_average=[round(value, 2) for value in psutil.getloadavg()],
        memory_total_gb=_gb(virtual.total),
        memory_used_gb=_gb(virtual.used),
        memory_percent=virtual.percent,
        swap_percent=swap.percent,
        disks=disks,
        warnings=warnings,
    )


def _mounts_of_interest(settings: Settings) -> list[str]:
    """Root, plus wherever the assistant's own data lives.

    Reporting every mount would include snap loopbacks and container overlays,
    which is noise in a spoken summary.
    """
    candidates = ["/", str(settings.app_data_dir)]
    for extra in (settings.wardrobe_image_root, settings.backup_root):
        if extra is not None:
            candidates.append(str(extra))

    seen: dict[str, None] = {}
    for candidate in candidates:
        try:
            mount = _mount_point(candidate)
        except OSError:
            continue
        seen.setdefault(mount, None)
    return list(seen)


def _mount_point(path: str) -> str:
    import os

    current = os.path.abspath(path)
    while not os.path.ismount(current):
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return current


async def _collect_gpus() -> tuple[list[GpuStatus], str | None]:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return [], "nvidia-smi is not installed; no GPU information available."

    try:
        process = await asyncio.create_subprocess_exec(
            binary,
            f"--query-gpu={NVIDIA_QUERY}",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=NVIDIA_SMI_TIMEOUT)
    except TimeoutError:
        return [], "nvidia-smi did not respond in time."
    except OSError as exc:
        return [], f"nvidia-smi could not be run: {exc}"

    if process.returncode != 0:
        detail = stderr.decode("utf-8", "replace").strip().splitlines()
        return [], detail[0] if detail else "nvidia-smi reported an error."

    return _parse_nvidia_smi(stdout.decode("utf-8", "replace")), None


def _parse_nvidia_smi(output: str) -> list[GpuStatus]:
    gpus: list[GpuStatus] = []
    for line in output.strip().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 5:
            continue
        try:
            gpus.append(
                GpuStatus(
                    index=int(fields[0]),
                    name=fields[1],
                    utilisation_percent=float(fields[2]),
                    memory_used_mb=float(fields[3]),
                    memory_total_mb=float(fields[4]),
                    temperature_c=float(fields[5]) if len(fields) > 5 and fields[5] else None,
                )
            )
        except ValueError:
            # A driver in a bad state prints "[N/A]" in some columns.
            continue
    return gpus


def _warn_about_gpus(gpus: list[GpuStatus]) -> list[str]:
    warnings: list[str] = []
    for gpu in gpus:
        if gpu.memory_total_mb and gpu.memory_used_mb / gpu.memory_total_mb >= 0.95:
            warnings.append(f"GPU {gpu.index} memory is nearly full.")
        if gpu.temperature_c is not None and gpu.temperature_c >= 85:
            warnings.append(f"GPU {gpu.index} is running at {gpu.temperature_c:.0f} degrees.")
    return warnings


# ------------------------------------------------------------------ services
class ServiceStatus(BaseModel):
    name: str
    scope: str
    active_state: str
    sub_state: str | None = None
    enabled: bool | None = None
    since: str | None = None
    detail: str | None = None


async def service_status(name: str, settings: Settings) -> ServiceStatus:
    """Report on one allowlisted systemd unit.

    Only units named in WORKSTATION_ALLOWED_SERVICES can be queried, and only
    queried: starting and stopping services is not something this assistant does.
    """
    allowed = [entry.strip() for entry in settings.workstation_allowed_services if entry.strip()]
    if name not in allowed:
        raise ValidationError(
            f"'{name}' is not in WORKSTATION_ALLOWED_SERVICES, so its status cannot be read."
        )

    scope = "user" if settings.workstation_service_scope == "user" else "system"
    argv = ["systemctl"]
    if scope == "user":
        argv.append("--user")
    argv.extend(["show", name, "--no-page", "--property=ActiveState,SubState,UnitFileState"])

    stdout, stderr, code = await _run_readonly(argv)
    if code != 0:
        return ServiceStatus(
            name=name,
            scope=scope,
            active_state="unknown",
            detail=(stderr.strip().splitlines() or ["systemctl failed"])[0],
        )

    values: dict[str, str] = {}
    for line in stdout.splitlines():
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()

    unit_file_state = values.get("UnitFileState", "")
    return ServiceStatus(
        name=name,
        scope=scope,
        active_state=values.get("ActiveState", "unknown"),
        sub_state=values.get("SubState") or None,
        enabled=unit_file_state == "enabled" if unit_file_state else None,
    )


async def all_service_statuses(settings: Settings) -> list[ServiceStatus]:
    allowed = [entry.strip() for entry in settings.workstation_allowed_services if entry.strip()]
    return [await service_status(name, settings) for name in allowed]


async def _run_readonly(argv: list[str]) -> tuple[str, str, int]:
    """Run a fixed read-only command with no shell and a short deadline."""
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
    except TimeoutError:
        return "", "The command did not respond in time.", 1
    except OSError as exc:
        return "", str(exc), 1
    return (
        stdout.decode("utf-8", "replace"),
        stderr.decode("utf-8", "replace"),
        process.returncode or 0,
    )


def summarise(status: WorkstationStatus) -> str:
    """One or two sentences, phrased for speech."""
    parts = [
        f"CPU at {status.cpu_percent:.0f} percent",
        f"memory at {status.memory_percent:.0f} percent",
    ]
    if status.gpus:
        busiest = max(status.gpus, key=lambda gpu: gpu.utilisation_percent)
        parts.append(f"GPU {busiest.index} at {busiest.utilisation_percent:.0f} percent")
    root = next((disk for disk in status.disks if disk.mount == "/"), None)
    if root:
        parts.append(f"{root.free_gb:.0f} gigabytes free on the root disk")

    sentence = ", ".join(parts) + "."
    if status.warnings:
        sentence += " " + " ".join(status.warnings)
    return sentence


def as_dict(status: WorkstationStatus) -> dict[str, Any]:
    return status.model_dump(mode="json")
