"""Uses the configured workflow."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

import psutil

from app.logging_config import get_logger

logger = get_logger("vision_sidecar.metrics")

_NVIDIA_SMI = shutil.which("nvidia-smi")


@dataclass(frozen=True)
class GpuSnapshot:
    index: int
    name: str
    uuid: str | None
    memory_total_mib: float
    memory_used_mib: float
    memory_free_mib: float
    utilization_gpu: int | None
    pci_bus_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "uuid": self.uuid,
            "memory_total_mib": self.memory_total_mib,
            "memory_used_mib": self.memory_used_mib,
            "memory_free_mib": self.memory_free_mib,
            "utilization_gpu": self.utilization_gpu,
            "pci_bus_id": self.pci_bus_id,
        }


@dataclass(frozen=True)
class ComputeApp:
    pid: int
    process_name: str
    used_gpu_memory_mib: float

    @property
    def name(self) -> str:
        return self.process_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "process_name": self.process_name,
            "used_gpu_memory_mib": self.used_gpu_memory_mib,
        }


@dataclass
class ResourceSnapshot:
    cpu_percent: float
    rss_mib: float
    gpu_count: int
    gpus: list[GpuSnapshot] = field(default_factory=list)
    compute_apps: list[ComputeApp] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "rss_mib": self.rss_mib,
            "gpu_count": self.gpu_count,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "compute_apps": [app.to_dict() for app in self.compute_apps],
        }


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def read_gpu() -> GpuSnapshot | None:
    gpus = list_gpus()
    return gpus[0] if gpus else None


def list_gpus() -> list[GpuSnapshot]:
    if _NVIDIA_SMI is None:
        return []
    try:
        completed = subprocess.run(
            [
                _NVIDIA_SMI,
                "--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("nvidia-smi failed", extra={"error": type(exc).__name__})
        return []
    if completed.returncode != 0:
        return []
    snapshots: list[GpuSnapshot] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8:
            continue
        try:
            snapshots.append(
                GpuSnapshot(
                    index=int(parts[0]),
                    name=parts[1],
                    uuid=parts[2],
                    pci_bus_id=parts[3],
                    memory_total_mib=float(parts[4]),
                    memory_used_mib=float(parts[5]),
                    memory_free_mib=float(parts[6]),
                    utilization_gpu=int(float(parts[7])),
                )
            )
        except ValueError:
            continue
    return snapshots


def list_compute_apps() -> list[ComputeApp]:
    if _NVIDIA_SMI is None:
        return []
    try:
        completed = subprocess.run(
            [
                _NVIDIA_SMI,
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    apps: list[ComputeApp] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=2)]
        if len(parts) < 3:
            continue
        try:
            apps.append(
                ComputeApp(
                    pid=int(parts[0]),
                    process_name=parts[1],
                    used_gpu_memory_mib=float(parts[2]),
                )
            )
        except ValueError:
            continue
    return apps


def snapshot_resources() -> ResourceSnapshot:
    gpus = list_gpus()
    return ResourceSnapshot(
        cpu_percent=psutil.cpu_percent(interval=None),
        rss_mib=round(psutil.Process().memory_info().rss / (1024 * 1024), 1),
        gpu_count=len(gpus),
        gpus=gpus,
        compute_apps=list_compute_apps(),
    )


def snapshot_as_dict(snapshot: ResourceSnapshot | dict[str, Any]) -> dict[str, Any]:
    if isinstance(snapshot, ResourceSnapshot):
        return snapshot.to_dict()
    return snapshot


def resource_snapshot() -> dict[str, Any]:
    return snapshot_resources().to_dict()
