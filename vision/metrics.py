"""Disk and budget metrics for visual evidence."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from app.config import AppEnv, Settings
from vision.storage import visual_evidence_root

FREE_BYTES_FLOOR = 20 * 1024 * 1024 * 1024
FREE_RATIO_FLOOR = 0.10


@dataclass(frozen=True)
class VisualDiskMetrics:
    evidence_bytes: int
    evidence_file_count: int
    budget_bytes: int
    budget_used_ratio: float
    volume_path: str
    volume_total_bytes: int
    volume_free_bytes: int
    volume_free_ratio: float
    below_floor: bool
    capture_allowed: bool


def directory_byte_size(root: Path) -> tuple[int, int]:
    total = 0
    count = 0
    if not root.exists():
        return 0, 0
    for path in root.rglob("*"):
        if path.is_file() and not path.name.startswith(".tmp-"):
            total += path.stat().st_size
            count += 1
    return total, count


def collect_disk_metrics(settings: Settings) -> VisualDiskMetrics:
    root = visual_evidence_root(settings)
    evidence_bytes, file_count = directory_byte_size(root)
    usage = shutil.disk_usage(settings.app_data_dir)
    free_ratio = usage.free / usage.total if usage.total else 0.0
    below_floor = usage.free < FREE_BYTES_FLOOR or free_ratio < FREE_RATIO_FLOOR
    budget = settings.visual_evidence_max_bytes
    used_ratio = (evidence_bytes / budget) if budget else 0.0
    capture_allowed = True
    if settings.visual_enabled:
        over_budget = budget > 0 and evidence_bytes >= budget
        if over_budget:
            capture_allowed = False
        if below_floor and settings.app_env is AppEnv.PRODUCTION:
            capture_allowed = False
    return VisualDiskMetrics(
        evidence_bytes=evidence_bytes,
        evidence_file_count=file_count,
        budget_bytes=budget,
        budget_used_ratio=round(used_ratio, 4),
        volume_path=str(settings.app_data_dir),
        volume_total_bytes=usage.total,
        volume_free_bytes=usage.free,
        volume_free_ratio=round(free_ratio, 4),
        below_floor=below_floor,
        capture_allowed=capture_allowed,
    )


def metrics_as_dict(metrics: VisualDiskMetrics) -> dict[str, int | float | str | bool]:
    return {
        "evidence_bytes": metrics.evidence_bytes,
        "evidence_file_count": metrics.evidence_file_count,
        "budget_bytes": metrics.budget_bytes,
        "budget_used_ratio": metrics.budget_used_ratio,
        "volume_path": metrics.volume_path,
        "volume_total_bytes": metrics.volume_total_bytes,
        "volume_free_bytes": metrics.volume_free_bytes,
        "volume_free_ratio": metrics.volume_free_ratio,
        "below_floor": metrics.below_floor,
        "capture_allowed": metrics.capture_allowed,
    }
