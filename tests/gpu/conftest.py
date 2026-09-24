"""Triple-gated GPU tests. Skipped unless VISION_GPU_TESTS=1, nvidia-smi, and torch+CUDA."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest


def _gpu_env_enabled() -> bool:
    return os.environ.get("VISION_GPU_TESTS") == "1"


def _nvidia_smi_ok() -> bool:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False
    try:
        completed = subprocess.run(
            [exe, "-L"], check=False, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and "GPU" in completed.stdout


def _torch_cuda_ok() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not _gpu_env_enabled():
        skip = pytest.mark.skip(reason="GPU tests require VISION_GPU_TESTS=1")
    elif not _nvidia_smi_ok():
        skip = pytest.mark.skip(reason="GPU tests require nvidia-smi to enumerate a device")
    elif not _torch_cuda_ok():
        skip = pytest.mark.skip(reason="GPU tests require torch with CUDA")
    else:
        return
    for item in items:
        path = str(getattr(item, "path", "") or item.fspath)
        if "tests/gpu/" not in path.replace("\\", "/"):
            continue
        item.add_marker(skip)
