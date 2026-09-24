"""Live GPU checks. Triple-gated like tests/pi: env, nvidia-smi, torch+CUDA."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from vision_sidecar.metrics import list_gpus, snapshot_resources

if os.environ.get("VISION_GPU_TESTS") != "1":
    pytest.skip("GPU tests require VISION_GPU_TESTS=1", allow_module_level=True)

_nvidia = shutil.which("nvidia-smi")
if _nvidia is None:
    pytest.skip("GPU tests require nvidia-smi", allow_module_level=True)
try:
    _smi = subprocess.run([_nvidia, "-L"], check=False, capture_output=True, text=True, timeout=5)
except (OSError, subprocess.TimeoutExpired):
    pytest.skip("GPU tests require nvidia-smi to enumerate a device", allow_module_level=True)
else:
    if _smi.returncode != 0 or "GPU" not in _smi.stdout:
        pytest.skip("GPU tests require nvidia-smi to enumerate a device", allow_module_level=True)

try:
    import torch
except ImportError:
    pytest.skip("GPU tests require torch with CUDA", allow_module_level=True)
else:
    if not torch.cuda.is_available():
        pytest.skip("GPU tests require torch with CUDA", allow_module_level=True)


def test_visible_gpu_is_not_claimed_to_be_a_3090() -> None:
    gpus = list_gpus()
    assert gpus, "nvidia-smi enumerated no GPU"
    for gpu in gpus:
        assert "3090" not in gpu.name
        assert gpu.index == 0 or gpu.index >= 0


def test_resource_snapshot_reports_free_vram() -> None:
    snap = snapshot_resources()
    assert snap.gpu_count == len(snap.gpus)
    if snap.gpus:
        assert snap.gpus[0].memory_total_mib > 0
