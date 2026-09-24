"""Ollama co-tenant policy for a bounded bake-off.

May ``ollama stop`` a resident qwen model only when there is no evidence of
Uses the configured workflow.
``OLLAMA_KEEP_ALIVE`` permanently. Always restores the previous model.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.logging_config import get_logger
from vision_sidecar.metrics import snapshot_resources

logger = get_logger(__name__)

OLLAMA_HOST = "http://127.0.0.1:11434"
BUSY_SM_THRESHOLD = 15
BUSY_SAMPLES = 3
BUSY_SAMPLE_PAUSE_S = 1.0


@dataclass
class OllamaState:
    models: list[dict[str, Any]] = field(default_factory=list)
    busy: bool = False
    busy_reason: str = ""
    raw_ps: str = ""


def read_ollama_state(*, sample_traffic: bool = True) -> OllamaState:
    models: list[dict[str, Any]] = []
    raw_ps = ""
    try:
        raw_ps = subprocess.check_output([_ollama_bin(), "ps"], text=True, timeout=10)
    except Exception as exc:
        return OllamaState(busy=True, busy_reason=f"ollama ps failed: {exc}", raw_ps="")
    try:
        with httpx.Client(timeout=5.0) as client:
            payload = client.get(f"{OLLAMA_HOST}/api/ps").json()
            models = list(payload.get("models") or [])
    except Exception:
        models = []

    busy = False
    reason = ""
    if sample_traffic:
        busy, reason = _traffic_busy()
    return OllamaState(models=models, busy=busy, busy_reason=reason, raw_ps=raw_ps)


def maybe_stop_for_bakeoff(state: OllamaState) -> dict[str, Any]:
    """Unload resident qwen only when idle. Caller must restore."""
    if state.busy:
        return {"stopped": False, "reason": state.busy_reason or "ollama is busy"}
    if not state.models:
        return {"stopped": False, "reason": "no ollama model resident"}
    names = [str(item.get("name") or item.get("model") or "") for item in state.models]
    qwen = next((name for name in names if "qwen" in name.lower()), None)
    if qwen is None:
        return {"stopped": False, "reason": "no qwen model resident", "names": names}
    logger.warning("Temporarily unloading ollama model for bake-off", extra={"model": qwen})
    subprocess.check_call([_ollama_bin(), "stop", qwen], timeout=60)
    return {"stopped": True, "model": qwen, "prior_models": names}


def restore_ollama(model: str, *, keep_alive: str = "-1") -> dict[str, Any]:
    """Reload the previous model. Relies on the existing service KEEP_ALIVE.

    ``keep_alive`` is sent only on this generate call so the service unit is
    not edited. Default ``-1`` matches the measured unit environment.
    """
    with httpx.Client(timeout=180.0) as client:
        response = client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": " ", "keep_alive": keep_alive, "stream": False},
        )
        response.raise_for_status()
    time.sleep(1.0)
    after = read_ollama_state(sample_traffic=False)
    names = [str(item.get("name") or item.get("model") or "") for item in after.models]
    loaded = any(model in name or name in model for name in names) or model in after.raw_ps
    return {"restored": loaded, "model": model, "ps": names, "raw_ps": after.raw_ps}


def _traffic_busy() -> tuple[bool, str]:
    samples: list[int] = []
    for _ in range(BUSY_SAMPLES):
        snap = snapshot_resources()
        util = 0
        for gpu in snap.gpus:
            if gpu.utilization_gpu is not None:
                util = max(util, gpu.utilization_gpu)
        samples.append(util)
        time.sleep(BUSY_SAMPLE_PAUSE_S)
    peak = max(samples) if samples else 0
    mean = sum(samples) / len(samples) if samples else 0
    if peak >= BUSY_SM_THRESHOLD:
        return True, f"GPU SM util peak={peak}% mean={mean:.0f}% over {BUSY_SAMPLES}s"
    cpu_busy, cpu_reason = _llama_cpu_busy()
    if cpu_busy:
        return True, cpu_reason
    return False, ""


def _llama_cpu_busy(*, threshold: float = 10.0) -> tuple[bool, str]:
    """llama-server can be serving while SM util is near zero (decode / KV)."""
    import psutil

    apps = snapshot_resources().compute_apps
    for app in apps:
        name = app.name.lower()
        if "llama" not in name and "ollama" not in name:
            continue
        try:
            proc = psutil.Process(app.pid)
            cpu = proc.cpu_percent(interval=0.4)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if cpu >= threshold:
            return True, f"{app.process_name} pid={app.pid} cpu={cpu:.0f}%"
    return False, ""


def _ollama_bin() -> str:
    path = shutil.which("ollama")
    if path is None:
        raise FileNotFoundError("ollama is not on PATH")
    return path
