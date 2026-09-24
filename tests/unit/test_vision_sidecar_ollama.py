"""Uses the configured workflow."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from vision_sidecar.bakeoff import _maybe_gpu_bench
from vision_sidecar.config import SidecarSettings
from vision_sidecar.ollama import OllamaState, maybe_stop_for_bakeoff


def test_busy_state_does_not_stop_qwen() -> None:
    state = OllamaState(
        models=[{"name": "qwen3.8:27b"}],
        busy=True,
        busy_reason="llama-server pid=18996 cpu=122%",
        raw_ps="qwen3.8:27b",
    )
    result = maybe_stop_for_bakeoff(state)
    assert result["stopped"] is False
    assert "122" in result["reason"]


def test_gpu_bench_skips_when_busy(tmp_path: Path) -> None:
    settings = SidecarSettings(
        app_env="test",
        app_data_dir=tmp_path / "pa-data",
        vision_sidecar_token="a" * 48,
        mock_mode=True,
    )
    state = OllamaState(models=[{"name": "qwen3.8:27b"}], busy=True, busy_reason="cpu")
    with patch("vision_sidecar.bakeoff.read_ollama_state", return_value=state):
        report = _maybe_gpu_bench(settings, allow_download=True)
    assert report["ran"] is False
    assert "busy" in report["reason"]
