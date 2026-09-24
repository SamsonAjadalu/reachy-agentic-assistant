"""Bounded bake-off: verify upstream cards, optionally bench GPU if ollama is idle."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_sidecar import __version__
from vision_sidecar.catalog import CATALOG, verify_catalog
from vision_sidecar.config import SidecarSettings
from vision_sidecar.metrics import snapshot_as_dict, snapshot_resources
from vision_sidecar.ollama import maybe_stop_for_bakeoff, read_ollama_state, restore_ollama


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify perception models; GPU bench only if ollama is idle"
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = SidecarSettings()
    if args.allow_download:
        settings.vision_allow_downloads = True
    settings.ensure_directories()
    settings.apply_hf_env()

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "sidecar_version": __version__,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "hf_home": str(settings.hf_cache_dir),
        "allow_download": args.allow_download,
        "upstream": verify_catalog(),
        "catalog_pins": {
            key: {
                "hf_id": rec.hf_id,
                "repo_sha": rec.repo_sha,
                "licence": rec.licence,
                "commercial_ok": rec.commercial_ok,
                "weight_sha256": rec.weight_sha256,
            }
            for key, rec in CATALOG.items()
        },
        "gpu": snapshot_as_dict(snapshot_resources()),
        "models_downloaded": [],
        "gpu_bench": {"ran": False, "reason": "not requested"},
    }

    state = read_ollama_state()
    report["ollama"] = {
        "models": state.models,
        "busy": state.busy,
        "busy_reason": state.busy_reason,
        "raw_ps": state.raw_ps,
    }
    if args.gpu:
        report["gpu_bench"] = _maybe_gpu_bench(settings, allow_download=args.allow_download)

    text = json.dumps(report, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _maybe_gpu_bench(settings: SidecarSettings, *, allow_download: bool) -> dict[str, Any]:
    state = read_ollama_state()
    if state.busy:
        return {
            "ran": False,
            "reason": f"ollama busy ({state.busy_reason}); skipped live GPU bench",
            "prior_models": [item.get("name") for item in state.models],
        }
    if not allow_download:
        snap = snapshot_resources()
        gpu0 = snap.gpus[0].to_dict() if snap.gpus else None
        return {
            "ran": False,
            "reason": (
                "ollama idle, but downloads are disabled; not unloading qwen "
                "without a real model load"
            ),
            "free_mib": gpu0.get("memory_free_mib") if gpu0 else None,
            "gpu": gpu0,
            "prior_models": [item.get("name") for item in state.models],
        }
    decision = maybe_stop_for_bakeoff(state)
    restore: dict[str, Any] = {}
    try:
        snap = snapshot_resources()
        gpu0 = snap.gpus[0].to_dict() if snap.gpus else None
        return {
            "ran": bool(decision.get("stopped")),
            "reason": decision.get("reason") or "bounded ollama unload window",
            "free_mib": gpu0.get("memory_free_mib") if gpu0 else None,
            "gpu": gpu0,
            "allow_download": allow_download,
            "hf_home": str(settings.hf_cache_dir),
            "models_loaded": False,
            "note": "Weights load on demand through the real wrappers selected by the catalog.",
            "restore": restore,
            "decision": {k: v for k, v in decision.items() if k != "restore"},
        }
    finally:
        model = decision.get("model")
        if decision.get("stopped") and isinstance(model, str):
            restore.update(restore_ollama(model))


if __name__ == "__main__":
    raise SystemExit(main())
