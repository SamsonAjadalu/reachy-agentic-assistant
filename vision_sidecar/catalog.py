"""Upstream model catalog. Versions and licences recorded before any download."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

# Repo commit SHAs and licences verified against Hugging Face on 2026-09-02
# (GET https://huggingface.co/api/models/{id}). Weight sha256 is filled in only
# after a local download; it is left empty until then.


@dataclass(frozen=True)
class ModelRecord:
    purpose: str
    hf_id: str
    repo_sha: str
    licence: str
    commercial_ok: bool
    weight_filename: str
    weight_bytes: int
    notes: str
    default: bool = True
    weight_sha256: str | None = None


CATALOG: dict[str, ModelRecord] = {
    "grounding_dino": ModelRecord(
        purpose="open-vocab detection",
        hf_id="IDEA-Research/grounding-dino-tiny",
        repo_sha="a2bb814dd30d776dcf7e30523b00659f4f141c71",
        licence="apache-2.0",
        commercial_ok=True,
        weight_filename="model.safetensors",
        weight_bytes=689_359_096,
        notes="Transformers AutoModelForZeroShotObjectDetection. Envelope-A serial load.",
    ),
    "yolo_world": ModelRecord(
        purpose="low-latency open-vocab detection",
        hf_id="wondervictor/YOLO-World-V2.1",
        repo_sha="",
        licence="gpl-3.0",
        commercial_ok=False,
        weight_filename="yolov8s-worldv2.pt",
        weight_bytes=0,
        notes=(
            "AILab-CVC/YOLO-World is GPL-3.0; Ultralytics-distributed variants are AGPL-3.0. "
            "Disabled by default until a licence decision. Wrapper loads ultralytics YOLOWorld "
            "only when explicitly enabled."
        ),
        default=False,
    ),
    "sam2": ModelRecord(
        purpose="prompted segmentation",
        hf_id="facebook/sam2.1-hiera-tiny",
        repo_sha="de431c4043854a71d8101e17995dfe596bf101a5",
        licence="apache-2.0",
        commercial_ok=True,
        weight_filename="model.safetensors",
        weight_bytes=155_908_064,
        notes="SAM 2.1 Hiera-Tiny via transformers Sam2Model. Apache-2.0.",
    ),
    "da3mono_large": ModelRecord(
        purpose="relative monocular depth",
        hf_id="depth-anything/DA3MONO-LARGE",
        repo_sha="f465978e618db8cc79c83b8bbf24964857db1875",
        licence="apache-2.0",
        commercial_ok=True,
        weight_filename="model.safetensors",
        weight_bytes=1_336_734_448,
        notes=(
            "ByteDance-Seed Depth Anything 3 monocular-large. Direct relative depth + confidence. "
            "DA3-LARGE any-view family is CC BY-NC and uses its catalogued weights."
        ),
    ),
    "dinov2": ModelRecord(
        purpose="instance and scene embeddings",
        hf_id="facebook/dinov2-small",
        repo_sha="ed25f3a31f01632728cabb09d1542f84ab7b0056",
        licence="apache-2.0",
        commercial_ok=True,
        weight_filename="model.safetensors",
        weight_bytes=88_249_960,
        notes="384-d CLS token. Cross-version comparison with dinov2-base is refused.",
    ),
}


HF_API = "https://huggingface.co/api/models/{id}"


def verify_upstream(record: ModelRecord, *, timeout: float = 20.0) -> dict[str, Any]:
    """Fetch the Hub card. Does not download weights."""
    url = HF_API.format(id=record.hf_id)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        payload = response.json()
    card = payload.get("cardData") or {}
    licence = (card.get("license") or payload.get("license") or "").lower()
    sha = payload.get("sha") or ""
    return {
        "hf_id": payload.get("id") or record.hf_id,
        "repo_sha": sha,
        "licence": licence,
        "gated": bool(payload.get("gated")),
        "private": bool(payload.get("private")),
        "sha_matches_catalog": sha == record.repo_sha if record.repo_sha else None,
        "licence_matches_catalog": licence == record.licence if licence else None,
        "last_modified": payload.get("lastModified"),
    }


def verify_catalog(*, timeout: float = 20.0) -> dict[str, Any]:
    results = []
    for key, record in CATALOG.items():
        if not record.hf_id or key == "yolo_world":
            results.append(
                {"key": key, "skipped": True, "reason": "no huggingface card or licence-blocked"}
            )
            continue
        try:
            info = verify_upstream(record, timeout=timeout)
            info["key"] = key
            info["ok"] = True
            results.append(info)
        except Exception as exc:
            results.append({"key": key, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return {"results": results}
