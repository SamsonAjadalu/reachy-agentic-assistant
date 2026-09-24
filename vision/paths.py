"""Content-addressed evidence paths under APP_DATA_DIR/visual."""

from __future__ import annotations

from pathlib import Path

from vision.enums import EvidenceKind

HEX64 = 64


def assert_content_hash(content_hash: str) -> str:
    digest = content_hash.strip().lower()
    if len(digest) != HEX64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("content_hash must be a 64-character lowercase hex SHA-256 digest.")
    return digest


def evidence_relative_path(kind: EvidenceKind | str, content_hash: str) -> str:
    digest = assert_content_hash(content_hash)
    kind_value = kind.value if isinstance(kind, EvidenceKind) else kind
    return f"{kind_value}/{digest[:2]}/{digest[2:4]}/{digest}.jpg"


def evidence_absolute_path(root: Path, kind: EvidenceKind | str, content_hash: str) -> Path:
    return root / evidence_relative_path(kind, content_hash)


def visual_evidence_root(app_data_dir: Path) -> Path:
    return app_data_dir / "visual" / "evidence"


def visual_quarantine_root(app_data_dir: Path) -> Path:
    return app_data_dir / "visual" / "quarantine"
