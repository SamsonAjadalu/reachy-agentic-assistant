"""Uses the configured workflow."""

from __future__ import annotations

from pathlib import Path

from vision_sidecar.auth import require_token

REPO = Path(__file__).resolve().parents[2]


def test_auth_uses_constant_time_compare() -> None:
    text = (REPO / "vision_sidecar" / "auth.py").read_text(encoding="utf-8")
    assert "compare_digest" in text
    assert "credentials.credentials" in text
    assert require_token.__name__ == "require_token"


def test_auth_source_never_logs_the_presented_token() -> None:
    text = (REPO / "vision_sidecar" / "auth.py").read_text(encoding="utf-8")
    assert 'extra={"token"' not in text
    assert "presented" not in text.split("logger.warning")[-1]
