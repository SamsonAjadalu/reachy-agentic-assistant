"""Catch systemd unit templates that would not expand on real systemd."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SYSTEMD_DIR = REPO_ROOT / "deployment" / "main_pc" / "systemd"

# systemd supports ${VAR} / $VAR but not bash ${VAR:-default} / ${VAR-default}.
_BASH_DEFAULT = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*[:-][^}]+\}")


def test_unit_files_avoid_bash_parameter_defaults() -> None:
    units = sorted(SYSTEMD_DIR.glob("*.service")) + sorted(SYSTEMD_DIR.glob("*.timer"))
    assert units, "expected systemd unit templates under deployment/main_pc/systemd"
    offenders: list[str] = []
    for path in units:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for match in _BASH_DEFAULT.finditer(line):
                offenders.append(f"{path.name}:{line_no}: {match.group(0)}")
    assert not offenders, (
        "systemd does not expand bash ${VAR:-default} forms; use Environment= defaults "
        f"and plain ${{VAR}} instead. Found: {offenders}"
    )


def test_api_unit_sets_host_and_port_defaults_before_env_file() -> None:
    text = (SYSTEMD_DIR / "reachy-personal-assistant-api.service").read_text(encoding="utf-8")
    host_pos = text.index("Environment=APP_HOST=")
    port_pos = text.index("Environment=APP_PORT=")
    env_file_pos = text.index("EnvironmentFile=")
    assert host_pos < env_file_pos
    assert port_pos < env_file_pos
    assert "--host ${APP_HOST}" in text
    assert "--port ${APP_PORT}" in text
