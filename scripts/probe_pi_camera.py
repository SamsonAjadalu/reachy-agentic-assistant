#!/usr/bin/env python3
"""Probe the Reachy Pi camera HTTP surface without printing secrets.

Exit 0 when a camera status or frame endpoint answers. Tokens stay out of logs.
"""

from __future__ import annotations

import os
import sys
from urllib.parse import urlparse

import httpx

DEFAULT_URL = "http://reachy-mini.local:7861"


def _redacted(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or "unknown"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return f"{parsed.scheme}://{host}:{port}"


def main() -> int:
    base = (os.environ.get("REACHY_CAMERA_URL") or DEFAULT_URL).rstrip("/")
    vision_token = os.environ.get("REACHY_VISION_TOKEN") or ""
    camera_token = os.environ.get("REACHY_CAMERA_TOKEN") or ""
    text_token = os.environ.get("REACHY_TEXT_TURN_TOKEN") or ""
    token = (vision_token or camera_token or text_token).strip()
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer <redacted>"
        send_headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    else:
        send_headers = headers
    status_path = os.environ.get("REACHY_CAMERA_STATUS_PATH") or "/api/v1/camera/status"
    print(f"probing {_redacted(base)}{status_path} token={'set' if token else 'unset'}")
    try:
        with httpx.Client(timeout=5.0, follow_redirects=False) as client:
            response = client.get(f"{base}{status_path}", headers=send_headers)
    except httpx.HTTPError as exc:
        print(f"unreachable: {type(exc).__name__}")
        return 2
    print(f"status_http={response.status_code}")
    if response.status_code == 401:
        print("auth_required")
        return 3
    if response.status_code >= 400:
        print("endpoint_not_ready")
        return 4
    print("camera_status_ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
