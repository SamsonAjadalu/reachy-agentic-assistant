#!/usr/bin/env python3
"""Interactive Google OAuth setup.

Run once on the workstation, with a browser available:

    python scripts/setup_google_oauth.py

It opens the consent screen, catches the redirect on a loopback listener, and
writes the refresh token to the encrypted store. The loopback redirect is used
rather than the out-of-band flow because Google retired OOB, and because a code
that never leaves the machine cannot be shoulder-surfed off a terminal.

Nothing here writes a token to the terminal, the shell history or a log.
"""

from __future__ import annotations

import argparse
import asyncio
import http.server
import os
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any, ClassVar

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings  # noqa: E402
from integrations.google.oauth import (  # noqa: E402
    ALL_SCOPES,
    AuthorizationRequest,
    build_authorization_request,
    exchange_code,
    forget_credentials,
    load_credentials,
    save_credentials,
)
from shared.errors import AssistantError  # noqa: E402

SUCCESS_PAGE = b"""<!doctype html>
<html><head><title>Authorised</title></head>
<body style="font-family:system-ui;margin:4rem;max-width:32rem">
<h1>Google access granted</h1>
<p>The refresh token has been stored, encrypted, on the workstation.
You can close this tab and return to the terminal.</p>
</body></html>
"""

FAILURE_PAGE = b"""<!doctype html>
<html><head><title>Not authorised</title></head>
<body style="font-family:system-ui;margin:4rem;max-width:32rem">
<h1>Authorisation failed</h1>
<p>Nothing was stored. The terminal has the details.</p>
</body></html>
"""


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Single-shot handler that captures the authorization code."""

    result: ClassVar[dict[str, str]] = {}
    expected_state: ClassVar[str] = ""

    def do_GET(self) -> None:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        state = (query.get("state") or [""])[0]
        code = (query.get("code") or [""])[0]
        error = (query.get("error") or [""])[0]

        # A mismatched state means this redirect did not come from the request
        # this process started, so the code is not ours to redeem.
        if error or not code or state != type(self).expected_state:
            type(self).result = {"error": error or "state_mismatch"}
            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(FAILURE_PAGE)
            return

        type(self).result = {"code": code}
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(SUCCESS_PAGE)

    def log_message(self, format: str, *args: Any) -> None:
        """Silence the default stderr logging; the URL contains the code."""


def _port_of(redirect_uri: str) -> int:
    parsed = urllib.parse.urlparse(redirect_uri)
    if parsed.hostname not in {"localhost", "127.0.0.1"}:
        raise SystemExit(
            f"GOOGLE_REDIRECT_URI must point at localhost for this flow, got {redirect_uri!r}."
        )
    return parsed.port or 80


def _wait_for_code(request: AuthorizationRequest, timeout: int) -> str:
    port = _port_of(request.redirect_uri)
    _CallbackHandler.expected_state = request.state
    _CallbackHandler.result = {}

    try:
        server = http.server.HTTPServer(("127.0.0.1", port), _CallbackHandler)
    except OSError as exc:
        raise SystemExit(
            f"Cannot listen on 127.0.0.1:{port} ({exc}). Is another setup running?"
        ) from exc

    server.timeout = timeout
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    print("\nOpening the Google consent screen in your browser.")
    print("If it does not open, paste this URL:\n")
    print(request.url + "\n")
    with open(os.devnull, "w") as devnull:  # webbrowser chatters on stderr
        stderr, sys.stderr = sys.stderr, devnull
        try:
            webbrowser.open(request.url)
        finally:
            sys.stderr = stderr

    thread.join(timeout=timeout)
    server.server_close()

    result = _CallbackHandler.result
    if "code" not in result:
        raise SystemExit(
            f"No authorization code received ({result.get('error', 'timed out')}). Nothing stored."
        )
    return result["code"]


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()

    if args.status:
        credentials = load_credentials(settings)
        if credentials is None:
            print("Google is not authorised on this machine.")
            return 1
        print("Google is authorised.")
        print(f"  account : {credentials.account_email or 'unknown'}")
        print(f"  obtained: {credentials.obtained_at}")
        print(f"  scopes  : {len(credentials.scopes)}")
        for scope in sorted(credentials.scopes):
            print(f"            {scope}")
        return 0

    if args.revoke:
        removed = forget_credentials(settings)
        print(
            "Removed the stored Google credentials."
            if removed
            else "There were no stored Google credentials."
        )
        print(
            "Also revoke the grant at https://myaccount.google.com/permissions "
            "if you want Google's copy gone too."
        )
        return 0

    if not settings.google_client_id or not settings.google_client_secret.get_secret_value():
        print(
            "GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET must be set first.\n"
            "docs/setup/google.md walks through creating the OAuth client.",
            file=sys.stderr,
        )
        return 2

    if load_credentials(settings) is not None and not args.force:
        print("Google is already authorised. Re-run with --force to replace the grant.")
        return 0

    request = build_authorization_request(settings, scopes=tuple(args.scopes or ALL_SCOPES))
    code = _wait_for_code(request, args.timeout)

    print("Exchanging the authorization code...")
    credentials = await exchange_code(code, request, settings)
    save_credentials(credentials, settings)

    print("\nStored the refresh token, encrypted.")
    print(f"  account: {credentials.account_email or 'unknown'}")
    print(f"  store  : {settings.google_token_store_path}")
    print("\nThe encryption key is not stored alongside it. Keep a backup of the key;")
    print("without it the store cannot be read and you will have to re-authorise.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="Show the stored grant and exit.")
    parser.add_argument("--revoke", action="store_true", help="Delete the stored credentials.")
    parser.add_argument("--force", action="store_true", help="Replace an existing grant.")
    parser.add_argument(
        "--scope",
        dest="scopes",
        action="append",
        help="Request a specific scope. Repeatable. Defaults to every scope the app uses.",
    )
    parser.add_argument("--timeout", type=int, default=300, help="Seconds to wait for consent.")
    args = parser.parse_args()

    try:
        return asyncio.run(_run(args))
    except AssistantError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled. Nothing was stored.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
