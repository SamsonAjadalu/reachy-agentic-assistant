#!/usr/bin/env python3
"""Generate the API bearer token and the token-store encryption key.

    python scripts/generate_api_token.py                     # print a bearer token
    python scripts/generate_api_token.py --secret-key        # print a Fernet key
    python scripts/generate_api_token.py --write             # write both into .env
    python scripts/generate_api_token.py --secret-key \\
        --key-file /etc/reachy-personal-assistant/secret.key # write the key out of band

The encryption key is deliberately written somewhere other than APP_DATA_DIR:
storing it beside the ciphertext it protects would mean one stolen backup
exposes both. See SECURITY.md.
"""

from __future__ import annotations

import argparse
import os
import re
import secrets
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cryptography.fernet import Fernet  # noqa: E402

TOKEN_BYTES = 32


def generate_bearer_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def generate_fernet_key() -> str:
    return Fernet.generate_key().decode("ascii")


def write_env_value(env_path: Path, key: str, value: str) -> None:
    """Set ``key`` in a dotenv file, replacing any existing assignment."""
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    replaced = False
    for index, line in enumerate(lines):
        if pattern.match(line):
            lines[index] = f"{key}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    env_path.chmod(0o600)


def write_key_file(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(value + "\n")
    path.chmod(0o600)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--secret-key",
        action="store_true",
        help="Generate a Fernet encryption key instead of a bearer token.",
    )
    parser.add_argument(
        "--write", action="store_true", help="Write the generated value(s) into .env."
    )
    parser.add_argument(
        "--env-file", type=Path, default=REPO_ROOT / ".env", help="Dotenv path for --write."
    )
    parser.add_argument(
        "--key-file",
        type=Path,
        help="Write the Fernet key to this file (chmod 600) instead of printing it.",
    )
    args = parser.parse_args()

    if args.write:
        token = generate_bearer_token()
        key = generate_fernet_key()
        write_env_value(args.env_file, "PA_API_TOKEN", token)
        if args.key_file:
            write_key_file(args.key_file, key)
            write_env_value(args.env_file, "PA_SECRET_KEY_FILE", str(args.key_file))
            write_env_value(args.env_file, "PA_SECRET_KEY", "")
            print(f"Wrote PA_API_TOKEN to {args.env_file}")
            print(f"Wrote the encryption key to {args.key_file} (mode 600)")
        else:
            write_env_value(args.env_file, "PA_SECRET_KEY", key)
            print(f"Wrote PA_API_TOKEN and PA_SECRET_KEY to {args.env_file}")
            print(
                "For production, move the key out of .env: re-run with "
                "--key-file /etc/reachy-personal-assistant/secret.key, or use systemd "
                "LoadCredential. See SECURITY.md."
            )
        print("\nGive this token to the Reachy client as PA_API_TOKEN:")
        print(f"  {token}")
        return 0

    value = generate_fernet_key() if args.secret_key else generate_bearer_token()

    if args.key_file:
        write_key_file(args.key_file, value)
        print(f"Wrote the key to {args.key_file} (mode 600).")
        print("Point PA_SECRET_KEY_FILE at it and leave PA_SECRET_KEY unset.")
        return 0

    print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
