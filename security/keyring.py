"""Resolution of the master encryption key.

The key that decrypts the OAuth token store must not sit next to the store, or
the encryption buys nothing: anyone who can read one file can read both. Three
sources are supported, checked in this order:

1. ``PA_SECRET_KEY_FILE`` - a file outside ``APP_DATA_DIR``. Under systemd this
   points at ``$CREDENTIALS_DIRECTORY/pa_secret_key``, which the kernel exposes
   as a tmpfs readable only by the service and never written to disk.
2. ``PA_SECRET_KEY`` - the raw key in the environment. Used for development and
   for the test suite.
3. The OS keyring, when ``PA_SECRET_KEY_KEYRING=true``.

The resolved key is held in memory only. ``SECURITY.md`` documents the trade-offs
of each option.
"""

from __future__ import annotations

import os
from pathlib import Path

from cryptography.fernet import Fernet

from app.config import Settings
from shared.errors import ConfigurationError

KEYRING_SERVICE = "reachy-personal-assistant"
KEYRING_USERNAME = "pa_secret_key"


def generate_key() -> str:
    """Create a new URL-safe base64 Fernet key."""
    return Fernet.generate_key().decode("ascii")


def _read_key_file(path: Path) -> str:
    if not path.exists():
        raise ConfigurationError(
            f"PA_SECRET_KEY_FILE points at {path}, which does not exist. "
            "Create it with scripts/generate_api_token.py --secret-key."
        )
    mode = path.stat().st_mode & 0o777
    # A credentials-directory file is already 0400 on tmpfs; anything else must
    # not be group or world readable.
    if mode & 0o077:
        raise ConfigurationError(
            f"{path} has permissions {mode:o}. The encryption key must be owner-only (chmod 600)."
        )
    return path.read_text(encoding="utf-8").strip()


def _read_keyring() -> str:
    try:
        import keyring
    except ImportError as exc:  # pragma: no cover - exercised only when opted in
        raise ConfigurationError(
            "PA_SECRET_KEY_KEYRING=true but the 'keyring' package is not installed. "
            "Install it with: uv pip install keyring"
        ) from exc
    value = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    if not value:
        raise ConfigurationError(
            f"No key found in the OS keyring under {KEYRING_SERVICE}/{KEYRING_USERNAME}."
        )
    return value


def resolve_secret_key(settings: Settings) -> str:
    """Return the Fernet key, or raise a ``ConfigurationError`` explaining how to set one."""
    if settings.pa_secret_key_file is not None:
        key = _read_key_file(settings.pa_secret_key_file)
    elif settings.pa_secret_key.get_secret_value():
        key = settings.pa_secret_key.get_secret_value()
    elif settings.pa_secret_key_keyring:
        key = _read_keyring()
    else:
        raise ConfigurationError(
            "No encryption key configured. Set PA_SECRET_KEY, PA_SECRET_KEY_FILE or "
            "PA_SECRET_KEY_KEYRING. Generate one with: "
            "python scripts/generate_api_token.py --secret-key"
        )

    _assert_key_outside_store(settings)

    try:
        Fernet(key.encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(
            "The configured encryption key is not a valid Fernet key. "
            "Generate one with: python scripts/generate_api_token.py --secret-key"
        ) from exc
    return key


def _assert_key_outside_store(settings: Settings) -> None:
    """Refuse a key file stored in the same directory as the ciphertext it protects."""
    key_file = settings.pa_secret_key_file
    if key_file is None:
        return
    store = settings.google_token_store_path
    key_parent = key_file.expanduser().resolve().parent
    if store is not None and key_parent == store.expanduser().resolve().parent:
        raise ConfigurationError(
            f"PA_SECRET_KEY_FILE ({key_file}) sits in the same directory as the encrypted "
            f"token store ({store}). Move the key outside APP_DATA_DIR, or supply it via "
            "systemd LoadCredential or the OS keyring."
        )
    if settings.app_data_dir in key_parent.parents or key_parent == settings.app_data_dir:
        raise ConfigurationError(
            f"PA_SECRET_KEY_FILE ({key_file}) is inside APP_DATA_DIR ({settings.app_data_dir}). "
            "A backup of the data directory would then contain both the ciphertext and its key."
        )


def systemd_credentials_path(name: str = KEYRING_USERNAME) -> Path | None:
    """Path to a systemd-provided credential, when running under systemd."""
    directory = os.environ.get("CREDENTIALS_DIRECTORY")
    return Path(directory) / name if directory else None
