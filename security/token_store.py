"""Encrypted credential store.

OAuth refresh tokens are the highest-value secret this system holds, so they live
in their own Fernet-encrypted file rather than in the application database. That
keeps them out of ordinary database backups and out of any query a bug could
accidentally serialise into an API response.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from security.redaction import register_secret
from shared.errors import ConfigurationError


class EncryptedTokenStore:
    """A small encrypted key/value file keyed by provider name."""

    def __init__(self, path: Path, key: str) -> None:
        self._path = path
        try:
            self._fernet = Fernet(key.encode("ascii"))
        except (ValueError, TypeError) as exc:
            raise ConfigurationError("Invalid Fernet key supplied to the token store.") from exc

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.exists()

    def _read_all(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        raw = self._path.read_bytes()
        if not raw.strip():
            return {}
        try:
            plaintext = self._fernet.decrypt(raw)
        except InvalidToken as exc:
            raise ConfigurationError(
                f"Cannot decrypt {self._path}. The encryption key does not match the one used "
                "to write this store. Restore the original key or delete the store and re-run "
                "the OAuth setup script."
            ) from exc
        data = json.loads(plaintext.decode("utf-8"))
        return data if isinstance(data, dict) else {}

    def _write_all(self, data: dict[str, Any]) -> None:
        """Encrypt and replace atomically so a crash cannot leave a truncated store."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.parent.chmod(0o700)
        ciphertext = self._fernet.encrypt(json.dumps(data).encode("utf-8"))

        fd, tmp_name = tempfile.mkstemp(dir=str(self._path.parent), prefix=".tokens-")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(ciphertext)
                handle.flush()
                os.fsync(handle.fileno())
            tmp_path.chmod(0o600)
            tmp_path.replace(self._path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def get(self, provider: str) -> dict[str, Any] | None:
        entry = self._read_all().get(provider)
        if isinstance(entry, dict):
            for field in ("token", "refresh_token", "access_token", "client_secret"):
                value = entry.get(field)
                if isinstance(value, str):
                    register_secret(value)
            return entry
        return None

    def put(self, provider: str, payload: dict[str, Any]) -> None:
        data = self._read_all()
        data[provider] = payload
        self._write_all(data)

    def delete(self, provider: str) -> bool:
        data = self._read_all()
        if provider not in data:
            return False
        del data[provider]
        self._write_all(data)
        return True

    def providers(self) -> list[str]:
        return sorted(self._read_all())

    def health(self) -> dict[str, Any]:
        """Non-sensitive status, safe to expose through the diagnostics endpoint."""
        if not self.exists():
            return {"exists": False, "providers": [], "path": str(self._path)}
        mode = self._path.stat().st_mode & 0o777
        return {
            "exists": True,
            "providers": self.providers(),
            "path": str(self._path),
            "permissions": f"{mode:o}",
            "permissions_ok": not (mode & 0o077),
        }
