"""Secrets must not reach logs, error payloads or the token store's neighbours."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from app.config import AppEnv, Settings
from app.logging_config import JsonFormatter, RedactionFilter
from security.keyring import resolve_secret_key
from security.redaction import redact_text, redact_value, register_secret
from security.token_store import EncryptedTokenStore
from shared.errors import ConfigurationError


def _emit(record_factory) -> str:
    """Run a record through the redaction filter and JSON formatter."""
    record = record_factory()
    RedactionFilter().filter(record)
    return JsonFormatter().format(record)


def _record(msg: str, *args: object, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, msg, args or None, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


class TestRedaction:
    def test_registered_secret_is_scrubbed_from_a_message(self) -> None:
        register_secret("hunter2-the-actual-password")
        assert "hunter2" not in redact_text("password is hunter2-the-actual-password")

    @pytest.mark.parametrize(
        "secret",
        [
            "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw",
            "GOCSPX-abcdefghijklmnop12345",
            "ya29.a0AfH6SMBexampleaccesstoken",
            "secret_abcdefghijklmnopqrstuvwxyz012345",
            "ntn_abcdefghijklmnopqrstuvwxyz012345",
        ],
    )
    def test_credential_shapes_are_scrubbed_without_registration(self, secret: str) -> None:
        assert secret not in redact_text(f"provider replied with {secret} today")

    def test_private_key_blocks_are_scrubbed(self) -> None:
        pem = (
            "-----BEGIN OPENSSH PRIVATE KEY-----\n"
            "b3BlbnNzaC1rZXktdjEAAAAABG5vbmU=\n"
            "-----END OPENSSH PRIVATE KEY-----"
        )
        assert "b3BlbnNzaC" not in redact_text(f"key material: {pem}")

    def test_sensitive_dictionary_keys_are_replaced_wholesale(self) -> None:
        redacted = redact_value(
            {"authorization": "anything at all", "refresh_token": "x", "subject": "Lunch"}
        )
        assert redacted["authorization"] == "[REDACTED]"
        assert redacted["refresh_token"] == "[REDACTED]"
        assert redacted["subject"] == "Lunch"

    def test_nested_structures_are_traversed(self) -> None:
        redacted = redact_value({"outer": [{"api_key": "abc"}, {"safe": "value"}]})
        assert redacted["outer"][0]["api_key"] == "[REDACTED]"
        assert redacted["outer"][1]["safe"] == "value"

    def test_recursion_is_bounded(self) -> None:
        payload: dict = {}
        cursor = payload
        for _ in range(30):
            cursor["next"] = {}
            cursor = cursor["next"]
        assert "TRUNCATED" in json.dumps(redact_value(payload))


class TestLogPipeline:
    def test_secret_in_a_log_message_never_reaches_the_formatter(self) -> None:
        register_secret("tok_live_verysecretvalue")
        output = _emit(lambda: _record("calling provider with tok_live_verysecretvalue"))
        assert "verysecretvalue" not in output
        assert "[REDACTED]" in output

    def test_secret_in_a_log_extra_field_is_scrubbed(self) -> None:
        output = _emit(lambda: _record("outbound", api_key="abcd1234efgh5678"))
        assert "abcd1234efgh5678" not in output

    def test_secret_in_a_format_argument_is_scrubbed(self) -> None:
        register_secret("s3cr3t-token-value-here")
        output = _emit(lambda: _record("token=%s", "s3cr3t-token-value-here"))
        assert "s3cr3t-token-value-here" not in output

    def test_ordinary_content_survives_intact(self) -> None:
        output = _emit(lambda: _record("reminder created", reminder_text="Buy milk"))
        assert "Buy milk" in output


class TestSecretKeySeparation:
    def test_key_file_beside_the_token_store_is_refused(self, tmp_path: Path) -> None:
        """The whole point of encrypting the store is defeated if the key sits next to it."""
        data_dir = tmp_path / "data"
        store = data_dir / "secrets" / "google_tokens.enc"
        store.parent.mkdir(parents=True)
        key_file = store.parent / "secret.key"
        key_file.write_text(Fernet.generate_key().decode(), encoding="utf-8")
        key_file.chmod(0o600)

        settings = Settings(
            app_env=AppEnv.TEST,
            app_data_dir=data_dir,
            pa_api_token="t" * 40,
            pa_secret_key_file=key_file,
            google_token_store_path=store,
        )
        with pytest.raises(ConfigurationError, match="same directory"):
            resolve_secret_key(settings)

    def test_key_file_anywhere_inside_the_data_dir_is_refused(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        key_file = data_dir / "keys" / "secret.key"
        key_file.parent.mkdir(parents=True)
        key_file.write_text(Fernet.generate_key().decode(), encoding="utf-8")
        key_file.chmod(0o600)

        settings = Settings(
            app_env=AppEnv.TEST,
            app_data_dir=data_dir,
            pa_api_token="t" * 40,
            pa_secret_key_file=key_file,
        )
        with pytest.raises(ConfigurationError, match="inside APP_DATA_DIR"):
            resolve_secret_key(settings)

    def test_group_readable_key_file_is_refused(self, tmp_path: Path, settings: Settings) -> None:
        key_file = tmp_path / "loose.key"
        key_file.write_text(Fernet.generate_key().decode(), encoding="utf-8")
        key_file.chmod(0o644)
        settings.pa_secret_key_file = key_file
        with pytest.raises(ConfigurationError, match="owner-only"):
            resolve_secret_key(settings)

    def test_missing_key_file_explains_how_to_create_one(self, settings: Settings) -> None:
        settings.pa_secret_key_file = Path("/nonexistent/secret.key")
        with pytest.raises(ConfigurationError, match="generate_api_token"):
            resolve_secret_key(settings)

    def test_no_configured_source_is_an_explicit_error(self, settings: Settings) -> None:
        settings.pa_secret_key_file = None
        settings.pa_secret_key_keyring = False
        with pytest.raises(ConfigurationError, match="No encryption key configured"):
            resolve_secret_key(settings)

    def test_a_correctly_separated_key_resolves(self, settings: Settings) -> None:
        assert resolve_secret_key(settings)


class TestEncryptedTokenStore:
    def test_round_trips_a_credential(self, tmp_path: Path) -> None:
        store = EncryptedTokenStore(tmp_path / "tokens.enc", Fernet.generate_key().decode())
        store.put("google", {"refresh_token": "1//abc", "scopes": ["gmail.readonly"]})
        assert store.get("google") == {"refresh_token": "1//abc", "scopes": ["gmail.readonly"]}

    def test_ciphertext_on_disk_does_not_contain_the_plaintext(self, tmp_path: Path) -> None:
        path = tmp_path / "tokens.enc"
        store = EncryptedTokenStore(path, Fernet.generate_key().decode())
        store.put("google", {"refresh_token": "1//super-secret-refresh"})
        assert b"super-secret-refresh" not in path.read_bytes()

    def test_file_is_owner_only(self, tmp_path: Path) -> None:
        path = tmp_path / "tokens.enc"
        EncryptedTokenStore(path, Fernet.generate_key().decode()).put("google", {"a": 1})
        assert path.stat().st_mode & 0o077 == 0

    def test_wrong_key_produces_an_actionable_error(self, tmp_path: Path) -> None:
        path = tmp_path / "tokens.enc"
        EncryptedTokenStore(path, Fernet.generate_key().decode()).put("google", {"a": 1})
        other = EncryptedTokenStore(path, Fernet.generate_key().decode())
        with pytest.raises(ConfigurationError, match="does not match"):
            other.get("google")

    def test_delete_is_idempotent(self, tmp_path: Path) -> None:
        store = EncryptedTokenStore(tmp_path / "tokens.enc", Fernet.generate_key().decode())
        store.put("notion", {"token": "x"})
        assert store.delete("notion") is True
        assert store.delete("notion") is False

    def test_health_reports_no_secret_material(self, tmp_path: Path) -> None:
        store = EncryptedTokenStore(tmp_path / "tokens.enc", Fernet.generate_key().decode())
        store.put("google", {"refresh_token": "1//abc-secret"})
        health = store.health()
        assert health["providers"] == ["google"]
        assert "abc-secret" not in json.dumps(health)
        assert health["permissions_ok"] is True
