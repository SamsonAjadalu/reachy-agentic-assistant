"""Secret redaction for logs, error payloads and diagnostics.

Two layers, because either one alone leaks eventually:

* A registry of exact secret values, populated at startup from settings. Catches
  a token that reaches a log line through any code path.
* Pattern matching on well-known credential shapes. Catches secrets this process
  never held in settings, such as a bearer token echoed from a request header.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

REDACTED = "[REDACTED]"

# Key names whose values are always replaced, regardless of content.
SENSITIVE_KEY_PATTERN = re.compile(
    r"(token|secret|password|passwd|api[_-]?key|authorization|auth|credential|"
    r"client[_-]?secret|refresh[_-]?token|access[_-]?token|private[_-]?key|cookie|session)",
    re.IGNORECASE,
)

_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Authorization: Bearer <token>
    re.compile(r"(?i)\b(bearer|token)\s+[A-Za-z0-9._\-]{12,}"),
    # Telegram bot tokens: 123456789:AA...
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_\-]{30,}\b"),
    # Google OAuth client secrets and refresh tokens
    re.compile(r"\bGOCSPX-[A-Za-z0-9_\-]{10,}\b"),
    re.compile(r"\b1//[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bya29\.[A-Za-z0-9_\-]{10,}\b"),
    # Notion integration tokens
    re.compile(r"\b(secret_|ntn_)[A-Za-z0-9]{20,}\b"),
    # PEM private key blocks
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
)

_MIN_REGISTERED_LENGTH = 8
_registered_secrets: set[str] = set()


def register_secret(value: str | None) -> None:
    """Record a literal secret so it is scrubbed anywhere it appears."""
    if value and len(value) >= _MIN_REGISTERED_LENGTH:
        _registered_secrets.add(value)


def register_secrets(values: Iterable[str | None]) -> None:
    for value in values:
        register_secret(value)


def clear_registered_secrets() -> None:
    _registered_secrets.clear()


def redact_text(text: str) -> str:
    """Remove known secrets and credential-shaped substrings from free text."""
    if not text:
        return text
    result = text
    # Longest first so a prefix of another secret cannot leave a tail behind.
    for secret in sorted(_registered_secrets, key=len, reverse=True):
        if secret in result:
            result = result.replace(secret, REDACTED)
    for pattern in _PATTERNS:
        result = pattern.sub(REDACTED, result)
    return result


def redact_value(value: Any, *, key: str | None = None, _depth: int = 0) -> Any:
    """Recursively redact a structure destined for a log record or API response."""
    if _depth > 8:
        return "[TRUNCATED]"
    if key is not None and SENSITIVE_KEY_PATTERN.search(key) and value not in (None, "", [], {}):
        return REDACTED
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact_value(v, key=str(k), _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact_value(item, key=key, _depth=_depth + 1) for item in value]
    return value
