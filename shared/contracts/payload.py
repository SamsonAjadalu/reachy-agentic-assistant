"""Canonical payload serialisation and hashing.

An approval is only meaningful if the bytes the owner saw are provably the bytes
that get executed. Canonical JSON gives a stable representation regardless of
dict ordering or float formatting, so the hash stored with the pending action
can be re-verified immediately before the side effect fires.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from shared.timeutils import isoformat_utc


def _normalise(value: Any) -> Any:
    """Convert to JSON-native types with a single deterministic representation."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        # Round-trip through repr so 1.0 and 1 never collide accidentally while
        # remaining stable across platforms.
        return repr(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return _normalise(value.value)
    if isinstance(value, datetime):
        return isoformat_utc(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, list | tuple):
        return [_normalise(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(_normalise(item) for item in value)
    if hasattr(value, "model_dump"):
        return _normalise(value.model_dump())
    return str(value)


def canonical_json(payload: Any) -> str:
    """Deterministic JSON text for hashing and for the approval preview."""
    return json.dumps(
        _normalise(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def payload_hash(payload: Any) -> str:
    """SHA-256 of the canonical form, hex encoded."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def payload_fingerprint(payload: Any, *, length: int = 12) -> str:
    """Short prefix of the payload hash, for human-readable previews."""
    return payload_hash(payload)[:length]
