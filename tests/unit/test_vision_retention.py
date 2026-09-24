"""Retention classes, TTL, person pinning, and reference recomputation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from vision.enums import EvidenceKind, RetentionClass
from vision.storage import expires_at_for, resolve_retention

NOW = datetime(2026, 9, 2, 18, 0, tzinfo=UTC)


class _Settings:
    visual_ephemeral_retention_minutes = 60
    visual_thumbnail_retention_days = 30
    visual_evidence_retention_days = 90


class TestRetentionClasses:
    def test_person_crops_are_forced_ephemeral(self) -> None:
        assert (
            resolve_retention(
                RetentionClass.EVIDENCE, contains_person=True, privacy_sensitive=False
            )
            is RetentionClass.EPHEMERAL
        )

    def test_privacy_sensitive_is_forced_ephemeral(self) -> None:
        assert (
            resolve_retention(RetentionClass.PINNED, contains_person=False, privacy_sensitive=True)
            is RetentionClass.EPHEMERAL
        )

    def test_pinned_and_evaluation_never_expire(self) -> None:
        settings = _Settings()
        assert (
            expires_at_for(RetentionClass.PINNED, now=NOW, settings=settings) is None  # type: ignore[arg-type]
        )
        assert (
            expires_at_for(RetentionClass.EVALUATION, now=NOW, settings=settings) is None  # type: ignore[arg-type]
        )

    def test_ephemeral_ttl_is_one_hour(self) -> None:
        expires = expires_at_for(
            RetentionClass.EPHEMERAL,
            now=NOW,
            settings=_Settings(),  # type: ignore[arg-type]
        )
        assert expires == NOW + timedelta(minutes=60)

    def test_thumbnail_ttl_is_thirty_days(self) -> None:
        expires = expires_at_for(
            RetentionClass.THUMBNAIL,
            now=NOW,
            settings=_Settings(),  # type: ignore[arg-type]
        )
        assert expires == NOW + timedelta(days=30)

    def test_evidence_ttl_is_ninety_days(self) -> None:
        expires = expires_at_for(
            RetentionClass.EVIDENCE,
            now=NOW,
            settings=_Settings(),  # type: ignore[arg-type]
        )
        assert expires == NOW + timedelta(days=90)

    def test_person_pin_request_becomes_ephemeral_with_ttl(self) -> None:
        expires = expires_at_for(
            RetentionClass.PINNED,
            now=NOW,
            settings=_Settings(),  # type: ignore[arg-type]
            contains_person=True,
        )
        assert expires == NOW + timedelta(minutes=60)

    def test_kinds_include_mask_refs(self) -> None:
        assert EvidenceKind.MASK.value == "mask"
        assert EvidenceKind.THUMBNAIL.value == "thumbnail"
