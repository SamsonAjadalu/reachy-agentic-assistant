"""Owner profile, preferences, contacts and integration health."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base, TimestampMixin, UtcDateTime, UUIDPrimaryKeyMixin


class OwnerProfile(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """The single human this assistant serves.

    Modelled as a table rather than config so preferences, VIP contacts and
    briefing settings can reference it by foreign key.
    """

    __tablename__ = "owner_profile"

    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str | None] = mapped_column(String(320))
    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="America/Toronto")
    telegram_chat_id: Mapped[int | None] = mapped_column(Integer)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    preferences: Mapped[list[UserPreference]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )


class UserPreference(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "user_preferences"
    __table_args__ = (UniqueConstraint("owner_id", "key", name="uq_user_preferences_owner_id_key"),)

    owner_id: Mapped[str] = mapped_column(
        ForeignKey("owner_profile.id", ondelete="CASCADE"), nullable=False
    )
    key: Mapped[str] = mapped_column(String(120), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    value_type: Mapped[str] = mapped_column(String(20), nullable=False, default="string")

    owner: Mapped[OwnerProfile] = relationship(back_populates="preferences")


class Contact(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "contacts"
    __table_args__ = (Index("ix_contacts_normalised_name", "normalised_name"),)

    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    normalised_name: Mapped[str] = mapped_column(String(200), nullable=False)
    primary_email: Mapped[str | None] = mapped_column(String(320), index=True)
    secondary_emails: Mapped[str | None] = mapped_column(Text, comment="JSON array")
    phone: Mapped[str | None] = mapped_column(String(40))
    relationship_tag: Mapped[str | None] = mapped_column(String(60), index=True)
    organisation: Mapped[str | None] = mapped_column(String(200))
    notes: Mapped[str | None] = mapped_column(Text)
    is_vip: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source: Mapped[str] = mapped_column(String(30), nullable=False, default="local")
    external_id: Mapped[str | None] = mapped_column(String(200), index=True)
    last_synced_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    aliases: Mapped[list[ContactAlias]] = relationship(
        back_populates="contact", cascade="all, delete-orphan", lazy="selectin"
    )


class ContactAlias(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """How the owner actually refers to someone, e.g. "my supervisor"."""

    __tablename__ = "contact_aliases"
    __table_args__ = (
        UniqueConstraint("normalised_alias", name="uq_contact_aliases_normalised_alias"),
    )

    contact_id: Mapped[str] = mapped_column(
        ForeignKey("contacts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    alias: Mapped[str] = mapped_column(String(200), nullable=False)
    normalised_alias: Mapped[str] = mapped_column(String(200), nullable=False)
    confidence: Mapped[float] = mapped_column(nullable=False, default=1.0)

    contact: Mapped[Contact] = relationship(back_populates="aliases")


class IntegrationHealth(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "integration_health"
    __table_args__ = (UniqueConstraint("name", name="uq_integration_health_name"),)

    name: Mapped[str] = mapped_column(String(50), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    mode: Mapped[str] = mapped_column(String(20), nullable=False, default="mock")
    healthy: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_checked_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    last_success_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    last_error: Mapped[str | None] = mapped_column(Text)
    consecutive_failures: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class ServiceHeartbeat(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Liveness and single-instance leases for the scheduler and workers."""

    __tablename__ = "service_heartbeats"
    __table_args__ = (UniqueConstraint("service_name", name="uq_service_heartbeats_service_name"),)

    service_name: Mapped[str] = mapped_column(String(60), nullable=False)
    instance_id: Mapped[str] = mapped_column(String(80), nullable=False)
    pid: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="starting")
    last_heartbeat_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    lease_expires_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    detail: Mapped[str | None] = mapped_column(Text)


class IdempotencyKey(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Replay protection for write endpoints.

    Storing the request fingerprint alongside the key lets a genuine retry return
    the original response while a different body reusing the same key is
    rejected as a conflict rather than silently executed.
    """

    __tablename__ = "idempotency_keys"
    __table_args__ = (
        UniqueConstraint("scope", "key", name="uq_idempotency_keys_scope_key"),
        Index("ix_idempotency_keys_expires_at", "expires_at"),
    )

    scope: Mapped[str] = mapped_column(String(80), nullable=False)
    key: Mapped[str] = mapped_column(String(200), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    response_json: Mapped[str | None] = mapped_column(Text)
    resource_id: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="completed")
    expires_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)


class ActionAuditLog(UUIDPrimaryKeyMixin, Base):
    """Append-only record of every consequential operation.

    Deliberately has no ``updated_at``: rows are written once and never edited.
    """

    __tablename__ = "action_audit_log"

    occurred_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, index=True)
    action_type: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    risk_level: Mapped[str] = mapped_column(String(30), nullable=False)
    actor: Mapped[str] = mapped_column(String(60), nullable=False, default="reachy")
    outcome: Mapped[str] = mapped_column(String(30), nullable=False)
    resource_type: Mapped[str | None] = mapped_column(String(60))
    resource_id: Mapped[str | None] = mapped_column(String(200))
    approval_id: Mapped[str | None] = mapped_column(String(36), index=True)
    correlation_id: Mapped[str | None] = mapped_column(String(64), index=True)
    payload_hash: Mapped[str | None] = mapped_column(String(64))
    payload_preview: Mapped[str | None] = mapped_column(
        Text, comment="Redacted canonical payload, truncated"
    )
    error_code: Mapped[str | None] = mapped_column(String(60))
    detail: Mapped[str | None] = mapped_column(Text)
