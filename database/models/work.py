"""Documents, registered scripts and briefing preferences."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base, TimestampMixin, UtcDateTime, UUIDPrimaryKeyMixin
from shared.enums import ScriptRunStatus


class DocumentSource(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An approved root that the indexer is allowed to walk."""

    __tablename__ = "document_sources"
    __table_args__ = (UniqueConstraint("root_path", name="uq_document_sources_root_path"),)

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    root_path: Mapped[str] = mapped_column(String(500), nullable=False)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False, default="local")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    follow_symlinks: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    last_indexed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    last_index_duration_ms: Mapped[int | None] = mapped_column(Integer)
    document_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)

    documents: Mapped[list[DocumentRecord]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )


class DocumentRecord(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Indexed document metadata.

    Full text lives in a separate FTS5 database under APP_DATA_DIR, keyed by this
    row's id, so the relational schema stays migratable independently of the
    index.
    """

    __tablename__ = "document_records"
    __table_args__ = (
        UniqueConstraint("source_id", "relative_path", name="uq_document_records_source_path"),
        Index("ix_document_records_content_hash", "content_hash"),
        Index("ix_document_records_kind", "kind"),
    )

    source_id: Mapped[str] = mapped_column(
        ForeignKey("document_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    relative_path: Mapped[str] = mapped_column(String(500), nullable=False)
    absolute_path: Mapped[str] = mapped_column(String(700), nullable=False)
    filename: Mapped[str] = mapped_column(String(300), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    modified_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    content_hash: Mapped[str | None] = mapped_column(String(64))
    indexed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    extraction_ok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    extraction_error: Mapped[str | None] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    title: Mapped[str | None] = mapped_column(String(400))
    excerpt: Mapped[str | None] = mapped_column(Text)

    source: Mapped[DocumentSource] = relationship(back_populates="documents")


class RegisteredScript(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Database mirror of the YAML script registry.

    The YAML file is the source of truth so the allowlist is reviewable in git;
    this table exists so runs can foreign-key to a stable id and so the API can
    report which entries failed validation at load time.
    """

    __tablename__ = "registered_scripts"
    __table_args__ = (UniqueConstraint("name", name="uq_registered_scripts_name"),)

    name: Mapped[str] = mapped_column(String(80), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    executable: Mapped[str] = mapped_column(String(500), nullable=False)
    working_directory: Mapped[str | None] = mapped_column(String(500))
    parameters_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    risk_level: Mapped[str] = mapped_column(String(30), nullable=False, default="reversible_write")
    requires_approval: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    timeout_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=300)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    checksum: Mapped[str | None] = mapped_column(String(64))
    load_error: Mapped[str | None] = mapped_column(Text)

    runs: Mapped[list[ScriptRun]] = relationship(back_populates="script")


class ScriptRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "script_runs"
    __table_args__ = (Index("ix_script_runs_status_created_at", "status", "created_at"),)

    script_id: Mapped[str | None] = mapped_column(
        ForeignKey("registered_scripts.id", ondelete="SET NULL"), index=True
    )
    script_name: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    arguments_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    resolved_argv_json: Mapped[str | None] = mapped_column(
        Text, comment="Exact argv array executed. Never a shell string."
    )
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default=ScriptRunStatus.PENDING_APPROVAL.value
    )
    approval_id: Mapped[str | None] = mapped_column(String(36), index=True)
    pid: Mapped[int | None] = mapped_column(Integer)
    exit_code: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    stdout_tail: Mapped[str | None] = mapped_column(Text)
    stderr_tail: Mapped[str | None] = mapped_column(Text)
    log_path: Mapped[str | None] = mapped_column(String(500))
    error: Mapped[str | None] = mapped_column(Text)

    script: Mapped[RegisteredScript | None] = relationship(back_populates="runs")


class BriefingPreference(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "briefing_preferences"
    __table_args__ = (UniqueConstraint("owner_id", name="uq_briefing_preferences_owner_id"),)

    owner_id: Mapped[str] = mapped_column(
        ForeignKey("owner_profile.id", ondelete="CASCADE"), nullable=False
    )
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    send_hour: Mapped[int] = mapped_column(Integer, nullable=False, default=7)
    send_minute: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="America/Toronto")
    sections_json: Mapped[str] = mapped_column(
        Text, nullable=False, default='["calendar","tasks","reminders","weather","outfit"]'
    )
    channels_json: Mapped[str] = mapped_column(Text, nullable=False, default='["telegram"]')
    quiet_hours_start: Mapped[int | None] = mapped_column(Integer)
    quiet_hours_end: Mapped[int | None] = mapped_column(Integer)
    urgency_threshold: Mapped[str] = mapped_column(String(20), nullable=False, default="warning")
    vip_contact_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    commute_start_hour: Mapped[int | None] = mapped_column(Integer)
    commute_end_hour: Mapped[int | None] = mapped_column(Integer)
    include_outfit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    formatter: Mapped[str] = mapped_column(String(30), nullable=False, default="template")
    disk_warning_percent: Mapped[float] = mapped_column(Float, nullable=False, default=90.0)
