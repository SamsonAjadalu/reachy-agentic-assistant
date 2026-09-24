"""Canonical enumerations.

Every value is stored in the database as its string form, so renaming a member
requires a migration.
"""

from __future__ import annotations

from enum import StrEnum


class RiskLevel(StrEnum):
    """Risk tier for an operation.

    Drives whether an approval is required before the side effect occurs.
    """

    READ = "read"
    """No side effect outside the assistant."""

    REVERSIBLE_WRITE = "reversible_write"
    """Local or draft-only state that the owner can undo without contacting anyone."""

    EXTERNAL_WRITE = "external_write"
    """Visible to third parties or destructive. Always requires approval."""


class ReminderStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    MISSED = "missed"
    SNOOZED = "snoozed"


class TaskStatus(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationChannel(StrEnum):
    TELEGRAM = "telegram"
    REACHY = "reachy"
    NONE = "none"


class DeliveryStatus(StrEnum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    SUPPRESSED = "suppressed"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTED = "executed"
    FAILED = "failed"


class ApprovalDecision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    EXPIRE = "expire"


class PendingActionStatus(StrEnum):
    """Lifecycle of the action bound to an approval.

    ``EXECUTING`` is claimed atomically so an action can only ever run once.
    """

    PENDING = "pending"
    EXECUTING = "executing"
    EXECUTED = "executed"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class BackgroundTaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class IntegrationName(StrEnum):
    GOOGLE = "google"
    GMAIL = "gmail"
    CALENDAR = "calendar"
    CONTACTS = "contacts"
    DRIVE = "drive"
    NOTION = "notion"
    TELEGRAM = "telegram"
    WEATHER = "weather"
    WORKSTATION = "workstation"
    DOCUMENTS = "documents"
    WARDROBE = "wardrobe"


class DocumentKind(StrEnum):
    TXT = "txt"
    MARKDOWN = "markdown"
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    CSV = "csv"
    XLSX = "xlsx"
    JSON = "json"


class LaundryStatus(StrEnum):
    CLEAN = "clean"
    WORN = "worn"
    IN_LAUNDRY = "in_laundry"
    NEEDS_REPAIR = "needs_repair"



class WatchStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERRORED = "errored"


class ScriptRunStatus(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    STARTING = "starting"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    STOPPED = "stopped"
    REJECTED = "rejected"
