"""All ORM models.

Imported as one module so ``Base.metadata`` is complete before Alembic
autogenerate or ``create_all`` runs.
"""

from database.base import Base
from database.models.approvals import (
    AlertDeduplication,
    Approval,
    BackgroundTask,
    BackgroundTaskEvent,
    Notification,
    NotificationDeliveryAttempt,
    PendingAction,
    PendingReachyNotification,
)
from database.models.core import (
    ActionAuditLog,
    Contact,
    ContactAlias,
    IdempotencyKey,
    IntegrationHealth,
    OwnerProfile,
    ServiceHeartbeat,
    UserPreference,
)
from database.models.scheduling import Reminder, ScheduledJob, Task, TaskRecurrence
from database.models.wardrobe import (
    Outfit,
    OutfitHistory,
    OutfitItem,
    SocialPostHistory,
    WardrobeAvailabilityEvent,
    WardrobeImage,
    WardrobeItem,
)
from database.models.work import (
    BriefingPreference,
    DocumentRecord,
    DocumentSource,
    RegisteredScript,
    ScriptRun,
)

__all__ = [
    "ActionAuditLog",
    "AlertDeduplication",
    "Approval",
    "BackgroundTask",
    "BackgroundTaskEvent",
    "Base",
    "BriefingPreference",
    "Contact",
    "ContactAlias",
    "DocumentRecord",
    "DocumentSource",
    "IdempotencyKey",
    "IntegrationHealth",
    "Notification",
    "NotificationDeliveryAttempt",
    "Outfit",
    "OutfitHistory",
    "OutfitItem",
    "OwnerProfile",
    "PendingAction",
    "PendingReachyNotification",
    "RegisteredScript",
    "Reminder",
    "ScheduledJob",
    "ScriptRun",
    "ServiceHeartbeat",
    "SocialPostHistory",
    "Task",
    "TaskRecurrence",
    "UserPreference",
    "WardrobeAvailabilityEvent",
    "WardrobeImage",
    "WardrobeItem",
]
