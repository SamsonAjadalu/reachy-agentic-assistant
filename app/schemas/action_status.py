"""Safe approval/action status responses for Reachy polling."""

from __future__ import annotations

from datetime import datetime

from app.schemas.common import ApiModel


class ActionStatusOut(ApiModel):
    approval_id: str
    action_id: str
    action_type: str
    status: str
    created_at: datetime
    approved_at: datetime | None = None
    rejected_at: datetime | None = None
    executed_at: datetime | None = None
    result_summary: str | None = None
