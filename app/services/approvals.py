"""Approval service facade.

Re-exports the state machine so callers depend on a service-layer name rather
than reaching into ``approvals.state_machine`` directly.
"""

from __future__ import annotations

from approvals.state_machine import (
    execute_approved_action,
    expire_stale_approvals,
    get_executor,
    register_executor,
    request_approval,
    resolve_by_token,
)

__all__ = [
    "execute_approved_action",
    "expire_stale_approvals",
    "get_executor",
    "register_executor",
    "request_approval",
    "resolve_by_token",
]
