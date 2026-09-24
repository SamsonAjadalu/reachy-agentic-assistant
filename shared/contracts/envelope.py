"""Response envelopes.

Every endpoint returns the same shape so the Reachy client can branch on
``ok`` alone, and every error carries a machine-readable code plus the request
id needed to find the corresponding server log line.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SuccessResponse[T](BaseModel):
    ok: bool = True
    data: T
    request_id: str | None = Field(
        default=None, description="Correlates this response with the server log entry."
    )


class ErrorBody(BaseModel):
    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable summary. Never contains secrets.")
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    ok: bool = False
    error: ErrorBody
    request_id: str | None = None


class Page[T](BaseModel):
    """Offset pagination.

    Cursor pagination is unnecessary at single-user scale and would complicate
    every Reachy adapter for no practical benefit.
    """

    items: list[T]
    total: int = Field(description="Total rows matching the filter, ignoring limit/offset.")
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.items) < self.total
