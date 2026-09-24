"""Gmail label and attachment safety bounds."""

from __future__ import annotations

from pathlib import Path

import pytest

from integrations.google.gmail import FORBIDDEN_LABEL_IDS, validate_label_ids
from integrations.google.mock import MockGmailService
from shared.errors import ValidationError


def test_trash_and_spam_are_forbidden() -> None:
    assert "TRASH" in FORBIDDEN_LABEL_IDS
    assert "SPAM" in FORBIDDEN_LABEL_IDS
    with pytest.raises(ValidationError, match="Trash and spam"):
        validate_label_ids(["TRASH"])
    with pytest.raises(ValidationError, match="Trash and spam"):
        validate_label_ids(["spam"])


def test_path_traversal_label_ids_are_refused() -> None:
    with pytest.raises(ValidationError):
        validate_label_ids(["../etc"])


@pytest.mark.asyncio
async def test_oversized_attachment_download_is_refused(tmp_path: Path) -> None:
    gmail = MockGmailService()
    gmail._attachments["msg-004"] = [
        (
            gmail._attachments["msg-004"][0][0],
            b"x" * 100,
        )
    ]
    with pytest.raises(ValidationError, match="limit"):
        await gmail.download_attachment(
            "msg-004",
            "att-invoice-1",
            dest_dir=tmp_path,
            max_bytes=10,
        )
