"""Unit tests for Gmail reply header helpers."""

from __future__ import annotations

from integrations.google.gmail import build_references_header, build_reply_subject


class TestReplyHelpers:
    def test_build_reply_subject_adds_prefix(self) -> None:
        assert build_reply_subject("Coffee?") == "Re: Coffee?"

    def test_build_reply_subject_preserves_existing_prefix(self) -> None:
        assert build_reply_subject("Re: Coffee?") == "Re: Coffee?"
        assert build_reply_subject("Fwd: Coffee?") == "Fwd: Coffee?"

    def test_build_references_header_appends_message_id(self) -> None:
        assert (
            build_references_header("<a@example.com>", "<b@example.com>")
            == "<a@example.com> <b@example.com>"
        )

    def test_build_references_header_without_existing(self) -> None:
        assert build_references_header("", "<b@example.com>") == "<b@example.com>"
