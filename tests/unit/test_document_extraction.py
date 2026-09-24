"""Extraction and summarisation.

One case per supported format, plus the failure modes that matter when the
input is somebody's real documents folder: unreadable files, wrong extensions
and files large enough to matter.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from documents.extractors import (
    SUPPORTED_EXTENSIONS,
    clean_text,
    extract,
    is_supported,
    kind_for,
)
from documents.summarise import summarise
from shared.enums import DocumentKind
from shared.errors import ValidationError
from tests.fixtures.make_documents import RESEARCH_TEXT, build_all


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_all(tmp_path_factory.mktemp("corpus"))


class TestFormats:
    @pytest.mark.parametrize(
        ("filename", "kind", "needle"),
        [
            ("research_notes.txt", DocumentKind.TEXT, "Tactile feedback"),
            ("meeting.md", DocumentKind.MARKDOWN, "Rerun the ablation"),
            ("config.yaml", DocumentKind.TEXT, "batch_size"),
            ("train.py", DocumentKind.CODE, "def main"),
            ("results.csv", DocumentKind.CSV, "sponge"),
            ("run.json", DocumentKind.JSON, "abc123"),
            ("paper.pdf", DocumentKind.PDF, "Tactile feedback"),
            ("summary.docx", DocumentKind.DOCX, "Quarterly research summary"),
            ("slides.pptx", DocumentKind.PPTX, "Grasp benchmark results"),
            ("trials.xlsx", DocumentKind.XLSX, "sponge"),
        ],
    )
    def test_every_supported_format_yields_text(
        self, corpus: Path, filename: str, kind: DocumentKind, needle: str
    ) -> None:
        result = extract(corpus / filename)
        assert result.kind is kind
        assert needle in result.text

    def test_a_docx_table_is_included(self, corpus: Path) -> None:
        """Numbers in tables are often the part worth searching for."""
        assert "Tactile | 94%" in extract(corpus / "summary.docx").text

    def test_a_spreadsheet_names_its_sheets(self, corpus: Path) -> None:
        assert "[Sheet: Trials]" in extract(corpus / "trials.xlsx").text

    def test_a_deck_numbers_its_slides(self, corpus: Path) -> None:
        result = extract(corpus / "slides.pptx")
        assert "[Slide 1]" in result.text
        assert result.page_count == 1

    def test_an_unsupported_format_is_refused_by_name(self, corpus: Path) -> None:
        with pytest.raises(ValidationError, match="no text extractor"):
            extract(corpus / "ignored.bin")

    def test_the_supported_list_and_the_dispatcher_agree(self) -> None:
        for extension in SUPPORTED_EXTENSIONS:
            assert is_supported(Path(f"x{extension}"))
            assert kind_for(Path(f"x{extension}")) is not DocumentKind.OTHER


class TestRobustness:
    def test_a_file_pretending_to_be_a_pdf_is_reported(self, tmp_path: Path) -> None:
        """Extension is what the owner sees, so a mismatch is reported, not guessed at."""
        fake = tmp_path / "not-really.pdf"
        fake.write_text("this is plain text", encoding="utf-8")
        with pytest.raises(ValidationError, match="could not be read as a PDF"):
            extract(fake)

    def test_a_truncated_docx_is_reported(self, tmp_path: Path, corpus: Path) -> None:
        broken = tmp_path / "broken.docx"
        broken.write_bytes((corpus / "summary.docx").read_bytes()[:200])
        with pytest.raises(ValidationError, match="Word document"):
            extract(broken)

    def test_malformed_json_is_still_searchable_text(self, tmp_path: Path) -> None:
        path = tmp_path / "half.json"
        path.write_text('{"run_id": "abc123", "epochs":', encoding="utf-8")
        assert "abc123" in extract(path).text

    def test_invalid_encoding_does_not_raise(self, tmp_path: Path) -> None:
        path = tmp_path / "latin.txt"
        path.write_bytes(b"caf\xe9 results are stable")
        assert "results are stable" in extract(path).text

    def test_an_empty_file_extracts_to_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")
        assert extract(path).text == ""

    def test_extraction_stops_at_the_character_budget(self, tmp_path: Path) -> None:
        path = tmp_path / "big.txt"
        path.write_text("word " * 50_000, encoding="utf-8")
        result = extract(path, max_chars=1000)
        assert len(result.text) == 1000
        assert result.truncated is True

    def test_control_characters_are_stripped(self) -> None:
        """They break both the FTS index and speech synthesis."""
        assert clean_text("clean\x00text\x07here") == "cleantexthere"

    def test_whitespace_is_collapsed_but_paragraphs_survive(self) -> None:
        assert clean_text("a    b\n\n\n\nc") == "a b\n\nc"


class TestSummariser:
    def test_every_sentence_comes_from_the_document(self) -> None:
        """An extractive summary cannot state something the source does not."""
        result = summarise(RESEARCH_TEXT, max_sentences=2)
        for sentence in result.sentences:
            assert sentence in RESEARCH_TEXT

    def test_the_sentence_budget_is_respected(self) -> None:
        assert len(summarise(RESEARCH_TEXT, max_sentences=2).sentences) == 2

    def test_sentences_keep_their_original_order(self) -> None:
        """Score order reads as a series of non-sequiturs when spoken."""
        result = summarise(RESEARCH_TEXT, max_sentences=3)
        positions = [RESEARCH_TEXT.index(sentence) for sentence in result.sentences]
        assert positions == sorted(positions)

    def test_keywords_exclude_filler_words(self) -> None:
        result = summarise(RESEARCH_TEXT)
        assert "tactile" in result.keywords
        assert "that" not in result.keywords

    def test_the_same_input_summarises_identically(self) -> None:
        assert summarise(RESEARCH_TEXT).text == summarise(RESEARCH_TEXT).text

    def test_a_short_document_survives_intact(self) -> None:
        result = summarise("Short note.", max_sentences=3)
        assert "Short note." in result.text

    def test_empty_input_is_not_an_error(self) -> None:
        assert summarise("").is_empty
