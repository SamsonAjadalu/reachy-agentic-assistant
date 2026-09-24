"""Indexing, search and containment.

The containment cases are the important ones: the index walks a directory the
owner nominated, but the paths that come back out of it are then handed to a
Uses the configured workflow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from documents.index import DocumentIndex, sanitise_query
from shared.errors import SecurityViolationError, ValidationError
from tests.fixtures.make_documents import build_all


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return build_all(tmp_path / "docs")


@pytest.fixture
def index(settings: Settings, corpus: Path, tmp_path: Path) -> DocumentIndex:
    settings.document_index_roots = [str(corpus)]
    settings.app_data_dir = tmp_path / "data"
    (tmp_path / "data").mkdir(exist_ok=True)
    return DocumentIndex(settings)


class TestIndexing:
    def test_a_first_run_indexes_every_supported_file(self, index: DocumentIndex) -> None:
        stats = index.reindex()
        assert stats.indexed == 10  # every fixture except the binary
        assert stats.failed == 0

    def test_a_second_run_reads_nothing_again(self, index: DocumentIndex) -> None:
        """Incremental behaviour is what makes an hourly re-index affordable."""
        index.reindex()
        stats = index.reindex()
        assert stats.indexed == 0
        assert stats.skipped == 10

    def test_a_changed_file_is_reread(self, index: DocumentIndex, corpus: Path) -> None:
        index.reindex()
        target = corpus / "research_notes.txt"
        target.write_text("Completely different content about lidar.", encoding="utf-8")

        stats = index.reindex()

        assert stats.updated == 1
        assert index.search("lidar")[0].name == "research_notes.txt"

    def test_a_deleted_file_stops_being_a_result(self, index: DocumentIndex, corpus: Path) -> None:
        index.reindex()
        (corpus / "research_notes.txt").unlink()

        stats = index.reindex()

        assert stats.removed == 1
        assert all(hit.name != "research_notes.txt" for hit in index.search("tactile"))

    def test_a_full_rebuild_starts_from_empty(self, index: DocumentIndex) -> None:
        index.reindex()
        stats = index.reindex(full=True)
        assert stats.indexed == 10
        assert stats.skipped == 0

    def test_one_unreadable_file_does_not_stop_the_run(
        self, index: DocumentIndex, corpus: Path
    ) -> None:
        (corpus / "corrupt.docx").write_bytes(b"not a zip file at all")

        stats = index.reindex()

        assert stats.failed == 1
        assert stats.indexed == 10
        assert any("corrupt.docx" in error for error in stats.errors)

    def test_noise_directories_are_skipped(self, index: DocumentIndex, corpus: Path) -> None:
        """A .git or node_modules folder would dwarf the documents it sits beside."""
        noise = corpus / "node_modules" / "pkg"
        noise.mkdir(parents=True)
        (noise / "index.js").write_text("module.exports = {}", encoding="utf-8")

        stats = index.reindex()

        assert stats.scanned == 10

    def test_an_oversized_file_is_refused(
        self, index: DocumentIndex, corpus: Path, settings: Settings
    ) -> None:
        settings.document_max_bytes = 100
        (corpus / "huge.txt").write_text("x" * 500, encoding="utf-8")

        stats = index.reindex()

        assert any("huge.txt" in error for error in stats.errors)

    def test_no_configured_roots_is_reported_not_crashed(self, settings: Settings) -> None:
        settings.document_index_roots = []
        stats = DocumentIndex(settings).reindex()
        assert stats.scanned == 0
        assert "No document roots" in stats.errors[0]

    def test_progress_is_reported_for_every_file(self, index: DocumentIndex) -> None:
        seen: list[tuple[int, int]] = []
        index.reindex(progress=lambda position, total: seen.append((position, total)))
        assert len(seen) == 10
        assert seen[-1] == (10, 10)


class TestSearch:
    @pytest.fixture(autouse=True)
    def _indexed(self, index: DocumentIndex) -> None:
        index.reindex()

    def test_a_word_finds_its_document(self, index: DocumentIndex) -> None:
        assert any(hit.name == "paper.pdf" for hit in index.search("deformable"))

    def test_results_carry_a_snippet(self, index: DocumentIndex) -> None:
        assert "tactile" in index.search("tactile")[0].snippet.lower()

    def test_stemming_finds_related_forms(self, index: DocumentIndex) -> None:
        """The porter tokeniser is why 'trials' finds 'trial'."""
        assert index.search("trials")

    def test_results_can_be_narrowed_by_format(self, index: DocumentIndex) -> None:
        hits = index.search("tactile", kind="pdf")
        assert hits and all(hit.kind == "pdf" for hit in hits)

    def test_the_best_match_comes_first(self, index: DocumentIndex) -> None:
        hits = index.search("grasp success")
        assert hits[0].score >= hits[-1].score

    def test_an_unmatched_word_returns_nothing(self, index: DocumentIndex) -> None:
        assert index.search("xylophone") == []

    @pytest.mark.parametrize(
        "query",
        [
            'tactile" OR body:"secret',
            "tactile NEAR/5 grasp",
            "path: /etc/passwd",
            "tactile AND (grasp OR *",
            "'; DROP TABLE documents; --",
        ],
    )
    def test_fts_operators_in_a_query_are_neutralised(
        self, index: DocumentIndex, query: str
    ) -> None:
        """Uses the configured workflow."""
        index.search(query)  # no exception is the assertion

    def test_a_query_with_no_words_is_refused(self, index: DocumentIndex) -> None:
        with pytest.raises(ValidationError):
            index.search("!!! ***")

    def test_quotes_are_stripped_from_terms(self) -> None:
        assert '"' not in sanitise_query('tactile"').replace('"tactile"', "")


class TestReading:
    @pytest.fixture(autouse=True)
    def _indexed(self, index: DocumentIndex) -> None:
        index.reindex()

    def test_an_indexed_file_reads_back(self, index: DocumentIndex, corpus: Path) -> None:
        payload = index.get_text(str(corpus / "research_notes.txt"))
        assert "Tactile feedback" in payload["text"]
        assert payload["content_is_untrusted"] is True

    def test_a_path_outside_the_roots_is_refused(self, index: DocumentIndex) -> None:
        with pytest.raises(SecurityViolationError):
            index.get_text("/etc/passwd")

    def test_traversal_out_of_a_root_is_refused(self, index: DocumentIndex, corpus: Path) -> None:
        with pytest.raises(SecurityViolationError):
            index.get_text(str(corpus / ".." / ".." / "etc" / "passwd"))

    def test_a_symlink_pointing_out_is_refused(
        self, index: DocumentIndex, corpus: Path, tmp_path: Path
    ) -> None:
        """Uses the configured workflow."""
        secret = tmp_path / "secret.txt"
        secret.write_text("private", encoding="utf-8")
        (corpus / "link.txt").symlink_to(secret)

        with pytest.raises(SecurityViolationError):
            index.get_text(str(corpus / "link.txt"))

    def test_a_symlink_is_not_indexed_by_default(
        self, index: DocumentIndex, corpus: Path, tmp_path: Path
    ) -> None:
        secret = tmp_path / "secret.txt"
        secret.write_text("classified marker word", encoding="utf-8")
        (corpus / "link.txt").symlink_to(secret)

        index.reindex(full=True)

        assert index.search("classified") == []

    def test_a_missing_file_is_a_validation_error(self, index: DocumentIndex, corpus: Path) -> None:
        with pytest.raises(ValidationError):
            index.get_text(str(corpus / "not-here.txt"))


class TestStats:
    def test_stats_describe_the_index(self, index: DocumentIndex) -> None:
        index.reindex()
        stats = index.stats()
        assert stats["documents"] == 10
        assert stats["by_kind"]["text"] >= 1
        assert stats["last_indexed_at"]

    def test_stats_work_before_the_first_run(self, index: DocumentIndex) -> None:
        assert index.stats()["documents"] == 0

    def test_the_index_lives_outside_the_application_database(
        self, index: DocumentIndex, settings: Settings
    ) -> None:
        """It is rebuildable, so it has no business in a backup of things that are not."""
        assert index.path != settings.database_path
        assert index.path.name.endswith(".db")
