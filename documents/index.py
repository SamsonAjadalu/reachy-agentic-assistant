"""Full-text index over the owner's local documents.

Kept in its own SQLite file rather than in the application database, for three
reasons: it is rebuildable, so it has no business in a backup of things that are
not; FTS5 tables are large relative to what they index; and a corrupt index
should be deletable without touching reminders and approvals.

Indexing is incremental. A file is re-read only when its size or modification
time changed, which is what makes a scheduled re-index cheap enough to run
hourly over a large folder.

Every path is resolved through ``security.paths`` against DOCUMENT_INDEX_ROOTS
before it is opened, so neither a configured root with a symlink in it nor a
crafted search result can reach outside the approved directories.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import sqlite3
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.logging_config import get_logger
from documents.extractors import extract, is_supported
from security.paths import bounded_size, normalise_roots, resolve_within_roots
from shared.errors import ValidationError
from shared.timeutils import utcnow

logger = get_logger(__name__)

SCHEMA_VERSION = 1
SNIPPET_TOKENS = 24
MAX_RESULTS = 100

# Directories that are never worth indexing and are frequently enormous.
SKIP_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "site-packages",
        ".cache",
        "dist",
        "build",
        ".Trash",
    }
)


@dataclass
class IndexStats:
    scanned: int = 0
    indexed: int = 0
    updated: int = 0
    skipped: int = 0
    removed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "indexed": self.indexed,
            "updated": self.updated,
            "skipped": self.skipped,
            "removed": self.removed,
            "failed": self.failed,
            "errors": self.errors[:20],
            "duration_seconds": round(self.duration_seconds, 2),
        }


@dataclass
class SearchHit:
    path: str
    name: str
    kind: str
    snippet: str
    score: float
    size_bytes: int
    modified_at: str
    page_count: int | None = None


class DocumentIndex:
    """FTS5 index over the configured document roots."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._path = self._settings.document_index_path
        self._roots = normalise_roots(list(self._settings.document_index_roots))

    @property
    def roots(self) -> list[Path]:
        return list(self._roots)

    @property
    def path(self) -> Path:
        return self._path

    # ----------------------------------------------------------------- schema
    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    path         TEXT PRIMARY KEY,
                    name         TEXT NOT NULL,
                    kind         TEXT NOT NULL,
                    size_bytes   INTEGER NOT NULL,
                    mtime        REAL NOT NULL,
                    content_hash TEXT NOT NULL,
                    page_count   INTEGER,
                    truncated    INTEGER NOT NULL DEFAULT 0,
                    indexed_at   TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(
                    path UNINDEXED,
                    name,
                    body,
                    tokenize = 'porter unicode61'
                );

                CREATE TABLE IF NOT EXISTS index_meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    # ------------------------------------------------------------- traversal
    def iter_candidates(self) -> Iterator[Path]:
        """Walk the roots, skipping noise directories and unsupported files."""
        for root in self._roots:
            for entry in sorted(root.rglob("*")):
                if any(part in SKIP_DIRECTORIES for part in entry.parts):
                    continue
                if entry.is_symlink() and not self._settings.document_follow_symlinks:
                    continue
                if not entry.is_file() or not is_supported(entry):
                    continue
                yield entry

    # ------------------------------------------------------------- indexing
    def reindex(
        self,
        *,
        full: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> IndexStats:
        """Bring the index up to date. Blocking; call through ``reindex_async``."""
        started = time.monotonic()
        stats = IndexStats()
        self.ensure_schema()

        if not self._roots:
            stats.errors.append(
                "No document roots are configured. Set DOCUMENT_INDEX_ROOTS to index anything."
            )
            return stats

        with self._connect() as connection:
            if full:
                connection.execute("DELETE FROM documents")
                connection.execute("DELETE FROM document_fts")

            known = {
                row["path"]: (row["size_bytes"], row["mtime"])
                for row in connection.execute("SELECT path, size_bytes, mtime FROM documents")
            }
            seen: set[str] = set()
            candidates = list(self.iter_candidates())

            for position, file_path in enumerate(candidates, start=1):
                stats.scanned += 1
                key = str(file_path)
                seen.add(key)

                try:
                    info = file_path.stat()
                except OSError:
                    stats.failed += 1
                    continue

                previous = known.get(key)
                if previous and previous == (info.st_size, info.st_mtime):
                    stats.skipped += 1
                    if progress:
                        progress(position, len(candidates))
                    continue

                try:
                    self._index_one(connection, file_path, info.st_size, info.st_mtime)
                except ValidationError as exc:
                    stats.failed += 1
                    stats.errors.append(f"{file_path.name}: {exc.message}")
                except Exception as exc:
                    stats.failed += 1
                    stats.errors.append(f"{file_path.name}: {type(exc).__name__}")
                    logger.warning(
                        "Indexing failed for a file",
                        extra={"file": file_path.name, "error": type(exc).__name__},
                    )
                else:
                    if previous:
                        stats.updated += 1
                    else:
                        stats.indexed += 1

                if progress:
                    progress(position, len(candidates))

            # A file that vanished should stop being a search result.
            for stale in set(known) - seen:
                connection.execute("DELETE FROM documents WHERE path = ?", (stale,))
                connection.execute("DELETE FROM document_fts WHERE path = ?", (stale,))
                stats.removed += 1

        stats.duration_seconds = time.monotonic() - started
        logger.info("Document index updated", extra=stats.as_dict())
        return stats

    def _index_one(
        self, connection: sqlite3.Connection, path: Path, size: int, mtime: float
    ) -> None:
        bounded_size(path, self._settings.document_max_bytes)
        extracted = extract(path)
        content_hash = hashlib.sha256(extracted.text.encode("utf-8")).hexdigest()

        connection.execute(
            """
            INSERT INTO documents
                (path, name, kind, size_bytes, mtime, content_hash, page_count,
                 truncated, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                name = excluded.name,
                kind = excluded.kind,
                size_bytes = excluded.size_bytes,
                mtime = excluded.mtime,
                content_hash = excluded.content_hash,
                page_count = excluded.page_count,
                truncated = excluded.truncated,
                indexed_at = excluded.indexed_at
            """,
            (
                str(path),
                path.name,
                extracted.kind.value,
                size,
                mtime,
                content_hash,
                extracted.page_count,
                int(extracted.truncated),
                utcnow().isoformat(),
            ),
        )
        connection.execute("DELETE FROM document_fts WHERE path = ?", (str(path),))
        connection.execute(
            "INSERT INTO document_fts (path, name, body) VALUES (?, ?, ?)",
            (str(path), path.name, extracted.text),
        )

    async def reindex_async(
        self, *, full: bool = False, progress: Callable[[int, int], None] | None = None
    ) -> IndexStats:
        """Run the blocking scan off the event loop."""
        return await asyncio.to_thread(self.reindex, full=full, progress=progress)

    # --------------------------------------------------------------- queries
    def search(self, query: str, *, limit: int = 10, kind: str | None = None) -> list[SearchHit]:
        cleaned = sanitise_query(query)
        limit = max(1, min(limit, MAX_RESULTS))
        self.ensure_schema()

        sql = """
            SELECT d.path, d.name, d.kind, d.size_bytes, d.indexed_at, d.page_count,
                   snippet(document_fts, 2, '', '', ' ... ', ?) AS snippet,
                   bm25(document_fts) AS score
            FROM document_fts
            JOIN documents d ON d.path = document_fts.path
            WHERE document_fts MATCH ?
        """
        params: list[Any] = [SNIPPET_TOKENS, cleaned]
        if kind:
            sql += " AND d.kind = ?"
            params.append(kind)
        # bm25 returns lower-is-better, so ascending order is most relevant first.
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)

        with self._connect() as connection:
            try:
                rows = connection.execute(sql, params).fetchall()
            except sqlite3.OperationalError as exc:
                raise ValidationError(f"That search could not be parsed: {exc}") from exc

        return [
            SearchHit(
                path=row["path"],
                name=row["name"],
                kind=row["kind"],
                snippet=" ".join((row["snippet"] or "").split()),
                score=round(-float(row["score"]), 4),
                size_bytes=row["size_bytes"],
                modified_at=row["indexed_at"],
                page_count=row["page_count"],
            )
            for row in rows
        ]

    def get_text(self, raw_path: str, *, max_chars: int = 20000) -> dict[str, Any]:
        """Read one document, re-checking containment at read time."""
        resolved = resolve_within_roots(
            raw_path,
            self._roots,
            follow_symlinks=self._settings.document_follow_symlinks,
        )
        bounded_size(resolved, self._settings.document_max_bytes)
        extracted = extract(resolved, max_chars=max_chars)
        return {
            "path": str(resolved),
            "name": resolved.name,
            "kind": extracted.kind.value,
            "text": extracted.text,
            "truncated": extracted.truncated,
            "page_count": extracted.page_count,
            "content_is_untrusted": True,
        }

    def stats(self) -> dict[str, Any]:
        self.ensure_schema()
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            by_kind = {
                row["kind"]: row["n"]
                for row in connection.execute(
                    "SELECT kind, COUNT(*) AS n FROM documents GROUP BY kind ORDER BY n DESC"
                )
            }
            newest = connection.execute("SELECT MAX(indexed_at) AS last FROM documents").fetchone()[
                "last"
            ]

        return {
            "documents": total,
            "by_kind": by_kind,
            "last_indexed_at": newest,
            "roots": [str(root) for root in self._roots],
            "index_path": str(self._path),
            "index_size_bytes": self._path.stat().st_size if self._path.exists() else 0,
        }

    def clear(self) -> None:
        self.ensure_schema()
        with self._connect() as connection:
            connection.execute("DELETE FROM documents")
            connection.execute("DELETE FROM document_fts")


def sanitise_query(query: str) -> str:
    """Turn a spoken phrase into a safe FTS5 MATCH expression.

    FTS5 has its own syntax where a stray quote or a bare ``NEAR`` is a parse
    error, and column filters like ``path:`` would let a query reach past the
    body. Each word is quoted as a literal, which keeps phrase search working
    while removing every operator.
    """
    words = [word for word in re.findall(r"[\w'-]+", query, flags=re.UNICODE) if word]
    if not words:
        raise ValidationError("The search needs at least one word.")
    return " ".join(f'"{word.replace(chr(34), "")}"' for word in words[:20])
