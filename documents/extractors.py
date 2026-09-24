"""Text extraction, one function per format.

Every extractor obeys the same three rules:

Uses the configured workflow.
  scripts, no external entity resolution;
* it stops at a character budget rather than reading a whole file into memory,
  Uses the configured workflow.
* it fails soft. A corrupt PDF in the middle of a folder returns an error for
  that file and lets the indexing run continue.

Format is decided by extension. Sniffing content would be more thorough, but
extension is what the owner sees, and a mismatch is better reported than
silently worked around.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.logging_config import get_logger
from shared.enums import DocumentKind
from shared.errors import ValidationError

logger = get_logger(__name__)

DEFAULT_MAX_CHARS = 200_000
MAX_PDF_PAGES = 300
MAX_SHEET_ROWS = 2000
MAX_CSV_ROWS = 2000

TEXT_EXTENSIONS = {
    ".txt",
    ".text",
    ".log",
    ".rst",
    ".tex",
    ".cfg",
    ".ini",
    ".toml",
    ".yaml",
    ".yml",
}
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".rs",
    ".go",
    ".java",
    ".sh",
    ".bash",
    ".sql",
    ".r",
    ".m",
    ".jl",
}

EXTENSION_KINDS: dict[str, DocumentKind] = {
    ".pdf": DocumentKind.PDF,
    ".docx": DocumentKind.DOCX,
    ".pptx": DocumentKind.PPTX,
    ".xlsx": DocumentKind.XLSX,
    ".csv": DocumentKind.CSV,
    ".json": DocumentKind.JSON,
    ".md": DocumentKind.MARKDOWN,
    ".markdown": DocumentKind.MARKDOWN,
    **dict.fromkeys(TEXT_EXTENSIONS, DocumentKind.TEXT),
    **dict.fromkeys(CODE_EXTENSIONS, DocumentKind.CODE),
}

SUPPORTED_EXTENSIONS = frozenset(EXTENSION_KINDS)

_WHITESPACE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass
class ExtractedText:
    text: str
    kind: DocumentKind
    page_count: int | None = None
    truncated: bool = False
    metadata: dict[str, Any] | None = None


def kind_for(path: Path) -> DocumentKind:
    return EXTENSION_KINDS.get(path.suffix.lower(), DocumentKind.OTHER)


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def extract(path: Path, *, max_chars: int = DEFAULT_MAX_CHARS) -> ExtractedText:
    """Pull readable text out of a file, dispatching on extension."""
    suffix = path.suffix.lower()
    kind = kind_for(path)

    if kind is DocumentKind.OTHER:
        raise ValidationError(
            f"{path.name} has no text extractor. Supported formats: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    if suffix == ".pdf":
        return _extract_pdf(path, max_chars)
    if suffix == ".docx":
        return _extract_docx(path, max_chars)
    if suffix == ".pptx":
        return _extract_pptx(path, max_chars)
    if suffix == ".xlsx":
        return _extract_xlsx(path, max_chars)
    if suffix == ".csv":
        return _extract_csv(path, max_chars)
    if suffix == ".json":
        return _extract_json(path, max_chars)
    return _extract_plain(path, kind, max_chars)


def _finish(text: str, kind: DocumentKind, max_chars: int, **extra: Any) -> ExtractedText:
    cleaned = clean_text(text)
    truncated = len(cleaned) > max_chars
    return ExtractedText(text=cleaned[:max_chars], kind=kind, truncated=truncated, **extra)


def clean_text(text: str) -> str:
    """Normalise whitespace and strip control characters.

    Extracted text goes into an FTS index and then into speech; stray control
    characters break both, and PDF extraction produces plenty of them.
    """
    normalised = unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n"))
    without_controls = "".join(
        char for char in normalised if char == "\n" or unicodedata.category(char)[0] != "C"
    )
    collapsed = _WHITESPACE.sub(" ", without_controls)
    return _BLANK_LINES.sub("\n\n", collapsed).strip()


def _extract_plain(path: Path, kind: DocumentKind, max_chars: int) -> ExtractedText:
    # Read a little past the budget so truncation is detected without loading a
    # multi-gigabyte log file.
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        raw = handle.read(max_chars + 1024)
    return _finish(raw, kind, max_chars)


def _extract_pdf(path: Path, max_chars: int) -> ExtractedText:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            # An empty-password unlock covers the common "protected but not
            # really" case; anything else is left alone.
            try:
                reader.decrypt("")
            except (NotImplementedError, PdfReadError) as exc:
                raise ValidationError(f"{path.name} is encrypted and cannot be read.") from exc

        chunks: list[str] = []
        total = 0
        pages = reader.pages[:MAX_PDF_PAGES]
        for page in pages:
            try:
                chunks.append(page.extract_text() or "")
            except Exception:  # # a malformed page leaves the remaining document index intact
                logger.debug("Skipped an unreadable PDF page", extra={"file": path.name})
                continue
            total += len(chunks[-1])
            if total > max_chars:
                break

        return _finish(
            "\n\n".join(chunks), DocumentKind.PDF, max_chars, page_count=len(reader.pages)
        )
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"{path.name} could not be read as a PDF.") from exc


def _extract_docx(path: Path, max_chars: int) -> ExtractedText:
    import docx

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        # python-docx raises its own package errors that share no common base
        # with the zip and XML errors underneath them.
        raise ValidationError(f"{path.name} could not be read as a Word document.") from exc

    parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))

    return _finish("\n".join(parts), DocumentKind.DOCX, max_chars)


def _extract_pptx(path: Path, max_chars: int) -> ExtractedText:
    from pptx import Presentation

    try:
        presentation = Presentation(str(path))
    except Exception as exc:
        raise ValidationError(f"{path.name} could not be read as a presentation.") from exc

    parts: list[str] = []
    slide_count = 0
    for index, slide in enumerate(presentation.slides, start=1):
        slide_count = index
        slide_text = [
            shape.text.strip()
            for shape in slide.shapes
            if getattr(shape, "has_text_frame", False) and shape.text.strip()
        ]
        if slide_text:
            parts.append(f"[Slide {index}]\n" + "\n".join(slide_text))

    return _finish("\n\n".join(parts), DocumentKind.PPTX, max_chars, page_count=slide_count)


def _extract_xlsx(path: Path, max_chars: int) -> ExtractedText:
    from openpyxl import load_workbook

    try:
        # read_only streams rows instead of building the whole sheet, and
        # data_only takes cached values so no formula is ever evaluated.
        workbook = load_workbook(str(path), read_only=True, data_only=True)
    except Exception as exc:
        raise ValidationError(f"{path.name} could not be read as a spreadsheet.") from exc

    try:
        parts: list[str] = []
        total = 0
        for sheet in workbook.worksheets:
            parts.append(f"[Sheet: {sheet.title}]")
            for row_index, row in enumerate(sheet.iter_rows(values_only=True)):
                if row_index >= MAX_SHEET_ROWS:
                    parts.append("[further rows omitted]")
                    break
                cells = [str(value) for value in row if value is not None]
                if cells:
                    line = " | ".join(cells)
                    parts.append(line)
                    total += len(line)
            if total > max_chars:
                break
        return _finish("\n".join(parts), DocumentKind.XLSX, max_chars)
    finally:
        workbook.close()


def _extract_csv(path: Path, max_chars: int) -> ExtractedText:
    parts: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        try:
            dialect = csv.Sniffer().sniff(handle.read(4096)) if path.stat().st_size else None
        except csv.Error:
            dialect = None
        handle.seek(0)
        reader = csv.reader(handle, dialect) if dialect else csv.reader(handle)
        for index, row in enumerate(reader):
            if index >= MAX_CSV_ROWS:
                parts.append("[further rows omitted]")
                break
            parts.append(" | ".join(cell.strip() for cell in row))

    return _finish("\n".join(parts), DocumentKind.CSV, max_chars)


def _extract_json(path: Path, max_chars: int) -> ExtractedText:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Malformed JSON is still text worth searching.
        return _finish(raw, DocumentKind.JSON, max_chars)
    return _finish(json.dumps(parsed, indent=2, ensure_ascii=False), DocumentKind.JSON, max_chars)
