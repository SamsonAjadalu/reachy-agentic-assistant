"""Filesystem containment.

Every path that originates outside this process - an API argument, a registry
entry, a filename inside an archive - is resolved through here before it is
opened. The rule is that the fully resolved real path must sit under an
explicitly approved root.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

from shared.errors import SecurityViolationError, ValidationError

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._\- ]+")
_MAX_FILENAME_LENGTH = 120

# Windows reserves these regardless of extension; harmless to avoid on Linux too.
_RESERVED_STEMS = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
)


def normalise_roots(roots: list[str] | list[Path]) -> list[Path]:
    """Uses the configured workflow."""
    resolved: list[Path] = []
    for root in roots:
        path = Path(root).expanduser()
        try:
            real = path.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if real.is_dir():
            resolved.append(real)
    return resolved


def is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_within_roots(
    raw_path: str | Path,
    roots: list[Path],
    *,
    follow_symlinks: bool = False,
    must_exist: bool = True,
) -> Path:
    """Resolve ``raw_path`` and prove it lies inside one of ``roots``.

    ``..`` traversal is defeated by resolving before comparing. Symlink escape is
    defeated by comparing the real path, and by rejecting symlinks outright
    unless the deployment has opted in.
    """
    if not roots:
        raise SecurityViolationError(
            "No approved root directories are configured for this operation."
        )

    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        # A relative path is only meaningful against a root, and accepting one
        # would make the effective target depend on the process cwd.
        matches = [root / candidate for root in roots if (root / candidate).exists()]
        if not matches:
            raise SecurityViolationError("Path is not inside an approved directory.")
        candidate = matches[0]

    if not follow_symlinks:
        # Check every component: a symlinked parent directory is just as much an
        # escape as a symlinked leaf.
        probe = candidate
        while True:
            if probe.is_symlink():
                raise SecurityViolationError("Symlinked paths are not permitted.")
            if probe.parent == probe:
                break
            probe = probe.parent

    try:
        real = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise SecurityViolationError("Path could not be resolved.") from exc

    # Containment is checked before existence deliberately. Answering "that file
    # does not exist" for a path outside the roots would make this an existence
    # oracle for the rest of the filesystem.
    if not any(is_within(real, root) or real == root for root in roots):
        raise SecurityViolationError("Path is not inside an approved directory.")

    if must_exist and not real.exists():
        raise ValidationError("The requested file does not exist.")
    return real


def safe_filename(raw: str, *, default: str = "file", extension: str | None = None) -> str:
    """Reduce arbitrary text to a filename safe to create on disk.

    Used for attachment downloads and wardrobe uploads, where the name is
    Uses the configured workflow.
    """
    text = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    text = text.replace("\\", "/").split("/")[-1].strip()
    text = _UNSAFE_FILENAME.sub("_", text).strip("._ ")

    stem, _, ext = text.rpartition(".")
    if not stem:
        stem, ext = text, ""
    if not stem or stem.lower() in _RESERVED_STEMS:
        stem = default
    if extension is not None:
        ext = extension.lstrip(".")

    stem = stem[:_MAX_FILENAME_LENGTH]
    return f"{stem}.{ext}" if ext else stem


def assert_readable(path: Path) -> None:
    if not path.is_file():
        raise ValidationError("The requested path is not a regular file.")
    if not os.access(path, os.R_OK):
        raise SecurityViolationError("The file is not readable by the assistant service.")


def bounded_size(path: Path, max_bytes: int) -> int:
    """Return the file size, refusing anything over the configured ceiling."""
    size = path.stat().st_size
    if size > max_bytes:
        raise ValidationError(
            f"File is {size} bytes, over the {max_bytes} byte limit for this operation."
        )
    return size
