"""Content-addressed evidence files.

Write file → fsync → os.replace → then the caller inserts the database row.
Uses the configured workflow.
"""

from __future__ import annotations

import errno
import hashlib
import io
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image

from app.config import Settings
from vision.enums import EvidenceKind, RetentionClass
from vision.hashing import dhash64, phash64
from vision.paths import assert_content_hash, evidence_relative_path
from vision.schemas import StoredEvidence
from wardrobe.images import load_image

THUMBNAIL_EDGE = 224
JPEG_QUALITY = 88
FULL_FRAME_MAX_EDGE = 1280


class DiskFullError(OSError):
    """The evidence volume has no space left for a new file."""

    def __init__(self, message: str = "Visual evidence volume is full.") -> None:
        super().__init__(errno.ENOSPC, message)


@dataclass
class EncodedStill:
    image: Image.Image
    jpeg_bytes: bytes
    content_hash: str
    dhash64: str
    phash64: str
    width: int
    height: int


def visual_evidence_root(settings: Settings) -> Path:
    return settings.visual_evidence_path


def relative_evidence_path(kind: EvidenceKind | str, content_hash: str) -> str:
    return evidence_relative_path(kind, content_hash)


def encode_jpeg(
    image: Image.Image, *, max_edge: int | None = None, quality: int = JPEG_QUALITY
) -> bytes:
    working = image.convert("RGB")
    if max_edge is not None:
        working = working.copy()
        working.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    working.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def prepare_image(data: bytes) -> Image.Image:
    """Decode via the wardrobe path so EXIF (including GPS) is dropped."""
    return load_image(data)


def encode_still(data: bytes, *, max_edge: int | None = FULL_FRAME_MAX_EDGE) -> EncodedStill:
    image = prepare_image(data)
    jpeg = encode_jpeg(image, max_edge=max_edge)
    stored = Image.open(io.BytesIO(jpeg))
    stored.load()
    digest = hashlib.sha256(jpeg).hexdigest()
    dhash, phash = dhash64(image), phash64(image)
    return EncodedStill(
        image=image,
        jpeg_bytes=jpeg,
        content_hash=digest,
        dhash64=dhash,
        phash64=phash,
        width=stored.width,
        height=stored.height,
    )


def encode_thumbnail(image: Image.Image) -> EncodedStill:
    jpeg = encode_jpeg(image, max_edge=THUMBNAIL_EDGE, quality=80)
    stored = Image.open(io.BytesIO(jpeg))
    stored.load()
    digest = hashlib.sha256(jpeg).hexdigest()
    dhash, phash = dhash64(image), phash64(image)
    return EncodedStill(
        image=image,
        jpeg_bytes=jpeg,
        content_hash=digest,
        dhash64=dhash,
        phash64=phash,
        width=stored.width,
        height=stored.height,
    )


def resolve_retention(
    requested: RetentionClass,
    *,
    contains_person: bool,
    privacy_sensitive: bool = False,
) -> RetentionClass:
    if contains_person or privacy_sensitive:
        return RetentionClass.EPHEMERAL
    if requested is RetentionClass.PINNED and contains_person:
        return RetentionClass.EPHEMERAL
    return requested


def expires_at_for(
    retention: RetentionClass,
    *,
    now: datetime,
    settings: Settings,
    contains_person: bool = False,
) -> datetime | None:
    resolved = resolve_retention(retention, contains_person=contains_person)
    if resolved is RetentionClass.PINNED or resolved is RetentionClass.EVALUATION:
        return None
    if resolved is RetentionClass.EPHEMERAL:
        return now + timedelta(minutes=settings.visual_ephemeral_retention_minutes)
    if resolved is RetentionClass.THUMBNAIL:
        return now + timedelta(days=settings.visual_thumbnail_retention_days)
    return now + timedelta(days=settings.visual_evidence_retention_days)


def write_evidence_file(
    settings: Settings,
    encoded: EncodedStill,
    *,
    kind: EvidenceKind,
    retention: RetentionClass,
    contains_person: bool = False,
    privacy_sensitive: bool = False,
) -> StoredEvidence:
    digest = assert_content_hash(encoded.content_hash)
    resolved = resolve_retention(
        retention, contains_person=contains_person, privacy_sensitive=privacy_sensitive
    )
    relative = relative_evidence_path(kind, digest)
    destination = visual_evidence_root(settings) / relative
    if kind is EvidenceKind.FULL_FRAME and not settings.visual_keep_full_frames:
        from shared.errors import ValidationError

        raise ValidationError("Full-frame retention is disabled.")
    if resolved is RetentionClass.PINNED and (contains_person or privacy_sensitive):
        from shared.errors import ValidationError

        raise ValidationError("Person-containing evidence cannot be pinned.")
    already = destination.is_file()
    if not already:
        atomic_replace(destination, encoded.jpeg_bytes)
    byte_size = len(encoded.jpeg_bytes) if already else destination.stat().st_size
    return StoredEvidence(
        content_hash=digest,
        relative_path=relative,
        kind=kind,
        retention_class=resolved,
        byte_size=byte_size,
        width=encoded.width,
        height=encoded.height,
        dhash64=encoded.dhash64,
        phash64=encoded.phash64,
        reused=already,
        contains_person=contains_person,
    )


def atomic_replace(destination: Path, data: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for folder in (destination.parent, destination.parent.parent, destination.parent.parent.parent):
        if folder.exists():
            folder.chmod(0o700)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".jpg", dir=destination.parent)
    tmp_path = Path(tmp_name)
    try:
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, destination)
        _fsync_directory(destination.parent)
        destination.chmod(0o600)
    except OSError as exc:
        if fd >= 0:
            os.close(fd)
        tmp_path.unlink(missing_ok=True)
        if exc.errno == errno.ENOSPC:
            raise DiskFullError() from exc
        raise
    except BaseException:
        if fd >= 0:
            os.close(fd)
        tmp_path.unlink(missing_ok=True)
        raise


def unlink_evidence_file(settings: Settings, relative_path: str) -> bool:
    from security.paths import resolve_within_roots

    root = visual_evidence_root(settings).resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        path = resolve_within_roots(root / relative_path, [root], must_exist=False)
    except Exception:
        return False
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return True


def quarantine_orphan(settings: Settings, file_path: Path) -> Path:
    root = visual_evidence_root(settings)
    relative = file_path.relative_to(root)
    target = settings.visual_quarantine_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.parent.chmod(0o700)
    file_path.replace(target)
    target.chmod(0o600)
    return target


def read_evidence(settings: Settings, relative_path: str) -> bytes:
    from security.paths import resolve_within_roots

    root = visual_evidence_root(settings).resolve()
    resolved = resolve_within_roots(root / relative_path, [root])
    return resolved.read_bytes()


def _fsync_directory(directory: Path) -> None:
    dir_fd = os.open(directory, os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
