"""Frame decode and APP_DATA_DIR path confinement."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from shared.errors import NotFoundError, SecurityViolationError, ValidationError

MAX_PIXELS = 20_000_000


def decode_frame(
    *,
    image_b64: str | None,
    image_path: str | None,
    data_dir: Path,
) -> tuple[Image.Image, bytes, str]:
    raw = _read_bytes(image_b64=image_b64, image_path=image_path, data_dir=data_dir)
    digest = hashlib.sha256(raw).hexdigest()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except UnidentifiedImageError as exc:
        raise ValidationError(
            "Frame bytes are not a decodable image.", code="decode_error"
        ) from exc
    image = image.convert("RGB")
    if image.width * image.height > MAX_PIXELS:
        raise ValidationError("Frame exceeds the pixel ceiling.")
    return image, raw, digest


def _read_bytes(*, image_b64: str | None, image_path: str | None, data_dir: Path) -> bytes:
    if image_b64:
        try:
            return base64.b64decode(image_b64, validate=True)
        except binascii.Error as exc:
            raise ValidationError("image_b64 is not valid base64.") from exc
    if not image_path:
        raise ValidationError("No frame was supplied.")
    path = _confine_path(image_path, data_dir)
    return path.read_bytes()


def _confine_path(image_path: str, data_dir: Path) -> Path:
    data = data_dir.expanduser().resolve()
    resolved = Path(image_path).expanduser().resolve()
    if resolved != data and data not in resolved.parents:
        raise SecurityViolationError("Image path is outside APP_DATA_DIR.")
    if not resolved.is_file():
        raise NotFoundError("Image path does not exist.")
    return resolved
