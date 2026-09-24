"""Wardrobe image storage and colour analysis.

Images are the one place this service accepts arbitrary binary input, so they
get the strictest handling in the codebase:

* the format is decided by decoding the file, never by its name or the declared
  content type;
* pixel dimensions are checked before a full decode, because a small file can
  describe an enormous image;
* every saved file is re-encoded from the decoded pixels, which drops EXIF
  wholesale - including the GPS tag that says where the owner lives;
* the stored name is a content hash, so a crafted filename has nowhere to go.

Colour extraction is a small k-means over a downsampled image. It runs locally
in a few milliseconds and produces a proposal the owner confirms, rather than a
label the system asserts.
"""

from __future__ import annotations

import colorsys
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageOps, UnidentifiedImageError

from app.logging_config import get_logger
from shared.errors import ValidationError

logger = get_logger(__name__)

ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "HEIF", "HEIC"})
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 50_000_000
STORED_MAX_EDGE = 1600
THUMBNAIL_EDGE = 320
JPEG_QUALITY = 88

# Pillow's own guard against decompression bombs, set to our ceiling.
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

# Named colours the assistant can say out loud. Matching is done in HSV, which
# groups shades the way people describe them far better than RGB distance.
COLOUR_ANCHORS: list[tuple[str, tuple[float, float, float]]] = [
    ("black", (0.0, 0.0, 0.05)),
    ("charcoal", (0.0, 0.0, 0.22)),
    ("grey", (0.0, 0.0, 0.5)),
    ("light grey", (0.0, 0.0, 0.78)),
    ("white", (0.0, 0.0, 0.97)),
    ("red", (0.0, 0.75, 0.7)),
    ("burgundy", (0.97, 0.7, 0.35)),
    ("pink", (0.93, 0.35, 0.9)),
    ("orange", (0.07, 0.8, 0.85)),
    ("brown", (0.07, 0.6, 0.4)),
    ("tan", (0.09, 0.35, 0.75)),
    ("beige", (0.11, 0.18, 0.88)),
    ("yellow", (0.15, 0.75, 0.9)),
    ("olive", (0.2, 0.6, 0.45)),
    ("green", (0.33, 0.65, 0.55)),
    ("teal", (0.48, 0.6, 0.5)),
    ("light blue", (0.55, 0.35, 0.85)),
    ("blue", (0.6, 0.7, 0.6)),
    ("navy", (0.63, 0.75, 0.3)),
    ("purple", (0.78, 0.55, 0.5)),
]


@dataclass
class StoredImage:
    relative_path: str
    thumbnail_path: str
    width: int
    height: int
    bytes: int
    content_hash: str
    dominant_colours: list[dict[str, Any]]

    @property
    def primary_colour_name(self) -> str:
        return str(self.dominant_colours[0]["name"]) if self.dominant_colours else "unknown"


def load_image(data: bytes) -> Image.Image:
    """Decode and validate. Format comes from the bytes, not from a filename."""
    if not data:
        raise ValidationError("The image is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValidationError(
            f"The image is {len(data) // 1024} KB, over the "
            f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB limit."
        )

    try:
        probe = Image.open(io.BytesIO(data))
        image_format = (probe.format or "").upper()
        width, height = probe.size
        # verify() consumes the file object, so the real decode reopens below.
        probe.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValidationError("That file could not be decoded as an image.") from exc

    if image_format not in ALLOWED_FORMATS:
        raise ValidationError(
            f"{image_format or 'That format'} is not accepted. "
            f"Use one of: {', '.join(sorted(ALLOWED_FORMATS))}."
        )
    if width * height > MAX_PIXELS:
        # A few hundred kilobytes can describe a gigapixel image.
        raise ValidationError("The image dimensions are too large to process.")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (OSError, ValueError) as exc:
        raise ValidationError("The image data is truncated or corrupt.") from exc

    # Honour the orientation tag, then discard all metadata with it.
    return ImageOps.exif_transpose(image).convert("RGB")


def dominant_colours(image: Image.Image, *, count: int = 3) -> list[dict[str, Any]]:
    """Propose the main colours of a garment.

    Pillow's median-cut quantiser does the clustering; the work here is
    discarding near-white and near-black pixels, which are usually the wall and
    the shadow rather than the garment.
    """
    working = image.copy()
    working.thumbnail((160, 160))

    filtered = Image.new("RGB", working.size, (127, 127, 127))
    kept = 0
    # An RGB image always yields three-channel tuples; the signature covers modes
    # this function never sees.
    pixels = cast("list[tuple[int, int, int]]", list(working.get_flattened_data()))
    for index, pixel in enumerate(pixels):
        red, green, blue = pixel
        brightness = (red + green + blue) / 3
        if 18 <= brightness <= 240:
            filtered.putpixel((index % working.width, index // working.width), pixel)
            kept += 1

    source = filtered if kept > len(pixels) * 0.2 else working
    quantised = source.quantize(colors=max(count * 2, 6), method=Image.Quantize.MEDIANCUT)
    palette = quantised.getpalette() or []
    counts = sorted(quantised.getcolors() or [], reverse=True)

    results: list[dict[str, Any]] = []
    total = sum(pixel_count for pixel_count, _ in counts) or 1
    for pixel_count, entry in counts:
        if not isinstance(entry, int):  # a non-palette image would yield RGB tuples
            continue
        red, green, blue = palette[entry * 3 : entry * 3 + 3]
        name = name_colour(red, green, blue)
        if any(entry["name"] == name for entry in results):
            continue
        results.append(
            {
                "name": name,
                "hex": f"#{red:02x}{green:02x}{blue:02x}",
                "share": round(pixel_count / total, 3),
            }
        )
        if len(results) >= count:
            break
    return results


def name_colour(red: int, green: int, blue: int) -> str:
    """Nearest anchor in HSV, with hue distance treated as circular."""
    hue, saturation, value = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)

    best_name = "unknown"
    best_distance = float("inf")
    for name, (anchor_hue, anchor_saturation, anchor_value) in COLOUR_ANCHORS:
        hue_gap = abs(hue - anchor_hue)
        hue_gap = min(hue_gap, 1 - hue_gap)
        # Hue is meaningless on a grey pixel, so it is weighted by saturation.
        weight = min(saturation, anchor_saturation)
        distance = (
            (hue_gap * 3.0 * weight) ** 2
            + (saturation - anchor_saturation) ** 2
            + (value - anchor_value) ** 2
        )
        if distance < best_distance:
            best_distance = distance
            best_name = name
    return best_name


def store_image(data: bytes, root: Path, *, item_id: str) -> StoredImage:
    """Re-encode and save an image under ``root``.

    Saving the re-encoded pixels rather than the uploaded bytes is what removes
    EXIF, and with it the GPS coordinates most phone cameras attach.
    """
    image = load_image(data)
    content_hash = hashlib.sha256(data).hexdigest()

    stored = image.copy()
    stored.thumbnail((STORED_MAX_EDGE, STORED_MAX_EDGE), Image.Resampling.LANCZOS)

    thumbnail = image.copy()
    thumbnail.thumbnail((THUMBNAIL_EDGE, THUMBNAIL_EDGE), Image.Resampling.LANCZOS)

    directory = root / item_id
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / f"{content_hash[:16]}.jpg"
    thumbnail_path = directory / f"{content_hash[:16]}_thumb.jpg"

    stored.save(image_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    thumbnail.save(thumbnail_path, format="JPEG", quality=80, optimize=True)

    return StoredImage(
        relative_path=str(image_path.relative_to(root)),
        thumbnail_path=str(thumbnail_path.relative_to(root)),
        width=stored.width,
        height=stored.height,
        bytes=image_path.stat().st_size,
        content_hash=content_hash,
        dominant_colours=dominant_colours(image),
    )


def read_stored(root: Path, relative_path: str) -> bytes:
    """Read a stored image back, refusing anything outside the image root."""
    from security.paths import resolve_within_roots

    resolved = resolve_within_roots(root / relative_path, [root.resolve()])
    return resolved.read_bytes()
