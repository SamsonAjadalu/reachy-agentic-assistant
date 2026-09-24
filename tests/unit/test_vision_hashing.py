"""Perceptual hashing without the imagehash package."""

from __future__ import annotations

import io

from PIL import Image

from vision.hashing import dhash64, hamming_distance, near_duplicate, phash64


def _rgb(colour: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> Image.Image:
    return Image.new("RGB", size, colour)


class TestPerceptualHashes:
    def test_identical_images_have_distance_zero(self) -> None:
        image = _rgb((40, 80, 120))
        assert dhash64(image) == dhash64(image.copy())
        assert phash64(image) == phash64(image.copy())
        assert hamming_distance(dhash64(image), dhash64(image)) == 0
        assert hamming_distance(phash64(image), phash64(image)) == 0

    def test_hashes_are_64_bit_hex(self) -> None:
        digest = dhash64(_rgb((10, 20, 30)))
        assert len(digest) == 16
        int(digest, 16)
        digest = phash64(_rgb((10, 20, 30)))
        assert len(digest) == 16
        int(digest, 16)

    def test_very_different_images_are_not_near_duplicates(self) -> None:
        red = _rgb((220, 20, 20))
        blue = _rgb((20, 20, 220))
        assert not near_duplicate(dhash64(red), dhash64(blue), phash64(red), phash64(blue))

    def test_same_scene_small_resize_is_near_duplicate(self) -> None:
        base = _rgb((90, 110, 70), (128, 128))
        buffer = io.BytesIO()
        base.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        jpeg = Image.open(buffer)
        smaller = base.resize((96, 96))
        assert near_duplicate(dhash64(jpeg), dhash64(smaller), phash64(jpeg), phash64(smaller))
