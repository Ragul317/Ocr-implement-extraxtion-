import io

import numpy as np
import pytest
from PIL import Image

from backend.image_preprocessing import InvalidImageError, UnsupportedImageError, preprocess_image
from tests.conftest import SAMPLES, image_bytes


def test_small_image_not_upscaled():
    img = preprocess_image(image_bytes((300, 120)), max_side=2000)
    assert (img.width, img.height) == (300, 120)
    assert img.scale == 1.0 and not img.resized
    assert img.array.shape == (120, 300, 3) and img.array.dtype == np.uint8


def test_large_png_downscaled_keeping_aspect_ratio():
    img = preprocess_image(image_bytes((5000, 2500)), max_side=2000)
    assert (img.width, img.height) == (2000, 1000)
    assert (img.original_width, img.original_height) == (5000, 2500)
    assert img.resized and img.info()["resized"] is True


def test_large_jpeg_sample_downscaled():
    data = (SAMPLES / "invoice_large.jpg").read_bytes()
    img = preprocess_image(data, max_side=2000)
    assert max(img.width, img.height) == 2000
    assert (img.original_width, img.original_height) == (4960, 5200)
    assert img.format == "JPEG"


def test_input_bytes_unchanged():
    data = image_bytes((3000, 100))
    copy = bytes(data)
    preprocess_image(data, max_side=2000)
    assert data == copy


def test_bgr_channel_order():
    img = preprocess_image(image_bytes((10, 10), color=(255, 0, 0)))  # pure red
    assert tuple(img.array[0, 0]) == (0, 0, 255)


def test_rgba_flattened_on_white():
    img = preprocess_image(image_bytes((10, 10), color=(0, 0, 0, 0), mode="RGBA"))
    assert tuple(img.array[0, 0]) == (255, 255, 255)


def test_exif_rotation_applied():
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° CW on display
    Image.new("RGB", (400, 100), "white").save(buf, format="JPEG", exif=exif)
    img = preprocess_image(buf.getvalue())
    assert (img.width, img.height) == (100, 400)
    assert (img.original_width, img.original_height) == (100, 400)


@pytest.mark.parametrize("data", [b"", b"not an image at all", b"\x89PNG\r\n\x1a\n" + b"\x00" * 50])
def test_invalid_bytes(data):
    with pytest.raises(InvalidImageError):
        preprocess_image(data)


def test_truncated_jpeg():
    data = (SAMPLES / "invoice_blurry.jpg").read_bytes()
    with pytest.raises(InvalidImageError):
        preprocess_image(data[: len(data) // 3])


def test_unsupported_format():
    with pytest.raises(UnsupportedImageError):
        preprocess_image(image_bytes((20, 20), fmt="GIF"))


def test_pixel_limit():
    with pytest.raises(InvalidImageError):
        preprocess_image(image_bytes((3000, 3000)), max_pixels=1_000_000)
