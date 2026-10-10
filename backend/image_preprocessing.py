"""In-memory image decoding, validation and downscaling before OCR (PLAN.md §14).

No PaddleOCR import here, so this module is cheap to import and to unit-test.
The caller's bytes are never modified; a new array is produced.
"""

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.config import settings

ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}


class InvalidImageError(ValueError):
    """Not an image, corrupt, or too large in pixels."""


class UnsupportedImageError(ValueError):
    """A real image, but in a format we do not accept."""


@dataclass
class PreprocessedImage:
    array: np.ndarray            # H x W x 3, uint8, BGR (what PaddleOCR expects)
    original_width: int
    original_height: int
    width: int                   # after downscaling
    height: int
    scale: float                 # processed / original (<= 1.0)
    format: str

    @property
    def resized(self) -> bool:
        return self.scale < 1.0

    def info(self) -> dict:
        return {
            "original_width": self.original_width,
            "original_height": self.original_height,
            "processed_width": self.width,
            "processed_height": self.height,
            "resized": self.resized,
            "format": self.format,
        }


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")


def preprocess_image(
    data: bytes,
    max_side: int | None = None,
    max_pixels: int | None = None,
) -> PreprocessedImage:
    """Decode image bytes, validate them and downscale so the longest side <= max_side.

    Small images are never upscaled. Raises InvalidImageError / UnsupportedImageError.
    """
    max_side = max_side or settings.ocr_max_side
    max_pixels = max_pixels or settings.max_image_pixels
    if not data:
        raise InvalidImageError("The uploaded file is empty.")

    # 1. Cheap structural check (does not decode pixel data).
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("The file is not a valid image.") from exc
    if fmt not in ALLOWED_FORMATS:
        raise UnsupportedImageError(f"Unsupported image format {fmt}. Use JPG, PNG, BMP or WEBP.")

    # 2. Real decode (verify() leaves the object unusable, so reopen).
    try:
        img = Image.open(io.BytesIO(data))
        orig_w, orig_h = img.size
        if orig_w * orig_h > max_pixels:
            raise InvalidImageError(f"Image dimensions too large ({orig_w}x{orig_h}).")

        # JPEG only: let the decoder produce a reduced-size image directly (saves RAM on huge photos).
        # draft() picks a scale >= the requested size, so the final resize below still applies.
        if fmt == "JPEG" and max(orig_w, orig_h) > max_side:
            ratio = max_side / max(orig_w, orig_h)
            img.draft("RGB", (max(1, int(orig_w * ratio)), max(1, int(orig_h * ratio))))

        img = ImageOps.exif_transpose(img)
        img = _to_rgb(img)
    except InvalidImageError:
        raise
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("The image could not be decoded (corrupt or truncated).") from exc

    # EXIF rotation may have swapped width/height; express sizes in the upright orientation.
    if img.size != (orig_w, orig_h) and img.size[::-1] == (orig_w, orig_h):
        orig_w, orig_h = orig_h, orig_w

    # 3. Downscale (never upscale), keeping aspect ratio.
    long_side = max(orig_w, orig_h)
    if long_side > max_side:
        scale = max_side / long_side
        target = (max(1, round(orig_w * scale)), max(1, round(orig_h * scale)))
        if img.size != target:
            img = img.resize(target, Image.Resampling.LANCZOS)
    else:
        scale = 1.0
        if img.size != (orig_w, orig_h):  # draft() never runs here, but stay safe
            img = img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

    # RGB -> BGR without an extra full-size intermediate copy beyond the final array.
    array = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
    width, height = img.size
    img.close()

    return PreprocessedImage(
        array=array,
        original_width=orig_w,
        original_height=orig_h,
        width=width,
        height=height,
        scale=width / orig_w,
        format=fmt,
    )
