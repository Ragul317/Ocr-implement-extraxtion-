"""PaddleOCR 3.x wrapper (PLAN.md §14).

Owns the single PaddleOCR instance, runs it on a preprocessed image and converts the
native result into plain Python data. Nothing PaddleOCR-specific leaves this module.
"""

import logging
import os
import re
import statistics
import unicodedata
from concurrent.futures import ThreadPoolExecutor

from backend.config import settings
from backend.image_preprocessing import PreprocessedImage

# Quieten Paddle's C++ (glog) INFO logging; must be set before paddle is imported.
os.environ.setdefault("GLOG_minloglevel", "2")

logger = logging.getLogger(__name__)

ROW_JOINER = "    "  # same visual layout as the Phase 3 sample text ("Milk 1L    2    ₹50.00")

# ALL PaddleOCR work (creation + predict) runs on this one dedicated thread.
# Paddle's CPU (oneDNN) predictors keep per-thread state: calling one instance from
# changing threads, as FastAPI's worker pool does, failed intermittently with
# "RuntimeError: std::exception" and could leave the instance permanently broken
# (Phase 6 finding: 4/15 failures with a new thread per call, 0/15 on one thread).
# A single worker also serializes calls, so no separate predict lock is needed.
_paddle_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paddleocr")
_engine = None


class OcrError(RuntimeError):
    """PaddleOCR failed to load or to process an image."""


def _create_engine():
    """Runs on the PaddleOCR thread only."""
    global _engine
    if _engine is None:
        from paddleocr import PaddleOCR  # heavy import, deferred until needed

        kwargs = dict(
            lang=settings.ocr_lang,
            device=settings.ocr_device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
        if settings.ocr_det_model:
            kwargs["text_detection_model_name"] = settings.ocr_det_model
        if settings.ocr_rec_model:
            kwargs["text_recognition_model_name"] = settings.ocr_rec_model
        try:
            _engine = PaddleOCR(**kwargs)
        except Exception as exc:  # noqa: BLE001 - surface any load failure uniformly
            logger.exception("PaddleOCR could not be loaded")
            raise OcrError(f"Could not load PaddleOCR: {exc}") from exc
        logger.info("PaddleOCR loaded (det=%s, rec=%s)", settings.ocr_det_model, settings.ocr_rec_model)
    return _engine


def get_engine():
    """Create the PaddleOCR instance once per process (on the PaddleOCR thread)."""
    return _paddle_thread.submit(_create_engine).result()


def _predict(array):
    """Runs on the PaddleOCR thread only. Returns plain Python data, never Paddle objects."""
    global _engine
    engine = _create_engine()
    try:
        res = engine.predict(array)[0]
        return {k: res[k] for k in ("rec_texts", "rec_scores", "rec_polys", "rec_boxes")}
    except Exception:
        # Self-healing: a failed predictor may stay broken, so drop it; the next call reloads it.
        _engine = None
        raise


def normalize_text(text: str) -> str:
    """Light, lossless-in-meaning cleanup: NFC, drop zero-width/control chars, collapse spaces.

    Never replaces characters, so ₹, decimals, IDs, dates and phone numbers are untouched.
    """
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ch in "\t " or unicodedata.category(ch) not in ("Cc", "Cf"))
    return re.sub(r"[ \t]+", " ", text).strip()


def to_plain_lines(res, scale_x: float = 1.0, scale_y: float = 1.0) -> list[dict]:
    """Convert one PaddleOCR result into plain dicts, mapping coordinates to the original image."""
    lines = []
    for text, score, poly, box in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"], res["rec_boxes"]):
        lines.append(
            {
                "text": str(text),
                "confidence": round(float(score), 3),
                "polygon": [[round(float(x) * scale_x), round(float(y) * scale_y)] for x, y in poly],
                "box": [
                    round(float(box[0]) * scale_x),
                    round(float(box[1]) * scale_y),
                    round(float(box[2]) * scale_x),
                    round(float(box[3]) * scale_y),
                ],
            }
        )
    return lines


def reading_order_text(lines: list[dict], min_confidence: float = 0.0) -> tuple[str, int]:
    """Group detected segments into visual rows and join them top-to-bottom, left-to-right.

    Minimal row grouping: a segment joins the current row when its vertical centre is
    within half the median line height of the row centre. Skewed photos can still split
    or merge rows; there is no table/column reconstruction.
    Returns (text, number of segments dropped for low confidence).
    """
    kept = [ln for ln in lines if ln["confidence"] >= min_confidence and normalize_text(ln["text"])]
    dropped = sum(1 for ln in lines if ln["confidence"] < min_confidence)
    if not kept:
        return "", dropped

    heights = [max(1, ln["box"][3] - ln["box"][1]) for ln in kept]
    tolerance = statistics.median(heights) / 2

    def centre_y(ln: dict) -> float:
        return (ln["box"][1] + ln["box"][3]) / 2

    rows: list[list[dict]] = []
    row_centre = None
    for ln in sorted(kept, key=centre_y):
        cy = centre_y(ln)
        if rows and abs(cy - row_centre) <= tolerance:
            rows[-1].append(ln)
            row_centre = sum(centre_y(x) for x in rows[-1]) / len(rows[-1])
        else:
            rows.append([ln])
            row_centre = cy

    text_rows = [ROW_JOINER.join(normalize_text(ln["text"]) for ln in sorted(row, key=lambda x: x["box"][0]))
                 for row in rows]
    return "\n".join(text_rows), dropped


def run_ocr(image: PreprocessedImage, min_confidence: float | None = None) -> dict:
    """Run PaddleOCR on a preprocessed image and return plain data.

    Keys: text (reading order, sent to the LLM), raw_text (PaddleOCR order, unmodified),
    lines, line_count, average_confidence, low_confidence_dropped.
    """
    min_conf = settings.ocr_min_confidence if min_confidence is None else min_confidence
    try:
        native = _paddle_thread.submit(_predict, image.array).result()
        lines = to_plain_lines(
            native,
            scale_x=image.original_width / image.width,
            scale_y=image.original_height / image.height,
        )
    except OcrError:
        raise
    except Exception as exc:  # noqa: BLE001 - any PaddleOCR failure becomes OcrError
        logger.exception("PaddleOCR failed (engine discarded; it reloads on the next request)")
        raise OcrError(f"Text recognition failed: {exc}") from exc
    finally:
        native = None  # drop references to PaddleOCR's arrays early

    text, dropped = reading_order_text(lines, min_conf)
    confidences = [ln["confidence"] for ln in lines]
    return {
        "text": text,
        "raw_text": "\n".join(ln["text"] for ln in lines),
        "lines": lines,
        "line_count": len(lines),
        "average_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
        "low_confidence_dropped": dropped,
    }
