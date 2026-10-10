"""End-to-end processing: image bytes -> preprocess -> PaddleOCR -> DSPy -> result.

Phase 5's FastAPI endpoint calls `run_pipeline`; it never raises for expected failures,
it returns a PipelineResult whose `error.stage` says what failed. OCR output is kept
even when the LLM step fails.
"""

import logging
import time

from backend import ocr_service
from backend.dspy_service import ExtractionError, Extractor, get_extractor
from backend.fields import FieldSpec, FieldValidationError, resolve_fields
from backend.image_preprocessing import InvalidImageError, UnsupportedImageError, preprocess_image
from backend.schemas import PipelineError, PipelineResult

logger = logging.getLogger(__name__)


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def run_pipeline(
    image_bytes: bytes,
    field_keys: list[str] | None,
    *,
    extractor: Extractor | None = None,
    run_ocr=None,
) -> PipelineResult:
    """Process one image. `extractor` / `run_ocr` can be injected (tests, alternative engines)."""
    run_ocr = run_ocr or ocr_service.run_ocr
    total_start = time.perf_counter()
    timings: dict[str, int] = {}

    # 1. Requested fields (cheap; validate before any heavy work).
    try:
        fields: list[FieldSpec] = resolve_fields(field_keys)
    except FieldValidationError as exc:
        return PipelineResult(success=False, error=PipelineError(stage="fields", kind=exc.code, message=exc.message))
    result = PipelineResult(
        success=False,
        requested_fields=[f.key for f in fields],
        fields=[f.public() for f in fields],
    )

    # 2. Preprocess (decode, validate, downscale) in memory.
    start = time.perf_counter()
    try:
        image = preprocess_image(image_bytes)
    except (InvalidImageError, UnsupportedImageError) as exc:
        kind = "unsupported_format" if isinstance(exc, UnsupportedImageError) else "invalid_image"
        result.error = PipelineError(stage="preprocess", kind=kind, message=str(exc))
        result.timings_ms = {"preprocess": _ms(start), "total": _ms(total_start)}
        return result
    timings["preprocess"] = _ms(start)
    result.image = image.info()
    if image.resized:
        result.warnings.append(
            f"Image downscaled from {image.original_width}x{image.original_height} "
            f"to {image.width}x{image.height} before OCR"
        )

    # 3. OCR.
    start = time.perf_counter()
    try:
        ocr = run_ocr(image)
    except ocr_service.OcrError:
        # Details are logged by ocr_service; never send internal exception text to clients.
        result.error = PipelineError(stage="ocr", kind="ocr_failed",
                                     message="Text recognition failed. Try a clearer image or check the server log.")
        result.timings_ms = {**timings, "ocr": _ms(start), "total": _ms(total_start)}
        return result
    finally:
        del image  # release the pixel array before the (slow) LLM call
    timings["ocr"] = _ms(start)
    result.ocr = ocr
    if ocr["low_confidence_dropped"]:
        result.warnings.append(f"{ocr['low_confidence_dropped']} low-confidence OCR line(s) not sent to the LLM")

    # 4. No text: nothing to extract, the LLM is not called.
    if not ocr["text"].strip():
        result.success = True
        result.extracted = {f.key: None for f in fields}
        result.warnings.append("No text detected in image")
        result.timings_ms = {**timings, "extraction": 0, "total": _ms(total_start)}
        return result

    # 5. DSPy extraction. A failure here keeps the OCR result.
    start = time.perf_counter()
    try:
        extraction = (extractor or get_extractor()).extract(ocr["text"], fields)
    except ExtractionError as exc:
        result.error = PipelineError(stage="dspy", kind=exc.kind, message=exc.message)
        result.extracted = {}
        result.timings_ms = {**timings, "extraction": _ms(start), "total": _ms(total_start)}
        return result
    timings["extraction"] = _ms(start)

    result.success = True
    result.extracted = extraction.values
    result.value_adjustments = extraction.adjustments
    result.warnings.extend(extraction.warnings)
    result.timings_ms = {**timings, "total": _ms(total_start)}
    return result
