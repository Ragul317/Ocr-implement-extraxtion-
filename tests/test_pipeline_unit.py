"""Pipeline orchestration with stubs (fast; no PaddleOCR model, no LLM).

The stubs replace only the two slow engines. Real OCR + real Ollama are exercised
in test_pipeline_integration.py (marked slow).
"""

import json

from backend.dspy_service import ExtractionError, normalize_values
from backend.fields import PREDEFINED_FIELDS
from backend.ocr_service import OcrError, reading_order_text
from backend.pipeline import run_pipeline
from tests.conftest import SAMPLES, image_bytes


class StubOcr:
    """Returns recorded real OCR lines; remembers the image it was given."""

    def __init__(self, lines):
        self.lines, self.seen = lines, None

    def __call__(self, image):
        self.seen = image
        text, dropped = reading_order_text(self.lines, 0.5)
        return {"text": text, "raw_text": "\n".join(x["text"] for x in self.lines), "lines": self.lines,
                "line_count": len(self.lines), "average_confidence": 0.98 if self.lines else None,
                "low_confidence_dropped": dropped}


class StubExtractor:
    """Plays back a fixed LLM answer through the real normalization code."""

    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer or {}, error, []

    def extract(self, text, fields):
        self.calls.append((text, [f.key for f in fields]))
        if self.error:
            raise self.error
        return normalize_values(self.answer, fields, text)


CLEAR = (SAMPLES / "invoice_clear.png").read_bytes()


def test_success_shape_and_json(recorded_ocr_lines):
    ext = StubExtractor({"invoice_number": "INV-1001", "total": "619.5", "vendor_name": "SHREE GANESH TRADERS",
                         "unrequested": "x"})
    r = run_pipeline(CLEAR, ["invoice_number", "vendor_name", "total"], extractor=ext,
                     run_ocr=StubOcr(recorded_ocr_lines))
    assert r.success and r.error is None
    assert r.requested_fields == ["invoice_number", "vendor_name", "total"]
    assert r.extracted == {"invoice_number": "INV-1001", "vendor_name": "SHREE GANESH TRADERS", "total": "619.50"}
    assert r.value_adjustments[0].llm_value == "619.5"
    assert r.ocr.lines[0].box and r.ocr.line_count == 29
    assert "Total    619.50" in ext.calls[0][0]  # DSPy received the reading-order OCR text
    json.dumps(r.model_dump(), ensure_ascii=False)


def test_one_field(recorded_ocr_lines):
    r = run_pipeline(CLEAR, ["total"], extractor=StubExtractor({"total": "619.50"}),
                     run_ocr=StubOcr(recorded_ocr_lines))
    assert r.extracted == {"total": "619.50"}


def test_all_predefined_plus_custom(recorded_ocr_lines):
    keys = [f.key for f in PREDEFINED_FIELDS] + ["cgst_amount"]
    ext = StubExtractor({"cgst_amount": "14.75"})
    r = run_pipeline(CLEAR, keys, extractor=ext, run_ocr=StubOcr(recorded_ocr_lines))
    assert list(r.extracted) == keys
    assert r.extracted["cgst_amount"] == "14.75"
    assert r.fields[-1].custom is True and r.fields[-1].label == "Cgst Amount"
    assert ext.calls[0][1] == keys


def test_missing_fields_are_null(recorded_ocr_lines):
    r = run_pipeline(CLEAR, ["customer_name", "phone_number"], extractor=StubExtractor({"customer_name": ""}),
                     run_ocr=StubOcr(recorded_ocr_lines))
    assert r.success and r.extracted == {"customer_name": None, "phone_number": None}


def test_dspy_failure_keeps_ocr(recorded_ocr_lines):
    err = ExtractionError("unavailable", "LLM unavailable: could not connect to Ollama")
    r = run_pipeline(CLEAR, ["total"], extractor=StubExtractor(error=err), run_ocr=StubOcr(recorded_ocr_lines))
    assert r.success is False
    assert r.error.stage == "dspy" and r.error.kind == "unavailable"
    assert r.ocr is not None and r.ocr.line_count == 29
    assert r.extracted == {} and r.requested_fields == ["total"]
    json.dumps(r.model_dump(), ensure_ascii=False)


def test_blank_image_skips_llm():
    ext = StubExtractor()
    r = run_pipeline(image_bytes((800, 600)), ["total", "date"], extractor=ext, run_ocr=StubOcr([]))
    assert r.success and ext.calls == []
    assert r.extracted == {"total": None, "date": None}
    assert "No text detected in image" in r.warnings


def test_large_image_is_resized_before_ocr(recorded_ocr_lines):
    ocr = StubOcr(recorded_ocr_lines)
    r = run_pipeline((SAMPLES / "invoice_large.jpg").read_bytes(), ["total"],
                     extractor=StubExtractor({"total": "619.50"}), run_ocr=ocr)
    assert max(ocr.seen.width, ocr.seen.height) == 2000
    assert r.image.resized and r.image.original_width == 4960
    assert any("downscaled" in w for w in r.warnings)


def test_invalid_image():
    r = run_pipeline(b"definitely not an image", ["total"], extractor=StubExtractor(), run_ocr=StubOcr([]))
    assert r.success is False and r.error.stage == "preprocess" and r.error.kind == "invalid_image"
    assert r.ocr is None


def test_unsupported_format():
    r = run_pipeline(image_bytes(fmt="GIF"), ["total"], extractor=StubExtractor(), run_ocr=StubOcr([]))
    assert r.error.stage == "preprocess" and r.error.kind == "unsupported_format"


def test_ocr_failure():
    def broken_ocr(image):
        raise OcrError("Text recognition failed: boom")

    r = run_pipeline(CLEAR, ["total"], extractor=StubExtractor(), run_ocr=broken_ocr)
    assert r.success is False and r.error.stage == "ocr"


def test_invalid_fields_fail_before_any_work():
    ocr = StubOcr([])
    r = run_pipeline(CLEAR, [], extractor=StubExtractor(), run_ocr=ocr)
    assert r.error.stage == "fields" and r.error.kind == "NO_FIELDS_SELECTED"
    assert ocr.seen is None
