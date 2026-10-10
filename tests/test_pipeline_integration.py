"""End-to-end with REAL PaddleOCR (PP-OCRv6_small) and REAL Ollama (qwen2.5:3b).

Marked slow. Run with Ollama started:  pytest -m slow -v
Only 4 LLM calls are made (DSPy cache on: re-runs are instant). One PaddleOCR
instance is shared by all tests in the process.
"""

import json

import pytest

from backend.dspy_service import Extractor
from backend.fields import PREDEFINED_FIELDS
from backend.pipeline import run_pipeline
from tests.conftest import SAMPLES, image_bytes, ollama_running

pytestmark = pytest.mark.slow
needs_ollama = pytest.mark.skipif(not ollama_running(), reason="Ollama is not running (ollama serve)")

# An extractor that cannot reach any LLM (closed port): proves when the LLM is (not) used.
UNREACHABLE = dict(api_base="http://127.0.0.1:9", timeout=5, cache=False)


def run(name, keys, **kw):
    return run_pipeline((SAMPLES / name).read_bytes(), keys, **kw)


def assert_json(result):
    json.dumps(result.model_dump(), ensure_ascii=False)


@needs_ollama
def test_clear_invoice_requested_fields():
    keys = ["invoice_number", "date", "vendor_name", "gst_number", "total"]
    r = run("invoice_clear.png", keys)
    assert r.success, r.error
    assert list(r.extracted) == keys                       # only the requested fields, in order
    assert r.extracted["invoice_number"] == "INV-1001"
    assert r.extracted["date"] == "06/10/2026"
    assert "SHREE GANESH TRADERS" in r.extracted["vendor_name"].upper()
    assert r.extracted["gst_number"] == "29ABCDE1234F1Z5"
    assert r.extracted["total"] == "619.50"
    assert r.ocr.line_count >= 25 and all(len(ln.box) == 4 for ln in r.ocr.lines)
    assert "₹" in r.ocr.raw_text                             # raw OCR keeps the currency symbol
    assert_json(r)


@needs_ollama
def test_receipt_with_missing_fields():
    r = run("receipt_mono.png", ["vendor_name", "customer_name", "gst_number", "total"])
    assert r.success, r.error
    assert "FRESH MART" in (r.extracted["vendor_name"] or "").upper()
    assert r.extracted["customer_name"] is None              # not on the receipt: must not be invented
    assert r.extracted["gst_number"] is None
    assert r.extracted["total"] == "413.00"
    assert_json(r)


@needs_ollama
def test_large_image_resized_one_field():
    r = run("invoice_large.jpg", ["total"])
    assert r.success, r.error
    assert r.image.resized and max(r.image.processed_width, r.image.processed_height) == 2000
    assert max(ln.box[2] for ln in r.ocr.lines) > 2000      # boxes are in ORIGINAL image coordinates
    assert r.extracted == {"total": "619.50"}
    assert_json(r)


@needs_ollama
def test_all_predefined_fields_plus_custom():
    keys = [f.key for f in PREDEFINED_FIELDS] + ["cgst_amount"]
    r = run("invoice_clear.png", keys)
    assert r.success, r.error
    assert list(r.extracted) == keys
    assert r.extracted["invoice_number"] == "INV-1001"
    assert r.extracted["total"] == "619.50"
    assert r.extracted["customer_name"] and "ABC" in r.extracted["customer_name"]
    assert isinstance(r.extracted["product"], list) and any("Milk" in p for p in r.extracted["product"])
    assert r.fields[-1].custom
    assert r.extracted["cgst_amount"] == "14.75"
    assert_json(r)


def test_blank_image_does_not_call_llm():
    r = run_pipeline(image_bytes((800, 600)), ["total"], extractor=Extractor(**UNREACHABLE))
    assert r.success and r.ocr.line_count == 0
    assert r.extracted == {"total": None} and r.error is None


def test_ocr_succeeds_when_ollama_unavailable():
    r = run("invoice_clear.png", ["invoice_number", "total"], extractor=Extractor(**UNREACHABLE))
    assert r.success is False
    assert r.error.stage == "dspy" and r.error.kind == "unavailable"
    assert r.ocr.line_count >= 25 and "INV-1001" in r.ocr.text   # OCR result is kept
    assert r.extracted == {}
    assert_json(r)


def test_invalid_image_real_pipeline():
    r = run_pipeline(b"\x89PNG\r\n\x1a\nbroken", ["total"])
    assert r.error.stage == "preprocess"


def test_real_ocr_from_many_threads():
    """Regression (Phase 6): FastAPI calls OCR from different worker threads. Before the
    dedicated PaddleOCR thread, 4/15 such calls failed with RuntimeError: std::exception."""
    import threading

    from backend import ocr_service
    from backend.image_preprocessing import preprocess_image

    names = ["receipt_mono.png", "invoice_clear.png", "invoice_blurry.jpg", "blank.png"] * 2
    images = {n: preprocess_image((SAMPLES / n).read_bytes()) for n in set(names)}
    outcomes = []

    def one(name):
        try:
            outcomes.append(ocr_service.run_ocr(images[name])["line_count"])
        except ocr_service.OcrError as exc:
            outcomes.append(exc)

    for name in names:
        t = threading.Thread(target=one, args=(name,))
        t.start()
        t.join()
    assert not [o for o in outcomes if isinstance(o, Exception)], outcomes


def test_downscaled_image_boxes_map_to_original_coordinates():
    """Phase 7: invoice_large.jpg is invoice_clear.png at 4x. It is downscaled to 2000 px before OCR,
    and its boxes must come back in ORIGINAL-image pixels: large / 4 ~= clear for every matching line."""
    from backend import ocr_service
    from backend.image_preprocessing import preprocess_image

    clear = preprocess_image((SAMPLES / "invoice_clear.png").read_bytes())
    large = preprocess_image((SAMPLES / "invoice_large.jpg").read_bytes())
    assert large.resized and (large.original_width, large.original_height) == (4960, 5200)
    def by_text(lines):  # repeated texts (two "₹14.75", two "1") are paired by order, top to bottom
        groups = {}
        for ln in sorted(lines, key=lambda ln: ln["box"][1]):
            groups.setdefault(ln["text"], []).append(ln["box"])
        return groups

    clear_boxes, large_boxes = by_text(ocr_service.run_ocr(clear)["lines"]), by_text(ocr_service.run_ocr(large)["lines"])
    diffs = [max(abs(a / 4 - b) for a, b in zip(lb, cb))
             for text, boxes in large_boxes.items() if len(boxes) == len(clear_boxes.get(text, []))
             for lb, cb in zip(boxes, clear_boxes[text])]
    assert len(diffs) >= 25, diffs
    assert max(diffs) <= 8, sorted(diffs)[-5:]


@needs_ollama
def test_real_llm_timeout_keeps_ocr():
    """PLAN.md §20 T11b with the REAL client stack: an LLM call that exceeds its timeout is reported as
    error.kind == 'timeout' (via the actual litellm/DSPy exception chain) and the OCR result is kept."""
    r = run("invoice_clear.png", ["invoice_number"], extractor=Extractor(timeout=0.01, cache=False))
    assert r.success is False
    assert r.error.stage == "dspy" and r.error.kind == "timeout", r.error
    assert r.ocr.line_count >= 25 and "INV-1001" in r.ocr.text
