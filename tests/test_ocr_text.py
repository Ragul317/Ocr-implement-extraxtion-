"""OCR plain-data conversion, light normalization and reading order (no model needed)."""

import json

import numpy as np

from backend.ocr_service import normalize_text, reading_order_text, to_plain_lines


def test_normalize_text_preserves_important_characters():
    for s in ["₹724.50", "INV-1001", "29ABCDE1234F1Z5", "06/10/2026", "+91 98450 12345", "CGST 2.5%"]:
        assert normalize_text(s) == s


def test_normalize_text_light_cleanup():
    assert normalize_text("  Total​   ₹619.50 \t") == "Total ₹619.50"


def test_to_plain_lines_converts_numpy_and_scales_to_original():
    native = {  # same shapes/types PaddleOCR 3.7 returns (verified in Phase 2)
        "rec_texts": ["Total"],
        "rec_scores": [np.float32(0.98765)],
        "rec_polys": [np.array([[10, 20], [110, 20], [110, 40], [10, 40]], dtype=np.int16)],
        "rec_boxes": np.array([[10, 20, 110, 40]], dtype=np.int16),
    }
    lines = to_plain_lines(native, scale_x=2.0, scale_y=2.0)
    assert lines == [{"text": "Total", "confidence": 0.988, "polygon": [[20, 40], [220, 40], [220, 80], [20, 80]],
                      "box": [20, 40, 220, 80]}]
    json.dumps(lines)


def test_reading_order_groups_rows(recorded_ocr_lines):
    text, dropped = reading_order_text(recorded_ocr_lines, min_confidence=0.5)
    rows = text.splitlines()
    assert dropped == 0
    assert rows[0] == "SHREE GANESH TRADERS"
    assert "Milk 1L    2    ₹50.00" in rows
    assert "Basmati Rice 5kg    1    ₹500.00" in rows
    assert rows.index("Subtotal    ₹590.00") < rows.index("Total    619.50")


def test_reading_order_drops_low_confidence_from_llm_text(recorded_ocr_lines):
    lines = [dict(ln) for ln in recorded_ocr_lines]
    lines[0]["confidence"] = 0.1
    text, dropped = reading_order_text(lines, min_confidence=0.5)
    assert dropped == 1 and "SHREE GANESH TRADERS" not in text


def test_reading_order_empty():
    assert reading_order_text([], 0.5) == ("", 0)


def test_paddle_work_runs_on_one_dedicated_thread(monkeypatch):
    """Regression (Phase 6): Paddle predictors broke when called from changing threads."""
    import threading

    from backend import ocr_service
    from backend.image_preprocessing import preprocess_image
    from tests.conftest import image_bytes

    seen = []

    class FakeEngine:
        def predict(self, array):
            seen.append(threading.current_thread().name)
            return [{"rec_texts": [], "rec_scores": [], "rec_polys": [], "rec_boxes": np.zeros((0, 4))}]

    monkeypatch.setattr(ocr_service, "_engine", FakeEngine())
    image = preprocess_image(image_bytes((50, 20)))
    threads = [threading.Thread(target=ocr_service.run_ocr, args=(image,)) for _ in range(5)]
    for t in threads:
        t.start()
        t.join()
    assert len(seen) == 5 and len(set(seen)) == 1 and seen[0].startswith("paddleocr")


def test_failed_engine_is_discarded(monkeypatch):
    import pytest

    from backend import ocr_service
    from backend.image_preprocessing import preprocess_image
    from tests.conftest import image_bytes

    class BrokenEngine:
        def predict(self, array):
            raise RuntimeError("std::exception")

    monkeypatch.setattr(ocr_service, "_engine", BrokenEngine())
    with pytest.raises(ocr_service.OcrError):
        ocr_service.run_ocr(preprocess_image(image_bytes((50, 20))))
    assert ocr_service._engine is None  # the next request reloads a fresh instance
