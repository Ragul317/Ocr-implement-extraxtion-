"""Post-LLM normalization and amount-format restoration (no LLM call)."""

import pytest

from backend.dspy_service import ExtractionError, _classify, normalize_values, restore_number_format
from backend.fields import resolve_fields

OCR = """Invoice No: INV-1001
Subtotal: ₹690.00
CGST 2.5%: ₹17.25
Total: ₹724.50
Qty 2
GSTIN: 29ABCDE1234F1Z5
Amount 1,250.00
Rs. 36.00
Ambiguous 590 and ₹590.00"""


@pytest.mark.parametrize(
    "llm, expected, token",
    [
        ("724.5", "724.50", "₹724.50"),
        ("690", "690.00", "₹690.00"),
        ("1250", "1,250.00", "1,250.00"),
        ("36", "36.00", "Rs. 36.00"),
        ("724.50", "724.50", None),       # already exact: no adjustment recorded
        ("2", "2", None),
    ],
)
def test_restore_number_format(llm, expected, token):
    value, used, warning = restore_number_format(llm, OCR)
    assert (value, used, warning) == (expected, token, None)


@pytest.mark.parametrize("llm", ["INV-1001", "29ABCDE1234F1Z5", "06/10/2026", "+919845012345", "₹724.50"])
def test_non_plain_numbers_untouched(llm):
    assert restore_number_format(llm, OCR) == (llm, None, None)


def test_ambiguous_format_kept_with_warning():
    value, used, warning = restore_number_format("590", OCR)
    assert value == "590" and used is None and "several formats" in warning


def test_number_not_in_ocr_flagged():
    value, used, warning = restore_number_format("999.99", OCR)
    assert value == "999.99" and "not found" in warning


def test_digits_inside_ids_are_not_amounts():
    assert "not found" in restore_number_format("29", OCR)[2]  # from GSTIN 29ABCDE...


def test_normalize_requested_only_in_order_missing_null_extra_dropped():
    fields = resolve_fields(["total", "invoice_number", "customer_name", "gst_number"])
    raw = {"invoice_number": "INV-1001", "total": "724.5", "customer_name": "", "gst_number": "N/A",
           "vendor_name": "Invented Ltd"}
    result = normalize_values(raw, fields, OCR)
    assert list(result.values) == ["total", "invoice_number", "customer_name", "gst_number"]
    assert result.values == {"total": "724.50", "invoice_number": "INV-1001", "customer_name": None, "gst_number": None}
    assert result.adjustments == [{"field": "total", "llm_value": "724.5", "ocr_token": "₹724.50", "value": "724.50"}]


def test_normalize_missing_key_becomes_null_and_case_insensitive_keys():
    fields = resolve_fields(["invoice_number", "total"])
    result = normalize_values({"Invoice_Number": "INV-1001"}, fields, OCR)
    assert result.values == {"invoice_number": "INV-1001", "total": None}


def test_multi_valued_split_and_restored():
    fields = resolve_fields(["product", "quantity"])
    result = normalize_values({"product": "Milk 1L | Bread", "quantity": "2 | 1"}, fields, "Milk 1L 2\nBread 1")
    assert result.values == {"product": ["Milk 1L", "Bread"], "quantity": ["2", "1"]}


def test_unverified_text_value_warns():
    result = normalize_values({"vendor_name": "Made Up Traders"}, resolve_fields(["vendor_name"]), OCR)
    assert result.values["vendor_name"] == "Made Up Traders"
    assert any("not found verbatim" in w for w in result.warnings)


def test_error_classification():
    class LMTransportError(Exception):
        pass

    class Timeout(Exception):
        pass

    assert _classify(LMTransportError("Connection refused"))[0] == "unavailable"
    assert _classify(Timeout("Request timed out"))[0] == "timeout"
    assert _classify(ValueError("boom"))[0] == "unknown"
    assert ExtractionError("parse", "x").kind == "parse"
