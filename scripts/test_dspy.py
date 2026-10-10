"""Phase 3: standalone DSPy + local Ollama extraction test (no OCR, no FastAPI).

Sends OCR-like invoice text plus a list of requested field keys to qwen2.5:3b
through DSPy (`dspy.Predict` over a field-agnostic Signature), normalizes the
answer against the requested list, and checks it against expected values.

Requires a running Ollama server (`~/.local/ollama/bin/ollama serve`).

Usage:
    python scripts/test_dspy.py                 # all cases
    python scripts/test_dspy.py --runs 1        # one run per case
    python scripts/test_dspy.py --error-test    # also check the "LLM unreachable" error path

Note: this is a runnable script, not a pytest module.
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dspy  # noqa: E402

from backend.config import settings  # noqa: E402


class FieldExtractionSignature(dspy.Signature):
    """Extract the requested fields from noisy OCR text of an invoice, bill, receipt or document.
    Return a value for EVERY key in requested_fields and for NO other keys.
    Use only information present in the text. If a field is not present, use an empty string.
    Do not calculate or invent values. Copy values exactly as written, except amounts:
    digits and decimal point only, no currency symbol. Currency symbols such as ₹ may be
    misread by OCR as 'R', 'Rs', 'Z' or '?'. For list-type fields, separate values with ' | '."""

    ocr_text: str = dspy.InputField(desc="Raw OCR text, one detected line per row")
    requested_fields: dict[str, str] = dspy.InputField(desc="Fields to extract: key -> what the field means")
    extracted_values: dict[str, str] = dspy.OutputField(
        desc="Same keys as requested_fields -> extracted value, or empty string if absent"
    )


# Temporary inline hints for this phase; Phase 4 moves the catalog to backend/fields.py.
HINTS = {
    "invoice_number": "Invoice / bill / receipt number",
    "date": "Invoice/bill date, exactly as written",
    "vendor_name": "Seller, shop or company issuing the document",
    "customer_name": "Buyer / billed-to name",
    "gst_number": "GSTIN, a 15-character alphanumeric tax ID (e.g. 29ABCDE1234F1Z5)",
    "phone": "Phone/mobile number as written",
    "subtotal": "Amount before tax, digits only, no currency symbol",
    "total": "Grand total, digits only, no currency symbol",
    "product": "Names of purchased items, one per line item",
    "quantity": "Quantity for each line item, same order as Product",
}
MULTI_VALUED = {"product", "quantity"}
AMOUNT_FIELDS = {"subtotal", "total"}
EMPTY_MARKERS = {"", "n/a", "na", "null", "none", "not found", "not present", "-", "unknown"}

INVOICE_TEXT = """Invoice No: INV-1001
Date: 06/10/2026
Vendor: ABC Supermarket
Customer: Ragul Kumar
GSTIN: 29ABCDE1234F1Z5
Phone: +919845012345

Basmati Rice 5kg    1    ₹590.00
Milk 1L             2    ₹50.00
Subtotal: ₹690.00
CGST 2.5%: ₹17.25
SGST 2.5%: ₹17.25
Total: ₹724.50"""

MISSING_TEXT = """Invoice No: INV-2002
Vendor: XYZ Store
Total: ₹450.00"""

# (name, text, requested keys, expected values; None = must be empty / not found)
CASES = [
    (
        "A: full invoice (8 fields)",
        INVOICE_TEXT,
        ["invoice_number", "date", "vendor_name", "customer_name", "gst_number", "phone", "subtotal", "total"],
        {
            "invoice_number": "INV-1001",
            "date": "06/10/2026",
            "vendor_name": "ABC Supermarket",
            "customer_name": "Ragul Kumar",
            "gst_number": "29ABCDE1234F1Z5",
            "phone": "+919845012345",
            "subtotal": "690.00",
            "total": "724.50",
        },
    ),
    (
        "B: missing fields (5 fields)",
        MISSING_TEXT,
        ["invoice_number", "vendor_name", "customer_name", "gst_number", "total"],
        {
            "invoice_number": "INV-2002",
            "vendor_name": "XYZ Store",
            "customer_name": None,
            "gst_number": None,
            "total": "450.00",
        },
    ),
]

# Latency vs field count (single run each; plan Phase 3 checkpoint).
SCALING_CASES = [
    ("C: 1 field", INVOICE_TEXT, ["total"], {"total": "724.50"}),
    (
        "D: 10 fields incl. multi-valued product/quantity",
        INVOICE_TEXT,
        ["invoice_number", "date", "vendor_name", "customer_name", "gst_number", "phone",
         "product", "quantity", "subtotal", "total"],
        {
            "invoice_number": "INV-1001",
            "date": "06/10/2026",
            "vendor_name": "ABC Supermarket",
            "customer_name": "Ragul Kumar",
            "gst_number": "29ABCDE1234F1Z5",
            "phone": "+919845012345",
            "product": ["Basmati Rice 5kg", "Milk 1L"],
            "quantity": ["1", "2"],
            "subtotal": "690.00",
            "total": "724.50",
        },
    ),
]


def ollama_get(path: str, payload: dict | None = None, timeout: float = 120) -> dict:
    url = settings.llm_api_base.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def ollama_model_name() -> str:
    return settings.llm_model.split("/", 1)[1]


def normalize(raw: dict, requested: list[str]) -> tuple[dict, list[str], list[str]]:
    """Return (values in request order, missing keys, extra keys). Enforces 'requested keys only'."""
    raw = raw or {}
    lower_map = {str(k).strip().lower(): v for k, v in raw.items()}
    values, missing = {}, []
    for key in requested:
        if key in raw:
            value = raw[key]
        elif key in lower_map:
            value = lower_map[key]
        else:
            missing.append(key)
            value = None
        if isinstance(value, list):
            value = " | ".join(str(v) for v in value)
        value = None if value is None else str(value).strip()
        if value is not None and value.lower() in EMPTY_MARKERS:
            value = None
        if value is not None and key in MULTI_VALUED:
            items = [v.strip() for v in value.split("|") if v.strip()]
            value = items or None
        values[key] = value
    extra = [k for k in raw if str(k).strip().lower() not in requested]
    return values, missing, extra


def amount(value: str) -> float | None:
    digits = re.sub(r"[^\d.]", "", value or "")
    try:
        return float(digits)
    except ValueError:
        return None


def matches(key: str, got, expected) -> bool:
    if expected is None:
        return got is None
    if got is None:
        return False
    if key in AMOUNT_FIELDS:
        return amount(got) == amount(expected)
    if key == "phone":
        return re.sub(r"\D", "", got) == re.sub(r"\D", "", expected)
    if isinstance(expected, list):
        return isinstance(got, list) and [g.lower() for g in got] == [e.lower() for e in expected]
    return got.strip().lower() == expected.strip().lower()


def run_case(predict, name, text, requested, expected, run_no) -> dict:
    t0 = time.perf_counter()
    prediction = predict(ocr_text=text, requested_fields={k: HINTS[k] for k in requested})
    elapsed = time.perf_counter() - t0

    raw = dict(prediction.extracted_values or {})  # DSPy Prediction -> plain dict (note: `.values` would be the dict method)
    values, missing, extra = normalize(raw, requested)
    output_json = json.dumps({"fields": values}, ensure_ascii=False)  # must not raise
    checks = {k: matches(k, values[k], expected[k]) for k in requested}
    passed = all(checks.values()) and not extra and not missing

    print(f"--- {name} | run {run_no} | {elapsed:.1f}s | {'PASS' if passed else 'FAIL'}")
    print(f"    raw DSPy values : {json.dumps(raw, ensure_ascii=False)}")
    print(f"    normalized JSON : {output_json}")
    if missing:
        print(f"    keys missing from LLM output (filled with null): {missing}")
    if extra:
        print(f"    UNREQUESTED keys from LLM (dropped): {extra}")
    for key, ok in checks.items():
        if not ok:
            print(f"    MISMATCH {key}: got {values[key]!r}, expected {expected[key]!r}")
    rupee_in_raw = [k for k, v in raw.items() if "₹" in str(v)]
    if rupee_in_raw:
        print(f"    note: ₹ kept in raw values for {rupee_in_raw} (stripped by amount comparison)")
    return {
        "case": name,
        "run": run_no,
        "seconds": round(elapsed, 1),
        "passed": passed,
        "missing": missing,
        "extra": extra,
        "mismatches": [k for k, ok in checks.items() if not ok],
        "raw": raw,
        "values": values,
    }


def error_test() -> None:
    """Call a closed port to learn which exception DSPy/LiteLLM raises when the LLM is down."""
    bad_lm = dspy.LM(settings.llm_model, api_base="http://127.0.0.1:9", api_key="",
                     temperature=0.0, max_tokens=50, cache=False, num_retries=0, timeout=5)
    predict = dspy.Predict(FieldExtractionSignature)
    t0 = time.perf_counter()
    try:
        with dspy.context(lm=bad_lm):
            predict(ocr_text="Total: 10", requested_fields={"total": HINTS["total"]})
        print("--- error test: UNEXPECTED success")
    except Exception as exc:  # noqa: BLE001 - we want to see exactly what is raised
        chain = []
        e = exc
        while e is not None and len(chain) < 4:
            chain.append(f"{type(e).__module__}.{type(e).__name__}")
            e = e.__cause__ or e.__context__
        print(f"--- error test (LLM unreachable) after {time.perf_counter() - t0:.1f}s")
        print(f"    exception chain: {' <- '.join(chain)}")
        print(f"    message: {str(exc).splitlines()[0][:200]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=3, help="runs per main case (consistency check)")
    parser.add_argument("--no-scaling", action="store_true", help="skip the 1-field / 10-field latency cases")
    parser.add_argument("--error-test", action="store_true", help="also test the LLM-unreachable error path")
    parser.add_argument("--show-prompt", action="store_true", help="print the first prompt/response DSPy sent")
    args = parser.parse_args()

    print(f"dspy {dspy.__version__} | model {settings.llm_model} @ {settings.llm_api_base}")

    # 1. Ollama reachable? Load the model and measure its cold-start time.
    try:
        version = ollama_get("/api/version", timeout=5)["version"]
    except OSError as exc:
        sys.exit(f"Ollama is not reachable at {settings.llm_api_base} ({exc}). Start it with: ollama serve")
    t0 = time.perf_counter()
    warm = ollama_get("/api/generate", {"model": ollama_model_name(), "prompt": "", "keep_alive": "10m"})
    load_s = time.perf_counter() - t0
    running = ollama_get("/api/ps").get("models", [])
    size_gb = next((m["size"] / 1e9 for m in running if m["name"].startswith(ollama_model_name())), None)
    print(f"ollama {version} | model load {warm.get('load_duration', 0) / 1e9:.1f}s "
          f"(request {load_s:.1f}s) | resident model size {size_gb:.2f} GB\n" if size_gb else
          f"ollama {version} | model load request {load_s:.1f}s\n")

    # 2. Configure DSPy once (cache off so every run really hits the model).
    lm = dspy.LM(
        settings.llm_model,
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key or "",
        temperature=0.0,
        max_tokens=1000,
        cache=False,
        num_retries=1,
        timeout=settings.llm_timeout_seconds,
    )
    dspy.configure(lm=lm)
    predict = dspy.Predict(FieldExtractionSignature)

    results = []
    total_t0 = time.perf_counter()
    for name, text, requested, expected in CASES:
        for run_no in range(1, args.runs + 1):
            results.append(run_case(predict, name, text, requested, expected, run_no))
            if args.show_prompt and len(results) == 1:
                dspy.inspect_history(n=1)
    if not args.no_scaling:
        for name, text, requested, expected in SCALING_CASES:
            results.append(run_case(predict, name, text, requested, expected, 1))
    total_s = time.perf_counter() - total_t0

    if args.error_test:
        print()
        error_test()

    print("\n=== summary")
    for r in results:
        flags = []
        if r["mismatches"]:
            flags.append(f"mismatch={r['mismatches']}")
        if r["extra"]:
            flags.append(f"extra={r['extra']}")
        if r["missing"]:
            flags.append(f"missing={r['missing']}")
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['seconds']:>5.1f}s  {r['case']} (run {r['run']})"
              + (f"  {' '.join(flags)}" if flags else ""))
    passed = sum(r["passed"] for r in results)
    print(f"  {passed}/{len(results)} passed | total extraction time {total_s:.1f}s")

    out = PROJECT_ROOT / "outputs" / "phase3_dspy_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  saved {out.relative_to(PROJECT_ROOT)}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
