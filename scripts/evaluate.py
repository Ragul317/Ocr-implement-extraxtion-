"""Evaluate OCR + field extraction on tests/samples (PLAN.md §23). Real PaddleOCR + real Ollama, no cache.

Usage (Ollama running, FastAPI NOT running, to keep a single PaddleOCR in memory):
    python scripts/evaluate.py

Per image: OCR metrics against <stem>.truth.txt, key-token checks, box/ink check, then two
extraction requests (10 predefined fields; custom fields). Each field is scored:
  correct       matches the expected value (amount formatting differences are noted)
  missing       a value was expected, null was returned
  incorrect     wrong value that does exist in the OCR text (misattribution)
  hallucinated  a value that does not appear in the OCR text at all
Fields whose expected answer is ambiguous for a document are not requested there.
Each scored field also records whether the expected value is present in the OCR text at all
(`ocr_has_expected`): if not, a wrong answer is caused by OCR, not by the LLM.
Results: printed tables + outputs/evaluation/phase8_results.json

    python scripts/evaluate.py --rescore   # re-score the saved LLM answers (re-runs OCR only, no LLM)
"""

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from backend import ocr_service  # noqa: E402
from backend.dspy_service import ExtractionError, Extractor  # noqa: E402
from backend.fields import PREDEFINED_FIELDS, resolve_fields  # noqa: E402
from backend.image_preprocessing import preprocess_image  # noqa: E402

SAMPLES = ROOT / "tests" / "samples"
OUT = ROOT / "outputs" / "evaluation"

INVOICE = {
    "invoice_number": "INV-1001", "vendor_name": "SHREE GANESH TRADERS", "customer_name": "ABC Stores",
    "date": "06/10/2026", "gst_number": "29ABCDE1234F1Z5", "phone_number": "+91 98450 12345",
    "product": ["Milk 1L", "Bread", "Basmati Rice 5kg"], "quantity": ["2", "1", "1"],
    "subtotal": "590.00", "total": "619.50",
    # custom
    "address": "12, MG Road, Bengaluru 560001", "payment_reference": None, "cgst_amount": "14.75",
    "purchase_order_number": None,
}
RECEIPT = {
    "invoice_number": "58213", "vendor_name": "FRESH MART SUPERMARKET", "customer_name": None,
    "date": "2026-09-28", "gst_number": None, "phone_number": "044-2615 7788",
    "product": ["Tomato 1kg", "Onion 2kg", "Toor Dal 1kg", "Sunflower Oil 1L"],
    "subtotal": None, "total": "413.00",
    # custom ("quantity" not scored: the receipt has no quantity column)
    "address": "Anna Nagar, Chennai 600040", "bill_no": "58213", "payment_reference": None, "cgst_amount": None,
    "purchase_order_number": None,
}
BLANK = {"invoice_number": None, "date": None, "total": None, "address": None, "cgst_amount": None}

CASES = [
    ("invoice_clear.png", INVOICE, ["INV-1001", "29ABCDE1234F1Z5", "06/10/2026", "619.50", "590.00"]),
    ("receipt_mono.png", RECEIPT, ["FRESH MART SUPERMARKET", "58213", "2026-09-28", "413.00"]),
    ("invoice_large.jpg", INVOICE, ["INV-1001", "29ABCDE1234F1Z5", "06/10/2026", "619.50", "590.00"]),
    ("invoice_blurry.jpg", INVOICE, ["INV-1001", "29ABCDE1234F1Z5", "06/10/2026", "619.50", "590.00"]),
    ("blank.png", BLANK, []),
]
PREDEFINED_KEYS = [f.key for f in PREDEFINED_FIELDS]


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def norm(value) -> str:
    v = re.sub(r"(₹|rs\.?)", "", str(value).lower())
    return re.sub(r"\s+", " ", v).strip(" .")


def same(key: str, got, exp) -> tuple[bool, bool]:
    """(equal, formatting_differs)."""
    if isinstance(exp, list):
        return isinstance(got, list) and [norm(g) for g in got] == [norm(e) for e in exp], False
    if key == "phone_number":
        return re.sub(r"\D", "", str(got)) == re.sub(r"\D", "", exp), False
    if re.fullmatch(r"[\d,]+\.\d+", exp):
        try:
            equal = float(re.sub(r"[^\d.]", "", str(got))) == float(exp.replace(",", ""))
        except ValueError:
            return False, False
        return equal, equal and str(got) != exp
    return norm(got) == norm(exp), False


def in_text(value, text: str) -> bool:
    """Every item (lists, and newline-separated values) occurs in the OCR text, ignoring spaces."""
    items = value if isinstance(value, list) else [value]
    parts = [p for i in items for p in str(i).splitlines() if p.strip()]
    return all(squash(norm(p)) in squash(norm(text)) for p in parts)


def score(key, got, exp, ocr_text) -> str:
    if exp is None:
        if got is None:
            return "correct"
        return "incorrect" if in_text(got, ocr_text) else "hallucinated"
    if got is None:
        return "missing"
    equal, fmt = same(key, got, exp)
    if equal:
        return "correct (format)" if fmt else "correct"
    return "incorrect" if in_text(got, ocr_text) else "hallucinated"


def ocr_metrics(name: str, ocr: dict, key_tokens: list[str]) -> dict:
    truth_path = SAMPLES / f"{Path(name).stem}.truth.txt"
    truth = truth_path.read_text(encoding="utf-8") if truth_path.exists() else ""
    raw = ocr["raw_text"]
    tc, oc = Counter(c for c in truth if not c.isspace()), Counter(c for c in raw if not c.isspace())
    tt, ot = Counter(truth.split()), Counter(raw.split())
    confs = [ln["confidence"] for ln in ocr["lines"]]
    img = Image.open(SAMPLES / name).convert("L")
    ink = 0
    for ln in ocr["lines"]:
        crop = img.crop(tuple(ln["box"]))
        px = crop.histogram()
        ink += sum(px[:170]) / max(1, crop.width * crop.height) > 0.03
    return {
        "lines": ocr["line_count"],
        "truth_lines": len([l for l in truth.splitlines() if l.strip()]),
        "char_recall": round(1 - sum((tc - oc).values()) / sum(tc.values()), 3) if tc else None,
        "token_recall": round(1 - sum((tt - ot).values()) / sum(tt.values()), 3) if tt else None,
        "missing_tokens": list((tt - ot).elements())[:12],
        "conf_min": min(confs, default=None), "conf_avg": ocr["average_confidence"], "conf_max": max(confs, default=None),
        "rupee_truth": truth.count("₹"), "rupee_ocr": raw.count("₹"),
        "key_tokens_verbatim": {t: (t in raw) for t in key_tokens},
        "boxes_on_ink": f"{ink}/{len(ocr['lines'])}",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    previous = None
    if "--rescore" in sys.argv:
        data = json.loads((OUT / "phase8_results.json").read_text(encoding="utf-8"))
        previous = {e["image"]: e for e in data["images"]}
        report_init = data["ocr_init_s"]
    t0 = time.perf_counter()
    ocr_service.get_engine()
    init_s = time.perf_counter() - t0
    extractor = Extractor(cache=False)
    report = {"ocr_init_s": report_init if previous else round(init_s, 1), "images": [],
              "rescored": bool(previous)}
    print(f"OCR engine init: {init_s:.1f}s\n")

    for name, truth, key_tokens in CASES:
        data = (SAMPLES / name).read_bytes()
        t = time.perf_counter(); image = preprocess_image(data); pre_s = time.perf_counter() - t
        t = time.perf_counter(); ocr = ocr_service.run_ocr(image); ocr_s = time.perf_counter() - t
        metrics = ocr_metrics(name, ocr, key_tokens)
        entry = {"image": name, "resized": image.resized, "preprocess_s": round(pre_s, 2), "ocr_s": round(ocr_s, 1),
                 "ocr": metrics, "ocr_raw_text": ocr["raw_text"], "ocr_ai_text": ocr["text"], "requests": [], "fields": {}}
        saved = previous.get(name) if previous else None
        predefined = [k for k in PREDEFINED_KEYS if k in truth]
        custom = [k for k in truth if k not in PREDEFINED_KEYS]
        for keys in (predefined, custom):
            if not keys:
                continue
            fields = resolve_fields(keys)
            if saved is not None:   # --rescore: reuse the recorded LLM answers
                values = {k: saved["fields"][k]["got"] for k in keys}
                req = next(r for r in saved["requests"] if r["fields"] == len(keys))
                llm_s, error = req["llm_s"], req["error"]
            elif not ocr["text"].strip():
                values, llm_s, error = {k: None for k in keys}, 0.0, None   # pipeline skips the LLM
            else:
                t = time.perf_counter()
                try:
                    values, error = extractor.extract(ocr["text"], fields).values, None
                except ExtractionError as exc:
                    values, error = {k: None for k in keys}, f"{exc.kind}: {exc.message}"
                llm_s = time.perf_counter() - t
            entry["requests"].append({"fields": len(keys), "llm_s": round(llm_s, 1), "error": error})
            for k in keys:
                entry["fields"][k] = {"expected": truth[k], "got": values.get(k),
                                      "score": "error" if error else score(k, values.get(k), truth[k], ocr["raw_text"]),
                                      "ocr_has_expected": truth[k] is None or in_text(truth[k], ocr["raw_text"])}
        if saved is not None:  # keep the timings of the run that produced the answers
            entry.update({k: saved[k] for k in ("preprocess_s", "ocr_s")})
        entry["total_s"] = round(entry["preprocess_s"] + entry["ocr_s"] + sum(r["llm_s"] for r in entry["requests"]), 1)
        report["images"].append(entry)

        m = metrics
        print(f"=== {name}  (resized={image.resized}, OCR {ocr_s:.1f}s, LLM "
              f"{' + '.join(str(r['llm_s']) + 's/' + str(r['fields']) + 'f' for r in entry['requests'])})")
        print(f"  OCR: {m['lines']} lines (truth {m['truth_lines']}), char recall {m['char_recall']}, token recall "
              f"{m['token_recall']}, conf {m['conf_min']}–{m['conf_max']} (avg {m['conf_avg']}), "
              f"₹ {m['rupee_ocr']}/{m['rupee_truth']}, boxes on ink {m['boxes_on_ink']}")
        print(f"  key tokens verbatim: {m['key_tokens_verbatim']}  missing tokens: {m['missing_tokens']}")
        for k, f in entry["fields"].items():
            cause = "" if f["score"].startswith("correct") else ("  [OCR text lacks expected]" if not f["ocr_has_expected"] else "")
            print(f"  {f['score']:<17} {k:<22} got={json.dumps(f['got'], ensure_ascii=False)}  "
                  f"expected={json.dumps(f['expected'], ensure_ascii=False)}{cause}")
        print()

    totals = Counter(f["score"].split(" ")[0] for e in report["images"] for f in e["fields"].values())
    report["field_totals"] = dict(totals)
    print("field totals:", dict(totals))
    (OUT / "phase8_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {(OUT / 'phase8_results.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
