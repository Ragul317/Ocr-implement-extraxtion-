"""Phase 2: standalone PaddleOCR 3.x test (no FastAPI, no DSPy).

Runs real PaddleOCR on one or more images and prints, per image:
detected text, confidence, polygon/box, timings, and (if a
`<stem>.truth.txt` file sits next to the image) a simple token-recall
score and a ₹ check. Results are saved as plain JSON plus PaddleOCR's
own box visualisation under outputs/phase2/<model>/.

Usage:
    python scripts/try_ocr.py tests/samples/invoice_clear.png
    python scripts/try_ocr.py tests/samples/*.png tests/samples/*.jpg
    python scripts/try_ocr.py --det PP-OCRv5_mobile_det --rec en_PP-OCRv5_mobile_rec tests/samples/invoice_clear.png
"""

import argparse
import json
import os
import resource
import sys
import time
from collections import Counter
from pathlib import Path

# Quieten Paddle's C++ (glog) INFO logging; must be set before paddle is imported.
os.environ.setdefault("GLOG_minloglevel", "2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings  # noqa: E402


def peak_rss_mb() -> float:
    # ru_maxrss is in KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def to_plain_lines(res) -> list[dict]:
    """Convert one PaddleOCR 3.x result into plain Python (JSON-serialisable) data."""
    texts = res["rec_texts"]
    scores = res["rec_scores"]
    polys = res["rec_polys"]
    boxes = res["rec_boxes"]
    lines = []
    for text, score, poly, box in zip(texts, scores, polys, boxes):
        lines.append(
            {
                "text": str(text),
                "confidence": round(float(score), 4),
                "polygon": [[int(x), int(y)] for x, y in poly],
                "box": [int(v) for v in box],
            }
        )
    return lines


def token_recall(truth_text: str, ocr_text: str) -> tuple[float, list[str]]:
    """Fraction of ground-truth tokens found in the OCR output (order-insensitive)."""
    truth_tokens = Counter(truth_text.split())
    ocr_tokens = Counter(ocr_text.split())
    missing = list((truth_tokens - ocr_tokens).elements())
    total = sum(truth_tokens.values())
    return (1 - len(missing) / total if total else 1.0), missing


def char_recall(truth_text: str, ocr_text: str) -> float:
    """Fraction of ground-truth characters (whitespace ignored) present in the OCR output.

    Token recall punishes merged words ("Phone:+91") as hard as wrong ones; this does not.
    """
    truth_chars = Counter(c for c in truth_text if not c.isspace())
    ocr_chars = Counter(c for c in ocr_text if not c.isspace())
    total = sum(truth_chars.values())
    return 1 - sum((truth_chars - ocr_chars).values()) / total if total else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--det", default=settings.ocr_det_model, help="text detection model name (default: PaddleOCR default)")
    parser.add_argument("--rec", default=settings.ocr_rec_model, help="text recognition model name (default: PaddleOCR default)")
    parser.add_argument("--lang", default=settings.ocr_lang)
    parser.add_argument("--device", default=settings.ocr_device)
    parser.add_argument("--no-textline-orientation", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="do not print every detected line")
    args = parser.parse_args()

    from paddleocr import PaddleOCR
    import paddle
    import paddleocr

    print(f"paddlepaddle {paddle.__version__} | paddleocr {paddleocr.__version__} | device={args.device} lang={args.lang}")

    init_kwargs = dict(
        lang=args.lang,
        device=args.device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=not args.no_textline_orientation,
    )
    if args.det:
        init_kwargs["text_detection_model_name"] = args.det
    if args.rec:
        init_kwargs["text_recognition_model_name"] = args.rec

    t0 = time.perf_counter()
    ocr = PaddleOCR(**init_kwargs)
    init_s = time.perf_counter() - t0

    # Report the models PaddleOCR actually resolved (not just what we asked for).
    # `_merged_paddlex_config` is private, so fall back to the requested names if it changes.
    try:
        sub_modules = ocr._merged_paddlex_config["SubModules"]
        det_name = sub_modules["TextDetection"]["model_name"]
        rec_name = sub_modules["TextRecognition"]["model_name"]
    except (AttributeError, KeyError, TypeError):
        det_name, rec_name = args.det or "default", args.rec or "default"
    model_tag = f"{det_name}+{rec_name}"
    print(f"models: det={det_name} rec={rec_name} | init {init_s:.1f}s | peak RSS {peak_rss_mb():.0f} MB\n")

    out_dir = PROJECT_ROOT / "outputs" / "phase2" / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for i, image in enumerate(args.images):
        t0 = time.perf_counter()
        results = ocr.predict(str(image))
        ocr_s = time.perf_counter() - t0
        res = results[0]

        lines = to_plain_lines(res)
        json.dumps(lines)  # proves the result is plain, serialisable Python data
        full_text = "\n".join(line["text"] for line in lines)
        avg_conf = sum(line["confidence"] for line in lines) / len(lines) if lines else None

        print(f"=== {image.name}  ({ocr_s:.1f}s{' incl. first-call warm-up' if i == 0 else ''}, {len(lines)} lines)")
        if not args.quiet:
            for n, line in enumerate(lines, 1):
                print(f"  {n:>2}. {line['confidence']:.3f}  box={line['box']}  {line['text']}")

        row = {
            "image": image.name,
            "ocr_s": round(ocr_s, 2),
            "lines": len(lines),
            "avg_conf": round(avg_conf, 3) if avg_conf is not None else None,
            "min_conf": min((line["confidence"] for line in lines), default=None),
        }

        truth_path = image.with_name(f"{image.stem}.truth.txt")
        if truth_path.exists():
            truth = truth_path.read_text(encoding="utf-8")
            recall, missing = token_recall(truth, full_text)
            row["token_recall"] = round(recall, 3)
            row["char_recall"] = round(char_recall(truth, full_text), 3)
            row["rupee_truth"] = truth.count("₹")
            row["rupee_ocr"] = full_text.count("₹")
            print(f"  char recall (no spaces): {row['char_recall']:.1%} | token recall: {recall:.1%}"
                  + (f" | missing/misread tokens: {missing}" if missing else ""))
            if row["rupee_truth"]:
                print(f"  ₹ in truth: {row['rupee_truth']} | ₹ in OCR: {row['rupee_ocr']}")
        print()

        (out_dir / f"{image.stem}.json").write_text(
            json.dumps({"image": image.name, "full_text": full_text, "lines": lines}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        res.save_to_img(str(out_dir))
        summary.append(row)

    print(f"--- summary ({model_tag}) | init {init_s:.1f}s | peak RSS {peak_rss_mb():.0f} MB | saved to {out_dir.relative_to(PROJECT_ROOT)}")
    for row in summary:
        print("  " + json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
