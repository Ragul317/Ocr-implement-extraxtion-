"""Run the Phase 4 pipeline (preprocess -> PaddleOCR -> DSPy/Ollama) on one image.

Usage:
    python scripts/run_pipeline.py tests/samples/invoice_clear.png \
        --fields invoice_number,date,vendor_name,gst_number,total
    python scripts/run_pipeline.py IMAGE --no-cache      # force a real LLM call (timing)
    python scripts/run_pipeline.py IMAGE --show-lines    # also print every OCR line

Requires `ollama serve` to be running.
"""

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.dspy_service import Extractor  # noqa: E402
from backend.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("--fields", help="comma-separated keys (default: the default-checked fields)")
    parser.add_argument("--no-cache", action="store_true", help="bypass the DSPy cache")
    parser.add_argument("--show-lines", action="store_true")
    args = parser.parse_args()

    keys = [k.strip() for k in args.fields.split(",")] if args.fields else None
    extractor = Extractor(cache=False) if args.no_cache else None

    t0 = time.perf_counter()
    result = run_pipeline(args.image.read_bytes(), keys, extractor=extractor)
    wall = time.perf_counter() - t0

    data = result.model_dump()
    text = json.dumps(data, ensure_ascii=False)  # proves JSON-serialisable
    if not args.show_lines and data["ocr"]:
        data["ocr"]["lines"] = f"<{len(data['ocr']['lines'])} lines; use --show-lines>"
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"\nwall {wall:.1f}s | timings_ms {result.timings_ms} | JSON {len(text)} bytes | "
          f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
