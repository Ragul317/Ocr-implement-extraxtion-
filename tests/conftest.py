import io
import json
import urllib.request
from pathlib import Path

import pytest
from PIL import Image

from backend.config import settings

TESTS_DIR = Path(__file__).resolve().parent
SAMPLES = TESTS_DIR / "samples"
FIXTURES = TESTS_DIR / "fixtures"


def image_bytes(size=(200, 100), color="white", fmt="PNG", mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format=fmt)
    return buf.getvalue()


def ollama_running() -> bool:
    try:
        with urllib.request.urlopen(settings.llm_api_base.rstrip("/") + "/api/version", timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def recorded_ocr_lines() -> list[dict]:
    """Real PaddleOCR (PP-OCRv6_small) output for samples/invoice_clear.png, recorded in Phase 2."""
    return json.loads((FIXTURES / "ocr_invoice_clear_small.json").read_text(encoding="utf-8"))["lines"]
