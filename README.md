# OCR Extractor

A local, one-page web app that reads an **invoice, receipt or document image** and extracts **the fields you choose**.

- **PaddleOCR** (PP-OCRv6_small, CPU) reads the text, with a confidence score and bounding box for every line.
- **DSPy** with a local **Ollama** model (**qwen2.5:3b**) extracts the requested fields from that text.

Everything runs on your machine: no cloud APIs, no API keys, no database.

> **Status: complete (Phases 1–8).** Final evaluation: [EVALUATION.md](EVALUATION.md). Design and phase log: [PLAN.md](PLAN.md).


run cmd 

~/.local/ollama/bin/ollama serve

 cd /home/finstein-emp/Documents/projects/dataviz/ocr-implementation
python3 -m venv .venv
source .venv/bin/activate


uvicorn backend.main:app --host 127.0.0.1 --port 8000

## Architecture

```
Browser: frontend/ (HTML + CSS + vanilla JS, served by FastAPI at /)
    │  POST /api/extract   (multipart: file + one `fields` part per field key)
    ▼
FastAPI (backend/main.py)  ── request checks, one response shape for everything
    ▼
Image preprocessing (backend/image_preprocessing.py)  ── validate, EXIF fix, downscale to ≤ 2000 px
    ▼
PaddleOCR 3.x (backend/ocr_service.py)  ── one engine, on one dedicated thread; boxes in original pixels
    ▼
DSPy (backend/dspy_service.py)  ── dspy.Predict(FieldExtractionSignature) → only the requested fields
    ▼
Ollama / qwen2.5:3b (local, CPU)
    ▼
Structured fields + OCR text + confidence + boxes  ──►  rendered in the browser
```

The OCR result is always returned, even if the AI step fails (for example, Ollama not running).

## Setup

Requirements: Linux, CPU only, Python 3.12, about 8 GB RAM.

```bash
cd /home/finstein-emp/Documents/ocr
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 1. PaddlePaddle (CPU) from the official Paddle index (it is not on the default index at this version)
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# 2. Everything else, pinned
python -m pip install -r requirements.txt

# 3. Settings (no secrets needed for Ollama)
cp .env.example .env
```

PaddleOCR downloads its models (about 180 MB) to `~/.paddlex/` on first use.

### Ollama

The AI step uses **qwen2.5:3b** through a local Ollama server (`http://localhost:11434`). In this project Ollama is installed per user, without sudo, in `~/.local/ollama`:

```bash
~/.local/ollama/bin/ollama serve &      # start the server
~/.local/ollama/bin/ollama pull qwen2.5:3b
~/.local/ollama/bin/ollama list         # qwen2.5:3b should be listed
```

The model is about 2 GB on disk and about 2.2 GB in RAM while loaded. Ollama unloads it after a few idle minutes.

### Verify

```bash
python -c "import paddle, paddleocr, dspy, fastapi; print(paddle.__version__, paddleocr.__version__, dspy.__version__, fastapi.__version__)"
pip check
```

## Run

```bash
source .venv/bin/activate
~/.local/ollama/bin/ollama serve &                        # local LLM
uvicorn backend.main:app --host 127.0.0.1 --port 8000     # API + web page (loads PaddleOCR at startup)
```

Then open **http://127.0.0.1:8000/**.

- **One process:** FastAPI serves the page on the same origin as the API, so there is no separate web server, no build step and no CORS setup. Opening `frontend/index.html` from disk (`file://`) is not supported.
- **One uvicorn worker (the default):** each worker would load its own PaddleOCR.
- **Without Ollama** the app still runs OCR and shows "AI service offline".

Stop with `Ctrl+C` (uvicorn) and `pkill -x ollama`.

## Usage

```
Upload image  →  Select fields  →  Extract  →  Review structured values  →  Inspect OCR details
```

1. **Upload:** choose a JPG, PNG, BMP or WEBP image (up to 10 MB), or drag and drop it. A preview, the file name and the size are shown.
2. **Select fields:** tick any of the 10 predefined fields (Invoice Number, Vendor Name, Customer Name, Date, GST Number, Phone Number, Product, Quantity, Subtotal, Total), or add custom fields such as "Payment Reference". Duplicates are prevented, and "Selected fields: N" shows the count.
3. **Extract Selected Fields:** a progress message with an elapsed-time counter appears. On CPU this takes about 15–90 s, depending on the fields and on free RAM.
4. **Review:** one row per requested field, in order. "Not found" means the value is absent (it is never invented). "Copy as JSON" copies the values.
5. **Inspect OCR details:** raw OCR text, the text sent to the AI, per-line confidence and box coordinates, and green boxes over the image (toggle).

## API

Interactive docs: http://127.0.0.1:8000/docs (also `/redoc` and `/openapi.json`).

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness: `status`, `ocr_loaded`, models. Does not contact Ollama. |
| `GET /health?check_llm=true` | Also `llm_reachable`, `llm_model_available` (asks Ollama for its model list; no model call). |
| `GET /api/fields` | The 10 predefined fields (key, label, multi, default) and `max_fields`. |
| `POST /api/extract` | OCR + extraction for one image. |

**Canonical request** (`multipart/form-data`, the format the web page sends): one `file` part, then **one `fields` part per field key, repeated, in the order you want**:

```
file   = image
fields = invoice_number
fields = gst_number
fields = total
```

```bash
curl -s -F "file=@tests/samples/invoice_clear.png" \
     -F "fields=invoice_number" -F "fields=gst_number" -F "fields=total" -F "fields=pan_no" \
     http://127.0.0.1:8000/api/extract | python -m json.tool
```

Field keys follow one rule: a label becomes lowercase with `_` (`PAN No.` → `pan_no`). Unknown keys become custom fields. For convenience the API also accepts a comma-separated list of labels (`fields=Invoice Number,GST Number,Total`) or a JSON array.

**Response:** every request returns the same JSON shape:
- `success`, `requested_fields`, `fields` (labels, custom flags)
- `extracted`: one value per requested field, in order; `null` when not found
- `ocr`: `text`, `raw_text`, and `lines`, each with `text`, `confidence`, `polygon` and `box` in original-image pixels
- `image`, `value_adjustments`, `warnings`, `timings_ms`
- `error`: `{stage, kind, message}`

| HTTP | When |
|------|------|
| 200 | Success. Also `success: false` with `error.stage = "dspy"` when the AI step failed but the OCR result is included. |
| 400 | No file, empty file, invalid or empty field list |
| 413 | File larger than 10 MB |
| 415 | Not a JPG/PNG/BMP/WEBP image (checked by name/type and by content) |
| 422 | Corrupt or undecodable image |
| 500 | OCR failure or unexpected server error (no internal details in the response) |

## Tests and checks

```bash
pytest -m "not slow"   # 106 fast tests, no models (~4 s)
pytest                 # 117 tests; slow ones need the PaddleOCR models and `ollama serve` (~75 s with cached AI answers)
```

Browser checks (headless Google Chrome, no extra packages). Run the backend with its log in `outputs/` so the checks can also confirm what FastAPI received:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 > outputs/uvicorn.log 2>&1 &
python scripts/ui_check.py ui            # UI, errors, timeout, 14-field round trip (Ollama STOPPED)
python scripts/ui_check.py smoke         # invoice_clear + 3 fields, real extraction (Ollama RUNNING)
python scripts/ui_check.py integration   # 6 sequential real extractions on all samples (Ollama RUNNING)
```

Sample evaluation (OCR vs `.truth.txt` files and field scoring; run with Ollama running and FastAPI stopped):

```bash
python scripts/evaluate.py               # writes outputs/evaluation/phase8_results.json
```

Other scripts: `scripts/try_ocr.py` (OCR only), `scripts/run_pipeline.py` (pipeline from the command line), `scripts/test_dspy.py` (Phase 3 DSPy check), `scripts/make_samples.py` (regenerates the synthetic samples).

## Configuration (`.env`)

The main settings are listed below; `.env.example` has them all with comments.

| Setting | Default | Meaning |
|---------|---------|---------|
| `LLM_MODEL` / `LLM_API_BASE` | `ollama_chat/qwen2.5:3b` / `http://localhost:11434` | Local model |
| `LLM_TIMEOUT_SECONDS` | 180 | AI step timeout (server side) |
| `LLM_CACHE` | true | DSPy cache: the same text and fields answer instantly |
| `OCR_DET_MODEL` / `OCR_REC_MODEL` | `PP-OCRv6_small_det` / `_rec` | OCR models (`PP-OCRv6_medium_*` is more accurate but slower) |
| `OCR_MAX_SIDE` | 2000 | Larger images are downscaled before OCR |
| `MAX_UPLOAD_MB` / `MAX_FIELDS` | 10 / 20 | Upload and field-count limits |
| `CORS_ALLOW_ORIGINS` | `http://localhost:8000,http://127.0.0.1:8000` | Browser origins allowed to call the API |

No API key is needed. `.env` is git-ignored.

## Limitations

- **Accuracy:** OCR and extraction are good on clean documents but not perfect. See [EVALUATION.md](EVALUATION.md); on the samples, 47/61 fields were correct and none were invented.
- **qwen2.5:3b may miss fields.** It sometimes returns `null` for values that are present, returns only the first item of list fields (Product, Quantity), and can be sensitive to which fields are requested together. It should not invent values, and in the evaluation it never did.
- **Rotated or blurred images can break row grouping.** On a 1.5° rotated, blurred invoice, the amount column shifted by one row, so the AI read `Subtotal = 14.75` instead of `590.00`. Blur also drops `₹` symbols and merges words.
- **Timeouts:** the browser gives up after about **210 s**; the server-side AI timeout is **180 s**, and the server finishes its work even if the browser has given up.
- **Performance depends on free RAM.** CPU-only OCR plus a 2.2 GB local model fit in about 7.5 GB, but with little free memory the system swaps and AI calls slow from about 15 s to 80–100 s.
- **Evaluation used synthetic images** (rendered invoices and receipts, with blurred, large and blank variants), not real phone photos.
- **Browser validation was automated with headless Chrome** (plus an emulated 375 px mobile viewport); other browsers were not tested.

## Troubleshooting

- **"AI service offline"**: start Ollama (`~/.local/ollama/bin/ollama serve &`) and check that `ollama list` shows `qwen2.5:3b`.
- **First request slow**: the model is loading cold, or the machine is swapping. Close heavy applications.
- **`ModuleNotFoundError: setuptools`**: run `pip install setuptools` (Python 3.12 venvs don't include it; Paddle needs it).
- **Do not install `opencv-python`**: PaddleOCR already brings `opencv-contrib-python`.
- **Version pins:** `dspy==3.3.1` (3.4.0 has a circular-import bug with FastAPI), `litellm==1.103.3` and `PyYAML==6.0.2` (required together by PaddleX). Do not upgrade them blindly.
# Ocr-implement-extraxtion-
