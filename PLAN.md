# Simple OCR — Implementation Plan

> **Status: IMPLEMENTATION COMPLETE (Phases 1–8, 2026-10-06).** All phases passed their checkpoints. Final results: EVALUATION.md ("Final Evaluation"). Phase 9's documentation and cleanup tasks were done within Phase 8; its fresh-clone checkpoint was not performed (see §19). The sections below keep the original design, updated with what was actually built.
> Plan date: 2026-10-06. Versions were verified against the official docs in Phase 1 and are pinned in `requirements.txt`.

## 1. Project Overview

A local, single-page web app for learning and demos. A user uploads a photo or scan of an invoice, bill, receipt or document and **chooses which fields to extract**, from predefined fields plus any custom fields they type. **PaddleOCR** reads the text, along with a confidence score and polygon for every line. **DSPy**, driven by a local LLM through Ollama, then extracts **only the selected fields** from that raw text. A small **FastAPI** backend connects the two steps, and one **HTML/CSS/vanilla JS** page shows the results.

## 2. Problem Statement

Invoices and receipts arrive as images. Plain OCR produces an unordered list of text lines, which is not usable data such as "invoice number" or "total". There are two separate problems:
1. **Reading:** turn pixels into text with a measurable confidence. This is OCR.
2. **Understanding:** decide which text is the invoice number, the date, the total, and so on. This is language understanding.

The project keeps these two problems in separate, inspectable stages.

## 3. Project Objective

Deliver a working end-to-end pipeline, `Image → FastAPI → PaddleOCR → raw text → DSPy (dspy.Predict + Signature) → structured JSON → browser`, that:
- runs fully locally on a CPU-only laptop
- uses the real PaddleOCR 3.x API and the real DSPy `Signature`/`Predict` API, with no mocks or fake results
- is small enough to read in one sitting and gives a base for later experiments (DSPy optimization, better OCR, and so on)

## 4. Scope

- One HTML page: image picker, preview, **field selector** (predefined checkboxes + custom fields), "Extract Text" button, loading state, results, errors.
- API endpoints: `POST /api/extract` (image + selected fields), `GET /api/fields` (predefined field catalog), `GET /health`.
- PaddleOCR general OCR pipeline (detection + recognition), on CPU.
- DSPy extraction of **user-selected fields** with one typed Signature and `dspy.Predict`.
- Ollama-served local LLM, configured through `.env`.
- Input validation, size limits, clear error responses, partial-success handling.
- pytest test suite and a manual evaluation sheet.
- README with setup and run instructions.

## 5. Out of Scope

Databases (PostgreSQL/MongoDB/Redis), authentication or user accounts, Docker/Kubernetes, React/Next.js or any frontend framework, cloud deployment, OCR training or fine-tuning, DSPy optimizers (BootstrapFewShot, MIPROv2, GEPA), background workers or queues, microservices, PDF/multi-page input, multi-user concurrency tuning, complex UI.

## 6. Functional Requirements

| ID | Requirement |
|----|-------------|
| FR1 | The user can choose one image file (JPG, JPEG, PNG, BMP, WEBP). |
| FR2 | A preview of the selected image is shown before upload. |
| FR3 | "Extract Text" sends the image as `multipart/form-data` to `POST /api/extract`. |
| FR4 | The button is disabled and a loading indicator is shown while the request runs. |
| FR5 | The backend validates presence, extension, MIME type, real image decodability, and size. |
| FR6 | The backend runs PaddleOCR and returns each line's text, confidence and polygon/box. |
| FR7 | The backend joins recognised lines into one `full_text` in reading order. |
| FR8 | The backend sends `full_text` **and the requested field list** to DSPy, which returns values for **only those fields**. |
| FR9 | Requested fields that are not in the document are returned as `null` (`found: false`), never guessed. |
| FR10 | The response is a single JSON document containing OCR results (raw text, confidence, boxes), the requested fields, and their extracted values. |
| FR11 | If DSPy/LLM fails but OCR succeeded, the OCR result and the requested field list are still returned (partial success). |
| FR12 | The UI shows raw text, a per-line confidence table, optional box overlay, and the extracted values for the selected fields. |
| FR13 | The UI shows clear, human-readable error messages. |
| FR14 | The UI lists predefined fields as checkboxes that the user can select and unselect. |
| FR15 | The user can add custom fields by name and remove them again. |
| FR16 | Only the selected (checked) fields are sent with the image; "Extract Text" is disabled when no field is selected. |
| FR17 | The backend validates the field list (format, count, duplicates) before running OCR. |

### Dynamic Field Selection

**Goal:** the user decides what to extract instead of getting a fixed invoice schema. OCR always runs on the whole image. Only the DSPy step is limited to the selected fields.

**User capabilities**
1. **Select predefined fields:** check any field in the predefined list.
2. **Unselect predefined fields:** uncheck any field; "Select all" / "Clear" shortcuts are provided.
3. **Add custom fields:** type a name (e.g. "Due Date", "Payment Mode"), press **Add** or Enter. It appears as a checked custom field with a remove (×) button.
4. **Submit only the selected fields:** clicking **Extract Text** sends the image plus the list of checked field keys. Unchecked fields are never sent.

**Predefined field catalog** (lives in `backend/fields.py`; served by `GET /api/fields` so the list exists in one place only):

| Key | Label | Hint given to the LLM | Multi-valued | Checked by default |
|-----|-------|-----------------------|:---:|:---:|
| `invoice_number` | Invoice Number | Invoice / bill / receipt number | no | ✔ |
| `vendor_name` | Vendor Name | Seller, shop or company issuing the document | no | ✔ |
| `customer_name` | Customer Name | Buyer / billed-to name | no | |
| `date` | Date | Invoice/bill date, exactly as written | no | ✔ |
| `gst_number` | GST Number | GSTIN, a 15-character alphanumeric tax ID (e.g. `29ABCDE1234F1Z5`) | no | |
| `phone_number` | Phone Number | Phone/mobile number as written | no | |
| `product` | Product | Names of purchased items, one per line item | **yes** | |
| `quantity` | Quantity | Quantity for each line item, same order as Product | **yes** | |
| `subtotal` | Subtotal | Amount before tax, digits only, no currency symbol | no | |
| `total` | Total | Grand total, digits only, no currency symbol | no | ✔ |

**Custom field rules**
- The label the user typed is converted to a key: lowercase, every run of non-alphanumeric characters becomes `_`, leading/trailing `_` are trimmed. Example: "Due Date" becomes `due_date`, and "PAN No." becomes `pan_no`.
- A valid key matches `^[a-z][a-z0-9_]{0,39}$`. The frontend shows an inline error for an empty or invalid name.
- If the key equals a predefined key, the predefined checkbox is checked instead of adding a duplicate. If it equals an existing custom key, it is ignored with a short notice.
- The backend treats any key that is not in the catalog as a custom field. Its label is derived from the key (`due_date` becomes "Due Date"), and the LLM hint is "Value of '<label>' if present in the document". Custom fields are always single-valued.

**Limits:** at least 1 and at most `MAX_FIELDS` (default 20, set in `.env`) fields per request. Duplicates are removed silently and the first occurrence keeps its order.

**What is sent.** For the example selection Invoice Number, Vendor Name, GST Number, Total, the logical payload is:
```json
{ "fields": ["invoice_number", "vendor_name", "gst_number", "total"] }
```
On the wire this is the `fields` form field of the same multipart request as the image, with a JSON array string value (§13).

**What the backend does with it:** it validates the list, runs OCR as usual, and then passes **raw OCR text + the requested fields (key, label, hint, multi-valued flag)** to DSPy (§15). The response echoes the resolved `requested_fields` and returns one extracted value per requested field, in the same order (§13).

**Selection persistence:** none on the server. The page may remember the last selection in `localStorage` (wrapped in try/catch) purely as a per-browser convenience.

## 7. Non-Functional Requirements

- **Local-only:** no cloud services. After first-time model downloads, the app works offline.
- **Simplicity:** about 5 small backend modules and 1 HTML file. No build step.
- **Performance (CPU, measured in Phases 2–3):** OCR takes about 2–6 s per image (PP-OCRv6_small, ≤ 2000 px). LLM extraction takes about 7–25 s for 1–8 fields and up to about 90 s for 10 fields. Timeouts: **LLM 180 s**, end-to-end request target **210 s**.
- **Memory:** must fit in about 7.5 GB RAM alongside the OS: PP-OCRv6_small (about 1.1 GB peak) + qwen2.5:3b (about 2.2 GB) + Python. Large images are downscaled to a 2000 px long side before OCR, there is one OCR instance per process, and one Ollama server.
- **Security:** no secrets in code; strict upload validation; no uploaded files left on disk.
- **Observability:** server logs record each stage's duration and any errors. The response includes `timings_ms`.
- **Readability:** type hints, Pydantic models, and short functions, because this is a learning project.

## 8. Technology Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Language | Python **3.12** | Already installed; supported by current PaddlePaddle 3.x wheels. Avoid 3.13 until Paddle wheels are confirmed. |
| Web framework | FastAPI + Uvicorn | `uvicorn[standard]` |
| Upload parsing | python-multipart | Required by FastAPI for `UploadFile` |
| OCR framework | PaddlePaddle **3.x** (CPU) | Docs on 2026-10-06 showed `paddlepaddle==3.2.0` from the Paddle CPU index. **Re-verify.** |
| OCR library | PaddleOCR **3.x** | `from paddleocr import PaddleOCR`; `.predict()` API (not 2.x `.ocr(img, cls=True)`) |
| Image handling | Pillow, NumPy | Decode/validate the upload in memory and convert it to an ndarray for PaddleOCR |
| LLM programming | DSPy **3.x** | `dspy.LM`, `dspy.configure`, `dspy.Signature`, `dspy.Predict` |
| LLM runtime | **Ollama** (local) | Model: `qwen2.5:3b` (default) or `llama3.2:3b`. About 2 GB each. |
| Config | python-dotenv | `.env` loaded in `config.py` |
| Validation | Pydantic v2 | Shared by FastAPI and DSPy |
| Tests | pytest + httpx (FastAPI `TestClient`) | |
| Frontend | HTML5, CSS, vanilla JS (`fetch`, `FormData`, `<canvas>`) | Served by FastAPI, so no separate frontend server and no CORS issues |

## 9. Architecture

```
┌──────────────────────────── Browser ─────────────────────────────────┐
│ frontend/index.html  (HTML + CSS + vanilla JS)                       │
│   GET /api/fields → field checkboxes + custom fields                 │
│   file input → preview → select fields → [Extract Text] → results    │
└───────────────┬──────────────────────────────────────▲───────────────┘
                │ POST /api/extract (multipart: file+fields)│ JSON
                ▼                                       │
┌──────────────────────── FastAPI (backend/main.py) ───────────────────┐
│  validate upload + fields ─► ocr_service.run_ocr()                   │
│                          ─► dspy_service.extract(text, fields)       │
│   fields.py  (predefined catalog + field resolution/validation)      │
│   schemas.py (Pydantic request/response models)                      │
│   config.py  (.env settings)                                         │
└────────────────────────────┬──────────────────────────┬──────────────┘
                             ▼                          ▼
                 ┌──────────────────────┐   ┌──────────────────────────────┐
                 │ PaddleOCR 3.x (CPU)  │   │ DSPy: FieldExtraction        │
                 │ det → rec → scores   │   │   Signature + dspy.Predict   │
                 │ → polys/boxes        │   │ in: ocr_text, requested_fields
                 └──────────────────────┘   │ out: extracted_values{k: v}  │
                                            │        │ (LiteLLM)           │
                                            │        ▼                     │
                                            │  Ollama @ :11434 qwen2.5:3b  │
                                            └──────────────────────────────┘
```

Key design decisions:
- **One process, one port (8000).** FastAPI serves both the API and `index.html`, which removes the need for CORS and a second server.
- **Models load once at startup** (FastAPI `lifespan`), not per request. Loading PaddleOCR takes seconds and loading the LLM client is cheap.
- **In-memory image handling.** Bytes go through Pillow to an RGB→BGR `numpy.ndarray` and then to `PaddleOCR.predict()`. Nothing is written to disk, so there is nothing to clean up (see §11).
- **Sync endpoint (`def`, not `async def`)** so FastAPI runs the CPU-bound OCR/LLM work in its threadpool. **All PaddleOCR work runs on one dedicated thread** (`ocr_service`, a single-worker executor). Paddle's CPU predictors keep per-thread state, and calling one instance from changing FastAPI worker threads failed with `RuntimeError: std::exception` and could leave the instance permanently broken (Phase 6 finding). The single thread also serializes OCR calls. If a call still fails, the engine is discarded and reloaded on the next request.
- **Field selection only affects DSPy.** OCR output does not depend on the selected fields. The field list is validated before OCR runs (it's cheap, so the request fails fast) and only shapes the DSPy input and the response.
- **One fixed Signature for any field set.** The requested fields are an *input* to the Signature, not its schema, so the same `dspy.Predict` module handles every selection and can be optimized later as a single program (§15).

## 10. End-to-End Data Flow

0. On page load, JS calls `GET /api/fields` and renders the predefined checkboxes (defaults pre-checked).
1. The user selects `receipt.jpg`. JS validates the type and size locally and shows a preview via `URL.createObjectURL`. The user checks/unchecks fields and optionally adds custom ones.
2. Clicking **Extract Text** builds a `FormData` with `file` and `fields` (JSON array of the checked keys, e.g. `["invoice_number","vendor_name","gst_number","total"]`) and calls `fetch('/api/extract', {method:'POST', body})`.
3. FastAPI receives the `UploadFile` and `fields`. It parses and validates the fields (resolving each key to `{key, label, hint, multi, custom}` via `fields.py`), reads the bytes (rejecting > `MAX_UPLOAD_MB`), and checks the extension and MIME type.
4. Pillow opens the bytes (`Image.open` + `verify()`, then re-open), applies EXIF orientation (`ImageOps.exif_transpose`), converts to RGB, then to a numpy BGR array.
5. `ocr_service.run_ocr(array)` calls `ocr.predict(array)`, takes `result[0]`, and reads `rec_texts`, `rec_scores`, `rec_polys` and `rec_boxes`.
6. Normalization produces `OcrLine{text, confidence, polygon, box}` objects, sorted top-to-bottom then left-to-right, with lines grouped into rows (see §14). These are joined into `full_text` with `\n`.
7. If `full_text` is empty, DSPy is skipped and a warning is returned.
8. Otherwise `dspy_service.extract(full_text, requested_fields)` calls `dspy.Predict(FieldExtractionSignature)(ocr_text=full_text, requested_fields=...)`. DSPy formats a prompt, LiteLLM calls Ollama, and DSPy parses the typed `extracted_values` dict.
9. Output is normalized **against the requested list**: one entry per requested key, in request order; keys the LLM did not return become `null`; keys the LLM returned but were not requested are dropped (and logged); empty strings and "N/A" become `null`; multi-valued fields are split into lists.
10. FastAPI returns `OcrResponse` JSON (OCR + requested fields + extracted values + warnings + timings).
11. JS renders the extracted-values table (selected fields only), raw text, confidence table and box overlay canvas, or an error banner.

## 11. Project Folder Structure

```
ocr/                         (project root = /home/finstein-emp/Documents/ocr)
├── frontend/
│   ├── index.html           # single page markup (served by FastAPI at /)
│   ├── style.css            # styles
│   └── app.js               # behaviour: fields, upload, request, rendering (API_BASE_URL constant)
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, lifespan, routes, static mount
│   ├── config.py            # Settings loaded from .env
│   ├── schemas.py           # Pydantic API models
│   ├── fields.py            # predefined field catalog + field-list parsing/validation
│   ├── image_preprocessing.py # decode/validate/downscale in memory (no Paddle import)
│   ├── ocr_service.py       # PaddleOCR wrapper + plain-data conversion + reading order
│   ├── dspy_service.py      # DSPy LM config, Signature, Predict, normalization, amount formatting
│   └── pipeline.py          # image → preprocess → OCR → DSPy → result dict (called by main.py)
├── scripts/
│   ├── make_samples.py      # renders the synthetic sample images + ground truth
│   ├── try_ocr.py           # Phase 2 standalone check (real PaddleOCR on a sample)
│   ├── test_dspy.py         # Phase 3 standalone check (real LLM on sample text)
│   ├── run_pipeline.py      # Phase 4 command-line pipeline run
│   ├── ui_check.py          # headless-Chrome checks: ui / smoke / real / integration (no extra packages)
│   └── evaluate.py          # Phase 8 sample evaluation (OCR vs truth, field scoring)
├── tests/
│   ├── conftest.py
│   ├── samples/             # synthetic invoice/receipt images + .truth.txt (scripts/make_samples.py)
│   ├── fixtures/            # recorded real PaddleOCR output (unit tests)
│   ├── test_fields.py              # field parsing/resolution
│   ├── test_image_preprocessing.py # decode, validate, downscale
│   ├── test_ocr_text.py            # plain-data conversion, reading order, OCR thread
│   ├── test_dspy_normalization.py  # request-scoped normalization, amount formatting, errors
│   ├── test_pipeline_unit.py       # orchestration with OCR/LLM stubs
│   ├── test_pipeline_integration.py# slow: real PaddleOCR + Ollama
│   └── test_api.py                 # HTTP layer, canonical multipart, static page
├── outputs/                 # optional saved JSON results for manual evaluation (git-ignored)
├── EVALUATION.md            # final evaluation (top) + per-phase results
├── .env                     # real local settings (git-ignored)
├── .env.example             # template, committed
├── .gitignore
├── requirements.txt
├── PLAN.md
└── README.md
```

**Changes from the suggested structure, and why:**
- **`uploads/` removed.** Images are processed in memory (bytes → Pillow → numpy), and PaddleOCR 3.x accepts numpy arrays directly. With no temp files there is nothing to clean up and nothing can leak, so the "temporary file cleanup" requirement is met without extra code. If a later need for disk files appears, use `tempfile.NamedTemporaryFile` inside a `try/finally` rather than a shared folder.
- **`outputs/` kept but optional.** When `SAVE_OUTPUTS=true`, each response JSON is written to `outputs/<timestamp>_<name>.json`. These files are used for the manual evaluation in §23.
- **`scripts/` added.** Gives Phases 2–3 a place for standalone experiments before the API exists.
- **`backend/__init__.py` added** so the app runs as `uvicorn backend.main:app` from the project root.
- **`backend/image_preprocessing.py` and `backend/pipeline.py` added (Phase 4).** Preprocessing has no Paddle dependency, so it can be unit-tested in milliseconds. The pipeline holds the orchestration and partial-failure rules, so Phase 5's `main.py` stays a thin HTTP layer.
- **`backend/fields.py` added** for Dynamic Field Selection. The predefined catalog and the key validation rules live here so the API, DSPy service and frontend (via `GET /api/fields`) share one definition.

## 12. Component Responsibilities

| Component | Responsibility | Does NOT do |
|-----------|---------------|-------------|
| **index.html** | Collect the file, preview it, validate it on the client (type/size, as a convenience only), render the field selector from `GET /api/fields`, manage checked/custom fields, POST the file + selected keys, show loading/results/errors, draw boxes on a canvas. | Any OCR or extraction logic; the server is the source of truth for validation. |
| **Field catalog (`fields.py`)** | Define the predefined fields (key, label, LLM hint, multi-valued, default-checked); parse the `fields` form value; validate keys/count; dedupe; resolve unknown keys to custom fields. | Calling the LLM or OCR. |
| **FastAPI (`main.py`)** | HTTP layer: routing, upload reading, server-side validation, calling the services in order, mapping exceptions to HTTP errors, building the response, serving the static page, loading models at startup. | OCR or LLM logic. |
| **OCR service (`ocr_service.py`)** | Own the single `PaddleOCR` instance. Turn an image array into a normalized `OcrResult` (lines, confidences, polygons, boxes, full text, average confidence). | Interpreting meaning (it does not know what an "invoice" is). |
| **PaddleOCR** | The actual OCR engine: text detection (where text is), optional text-line orientation, and recognition (what the text says) with scores. | Field extraction. |
| **DSPy service (`dspy_service.py`)** | Configure `dspy.LM` from settings, hold the `dspy.Predict(FieldExtractionSignature)` module, call it with OCR text + requested fields, normalize the output against the requested list (only requested keys, request order, missing → `null`), and wrap LLM errors in a typed `ExtractionError`. | Reading images. It only ever sees text. Deciding which fields exist. |
| **DSPy Signature (`FieldExtractionSignature`)** | The declarative contract: inputs `ocr_text` and `requested_fields` (key → hint), output `extracted_values` (key → value), and a docstring telling the LM to return only the requested keys and leave missing ones empty. DSPy compiles this into the prompt and parses the reply. | Hand-written prompt strings; a fixed field list. |
| **LLM (Ollama `qwen2.5:3b`)** | Read the noisy OCR text and fill in a value for each requested key. | Running OCR; it never sees the image. |
| **API response models (`schemas.py`)** | Pydantic models (`OcrLine`, `OcrResult`, `FieldSpec`, `ExtractedField`, `Extraction`, `OcrResponse`, `ErrorResponse`) that define and validate the JSON contract and appear in the `/docs` OpenAPI UI. | Business logic. |
| **config.py** | Load `.env` via python-dotenv into one `Settings` object (model name, API base, limits, flags). | Hardcoding secrets. |

## 13. API Design

Implemented in `backend/main.py` (Phase 5). The route is thin: HTTP checks + `pipeline.run_pipeline`. **Every response body is a `PipelineResult`** (§19 Phase 4), errors included, so clients handle one shape. Interactive docs: `/docs`, `/redoc`, `/openapi.json` (generated by FastAPI).

### `POST /api/extract`

**Request** (`multipart/form-data`)
- `file` (required): one image, `.jpg .jpeg .png .bmp .webp`; MIME `image/jpeg|png|bmp|webp` (or a generic `application/octet-stream`); at most `MAX_UPLOAD_MB` (10 MB).
- **Canonical browser format (Phase 7):** one `file` part, then **one `fields` part per selected field key, repeated, in display order**, e.g. `fields=invoice_number`, `fields=gst_number`, `fields=total`, `fields=pan_no`. The page builds it in `buildExtractForm()` (`frontend/app.js`). It sends keys, not labels: a label may contain a comma (`Address, Line 1`), which the backend would split into two fields. Locked by `test_canonical_browser_multipart_request` and `test_frontend_builds_canonical_request`. Sending no `fields` part means "use the defaults"; the page never does that because it blocks extraction with 0 fields.
- `fields` (optional, may be repeated): the fields to extract, **in order**. Accepted forms, which can be mixed:
  - comma-separated labels or keys: `Invoice Number, Customer Name, GST Number, Total`
  - a JSON array: `["invoice_number", "customer_name", "gst_number", "total"]`
  - repeated parts: `-F fields=Total -F fields=Date`

  Each name is converted to a key with the §6 rule (`GST Number` → `gst_number`, `PAN No.` → `pan_no`). Keys in the catalog are predefined fields; any other valid key becomes a custom field. If `fields` is omitted, the default-checked fields are used. An explicitly empty list is an error. `MAX_FIELDS` (20) is only a safety cap, so all 10 predefined fields plus custom ones are allowed.
- Images with a long side > `OCR_MAX_SIDE` (2000 px) are downscaled before OCR (§14). `MAX_IMAGE_PIXELS` (40 MP) guards against decompression bombs.

```bash
curl -F "file=@tests/samples/invoice_clear.png" \
     -F "fields=Invoice Number,Customer Name,GST Number,Total" \
     http://127.0.0.1:8000/api/extract
```

**Check order:** fields parse (key format) → file present → name/MIME allowed → non-empty → size ≤ limit (read at most limit+1 bytes) → pipeline: field resolution (count, duplicates) → Pillow decode, the real image check → pixel limit → OCR → DSPy.

**Response (200)**: the pipeline result unchanged (`success`, `image`, `ocr{text, raw_text, lines[{text, confidence, polygon, box}], line_count, average_confidence}`, `requested_fields`, `fields`, `extracted`, `value_adjustments`, `warnings`, `timings_ms`, `error`). See the Phase 4 example in §19. `extracted` has exactly the requested keys, in request order; missing values are `null`; `product`/`quantity` are lists. Boxes are in original-image pixels. Amounts are strings without currency symbols, with their OCR formatting kept (§15). Dates are as written.

**Status codes** (`error.stage` / `error.kind` in the body):

| HTTP | stage / kind | When |
|------|--------------|------|
| 200 | `error: null` | success (also "no text detected": all fields `null` plus a warning) |
| 200 | `dspy` / `unavailable`, `timeout`, `parse`, `unknown` | **partial success**: OCR succeeded and is included, but LLM extraction failed (`success: false`) |
| 400 | `request` / `NO_FILE`, `EMPTY_FILE`, `INVALID_REQUEST` | no `file` part, 0 bytes, malformed multipart |
| 400 | `fields` / `NO_FIELDS_SELECTED`, `INVALID_FIELDS` | empty list, bad JSON, invalid name, more than `MAX_FIELDS` |
| 413 | `request` / `FILE_TOO_LARGE` | larger than `MAX_UPLOAD_MB` |
| 415 | `request` / `UNSUPPORTED_FILE_TYPE` | wrong extension or MIME type |
| 415 | `preprocess` / `unsupported_format` | real content is not JPEG/PNG/BMP/WEBP (e.g. a GIF renamed `.png`) |
| 422 | `preprocess` / `invalid_image` | corrupt or undecodable image, or too many pixels |
| 500 | `ocr` / `ocr_failed` | PaddleOCR failed (or could not load) |
| 500 | `internal` / `INTERNAL_ERROR` | anything unexpected |

Messages are safe for users. Tracebacks and internal exception text only go to the server log. If Ollama is down, the request still returns 200 with the OCR result, and the server keeps running. The handler is a sync `def`, so FastAPI runs the slow OCR and LLM work in a worker thread and `/health` stays responsive. The end-to-end target is 210 s (`REQUEST_TIMEOUT_SECONDS`, enforced by the browser in Phase 6); the LLM itself times out at 180 s.

### `GET /api/fields`
Returns the predefined field catalog that the frontend uses to render checkboxes, plus the limits it needs:
```json
{
  "max_fields": 20,
  "custom_key_pattern": "^[a-z][a-z0-9_]{0,39}$",
  "fields": [
    { "key": "invoice_number", "label": "Invoice Number", "multi": false, "custom": false, "default": true },
    { "key": "product",        "label": "Product",        "multi": true,  "custom": false, "default": false }
  ]
}
```
(The list is abbreviated here; the real response contains all 10 predefined fields from §6. LLM hints stay server-side.)

### `GET /health`
A cheap liveness check: no OCR, and Ollama is not contacted by default:
`{"status":"ok","ocr_model":"PP-OCRv6_small_det","ocr_loaded":true,"ocr_error":null,"llm_model":"ollama_chat/qwen2.5:3b"}`.
`GET /health?check_llm=true` also asks Ollama's `/api/tags` (about 1 s, no model call) and adds `llm_reachable` and `llm_model_available`, so the Phase 6 UI can show "LLM offline".

### Startup and CORS
- `lifespan` preloads PaddleOCR (`OCR_PRELOAD=true`) so the first request is fast. If loading fails, the server still starts and `/health` reports `ocr_error`.
- CORS origins come from `CORS_ALLOW_ORIGINS` in `.env` (default `http://localhost:8000,http://127.0.0.1:8000`). Methods `GET`/`POST`, header `Content-Type`, no credentials. Add e.g. `http://localhost:5500` if the page is served by a separate dev server.

### `GET /`
Phase 6: serves `frontend/index.html` (FastAPI `StaticFiles(directory="frontend", html=True)` mounted after the API routes).

## 14. OCR Processing Flow

```
Image bytes
 → image_preprocessing.preprocess_image(bytes)        (backend/image_preprocessing.py)
     verify + format allow-list → pixel limit → JPEG draft decode → EXIF orientation fix
     → RGB (alpha flattened on white) → downscale if long side > OCR_MAX_SIDE (2000) → numpy BGR array
 → PaddleOCR.predict(array)
     1. (optional) document orientation classification   [disabled: use_doc_orientation_classify=False]
     2. (optional) document unwarping                     [disabled: use_doc_unwarping=False]
     3. text DETECTION  → dt_polys  (where text lines are)
     4. (optional) text-line orientation                  [enabled: use_textline_orientation=True, handles upside-down lines cheaply]
     5. text RECOGNITION → rec_texts + rec_scores (what each line says, confidence 0..1)
     6. filter by text_rec_score_thresh                    → rec_polys, rec_boxes (kept lines only)
 → result[0] (one image = one result)
 → normalized OcrResult (plain dicts; boxes scaled back to ORIGINAL image coordinates)
```

**Image preprocessing (decided after Phase 2).** Phase 2 showed that PaddleOCR's built-in 4000 px limit is far too high for this machine. A 4960×5200 image used 4.27 GB and was killed with `medium`, and needed 2.8 GB with `small`. So preprocessing runs **before** PaddleOCR:
- Read the bytes in memory only; the caller's bytes and file are never modified.
- `Image.verify()` is the real "is this an image" check. Allowed formats: JPEG, PNG, BMP, WEBP.
- Reject images over `MAX_IMAGE_PIXELS` (40 MP) using the header size, before the full decode.
- JPEGs use Pillow's `draft()` so a huge photo is decoded at reduced scale directly, which saves memory.
- Apply EXIF orientation. Convert to RGB, flattening transparency onto white.
- If the longest side is greater than **`OCR_MAX_SIDE` (default 2000 px, configurable)**, resize with LANCZOS, keeping the aspect ratio. Small images are **never upscaled**.
- Return the BGR array plus `original` and `processed` sizes and the scale. OCR polygons and boxes are divided by the scale, so they always refer to the **original** image (needed for the Phase 6 overlay).
- Errors: `InvalidImageError` (corrupt, not an image, too many pixels) and `UnsupportedImageError` (wrong format).

**PaddleOCR 3.x construction (Phase 2 result):**
```python
PaddleOCR(
    lang="en",
    device="cpu",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    text_detection_model_name="PP-OCRv6_small_det",     # OCR_DET_MODEL (Phase 2 decision)
    text_recognition_model_name="PP-OCRv6_small_rec",   # OCR_REC_MODEL (Phase 2 decision)
)
```
`device` is passed through `**kwargs` in 3.7. Results are read with dict-style access on `predict(...)[0]`: `rec_texts` (list[str]), `rec_scores` (list[float]), `rec_polys` (list of 4×2 numpy int arrays) and `rec_boxes` (N×4 numpy array). These were confirmed in Phase 2. **Do not** use the 2.x pattern `ocr.ocr(img, cls=True)` with `[[box, (text, score)], ...]`. Only one `PaddleOCR` instance exists per process (a lazy singleton). It is created and called **only on one dedicated thread**, which also serializes calls, because Paddle predictors break when called from changing threads (§9, Phase 6 finding).

**Normalization rules**
- `confidence` is rounded to 3 decimals. `polygon` becomes a list of `[int,int]` and `box` a list of 4 ints (numpy types are converted to plain Python so they serialize to JSON).
- **Raw text is kept unchanged.** Each line's `text` is exactly what PaddleOCR returned, and `ocr.raw_text` joins the lines in PaddleOCR's own order.
- **Light text normalization only (for the LLM text):** Unicode NFC, removal of zero-width/control characters, collapsing runs of spaces, and trimming. Characters are never replaced, so `₹`, decimal points, invoice numbers, GSTINs, dates and phone numbers are untouched.
- **Reading order (minimal row grouping, required by Phase 2 finding 6):** sort by box centre-y. A segment joins the current row when its centre-y is within half the median line height of the row's centre. Within a row, segments are sorted by x and joined with 4 spaces, the same layout the Phase 3 sample used (`Milk 1L    2    ₹50.00`). `ocr.text` is these rows joined with `\n`, and it is what DSPy receives. Limitation: skewed or rotated photos can split or merge rows, and there is no table or column reconstruction (§26). Phase 7 observed it end to end: on `invoice_blurry.jpg` (rotated 1.5°) the right-hand amount column shifts up one row (`Subtotal    14.75`), so the LLM correctly reads a wrong row and returns Subtotal = 14.75. Fix proposal: skew-aware grouping, i.e. estimate the page angle from the polygons and group rows on the de-rotated y.
- `average_confidence` is the mean of the line confidences (or `null` if there are no lines).
- Lines below `OCR_MIN_CONFIDENCE` (default 0.5) are **kept** in `lines` (for transparency) but **excluded** from `full_text` sent to the LLM. A warning lists how many were dropped.

**Model choice (decided in Phase 2, see EVALUATION.md):** **PP-OCRv6_small** det + rec is the default. It reaches 99.6% character recall on the clear invoice, takes about 2–6 s per image, and peaks at about 1.1 GB. `PP-OCRv6_medium` (PaddleOCR's own default) is the accuracy fallback: about 3–4× slower and about 2× the RAM. **PP-OCRv5 mobile English is not used**: it never outputs ₹ and misread `₹14.75` as `314.75`.

**₹ symbol risk:** on clean images PP-OCRv6 recognizes ₹ (small: 8/9). It sometimes drops the ₹ on large bold totals, and on blurred images it drops it every time while still reporting about 0.99 confidence. Digits stayed correct. Mitigation: the Signature tells the LLM that amounts may be prefixed by OCR artefacts. The amount formatting is restored from the OCR text (§15), and the raw OCR text is always returned. A user who wants the currency can add a custom field `currency`.

## 15. DSPy Processing Flow

```
full_text (str) + requested fields [FieldSpec{key, label, hint, multi, custom}]  (from fields.py)
 → build requested_fields input: {key: "<label>: <hint>[ (list every value, separated by ' | ')]"}
 → dspy.Predict(FieldExtractionSignature)(ocr_text=full_text, requested_fields=...)
     DSPy adapter (ChatAdapter, falls back to JSONAdapter) builds the prompt from:
       - Signature docstring (task instructions)
       - input field descriptions + values (the OCR text and the requested key→hint map)
       - output field name, type (dict[str, str]), description
 → dspy.LM("ollama_chat/qwen2.5:3b", api_base=..., temperature=0)  (via LiteLLM)
 → Ollama returns text
 → DSPy parses it into the typed `extracted_values` dict (Pydantic-validated)
 → dspy.Prediction(extracted_values={...})      # read as prediction.extracted_values
 → normalize AGAINST THE REQUEST (in Python, not trusted to the LLM):
     for each requested key, in order:
        raw = extracted_values.get(key)       # exact key; then case-insensitive fallback
        ""/"N/A"/"null"/"none"/"not found" → None; trim whitespace
        multi-valued → split on "|" → list[str] (empty list → None)
        numeric value → restore its OCR formatting (see "Amount formatting" below)
     keys in `extracted_values` that were not requested → dropped + logged at DEBUG
 → extracted {key: value | [values] | None} in request order (+ value_adjustments, warnings)
```

**Signature (class-based, typed, field-agnostic; validated in Phase 3):**
```python
class FieldExtractionSignature(dspy.Signature):
    """Extract the requested fields from noisy OCR text of an invoice, bill, receipt or document.
    Return a value for EVERY key in requested_fields and for NO other keys.
    Use only information present in the text. If a field is not present, use an empty string.
    Do not calculate or invent values. Copy values exactly as written, except amounts:
    remove the currency symbol but keep every digit and decimal exactly as written
    (e.g. '₹724.50' -> '724.50'). Currency symbols such as ₹ may be misread by OCR
    as 'R', 'Rs', 'Z' or '?'. For list-type fields, separate values with ' | '."""
    ocr_text: str = dspy.InputField(desc="Raw OCR text, one detected line per row")
    requested_fields: dict[str, str] = dspy.InputField(
        desc="Fields to extract: key -> what the field means")
    extracted_values: dict[str, str] = dspy.OutputField(
        desc="Same keys as requested_fields -> extracted value, or empty string if absent")
```
- **Why `extracted_values` and not `values` (Phase 3 finding):** `dspy.Prediction` is dict-like, so `prediction.values` is the dict *method*. Worse, with an output field called `values`, qwen2.5:3b echoed the input sections back. `ChatAdapter` couldn't parse that, so DSPy fell back to a second `JSONAdapter` call, which doubled latency and hit the timeout. With `extracted_values` there was 1 clean call per extraction (10 of 10).
- **Amount formatting (Phase 3 finding: the LLM writes `724.5` for `₹724.50` and `690` for `₹690.00`).** This uses a safe restoration step that only ever *copies formatting from the OCR text* and never computes values:
  1. Only values that are a plain number (`^\d[\d,]*(\.\d+)?$`) are considered. Invoice numbers, dates, GSTINs and phone numbers are never touched.
  2. Find every number token in the OCR text whose numeric value equals the LLM's value.
  3. If exactly **one distinct formatting** exists (e.g. only `724.50`), use it with the currency symbol removed. Record `{field, llm_value, ocr_token}` in `value_adjustments`.
  4. If several different formattings match, or none do, keep the LLM value unchanged and add a warning. "Not found in OCR text" is also a useful sign that the value may be invented.

  The raw OCR text and lines are always in the result, so the original printed value (e.g. `₹724.50`) is never lost.
- **Why one fixed Signature, not a new Signature per request.** DSPy can build signatures dynamically (one `OutputField` per requested key), but a fixed Signature with the field list as *input* is simpler, needs no per-request class creation, keeps custom field names out of Python identifiers, and stays a single program that a future optimizer (`dspy.Evaluate`, `BootstrapFewShot`) can tune across all field combinations.
- **Why `dict[str, str]` values.** Small local models handle nullable and union types poorly, so they always return strings. Python maps empty strings to `None` and splits multi-valued fields into lists.
- **"Only the requested fields" is guaranteed in two layers:** the Signature instructs it, and the normalization step enforces it (drops extra keys and fills missing ones with `null`).
- **Multi-valued fields (`product`, `quantity`):** returned as lists in document order, so `product[i]` pairs with `quantity[i]` when both are selected. Proper line-item table extraction is a future improvement (§26).
- **The dict output parses reliably (Phase 3):** the string/batching fallback is not needed.
- **Latency (Phase 3, qwen2.5:3b at about 4.5 tokens/s; the prompt alone is about 480–650 tokens):** 1–8 fields take 7–23 s, but 10 fields took 87 s. So **`LLM_TIMEOUT_SECONDS=180`** and the end-to-end request target is **210 s** (`REQUEST_TIMEOUT_SECONDS`), which also covers OCR. The field count is not capped low: all predefined fields can be selected. `MAX_FIELDS=20` is only a safety limit.
- **LM configuration:** `lm = dspy.LM(settings.llm_model, api_base=..., api_key=settings.llm_api_key or "", temperature=0.0, max_tokens=1000, timeout=180, num_retries=0, cache=settings.llm_cache)`. It is created once, and each call runs inside `with dspy.context(lm=lm):`, which is thread-safe for FastAPI worker threads (unlike changing `dspy.configure` from several threads). `num_retries=0` prevents a 180 s timeout being retried into 360 s.
- **Errors (Phase 3):** an unreachable Ollama raises `dspy.utils.exceptions.LMTransportError` (← `litellm.APIConnectionError`). The service wraps every LLM or parse failure in `ExtractionError` (kinds: `unavailable`, `timeout`, `parse`, `unknown`), and the pipeline keeps the OCR result.
- **Caching:** DSPy caches LM calls by default (keyed on the full prompt, so the same image with a *different* field selection is a new call), so re-uploading the same image with the same fields returns instantly. This is fine for a demo, but note it during evaluation (disable with `cache=False` on `dspy.LM` when measuring latency).
- **Debugging:** `dspy.inspect_history(n=1)` (`scripts/test_dspy.py --show-prompt`) shows the exact prompt and reply. This is very useful for learning.
- Only `dspy.Predict` is used. No `ChainOfThought` and no optimizers in v1.

## 16. Frontend Flow

Single file `frontend/index.html` with inline `<style>` and `<script>`. Layout: header; left column = upload card + preview + **Fields to extract** card; right column = results; responsive single column on narrow screens.

**Fields to extract card:** a checkbox grid of the predefined fields (default ones pre-checked); "Select all" / "Clear" links; a text input with an **Add** button for custom fields; custom fields shown as checked items with a "custom" badge and a remove (×) button; a live counter "4 of 20 fields selected".

| User action | What happens |
|-------------|-------------|
| **Page load** | `GET /health?check_llm=true` and `GET /api/fields` in parallel. Render the predefined checkboxes from `/api/fields` (default-checked as specified; or the last selection restored from `localStorage` if present). If `/api/fields` fails, show "Could not load field list"; custom fields can still be added. If the OCR model is not ready or the LLM is unreachable, show a yellow notice ("LLM offline: OCR will still work"). The Extract button stays disabled until a file is chosen **and** at least one field is selected. |
| **Selects an image** (file input or drag-and-drop) | JS checks `file.type` against the allowed list and `file.size` ≤ 10 MB. If invalid, show a red inline message and keep the button disabled. If valid, revoke any old object URL, set `<img src=URL.createObjectURL(file)>`, show the file name and size, clear previous results/errors, and enable **Extract Text** (if ≥ 1 field is selected). |
| **Checks / unchecks a predefined field** | Update the selected set and the counter. If the count reaches 0, disable **Extract Text** and show the hint "Select at least one field". If it reaches `max_fields`, disable the remaining unchecked boxes and the Add button. |
| **Adds a custom field** (types a name, presses Add or Enter) | Convert the name to a key (§6 rules). If it's empty or invalid, show an inline error. If it matches a predefined key, check that box instead. If it matches an existing custom key, show "Already added". Otherwise append a checked custom item (label shown as typed) and clear the input. |
| **Removes a custom field** (×) | Remove it from the list and the selected set; update the counter and button state. |
| **Clicks Extract Text** | Disable the button, file input and field controls; change the label to "Extracting…"; show a spinner and the stage text "Running OCR and AI extraction (about 20 s, up to 3 minutes for many fields on CPU)…" plus an elapsed-seconds counter. Build `FormData` with `file` and `fields = JSON.stringify(selectedKeys)` (checked fields only, in display order) and call `fetch('/api/extract', {method:'POST', body, signal})` with an `AbortController` timeout of 210 s (`REQUEST_TIMEOUT_SECONDS`). |
| **Waits** | The elapsed timer ticks every second. No polling and no fake progress percentage. |
| **Receives success** | Hide the spinner and re-enable controls. Render: (1) an **Extracted Fields** card with **one row per requested field, in request order**: label, value (lists shown as bullet items; `null` → greyed "Not found"), and a "custom" badge where applicable; (2) **Raw OCR Text** in a `<pre>` with a "Copy" button; (3) **OCR Lines** table: #, text, confidence shown as a % with colour (green ≥ 0.9, amber 0.7–0.9, red < 0.7); (4) **Bounding boxes**: a `<canvas>` over the preview drawing each polygon, scaled from `image.width/height` to the displayed size, toggled by a checkbox; (5) summary: line count, average confidence, timings, "4 fields requested · 2 found". If `warnings` is non-empty, show a yellow banner. If `extraction_error` is set, show it in the Extracted Fields card (listing the requested field names) while the OCR sections still render. |
| **Receives an error** | Hide the spinner and re-enable controls. If the response is JSON with `error.message`, show it in a red banner with the HTTP code. For a network failure, show "Cannot reach the server. Is the backend running on :8000?". For a timeout (AbortError), show "Request timed out after 210 s.". For non-JSON, show "Unexpected server response (status N).". The previous preview stays so the user can retry. |

All text from the server and all user-typed custom field labels are inserted with `textContent` (never `innerHTML`) to avoid XSS.

Changing the field selection after a result is shown does not re-run anything. The user clicks **Extract Text** again. OCR re-runs too in v1, because results are not cached server-side.

## 17. Environment Setup

- **OS:** Linux (verified). CPU-only (no NVIDIA GPU detected).
- **Python:** 3.12 (`/usr/bin/python3.12`, 3.12.3 installed). May need `sudo apt install python3.12-venv` if `venv` is missing.
- **Virtual environment:** `.venv/` in the project root (git-ignored).
- **Ollama:** 0.35.1, installed per user without sudo in `~/.local/ollama` (official tarball). Start it manually with `~/.local/ollama/bin/ollama serve`; `qwen2.5:3b` is pulled. Server at `http://localhost:11434`.
- **Disk:** about 1.5–2 GB for Python packages, about 0.2–0.5 GB for PaddleOCR models (downloaded on first run to `~/.paddlex/`), and about 2 GB for the LLM.
- **RAM:** about 7.5 GB total, with only about 1.8 GB free at inspection time. Close heavy apps (browsers or IDEs with many tabs) during testing. Budget: PaddleOCR mobile about 0.5–1 GB, qwen2.5:3b about 2.5–3 GB, FastAPI/Python about 0.3 GB.
- **Environment variables (`.env.example`):**
  ```dotenv
  # --- LLM (DSPy via LiteLLM) ---
  LLM_MODEL=ollama_chat/qwen2.5:3b
  LLM_API_BASE=http://localhost:11434
  LLM_API_KEY=                 # empty for Ollama; set if you later switch to a hosted provider
  LLM_TIMEOUT_SECONDS=180      # Phase 3: 10 fields took 87 s
  LLM_NUM_RETRIES=0            # never retry a 180 s timeout
  LLM_CACHE=true               # DSPy on-disk cache; same image + same fields returns instantly

  # --- OCR ---
  OCR_LANG=en
  OCR_DEVICE=cpu
  OCR_DET_MODEL=PP-OCRv6_small_det
  OCR_REC_MODEL=PP-OCRv6_small_rec
  OCR_MIN_CONFIDENCE=0.5
  OCR_MAX_SIDE=2000            # downscale longer images before OCR (Phase 2 RAM finding)

  # --- Upload / request limits ---
  MAX_UPLOAD_MB=10
  MAX_IMAGE_PIXELS=40000000
  REQUEST_TIMEOUT_SECONDS=210  # end-to-end target (OCR + LLM); used by the API/frontend

  # --- Field selection ---
  MAX_FIELDS=20                # safety cap only; all predefined fields can be selected

  # --- Misc ---
  SAVE_OUTPUTS=false
  LOG_LEVEL=INFO

  # --- HTTP API (Phase 5) ---
  CORS_ALLOW_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
  OCR_PRELOAD=true
  ```
- **API key requirements:** **none** for Ollama. The app still reads `LLM_API_KEY` from `.env` so that switching to a hosted provider later only needs a `.env` change, for example `LLM_MODEL=anthropic/claude-haiku-4-5` plus a key. No key is ever written in code.

## 18. Dependency Installation Plan

Order matters, because PaddlePaddle wheels are Python-version and platform specific:

1. **Create and activate the venv** with Python 3.12. Upgrade `pip`.
2. **Verify the current official PaddlePaddle CPU install command** at the PaddleOCR docs ("PaddlePaddle Framework Installation") and paddlepaddle.org.cn. On 2026-10-06 it was:
   `python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/`
   Use whatever the docs say on the day. Check: `python -c "import paddle; paddle.utils.run_check()"`.
3. **Install PaddleOCR 3.x:** `python -m pip install paddleocr` (base package only; `paddleocr[all]` pulls doc-parsing extras that are not needed). Check: `python -c "import paddleocr; print(paddleocr.__version__)"`, which must start with `3.`.
4. **Install the web, AI and test stack:** done in Phase 1, now simply `pip install -r requirements.txt`.
5. **Exact working versions are frozen in `requirements.txt`** (Phase 1, verified with `pip check` in Phase 8):
   `paddleocr==3.7.0`, `setuptools` (Paddle imports it; Python 3.12 venvs lack it), `dspy==3.3.1`
   (3.4.0 breaks `import dspy` before `import fastapi`), `litellm==1.103.3` + `PyYAML==6.0.2` (PaddleX pins
   PyYAML 6.0.2; newer litellm needs ≥ 6.0.3), `fastapi==0.142.2`, `uvicorn[standard]==0.54.0`,
   `python-multipart==0.0.32`, `python-dotenv==1.2.4`, `Pillow==12.1.0`, `numpy==2.3.5`, `pytest==9.1.1`,
   `httpx==0.28.1`, `websockets==17.2`. `paddlepaddle==3.2.0` is installed first from the Paddle CPU index.
6. **Ollama:** installed per user without sudo (official tarball into `~/.local/ollama`, v0.35.1), then `ollama pull qwen2.5:3b`.

**Known compatibility risks to watch:**
- **Python version vs Paddle wheels.** If pip says "no matching distribution", the Python version is unsupported. Stay on 3.12.
- **OpenCV conflicts.** PaddleOCR/PaddleX pull in an OpenCV build (for example `opencv-contrib-python`). Do **not** also install `opencv-python`; two OpenCV packages break `cv2` imports.
- **NumPy version.** Let PaddleOCR choose. Do not pin numpy before installing Paddle.
- **Outdated tutorials.** Most online examples use the PaddleOCR 2.x API (`ocr.ocr(img, cls=True)`, `use_angle_cls=True`, `show_log`). These args were removed or renamed in 3.x and raise errors or warnings. Use only the 3.x docs.
- **CPU oneDNN/MKL-DNN issues.** Some Paddle 3.x CPU builds have had oneDNN crashes on certain inputs. If Phase 2 hits segfaults or "onednn" errors, try `enable_mkldnn=False` in the PaddleOCR constructor.
- **First-run model download** needs internet (from HuggingFace/BOS; source configurable via the `PADDLE_PDX_MODEL_SOURCE` env var). It takes a long time on first startup, so do it in Phase 2, not during a demo.
- **DSPy major-version API drift.** Use `dspy.LM("provider/model")` + `dspy.configure(lm=...)`. Old examples with `dspy.OpenAI(...)`, `dspy.OllamaLocal(...)` or `dspy.settings.configure` with legacy clients are outdated.
- **Pydantic v2** is shared by FastAPI and DSPy, which is fine. Do not install anything that needs Pydantic v1.
- **LiteLLM Ollama prefix.** Use `ollama_chat/` (chat endpoint), not `ollama/`, for better instruction following.

## 19. Implementation Phases

Each phase ends with a checkpoint. **Do not start the next phase until the checkpoint passes.**

### Phase 1: Environment setup ✅ (done 2026-10-06)
- **Objective:** Reproducible local environment with OCR and LLM runtimes installed.
- **Files:** `.gitignore`, `.env.example`, `.env`, `requirements.txt` (draft), `README.md` (skeleton), `backend/config.py`.
- **Tasks:** verify the official Paddle/PaddleOCR install docs; create the venv; install packages in the §18 order; install Ollama and pull `qwen2.5:3b`; write `.gitignore` (`.venv/`, `.env`, `__pycache__/`, `outputs/*`, `.pytest_cache/`, `*.log`); write `.env.example`; implement `config.py` (dotenv → `Settings`); optionally `git init`.
- **Expected result:** all imports work; the Ollama server responds.
- **Checkpoint:** `python -c "import paddle, paddleocr, dspy, fastapi; print(paddle.__version__, paddleocr.__version__, dspy.__version__)"` prints 3.x versions; `paddle.utils.run_check()` passes; `ollama run qwen2.5:3b "Say OK"` replies; `python -c "from backend.config import settings; print(settings.llm_model)"` prints the model name; `.env` is listed in `.gitignore`.

### Phase 2: PaddleOCR standalone test ✅ (done 2026-10-06)
- **Objective:** Prove real OCR works on CPU and learn the exact 3.x result structure.
- **Files:** `scripts/try_ocr.py`, `tests/samples/*` (collect 6–8 real images: clear printed invoice, thermal receipt, ₹ bill, blurry photo, blank image, large phone photo, multi-column document).
- **Tasks:** construct `PaddleOCR` (§14 args); run `predict()` on each sample; print `rec_texts`/`rec_scores`/`rec_boxes`; call `save_to_img` to `outputs/` to see the boxes visually; time each call; compare mobile vs default models; check the ₹ rendering; record peak RAM (`/usr/bin/time -v`).
- **Expected result:** readable text with confidence > 0.9 on clear images, plus box overlay images.
- **Checkpoint:** a short table in `EVALUATION.md` (image, model, time, line count, average confidence, ₹ ok?); final det/rec model names written into `.env`; confirmed result access path (`res.json["res"][...]` or equivalent) noted for Phase 4.

### Phase 3: DSPy standalone test ✅ (done 2026-10-06)
- **Objective:** Prove `dspy.Predict(FieldExtractionSignature)` extracts **only the requested fields** from text using local Ollama.
- **Files:** `scripts/test_dspy.py` (named `try_dspy.py` in the original plan).
- **Tasks:** configure `dspy.LM` from `.env`; define `FieldExtractionSignature` and a temporary inline copy of the predefined hints; run on (a) the example invoice text from the requirements with several field selections: `[invoice_number, date, customer_name, total]`, `[invoice_number, vendor_name, gst_number, total]`, `[product, quantity]`, and one custom field (`due_date`); (b) 2–3 real OCR outputs saved in Phase 2; (c) text missing invoice number and customer; print the raw `values` dict; check for extra or missing keys; run `dspy.inspect_history(n=1)`; time each call against the number of fields (1, 4, 10); stop Ollama and observe the exception type.
- **Expected result:** with `[invoice_number, date, customer_name, total]` the example yields `INV-1001`, `06/10/2026`, `ABC Stores`, `590`; `vendor_name`/`gst_number` come back empty; `product` lists Milk, Bread, Rice.
- **Checkpoint:** the example text is extracted correctly in 3 of 3 runs (temperature 0); the raw LLM output contains every requested key and ideally no extra keys (extra keys are recorded; normalization will drop them); missing fields come back empty, not hallucinated; latency for 10 fields stays under `LLM_TIMEOUT_SECONDS`; the §15 dict-vs-string fallback is decided; the exception class for "Ollama down" and for timeout is known for Phase 5 error mapping.

### Phase 4: PaddleOCR + DSPy integration (service modules) ✅ (done 2026-10-06)
- **Objective:** Turn the scripts into reusable, testable services.
- **Files:** `backend/config.py` (+ `MAX_FIELDS`, `OCR_MAX_SIDE`, `LLM_NUM_RETRIES`, `LLM_CACHE`, `REQUEST_TIMEOUT_SECONDS`; `LLM_TIMEOUT_SECONDS=180`), `.env.example`, `backend/schemas.py`, `backend/fields.py`, `backend/image_preprocessing.py`, `backend/ocr_service.py`, `backend/dspy_service.py`, `backend/pipeline.py`, `scripts/run_pipeline.py`, `pytest.ini`, `tests/*`.
- **Tasks:**
  - `fields.py`: `PREDEFINED_FIELDS` catalog (§6); `resolve_fields(keys)` and `parse_fields_input(values)` (Phase 5: labels, comma lists, JSON arrays) (dedupe, count/key validation, custom-field resolution, defaults when omitted) raising `FieldValidationError`.
  - `image_preprocessing.py`: `preprocess_image(bytes) -> PreprocessedImage` (§14).
  - `ocr_service.py`: lazy singleton `PaddleOCR` on one dedicated thread (Phase 6 fix); `run_ocr(image) -> OcrResult` (plain dicts, original-image coordinates, raw text, normalized reading-order text).
  - `dspy_service.py`: `FieldExtractionSignature` (`extracted_values`), `Extractor.extract(text, fields)` → normalization against the request + amount-format restoration; `ExtractionError`.
  - `pipeline.py`: `run_pipeline(image_bytes, field_keys) -> PipelineResult`.
  - `scripts/run_pipeline.py`: a command-line run on one image with a chosen field list.
- **Pipeline result (Python dict via Pydantic `model_dump()`):**
  ```json
  {
    "success": true,
    "image":  {"original_width": 1240, "original_height": 1300, "processed_width": 1240, "processed_height": 1300, "resized": false},
    "ocr":    {"text": "<reading-order text sent to DSPy>", "raw_text": "<PaddleOCR order, unmodified>",
               "line_count": 29, "average_confidence": 0.984,
               "lines": [{"text": "Invoice No: INV-1001", "confidence": 0.98, "polygon": [[59,63],[631,63],[631,100],[59,100]], "box": [59, 63, 631, 100]}]},
    "requested_fields": ["invoice_number", "vendor_name", "gst_number", "total"],
    "fields": [{"key": "invoice_number", "label": "Invoice Number", "multi": false, "custom": false}],
    "extracted": {"invoice_number": "INV-1001", "vendor_name": "…", "gst_number": "…", "total": "724.50"},
    "value_adjustments": [{"field": "total", "llm_value": "724.5", "ocr_token": "₹724.50", "value": "724.50"}],
    "warnings": [],
    "timings_ms": {"preprocess": 40, "ocr": 4100, "extraction": 16000, "total": 20200},
    "error": null
  }
  ```
  Failure rules:
  - Invalid image: `success: false`, `error.stage: "preprocess"`, `ocr: null`.
  - OCR failure: `error.stage: "ocr"`.
  - DSPy/LLM failure: `success: false`, `error: {"stage": "dspy", "kind": "unavailable|timeout|parse|unknown", "message": …}`. **The OCR result is kept**, and `extracted` is `{}`.
  - No text detected: `success: true`, all requested fields `null`, warning, LLM not called.

  Phase 5 maps this result onto the HTTP response (§13).
- **Expected result:** one Python call goes from image bytes + field keys to a JSON-serializable result containing OCR output and exactly the requested fields.
- **Checkpoint:** the clear invoice with `[invoice_number, date, vendor_name, gst_number, total]` returns exactly those 5 fields in that order, with `total` = `619.50` (formatting restored from `₹619.50`). The result passes `json.dumps`. Fast unit tests pass (stubs for OCR/LLM orchestration). Slow integration tests with real PaddleOCR + Ollama pass.

### Phase 5: FastAPI API ✅ (done 2026-10-06)
- **Objective:** Expose the Phase 4 pipeline over HTTP with validation and consistent errors.
- **Files:** `backend/main.py` (new), `backend/fields.py` (`key_from_label`, `parse_fields_input`), `backend/schemas.py` (error stages `request`/`internal`), `backend/config.py` + `.env(.example)` (`CORS_ALLOW_ORIGINS`, `OCR_PRELOAD`), `backend/pipeline.py` and `backend/dspy_service.py` (client-safe error messages only), `tests/test_api.py`.
- **Tasks:**
  - `create_app()` factory with a `lifespan` that preloads OCR.
  - `GET /health` (optional `check_llm`), `GET /api/fields`, and `POST /api/extract` (`file: UploadFile`, `fields: list[str] | None = Form(None)`) following the §13 check order. The pipeline is injected through `Depends(get_pipeline)` so tests can stub it.
  - Exception handlers that keep the `PipelineResult` shape, map statuses from `error.stage`, and never leak tracebacks.
  - CORS from `.env`, optional `SAVE_OUTPUTS`, and per-request logging (no OCR text in logs).
- **Expected result:** the API works from curl and from Swagger at `/docs`.
- **Checkpoint:**
  - `curl -F file=@tests/samples/invoice_clear.png -F "fields=Invoice Number,Customer Name,GST Number,Total" localhost:8000/api/extract` returns 200 with exactly those 4 fields, in order.
  - Field input: omitting `fields` returns the default set; `fields=[]` gives 400 `NO_FIELDS_SELECTED`; `fields=Total,###` gives 400 `INVALID_FIELDS`.
  - `GET /api/fields` lists 10 predefined fields.
  - File checks: a `.txt` upload gives 415, an empty file gives 400, and a renamed non-image `.jpg` gives 422.
  - With Ollama stopped: 200, `success: false`, `error.stage: "dspy"`, OCR present, and the server keeps running.
  - The full `pytest` suite passes.

t.

### Phase 6: HTML frontend ✅ (done 2026-10-06)
- **Objective:** Build the single-page UI on top of the Phase 5 API.
- **Files:** `frontend/index.html`, `frontend/style.css`, `frontend/app.js` (vanilla JS, no framework, no build), `backend/main.py` (serves `frontend/` at `/` with `StaticFiles`, mounted after the API routes), `backend/ocr_service.py` (dedicated PaddleOCR thread, see §9), `scripts/ui_check.py`, `tests/test_api.py`, `tests/test_ocr_text.py`, `tests/test_pipeline_integration.py`.
- **Implemented:**
  - **Header** with a live service status line (`GET /health?check_llm=true`).
  - **Image card:** file picker + drag-and-drop, client-side type/size checks, preview, name, size, remove.
  - **Field selector:** predefined checkboxes from `GET /api/fields` (defaults pre-checked, "list" badge), select all / clear all, custom fields (add with the button or Enter, remove, duplicate and predefined-name detection, §6 key rule, `max_fields` limit), and a "Selected fields: N" counter.
  - **"Extract Selected Fields" button:** disabled until there is an image and at least one field, so no useless request is sent.
  - **Request:** multipart `file` + `fields` (JSON array of exact keys, in display order), with an `AbortController` timeout of 210 s.
  - **Loading card:** message plus elapsed-seconds counter; all controls locked while running.
  - **Status banner** with icon + text (not colour alone) for every outcome:
    - success / no text found
    - AI failure that still shows the OCR result (`unavailable`, `timeout`, `parse`)
    - HTTP 400 / 413 / 415 / 422 / 500
    - network failure, timeout, malformed response
  - **Extracted fields:** one row per requested field in request order; lists as bullets; "Not found"; custom badge with the label the user typed; formatting-restoration notes; "Copy as JSON".
  - **OCR details (collapsible):** summary (lines, average confidence, image size, timings), warnings, raw OCR text, the reading-order text sent to the AI, and a lines table (text, confidence %, box).
  - **Bounding boxes** drawn on a `<canvas>` over the preview, scaled from original-image pixels to the displayed size (handles `object-fit` letterboxing and device pixel ratio, redraws on resize), with a toggle.
  - **Responsive** single column below 900 px; on narrow screens the outcome scrolls into view.
  - **Accessibility:** labels, fieldset/legend, `aria-live` status, `role="alert"` for errors, visible focus.
  - All server text is inserted with `textContent`.
- **Checkpoint (passed):**
  - `scripts/ui_check.py ui`: 40/40 checks (with Ollama stopped).
  - `scripts/ui_check.py real`: 9/9 checks; real PaddleOCR + Ollama extraction through the browser.
  - Mobile at 375 px: no horizontal scroll.
  - No JavaScript exceptions.
  - `pytest`: 110 passed.

### Phase 7: Frontend/backend integration ✅ (done 2026-10-06)
- **Objective:** Harden and verify Browser → FastAPI → preprocessing → PaddleOCR → DSPy → Ollama → result → UI.
- **Contract audit:** the backend accepted repeated `fields` parts, comma lists and JSON arrays. The page sent one JSON-array part. The single browser format is now **repeated `fields` parts carrying keys** (§13). The backend's comma/JSON forms remain only as curl conveniences.
- **Files:** `frontend/app.js` (`buildExtractForm`), `backend/main.py` (field documentation only), `tests/test_api.py` (+5), `tests/test_pipeline_integration.py` (+1), `scripts/ui_check.py` (new `integration` mode, request checks from Chrome's network events, real timeout path, 14-field round trip, box-alignment checks).
- **Checkpoint (passed):**
  - Every `tests/samples` image went through the UI in 6 sequential real extractions. Each click produced exactly one POST.
  - The `fields` parts on the wire, and the keys FastAPI logged, equalled the checked keys in order. The result table mirrored the selection, using the typed labels for custom fields.
  - No stale text, results or boxes between runs; controls restored after every run.
  - Boxes are drawn at the scaled positions and all cover ink. On the downscaled 4× image they map back to original coordinates within 3.5 px.
  - Real 400/413/415/422 bodies and the AI-off path are rendered safely. The real 210 s abort timer was exercised.
  - No PaddleOCR failures; uvicorn memory stable; no leftover processes.
  - `pytest`: 116 passed.

### Phase 8: Testing, evaluation, documentation and cleanup ✅ (done 2026-10-06)
- **Objective:** Final validation of the complete application; no new features.
- **Done:**
  - Environment and dependency check (`pip check` clean, `requirements.txt` fully satisfied, `websockets` declared explicitly).
  - Security and code-quality review: no hard-coded secrets, no shell or `eval`, no debug code or unused imports; the uploaded filename is now logged quoted (`%r`).
  - §20 gap closed: a real LLM-timeout test (T11b).
  - `scripts/evaluate.py` (§23) on all samples; API validation against the live server.
  - Browser checks: `ui` (now also drag and drop, Copy JSON, box toggle), `integration` and a new `smoke` mode.
  - Resource and process check; README, PLAN and EVALUATION finalized.
- **Checkpoint (passed):**
  - `pytest -m "not slow"`: 106 passed in about 4 s. `pytest`: 117 passed, 0 skipped.
  - `ui_check`: ui 57/57, integration 73/73, smoke 13/13 (real uncached AI: `INV-1001`, `29ABCDE1234F1Z5`, `619.50`).
  - Evaluation recorded in EVALUATION.md: 47/61 fields correct, 0 hallucinated.

### Phase 9: Documentation and cleanup (folded into Phase 8)
- Its tasks (README with architecture, setup, run, test and troubleshooting; pinned versions; dead-code check; no secrets) were completed during Phase 8.
- **Not performed:** the checkpoint "a fresh clone, following the README only, reaches a working UI". The project is not a git repository, and no fresh environment was built. The setup commands in README.md are the ones actually used in Phase 1.

## 20. Testing Strategy

**Layers**
- **Unit (fast, no models):** upload validation, **field-list parsing/validation (`parse_fields`)**, image decoding, OCR result normalization (fed with a dict captured from a real PaddleOCR run in Phase 2 and saved as a fixture; this is recorded real output, not invented), **extraction normalization against the requested list** (a pure function tested with plain dicts: ordering, missing → null, extra keys dropped, multi split, empty → None), error mapping.
- **Integration (`@pytest.mark.slow`, real PaddleOCR and real Ollama):** service and API tests against `tests/samples/`.
- **Manual:** browser walkthrough + `EVALUATION.md`.

No mock OCR engines and no fake LLM responses are used. Failure cases are triggered for real (for example, pointing `LLM_API_BASE` at a closed port).

| # | Case | Input | Expected |
|---|------|-------|----------|
| T1 | Clear invoice | printed invoice image, fields `[invoice_number, date, total]` | 200; avg conf > 0.9; invoice_number, date, total correct |
| T2 | Receipt | thermal receipt photo, fields `[vendor_name, total]` | 200; total and vendor extracted; many lines in order |
| T3 | Blurry image | blurred photo, default fields | 200; lower avg confidence; low-confidence warning; extraction partial/null fields, no crash |
| T4 | Empty image | plain white PNG | 200; `lines=[]`; warning "No text detected"; `requested_fields` echoed; `extraction=null`; LLM not called |
| T5 | Unsupported file | `.txt`, `.pdf` | 415 `UNSUPPORTED_FILE_TYPE` |
| T5b | Fake image | text file renamed `.jpg` | 422 `INVALID_IMAGE` |
| T5c | Empty upload | 0-byte file | 400 `EMPTY_FILE` |
| T5d | No file field | POST without `file` | 400 `NO_FILE` |
| T6 | Large image | 12 MB file; and a 6000×8000 photo under 10 MB | 413 `FILE_TOO_LARGE`; second → 200, downscaled, boxes in original coordinates |
| T7 | Multiple lines | multi-line document | line count matches roughly; `full_text` keeps top-to-bottom order; same-row items joined |
| T8 | ₹ currency | bill with ₹ amounts, fields `[subtotal, total]` + custom `currency` | totals are digits without symbol/OCR artefacts; custom `currency` → "INR" or "₹" |
| T9 | Missing fields | invoice without number/customer, fields `[invoice_number, customer_name, total]` | those fields `null` / `found: false`, not invented; total present |
| T10 | OCR failure | force via a corrupted-but-decodable edge case, or by calling `run_ocr` with an invalid array | 500 `OCR_FAILED`, clean JSON error |
| T11 | DSPy/LLM failure | Ollama stopped / bad `LLM_API_BASE` | 200 with OCR data + `extraction_error` |
| T11b | LLM timeout | `LLM_TIMEOUT_SECONDS=1` | 200 partial with timeout message |
| T12 | API failure (frontend) | backend stopped | UI shows "Cannot reach the server" |
| T13 | Health | `GET /health` (`?check_llm=true`) | `status: ok`, `ocr_loaded`; `llm_reachable` reflects Ollama state |
| T14 | Example text (DSPy only) | the requirements' sample text, fields `[invoice_number, date, customer_name, total]` | exact expected values (`INV-1001`, `06/10/2026`, `ABC Stores`, `590`) |
| **Dynamic Field Selection** | | | |
| T15 | Subset only | fields `[invoice_number, vendor_name, gst_number, total]` | `extraction.fields` has exactly these 4 keys, in this order, no others; `requested_fields` echoes them |
| T16 | Order preserved | fields `[total, invoice_number]` | response order is `total`, `invoice_number` |
| T17 | Single field | fields `[total]` | exactly one extracted entry |
| T18 | Custom field | fields `[invoice_number]` + custom `due_date` on an invoice that has a due date | `due_date` returned with `custom: true`, label "Due Date", correct value |
| T19 | Custom field absent | custom `payment_mode` on a document without it | `value: null`, `found: false` |
| T20 | Predefined field absent | `gst_number` on the example text | `null`, not invented |
| T21 | Multi-valued fields | fields `[product, quantity]` on a multi-item bill | both values are lists, in document order, the same length when quantities are printed |
| T22 | No fields | `fields=[]` | 400 `NO_FIELDS_SELECTED`; UI: Extract button disabled when nothing is checked |
| T23 | Fields omitted | no `fields` form field | 200 using the default-checked fields |
| T24 | Invalid fields value | `fields=not-json`, `fields={"a":1}`, `fields=["Bad Key!"]`, `fields=[1,2]` | 400 `INVALID_FIELDS` each; OCR not run |
| T25 | Too many fields | 21 keys | 400 `INVALID_FIELDS` ("max 20"); UI blocks the 21st |
| T26 | Duplicates | `["total","total","date"]` | deduped to `[total, date]`, 200 |
| T27 | LLM returns extra / missing keys | normalization unit test with a dict containing an unrequested key and lacking a requested one | extra key dropped; missing key → `null`; order follows the request |
| T28 | Field catalog | `GET /api/fields` | 10 predefined fields, the correct defaults, `max_fields` = 20 |
| T29 | Custom key conversion (frontend) | "Due Date", "PAN No.", "  ", "123abc", "Invoice Number" | `due_date`, `pan_no`, error, error, checks the predefined box |

## 21. Error Handling Strategy

| Situation | Where detected | Handling | User sees |
|-----------|---------------|----------|-----------|
| No fields selected | `fields.parse_fields` (empty list) | `FieldValidationError` → 400 `NO_FIELDS_SELECTED`, before OCR | "Select at least one field to extract." (the UI also prevents this) |
| Invalid field list | `fields.parse_fields` (bad JSON, non-strings, bad key, > `MAX_FIELDS`) | `FieldValidationError` → 400 `INVALID_FIELDS`, before OCR; the message names the offending key | "Invalid field name 'Bad Key!'. Use letters, digits and underscores." |
| LLM omits requested keys / adds unrequested keys | `dspy_service` normalization | missing → `null` (`found: false`); extra → dropped and logged at DEBUG; **not** an error | Requested fields only, "Not found" where empty |
| Invalid file (not an image) | `decode_image` (Pillow `UnidentifiedImageError`/`OSError`) | Raise `InvalidImageError` → 422 | "The file is not a valid image." |
| Unsupported format | `main.py` extension/MIME check | 415 before decoding | "Only JPG, PNG, BMP, WEBP are supported." |
| Empty upload | `main.py` after reading bytes | 400 | "The uploaded file is empty." |
| Too large | `main.py` read with limit (read up to `limit+1` bytes) | 413 | "File exceeds 10 MB." |
| Decompression bomb | Pillow `MAX_IMAGE_PIXELS` / `DecompressionBombError` | 422 | "Image dimensions too large." |
| OCR failure | `run_ocr` catches exceptions from `predict` | log full traceback; raise `OcrError` → 500 | "Text recognition failed. Try a clearer image." |
| OCR model not loaded | lifespan sets `ocr_ready=False` | 503 | "OCR engine is not ready, check server logs." |
| DSPy failure (parse error, adapter error) | `dspy_service.extract` | catch → `ExtractionError("Could not parse model output")`; **partial success 200** | OCR results + notice in the fields card |
| LLM API failure (Ollama down, connection refused, model not pulled) | LiteLLM exception inside DSPy | catch → `ExtractionError("LLM unavailable: …")`; partial 200 | same as above, with a hint "Is `ollama serve` running?" |
| Timeout | LLM: `timeout=180` passed to `dspy.LM`/LiteLLM, `num_retries=0`; frontend: `AbortController` 210 s | LLM: partial 200; browser: abort message | "AI extraction timed out" / "Request timed out" |
| Malformed LLM response | DSPy parsing + Pydantic validation, then our normalization (wrong types → None) | strip bad fields; if the whole parse fails → `ExtractionError` | partial results / notice |
| Malformed server response (frontend) | `res.json()` throws | catch in JS | "Unexpected server response (status N)." |
| Unexpected exception | global FastAPI exception handler | log; 500 `INTERNAL_ERROR`; no stack trace in the response | "Something went wrong." |

Principles: fail early on input; never lose OCR results because the LLM failed; stack traces go in server logs only; every error has the same JSON shape.

## 22. Security Considerations

- **API keys / secrets:** none needed for Ollama. `LLM_API_KEY` exists only in `.env`. `config.py` reads it, and it is never logged or returned. `.env.example` contains no real values.
- **.env / .gitignore:** `.env`, `.venv/`, `outputs/*` (keep `outputs/.gitkeep`), `__pycache__/`, `*.log`, `.pytest_cache/` are ignored. Before any commit, check `git status` to confirm `.env` is untracked.
- **File validation:** allow-list of extensions and MIME types, **plus** the authoritative Pillow decode. Never trust the client filename; it is used only for display and `outputs/` naming after sanitizing (`Path(name).name`, restricted characters).
- **File size limits:** `MAX_UPLOAD_MB` enforced server-side while reading; client-side check is for UX only; Pillow `MAX_IMAGE_PIXELS` against decompression bombs; downscale large images.
- **Temporary file cleanup:** none created (in-memory processing). If disk is ever used, `tempfile` + `try/finally` unlink.
- **CORS:** `CORSMiddleware` limited to the origins in `CORS_ALLOW_ORIGINS` (default: the app's own `localhost:8000` / `127.0.0.1:8000`), `GET`/`POST` only, no credentials, never `*`. Add a dev-server origin explicitly if needed.
- **Network exposure:** run Uvicorn on `127.0.0.1` only (not `0.0.0.0`); Ollama listens on localhost by default.
- **XSS:** OCR text and user-typed custom field labels are untrusted, so render them with `textContent` only.
- **Field-name validation:** custom field keys reach the LLM prompt, so the backend accepts only `^[a-z][a-z0-9_]{0,39}$` and at most `MAX_FIELDS` keys. This blocks long or free-form instructions being smuggled in through field names. The derived label comes from the validated key, never from raw user text.
- **Prompt injection:** OCR text could contain instructions ("ignore previous…"). Impact is limited because output is constrained to a dict that is filtered to the requested keys, and the LLM has no tools or side effects. Note this as a known limitation.
- **Logging:** do not log full OCR text or images at INFO level (documents may contain personal data). Log sizes, timings and counts.

## 23. Evaluation Strategy

Start manual and simple. Record everything in `EVALUATION.md`, one row per sample image, using `SAVE_OUTPUTS=true` JSON files as evidence.

**OCR**
- *Detected text:* for each sample, hand-write the ground-truth text (key lines only for long receipts). Count lines correct / partially correct / missed / spurious. Optional simple metric: character accuracy via `difflib.SequenceMatcher(None, truth, predicted).ratio()` per image.
- *Confidence:* compare `average_confidence` with actual correctness. Do low scores (< 0.7) line up with wrong lines? This checks whether `OCR_MIN_CONFIDENCE` is set sensibly.
- *Bounding boxes:* visual check using the UI overlay or `save_to_img` output: boxes cover their text, none are badly shifted (catches coordinate scaling bugs), and no big text block is left without a box.

**DSPy**
- *Field extraction accuracy:* for each sample, record the expected value per predefined field (ground truth, `null` if absent) once. Each evaluation run uses a stated field selection, and only the requested fields are scored. Mark ✔ exact / ≈ acceptable (formatting difference) / ✘ wrong. Report per-field accuracy = ✔ / times requested, across samples.
- *Selection robustness:* run the same image with 3 different selections (e.g. 2 fields, the 4-field example, all 10 fields). The value of a field should not change depending on which other fields were requested. Note any field whose accuracy drops when many fields are selected; this tells us whether batching is needed (§15 fallback).
- *Scope adherence:* from `dspy.inspect_history`, count how often the raw LLM output included unrequested keys or omitted requested ones. The response is always correct because normalization enforces it, but this measures how well the Signature instructions work.
- *Custom fields:* evaluate 2–3 custom fields (e.g. `due_date`, `payment_mode`, `bank_account`) on the samples that contain them and on samples that don't.
- *Missing field handling:* for fields absent from the document, ✔ if `null`, ✘ if a value was invented (a hallucination). Track the hallucination count separately; it matters more than misses.
- *Incorrect extraction:* for each ✘, note the cause, either **OCR error** (the text was already wrong, so it is not DSPy's fault) or **LLM error** (the text was right, extraction wrong). This keeps the two components' responsibilities separate.
- Keep the sample set (about 8–10 images + truth) fixed. This is the dev set that future DSPy optimization (`dspy.Evaluate` + metrics) will reuse.

Not in v1: automated metrics pipelines, DSPy `Evaluate`, optimizers.

## 24. Expected Final Output

At `http://localhost:8000` the user sees:
1. A header "Simple OCR: Invoice & Receipt Reader" and a small status pill ("OCR ready · LLM: qwen2.5:3b ✓", or a yellow "LLM offline").
2. A drop zone / "Choose image" button. After choosing, a preview of the image with its filename and size.
3. A **Fields to extract** card below the preview:
   - Checkboxes for Invoice Number ☑, Vendor Name ☑, Customer Name ☐, Date ☑, GST Number ☐, Phone Number ☐, Product ☐, Quantity ☐, Subtotal ☐, Total ☑ (defaults), with "Select all · Clear" links.
   - An "Add custom field" input + **Add** button. Added fields (e.g. "Due Date") appear checked with a *custom* badge and an × to remove them.
   - A counter "4 of 20 fields selected". **Extract Text** is enabled only when an image is chosen and at least one field is checked.
4. While processing: a disabled button reading "Extracting…", a spinner, and "Running OCR and AI extraction… 12 s". The field controls are locked.
5. On success, the right-hand panel shows:
   - **Extracted Fields**: one row per selected field, in the order selected. For the example selection Invoice Number, Vendor Name, GST Number, Total on the example invoice: Invoice Number **INV-1001** · Vendor Name *Not found* · GST Number *Not found* · Total **590**. Multi-valued fields such as Product show a bullet list (Milk · Bread · Rice). Custom fields show a *custom* badge. Unselected fields do **not** appear.
   - **Summary**: 8 lines · avg confidence 96.2% · 4 fields requested · 2 found · OCR 3.1 s · AI 9.5 s.
   - **Raw OCR Text**: monospaced block with a Copy button.
   - **Detected Lines**: table of text + colour-coded confidence %.
   - **Show boxes** toggle: green polygons drawn over the preview image around each detected line.
6. On error: a red banner such as "Only JPG, PNG, BMP, WEBP are supported (415)" or "Invalid field name … (400)". If only the LLM failed: OCR sections are shown normally and the Extracted Fields card lists the requested fields with "AI extraction unavailable: LLM unavailable, is Ollama running?"

## 25. Commands to Run

```bash
# --- Setup (once) ---
cd /home/finstein-emp/Documents/ocr
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r requirements.txt
cp .env.example .env
~/.local/ollama/bin/ollama serve &   # user-level Ollama install (Phase 1)
~/.local/ollama/bin/ollama pull qwen2.5:3b

# --- Run (API + web page on one port) ---
uvicorn backend.main:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000/   (API docs: /docs, /redoc)

# --- API ---
curl -s "http://127.0.0.1:8000/health?check_llm=true"
curl -s http://127.0.0.1:8000/api/fields
curl -s -F "file=@tests/samples/invoice_clear.png" \
     -F "fields=invoice_number" -F "fields=gst_number" -F "fields=total" \
     http://127.0.0.1:8000/api/extract | python -m json.tool

# --- Tests and checks ---
pytest -m "not slow"                       # 106 fast tests
pytest                                     # 117 tests (needs models + Ollama)
python scripts/ui_check.py ui              # browser checks, Ollama stopped (server running)
python scripts/ui_check.py smoke           # final smoke test, Ollama running
python scripts/ui_check.py integration     # sequential real extractions, Ollama running
python scripts/evaluate.py                 # sample evaluation, Ollama running, server stopped

# --- Stop ---
pkill -x ollama                            # and Ctrl+C for uvicorn
```

## 26. Future Improvements (NOT part of v1)

Found during Phases 7–8 (documented, deliberately not implemented):
- **Skew-aware row grouping:** estimate the page angle from the OCR polygons and group rows on de-rotated y. This fixes the shifted amount column on rotated images (blurred sample: Subtotal read as 14.75).
- **Split multi-valued answers on newlines too:** the model sometimes returns list fields newline-separated instead of `|`, so the UI shows one bullet.
- **Better list extraction (Product, Quantity):** qwen2.5:3b often returns only the first item. Options: a dedicated line-item step or table recognition (below).
- **Field-order sensitivity:** with `total` requested first, the 3B model once returned `null` for it. Few-shot examples or DSPy optimization on the evaluation set could stabilize this.
- **Server-side cancellation:** the browser gives up after 210 s, but the server finishes its work (up to the 180 s AI timeout).
- **Real-world dataset:** add real phone photos and scans with ground truth to `tests/samples/`.

Original ideas:

- **Preprocessing:** deskew, denoise, contrast/CLAHE, adaptive thresholding for thermal receipts; enable PaddleOCR doc orientation and unwarping for skewed phone photos.
- **Better OCR:** larger/server PP-OCR models, multilingual (Hindi + English) models, GPU inference, PaddleOCR-VL / PP-StructureV3 for layout.
- **Document classification:** a first DSPy step that routes to invoice / receipt / ID / generic signatures.
- **Table extraction:** PP-StructureV3 table recognition for line items instead of LLM inference.
- **Multiple document types:** a Signature per type; PDF and multi-page support.
- **OCR evaluation metrics:** CER/WER against labelled ground truth, box IoU.
- **DSPy optimization:** `dspy.Evaluate` with a field-level metric on the dev set; `BootstrapFewShot` / `MIPROv2` / `GEPA`; `ChainOfThought`; compare local versus hosted models.
- **Post-validation:** numeric parsing, check that line items sum to the total, date normalization with locale hints.
- **Persistence:** database of results and history.
- **Authentication** and multi-user support.
- **Deployment:** Docker image, reverse proxy, cloud hosting, background job queue for long documents.

## Decisions Needed Before / During Implementation

| Decision | Default in this plan | When |
|----------|---------------------|------|
| LLM provider | **Ollama (decided)** | Done |
| Ollama model | `qwen2.5:3b` (alt: `llama3.2:3b`; `qwen2.5:7b` if RAM allows, which is tight at 7.5 GB) | Phase 1/3 |
| OCR det/rec models | mobile variants | Phase 2 |
| OCR language | `en` (consider multilingual if Hindi text appears) | Phase 2 |
| Field selection model | **User-selected fields (decided)**: predefined catalog + custom fields, one fixed Signature with fields as input | Done |
| Signature output format | **`extracted_values: dict[str, str]` (decided, Phase 3)** | Done |
| Timeouts | **LLM 180 s, end-to-end request target 210 s (decided)** | Done |
| Large images | **Downscale to `OCR_MAX_SIDE`=2000 px before OCR (decided, Phase 2)** | Done |
| OCR model | **PP-OCRv6_small (decided, Phase 2)**; PP-OCRv5 mobile rejected | Done |
| Default-checked fields | Invoice Number, Vendor Name, Date, Total | Phase 4 (easy to change in `fields.py`) |
| `MAX_FIELDS` | 20 (lower if Phase 3 latency at 10+ fields is too high) | Phase 3/4 |
| Behaviour when `fields` is omitted | use the defaults (API convenience) | Phase 5 |
| Save outputs to disk | off by default, on during evaluation | Phase 5/8 |
| Git repo | `git init` in Phase 1 (recommended) | Phase 1 |
