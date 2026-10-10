# Evaluation

- **Final Evaluation** (Phase 8, 2026-10-06): the summary of the finished project.
- **Appendix:** the per-phase logs (Phases 2–7), with the detailed measurements behind it.

---

# Final Evaluation (Phase 8)

**Verdict: PASS** for the intended scope: a local, CPU-only, one-page OCR + field-extraction demo. This is **not** a claim of production-grade OCR or extraction accuracy (see Known limitations).

## Environment

| Component | Version |
|-----------|---------|
| OS / hardware | Linux, 4-core CPU, 7.5 GB RAM, 4 GB swap, no GPU |
| Python / pip | 3.12.3 / 26.2.1 (`.venv`) |
| PaddlePaddle / PaddleOCR / PaddleX | 3.2.0 (CPU) / 3.7.0 / 3.7.2; models PP-OCRv6_small det + rec, PP-LCNet text-line orientation |
| DSPy / LiteLLM | 3.3.1 / 1.103.3 |
| FastAPI / Uvicorn / python-multipart / Pillow | 0.142.2 / 0.54.0 / 0.32 / 12.1.0 |
| Ollama / model | 0.35.1 (user-level install) / qwen2.5:3b (1.9 GB on disk, about 2.2 GB loaded) |
| Dependencies | `pip check`: no broken requirements. `requirements.txt` fully satisfied (`websockets` now declared explicitly) |

## Architecture (as built)

```
Browser (frontend/: HTML + CSS + vanilla JS, served by FastAPI at /)
  → POST /api/extract (multipart: file + one `fields` part per key)
  → FastAPI (backend/main.py: request checks, status mapping, one response shape)
  → preprocessing (validate, EXIF, downscale ≤ 2000 px)
  → PaddleOCR (one engine on one dedicated thread; plain dicts; boxes in original pixels; row grouping)
  → DSPy dspy.Predict(FieldExtractionSignature) → Ollama qwen2.5:3b (180 s timeout)
  → normalization: only the requested fields, in order; missing → null; OCR formatting restored for amounts
  → PipelineResult JSON → UI: fields table, raw OCR, AI text, confidence table, box overlay
```

## Functional results (all verified)

- **Upload:** file picker, drag and drop, preview, name, size, remove; client-side type/size checks.
- **Fields:** 10 predefined fields loaded from `GET /api/fields`; custom fields with typed labels; select all / clear all; duplicate and predefined-name detection; counter; Extract disabled with 0 fields or no image.
- **Request:** one canonical request per click (wire = FastAPI = checked keys, in order).
- **Results:** table mirrors the selection; "Not found"; Copy as JSON; loading state with elapsed counter, controls locked and then restored.
- **OCR details:** raw text, AI text, confidence per line, boxes drawn and aligned (toggle).
- **Partial success:** the OCR result is kept when the AI fails (Ollama down, or a real LLM timeout).
- **Endpoints:** `GET /health`, `GET /health?check_llm=true`, `GET /api/fields`, `POST /api/extract`, `/docs`, `/redoc`, `/openapi.json`. All return 200 against the live server.

## Test results

| Suite | Result |
|-------|--------|
| `pytest -m "not slow"` | **106 passed**, 11 deselected, 0 failed, 0 skipped, **4.0 s** |
| `pytest` (real PaddleOCR + Ollama for slow tests) | **117 passed**, 0 failed, 0 skipped, **73.1 s** (with cached AI answers) |
| `scripts/ui_check.py ui` (Ollama stopped) | **57/57** |
| `scripts/ui_check.py integration` (Ollama running) | **73/73** (6 sequential extractions over all 5 samples) |
| `scripts/ui_check.py smoke` (Ollama running, **DSPy cache off**) | **13/13** |

New in Phase 8:
- **Real LLM-timeout test** (`test_real_llm_timeout_keeps_ocr`, PLAN.md §20 T11b): the real litellm/DSPy timeout is classified as `kind: "timeout"` and the OCR result is kept.
- **Browser checks for:** drag and drop, Copy as JSON (exact payload), the bounding-box toggle (overlay cleared, then redrawn) and the OCR detail values.

## Final smoke test (browser → FastAPI → PaddleOCR → DSPy → Ollama → UI, no cache)

`invoice_clear.png` with Invoice Number, GST Number and Total, in headless Chrome at `http://127.0.0.1:8000/`:

| Field | Result |
|-------|--------|
| Invoice Number | **INV-1001** |
| GST Number | **29ABCDE1234F1Z5** |
| Total | **619.50** |

- **Banner:** "Extraction complete, 3 of 3 requested fields found".
- **Request:** one POST, fields `invoice_number, gst_number, total` in order.
- **OCR:** 29 lines, all boxes aligned.
- **Errors:** none in the browser console.
- **Time:** **34.5 s** in the browser (preprocess 0.09 s + OCR 8.4 s + AI 25.8 s).

## OCR results (`scripts/evaluate.py`, against the `.truth.txt` files)

| Image | Lines | Char recall* | Token recall | Confidence min–avg | ₹ found | Key tokens† | Boxes on ink |
|-------|-------|--------------|--------------|--------------------|---------|-------------|--------------|
| invoice_clear.png | 29 | 99.6% | 94.4% | 0.921–0.984 | 6/7 | all | 29/29 |
| receipt_mono.png | 21 | 100% | 100% | 0.943–0.985 | 2/2 | all | 21/21 |
| invoice_large.jpg (4960×5200 → 1908×2000) | 29 | 100% | 100% | 0.914–0.979 | 7/7 | all | 29/29 |
| invoice_blurry.jpg (blur + 1.5° rotation) | 29 | 97.5% | 50.0% | 0.970–0.995 | **0/7** | all | 29/29 |
| blank.png | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

\*Whitespace ignored. †`INV-1001`, `29ABCDE1234F1Z5`, `06/10/2026`, `619.50`, `590.00` (invoices) and `FRESH MART SUPERMARKET`, `58213`, `2026-09-28`, `413.00` (receipt) appear verbatim.

- **Lines vs truth:** "Lines" are detected segments. The truth files have 17/13 logical rows; table columns are detected separately and regrouped into rows before the AI.
- **Digits:** invoice numbers, GSTIN, dates, decimal values and totals were read correctly on every sample.
- **₹ symbol:** missed once on the bold total of the clear image, and lost entirely on the blurred image. Blur also merges words (`SHREEGANESH`, `Phone:+919845012345`) and token recall drops to 50%, while **confidence stays at about 0.99**, so confidence does not flag these errors.
- **Boxes:** on the downscaled large image, boxes ÷ 4 match the clear image's within **3.5 px**; this is also a slow pytest. All boxes are drawn at the scaled positions in the browser.

## Field extraction results (61 scored fields, real uncached AI calls)

Fields: the 10 predefined fields plus custom Address, Bill No, Payment Reference, CGST Amount and Purchase Order Number. Fields with an ambiguous answer were not requested on that document.

| Result | Count | Examples |
|--------|-------|----------|
| **Correct** | **47** (77%) | invoice number, date, GSTIN, phone, total on every readable invoice; receipt bill no (custom) `58213`, phone `044-2615 7788`, total `413.00`; absent fields (customer on the receipt, Payment Reference, PO number) correctly `null`; large image CGST `₹14.75` (correct value with a currency symbol) |
| **Missing** | **3** | Address on the clear invoice and the receipt; CGST Amount on the clear invoice |
| **Incorrect** | **11** | Product/Quantity returned only the first item (6×); receipt Product as one newline-joined item; receipt Subtotal = the item amounts (no subtotal printed); blurred invoice: Vendor `SHREEGANESH TRADERS` and Address `12,MG Road` (OCR merged words), **Subtotal `14.75`** (row grouping, below) |
| **Hallucinated** | **0** | Every value returned exists in the OCR text |

- Missing is preferred to invented, and that held: there were no hallucinations.
- The prompt was not tuned for this evaluation.
- Every wrong answer is traceable to the OCR text or to list handling.
- API check: with fields sent in the order `total, invoice_number, gst_number, cgst_amount`, the model returned `total: null` (it finds `619.50` in other orders). The small model is sensitive to the field combination and order.

## Performance

| Step | Measured |
|------|----------|
| OCR engine init (models cached) | 1.5–5.2 s (server startup) |
| OCR per image | 0.3–1.6 s blank; 1.4–8.4 s invoice/receipt; 8.2–12.4 s large image (downscale included) |
| AI (qwen2.5:3b on CPU) | 13.9–25.8 s for 3–5 fields with the model warm; 40–100 s for 9–10 fields or when cold or swapping; cache hit about 3–11 ms |
| End-to-end request | 34.5 s (smoke test, 3 fields, uncached); 15–60 s typical; up to about 100 s observed under memory pressure |

Measurements come from Phase 8's `evaluate.py`, the browser runs and the API checks (ranges include Phase 7). AI time is dominated by free RAM: with about 1.3 GB available the OS swaps, and the same request slows several-fold.

## Resource usage

- **Processes:** one uvicorn worker and one `ollama serve` (the model runner loads on demand and unloads when idle).
- **PaddleOCR:** one engine on one dedicated thread. There were **0 PaddleOCR failures** in all Phase 7–8 runs, and the Phase 6 thread bug is covered by regression tests.
- **uvicorn footprint (RSS + swap) over 6 sequential browser extractions:** 853 → 945 → 978 → 1083 → 1102 → 1176 → 1079 MB in Phase 8 (Phase 7: 1184–1282 MB). It settles at about 1.1–1.3 GB, with no leak trend.
- **Peaks:** the pytest process peaks at about 2.0 GB (models plus the large image); `evaluate.py` at about 1.15 GB.
- **After testing:** 0 uvicorn, 0 Ollama, 0 headless Chrome and 0 test processes were running.

## Error handling (verified against the live server and in the browser)

| Situation | Result |
|-----------|--------|
| No file / empty file / invalid or empty field list | 400 (`NO_FILE` / `EMPTY_FILE` / `INVALID_FIELDS` / `NO_FIELDS_SELECTED`) |
| 11 MB upload | 413 `FILE_TOO_LARGE` |
| `.txt`; GIF disguised as `.png` | 415 (`UNSUPPORTED_FILE_TYPE` by name; `unsupported_format` by content) |
| Corrupt image (`.jpg` with text) | 422 `invalid_image` |
| OCR failure / unexpected error | 500 with a generic message (tested with stubs) |
| Ollama unavailable | 200, `success: false`, `error.stage = "dspy"`, OCR kept; the UI explains it |
| Real LLM timeout | `kind: "timeout"`, OCR kept |
| Network error, browser timeout (real 210 s timer, fast-forwarded), malformed response | friendly messages, controls restored, retry works |

No traceback or internal exception text reached a response in any test.

## Security review (lightweight)

- **Secrets:** no API keys in code. `LLM_API_KEY` is empty in `.env` and `.env.example`; `.env` is git-ignored, along with `.venv/` and `outputs/`.
- **Uploads:**
  - read into memory only; never written to disk (except the optional `SAVE_OUTPUTS` JSON, with a sanitized name) and never executed
  - type checked by name/MIME **and** by decoding
  - 10 MB limit (reads at most 10 MB + 1 byte)
  - 40 MP pixel limit before decode
  - corrupt images rejected safely
- **No code execution:** no shell commands, `subprocess`, `eval` or `exec` in the backend or frontend.
- **Fixed in Phase 8:** the uploaded filename was logged raw; it is now logged quoted (`%r`), so a filename with newlines cannot forge log lines.
- **Frontend:** OCR text is inserted with `textContent` only (no XSS). Custom field names are restricted to `^[a-z][a-z0-9_]{0,39}$`. CORS is limited to the configured origins, with no credentials.
- **Not in scope:** authentication or authorization; the server binds to `127.0.0.1` only.

## Code quality and repository hygiene

- No debug prints, TODOs, commented-out code or unused imports in `backend/`, `frontend/`, `scripts/` or `tests/`.
- Phase 8 code changes: `requirements.txt` (`websockets` declared), `backend/main.py` (log formatting), the new `scripts/evaluate.py` and one new test. Plus `scripts/ui_check.py` checks and docs.
- **Not a git repository:** no repository was created, per instructions. `.gitignore` would exclude `.env`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `outputs/*` (screenshots, logs, evaluation JSON) and `*.log`.

## Known limitations

1. **Rotated or blurred images and row grouping:** the minimal row grouping misaligns amount columns on rotated pages (blurred sample: Subtotal `14.75` instead of `590.00`). The AI faithfully reads the misaligned text.
2. **qwen2.5:3b:** misses some fields (Address, CGST Amount), returns only the first item of list fields, is sensitive to field order and combination, and once ignored the "no currency symbol" instruction (`₹14.75`). It invented no values in the evaluation.
3. **OCR on blur:** loses `₹`, merges words, and confidence does not reflect it.
4. **Timeouts:** browser about 210 s; server-side AI 180 s. The server finishes its work after the browser gives up.
5. **CPU and RAM:** performance depends on free RAM; latency rises several-fold when swapping.
6. **Evaluation data:** synthetic images only (5 samples).
7. **Browsers:** validation used headless Chrome only (plus an emulated mobile viewport).

Future improvements are listed in PLAN.md §26 (skew-aware grouping, newline-split lists, better line-item extraction, field-order stability, server-side cancellation, a real-photo dataset).

---

# Appendix: per-phase evaluation log

Manual evaluation record (see PLAN.md §23). One section per phase.

## Phase 2: PaddleOCR standalone (2026-10-06)

**Environment:** Python 3.12.3, paddlepaddle 3.2.0 (CPU), paddleocr 3.7.0 (paddlex 3.7.2), 4-core CPU, 7.5 GB RAM, no GPU.
**Script:** `scripts/try_ocr.py`. **Samples:** `scripts/make_samples.py` → `tests/samples/` (synthetic invoice-like images rendered with DejaVu fonts, each with a `.truth.txt`).
**Pipeline settings:** `lang="en"`, `device="cpu"`, `use_doc_orientation_classify=False`, `use_doc_unwarping=False`, `use_textline_orientation=True`.

### Samples

| Image | Size | Content |
|-------|------|---------|
| `invoice_clear.png` | 1240×1300 | Tax invoice: vendor, phone, GSTIN, invoice no., date, customer, 3 items in Qty/Amount columns, subtotal, CGST/SGST, total. 7 × ₹ |
| `receipt_mono.png` | 520×678 | Monospace supermarket receipt; `Rs.` amounts and 2 × `₹` |
| `invoice_blurry.jpg` | 1274×1334 | The invoice rotated 1.5°, Gaussian blur 2.2, JPEG q55 |
| `invoice_large.jpg` | 4960×5200 | The invoice upscaled 4× |
| `blank.png` | 800×600 | Plain white |

### Results by model

*char recall* = share of ground-truth characters present in the OCR output, ignoring whitespace. *token recall* = share of ground-truth words found exactly; merged words like `Phone:+91` count as misses. Times are per image, steady state. The first image in a run also pays about 2–3 s of warm-up.

| Model (det + rec) | Image | Lines | Avg conf | Char recall | Token recall | ₹ found | OCR time |
|---|---|---|---|---|---|---|---|
| **PP-OCRv6_medium** (PaddleOCR default for `en`) | invoice_clear | 29 | 0.981 | 100% | 100% | 7/7 | 15.6–16.5 s |
| | receipt_mono | 21 | 0.987 | 100% | 100% | 2/2 | 6.9–9.9 s |
| | invoice_blurry | 29 | 0.995 | 97.5% | 77.8% | 0/7 | 15.4–16.3 s |
| | blank | 0 | n/a | n/a | n/a | n/a | 1.9–3.9 s |
| | invoice_large | n/a | n/a | n/a | n/a | n/a | **killed: out of memory (4.27 GB RSS)** |
| **PP-OCRv6_small** ✅ chosen | invoice_clear | 29 | 0.984 | 99.6% | 94.4% | 6/7 | 4.1–4.4 s |
| | receipt_mono | 21 | 0.985 | 100% | 100% | 2/2 | 2.1–2.4 s |
| | invoice_blurry | 29 | 0.995 | 97.5% | 50.0% | 0/7 | 5.8 s |
| | blank | 0 | n/a | n/a | n/a | n/a | 0.4 s |
| | invoice_large | 29 | 0.983 | 99.6% | 98.1% | 6/7 | 17.5 s (2.8 GB RSS) |
| PP-OCRv5_mobile + en_PP-OCRv5_mobile_rec | invoice_clear | 31 | 0.992 | 97.5% | 87.0% | **0/7** | 3.6 s |
| | receipt_mono | 41 | 0.994 | 99.2% | 96.0% | **0/2** | 2.7 s |
| | invoice_blurry | 29 | 0.977 | 97.5% | 72.2% | 0/7 | 2.9 s |
| | blank | 0 | n/a | n/a | n/a | n/a | 0.3 s |

**Init time** (models already cached): 1.5–2.2 s for v6, about 9.5 s for v5 mobile. The first run also downloads the models: about 15–18 s, ~140 MB for `medium`.
**Peak process memory** for the 4 normal-size samples: medium about 1.9–2.0 GB, small about 1.05–1.1 GB, v5 mobile about 1.05 GB.

### Findings

1. **OCR works on CPU with the 3.x API.** `PaddleOCR(...).predict(path)[0]` returns an `OCRResult`. Values are read with `res["rec_texts"]` (list[str]), `res["rec_scores"]` (list[float]), `res["rec_polys"]` (list of 4×2 int numpy arrays) and `res["rec_boxes"]` (N×4 int numpy array, `[x_min, y_min, x_max, y_max]`). The same data is in `res.json["res"]`. Converting to plain `int`/`float`/`str` gives JSON-serializable data (verified with `json.dumps`).
2. **₹ is recognized on clean images** by PP-OCRv6: medium 9/9, small 8/9 (small dropped the ₹ on the bold, larger `₹619.50` total). Digits were always correct. Spaced `₹ 413.00` on the receipt was read correctly by both.
3. **PP-OCRv5 English mobile is unsafe for ₹:** it never outputs ₹, and once misread `₹14.75` as **`314.75`**, a wrong amount. Rejected.
4. **Blur loses ₹ completely and merges words** (`Phone:+919845012345`) with every model, while confidence stays at about 0.99–1.00. **Confidence does not flag these errors**, so the `OCR_MIN_CONFIDENCE` filter will not catch them. Digits survived (char recall 97.5%), so amount *values* are still usable.
5. **Large images are dangerous on this machine.** PaddleOCR only shrinks inputs to a 4000 px long side, which used 4.27 GB and was killed (medium) or 2.8 GB (small). **Phase 4 should downscale to about a 2000 px long side** before OCR. PLAN.md currently says 4000; this needs revising.
6. **Reading order is detection order, not row order.** Columns come out as separate lines (`Milk 1L`, `2`, `₹50.00`), and on the rotated image the order jumps (`Qty`, `Amount`, `Item`; amounts before item names). This confirms that the planned row-grouping step (PLAN.md §14) is needed before text goes to DSPy.
7. **Bounding boxes are accurate.** In the visual check (`outputs/phase2/*/invoice_clear_ocr_res_img.png`), boxes are tight and correctly placed, including the right-aligned amounts. (PaddleX's preview font has no ₹ glyph and draws `□`; the recognized text does contain `₹`.)
8. **Noise:** Paddle prints many `ReduceMeanCheckIfOneDNNSupport` lines, plus a harmless `UserWarning` saying `lang`/`ocr_version` are ignored when model names are given. The first use of `save_to_img` downloads a font (`simfang.ttf`).

### Decision

`OCR_DET_MODEL=PP-OCRv6_small_det`, `OCR_REC_MODEL=PP-OCRv6_small_rec` (in `.env`).
- **Why:** character accuracy matches medium on these samples (99.6% vs 100% clear; 97.5% vs 97.5% blurred), it is 3–4× faster on this CPU, and it needs about half the memory. That matters because qwen2.5:3b needs about 2.2 GB in the same 7.5 GB.
- **Fallback:** switch to `PP-OCRv6_medium_*` in `.env` if real-world invoices show accuracy problems.

### Limitations / to do

- The samples are synthetic renders. Add real phone photos and scans (thermal receipts, skewed shots) to `tests/samples/` with `.truth.txt` files and re-run `scripts/try_ocr.py`.

## Phase 3: DSPy + Ollama standalone (2026-10-06)

**Environment:** dspy 3.3.1 (pinned; installed API matches the plan: `dspy.LM`, `dspy.configure`, `dspy.Predict`, default `ChatAdapter` with automatic `JSONAdapter` fallback), litellm 1.103.3, Ollama 0.35.1 (user-level install, `ollama serve` started manually), model `qwen2.5:3b` (Q4, 2.16 GB resident, 100% CPU, 4096-token context). No cloud API used.
**Script:** `scripts/test_dspy.py` (`--runs N`, `--no-scaling`, `--error-test`, `--show-prompt`). Results: `outputs/phase3_dspy_results.json`.
**LM settings:** `temperature=0`, `max_tokens=1000`, `cache=False` (so every run really reaches the model), `num_retries=1`, `timeout=LLM_TIMEOUT_SECONDS` (90).

**Mechanism:** `dspy.Predict(FieldExtractionSignature)`, the field-agnostic Signature from PLAN.md §15:
- inputs: `ocr_text: str`, `requested_fields: dict[str, str]` (key → hint)
- output: `extracted_values: dict[str, str]`

The result is then normalized in Python against the requested list: request order kept, missing keys → `null`, unrequested keys dropped, empty / "N/A" → `null`, multi-valued fields split on `|`.

### Results (final run: 8/8 passed)

| Case | Requested | Runs | Result | Time per run |
|------|-----------|------|--------|--------------|
| A: full invoice | invoice_number, date, vendor_name, customer_name, gst_number, phone, subtotal, total | 3 | 3/3 PASS, identical output each run | 22.6 s / 16.2 s / 16.4 s |
| B: missing fields | invoice_number, vendor_name, customer_name, gst_number, total | 3 | 3/3 PASS; customer_name & gst_number → `""` → `null` | 16.1 s / 7.8 s / 6.9 s |
| C: 1 field | total | 1 | PASS (`724.50`) | 15.5 s |
| D: 10 fields | A's 8 + product, quantity | 1 | PASS; `product` → `["Basmati Rice 5kg","Milk 1L"]`, `quantity` → `["1","2"]` | **87.1 s** |

Case A normalized output:
```json
{"invoice_number": "INV-1001", "date": "06/10/2026", "vendor_name": "ABC Supermarket", "customer_name": "Ragul Kumar",
 "gst_number": "29ABCDE1234F1Z5", "phone": "+919845012345", "subtotal": "690", "total": "724.5"}
```
Case B normalized output:
```json
{"invoice_number": "INV-2002", "vendor_name": "XYZ Store", "customer_name": null, "gst_number": null, "total": "450"}
```

- **Requested fields present:** every requested key appeared in every response. No `missing` keys.
- **Hallucinated or unrelated fields:** none. No extra keys appeared, and absent fields came back empty rather than invented (case B, 3/3).
- **Invoice number, GSTIN, phone, date:** copied exactly in every run.
- **₹ values:** the model always removed `₹` as instructed, and the **numeric value was always correct**. However, trailing zeros are not kept: `₹690.00` → `690`, `₹724.50` → `724.5`, `₹450.00` → `450`. In case C (1 field) it returned `724.50`.
- **JSON conversion:** `prediction.extracted_values` is a plain `dict[str, str]`; `json.dumps(..., ensure_ascii=False)` succeeded for every result.
- **Error path (LLM unreachable):** fails immediately with `dspy.utils.exceptions.LMTransportError` ← `litellm.exceptions.APIConnectionError` ← `httpx.ConnectError` ("Connection refused"). Phase 5 should map `LMTransportError` to the partial-success `extraction_error`.

### Timing and memory

- **Model load:** about 4.6 s (cold, `/api/generate` warm-up); 0 s while the model stays loaded (`keep_alive`).
- **Prompt size:** about 480–650 tokens per request, most of it DSPy's adapter instructions and field descriptions. Generation is about 4.5 tokens/s.
- **Latency:** 7–23 s for 1–8 fields. Repeat calls are faster because Ollama reuses the shared prompt prefix (KV cache). **10 fields took 87 s, only 3 s under the 90 s timeout.**
- **Memory:** the Ollama model is resident at 2.16 GB (`ollama ps`); the `ollama serve` process itself is about 34 MB; the DSPy script peaks at about 260 MB. System "available" memory dropped from about 3.5 GB to about 1.3 GB while the model was loaded and returned to 3.5 GB after `ollama stop`. Running PaddleOCR (about 1.1 GB with `small`) at the same time would leave very little headroom.

### Issues found and fixed

1. **The output field name `values` broke parsing (fixed by renaming it to `extracted_values`).**
   - **Python side:** a DSPy `Prediction` is dict-like, so `prediction.values` returns the dict *method*, not the field.
   - **Model side:** with the field named `values`, qwen2.5:3b began its reply by echoing the input sections (`[[ ## ocr_text ## ]] …`, about 275–400 output tokens). `ChatAdapter` could not parse that, and DSPy fell back to a second `JSONAdapter` call. On the first, cold run this ran into the 90 s timeout, and one extraction took about 180 s in total.
   - **Proof:** re-running with the old name reproduced it (2 LM calls, 60 s). With `extracted_values`: 1 call per extraction, 10 of 10 calls clean.
   - **PLAN.md §15 needs updating:** use `extracted_values`.

### Experiments (not adopted yet)

- **Keep decimals as written.** Instruction tried: "remove the currency symbol but keep every digit and decimal exactly as written (e.g. '₹724.50' -> '724.50')". Result: `total` became `724.50`, but `subtotal` stayed `690`, so this is only a partial fix. Options for Phase 4: adopt this wording, and/or normalize amount strings in Python (for example, re-attach the decimals found in the OCR text). The numeric value was correct in every case.

### Checkpoint status

- ✅ The example text is extracted correctly in 3 of 3 runs at temperature 0.
- ✅ Every requested key is returned, there are no extra keys, and missing fields come back empty, not hallucinated.
- ✅ The `dict[str, str]` output parses reliably with `ChatAdapter` once the field is named `extracted_values`. The §15 string fallback is **not needed**.
- ✅ The "LLM down" exception type is known: `LMTransportError`.
- ⚠️ 10-field latency is 87 s against a 90 s timeout. Before Phase 5, raise `LLM_TIMEOUT_SECONDS` (to 150–180 s), lower `MAX_FIELDS`, or warn in the UI when many fields are selected. The 120 s browser timeout in PLAN.md §16 is also too short for 10 fields plus OCR.

## Phase 4: Combined pipeline (2026-10-06)

**Pipeline:** image bytes → `image_preprocessing.preprocess_image` (validate, EXIF fix, RGB, downscale if the long side > `OCR_MAX_SIDE`=2000) → `ocr_service.run_ocr` (PP-OCRv6_small, CPU, one shared instance + lock; plain dicts; boxes in original-image pixels; light text normalization; minimal row grouping) → `dspy_service.Extractor.extract` (`dspy.Predict(FieldExtractionSignature)` with `extracted_values`, via `dspy.context`, Ollama qwen2.5:3b, `timeout=180`, `num_retries=0`) → normalization against the requested fields + amount-format restoration → `PipelineResult` (Pydantic, JSON-serializable). Entry point: `backend.pipeline.run_pipeline(image_bytes, field_keys)`; CLI: `scripts/run_pipeline.py`.

### Verification run

`scripts/run_pipeline.py tests/samples/invoice_clear.png --fields invoice_number,date,vendor_name,gst_number,total --no-cache` (real PaddleOCR + real Ollama, LLM cache off):

| Field | Extracted | Correct |
|-------|-----------|---------|
| invoice_number | `INV-1001` | ✔ |
| date | `06/10/2026` | ✔ |
| vendor_name | `SHREE GANESH TRADERS` | ✔ |
| gst_number | `29ABCDE1234F1Z5` | ✔ |
| total | `619.50` | ✔ (decimals kept; OCR read the bold total as `619.50` without ₹, see Phase 2) |

- Exactly the 5 requested keys, in request order. No unrelated fields, no warnings, `error: null`.
- OCR: 29 lines, average confidence 0.984, boxes present. `ocr.text` (row-grouped, sent to DSPy) contains `Milk 1L    2    ₹50.00`; `ocr.raw_text` keeps PaddleOCR's own line order and every ₹.
- **Timings:** preprocess 49 ms · OCR 7.6 s (includes about 2 s model init + first-call warm-up) · DSPy/LLM 54.3 s (cold model load + heavy swapping, see Memory) · **total 62.0 s**. The JSON result is 5.7 KB.

### Tests: 71 passed (`pytest`)

| Suite | Tests | Result | Time |
|-------|-------|--------|------|
| Fast unit tests (`-m "not slow"`): fields, preprocessing, OCR text/reading order (real recorded Phase 2 output), DSPy normalization/formatting, pipeline orchestration with OCR/LLM stubs | 64 | 64 passed | 2.9 s |
| Integration (`-m slow`): real PaddleOCR + real Ollama | 7 | 7 passed | 146.6 s |
| Full suite, second run (LLM answers served from DSPy cache) | 71 | 71 passed | 27.0 s |

Integration cases (real models):

| # | Case | Fields | Result | Time |
|---|------|--------|--------|------|
| 1, 7 | Clear invoice | invoice_number, date, vendor_name, gst_number, total | all correct, `total` = `619.50` | 20.0 s |
| 2, 5 | Receipt + missing fields | vendor_name, customer_name, gst_number, total | vendor `FRESH MART…`, total `413.00`; customer_name and gst_number = `null` (not invented) | 55.1 s |
| 3, 6 | Large image (4960×5200 JPEG) + one field | total | downscaled to 1908×2000 before OCR; boxes in original coordinates; `total` = `619.50` | 19.7 s |
| 8, 9 | All 10 predefined fields + custom `cgst_amount` | 11 fields | all 11 keys in order; invoice no., total, customer, product list correct; custom `cgst_amount` = `14.75` | 46.7 s |
| 4 | Blank image | total | 0 lines, `total: null`, LLM **not** called (pointed at a closed port) | 0.9 s |
| 10, 12 | Ollama unavailable, OCR succeeds | invoice_number, total | `success: false`, `error.stage: "dspy"`, `kind: "unavailable"`; **OCR result kept** (29 lines, text contains INV-1001) | 2.8 s |
| 11 | Invalid image | total | `error.stage: "preprocess"` | 0.05 s |

The unit tests also cover: an unsupported format (GIF), a truncated JPEG, the pixel limit, EXIF rotation, RGBA flattening, BGR channel order, the input bytes being left unchanged, invalid/empty/too-many/duplicate field lists, custom-field labels, and the error classification (`unavailable`, `timeout`, `unknown`).

### Formatting and ₹

- With the decimals-preserving instruction (from Phase 3), the LLM now returned `619.50`, `413.00` and `14.75` as written. **Restoration from OCR** (`value_adjustments`) is the safety net: it copies the formatting of the unique matching OCR number token (`724.5` → `724.50`, `690` → `690.00`, `1250` → `1,250.00`, `36` → `36.00` from `Rs. 36.00`). Unit-tested only; it was not triggered in the real runs.
- Never touched: invoice numbers, GSTINs, dates, phone numbers, or values that already contain symbols. Digits inside IDs (`29` in `29ABCDE…`) are not treated as amounts.
- **Ambiguous or missing values:** if several formats match (e.g. `590` and `₹590.00`), the LLM value is kept and a warning is added. A number not found in the OCR text gets a warning (a possible sign of an invented value). A text value not found verbatim in the OCR text also gets a warning.
- Amounts are returned without the currency symbol (as specified); the original `₹…` text is always in `ocr.raw_text` and `ocr.lines`.
- Remaining OCR-level limitation (Phase 2): PP-OCRv6_small dropped the ₹ on the bold total, so `ocr.raw_text` shows `619.50` for that line. The extracted value is still correct.

### Memory

- Pipeline process peak: about 0.84 GB (clear invoice), and 1.59 GB for the pytest run that included the 4960×5200 image (downscaled first; Phase 2 without downscaling: 2.8–4.3 GB / killed).
- Ollama model: 2.2 GB resident.
- Before the run, only about 2.9 GB was available (Chrome, Java and other apps were using about 2.8 GB). With PaddleOCR + qwen2.5:3b loaded together, the system used **up to 3.5 GB of the 4 GB swap**. Nothing was killed, but swapping slowed the LLM calls (54 s cold vs 16–23 s in Phase 3). After `ollama stop`, available memory returned to 4.3 GB.
- One `PaddleOCR` instance per process and one Ollama server. The pixel array is released before the LLM call.

### Known limitations

- Row grouping is minimal: rotated or skewed photos can split or merge rows. There is no table reconstruction (Phase 2 finding 6).
- Blur still loses ₹ with high confidence (Phase 2 finding 4).
- Latency depends heavily on free RAM. Close heavy apps before demos.
- Paddle prints `ReduceMeanCheckIfOneDNNSupport` log lines and a `lang`/`ocr_version` `UserWarning` (harmless).

### Checkpoint: ✅ passed

The clear invoice with `[invoice_number, date, vendor_name, gst_number, total]` returns exactly those 5 fields in order with correct values (`total` = `619.50`). The result is JSON-serializable. OCR is kept when the LLM fails. Large images are resized before OCR. All 71 tests pass.

## Phase 5: FastAPI API (2026-10-06)

**App:** `backend/main.py` (`uvicorn backend.main:app --host 127.0.0.1 --port 8000`). Endpoints: `GET /health`, `GET /api/fields`, `POST /api/extract`, plus the generated `/docs`, `/redoc` and `/openapi.json`. Every response body is the Phase 4 `PipelineResult`, errors included.

### Live check (real uvicorn + PaddleOCR + Ollama, curl)

| Request | HTTP | Result |
|---------|------|--------|
| `GET /health` | 200 | `ocr_loaded: true` (preloaded at startup); no Ollama call |
| `GET /health?check_llm=true` | 200 | `llm_reachable: true`, `llm_model_available: true` |
| `GET /api/fields` | 200 | 10 predefined fields, `max_fields` 20 |
| `invoice_clear.png` + `fields=Invoice Number,Customer Name,GST Number,Total` | 200 | `INV-1001`, `ABC Stores`, `29ABCDE1234F1Z5`, `619.50`, in request order; 29 OCR lines with boxes; preprocess 63 ms · OCR 7.0 s · LLM 50.7 s · **total 57.8 s** |
| `requirements.txt` | 415 | `request/UNSUPPORTED_FILE_TYPE` |
| empty `.png` | 400 | `request/EMPTY_FILE` |
| text file renamed `.jpg` | 422 | `preprocess/invalid_image` |
| no `file` part | 400 | `request/NO_FILE` |
| `fields=Total,###` | 400 | `fields/INVALID_FIELDS` |
| receipt with **Ollama stopped** | 200 | `success: false`, `dspy/unavailable`, **21 OCR lines kept**; `/health` still 200; 0 tracebacks in the server log |

Memory: the uvicorn process uses 491 MB after the OCR preload and 885 MB after an extraction. With Ollama loaded, available memory dropped to about 1.7 GB, and swap was almost full because other desktop apps were running.

### Tests: 105 passed (`pytest`, 30 s with cached LLM answers)

- 34 new tests: 30 API tests in `tests/test_api.py` (29 fast with a stub pipeline + 1 slow test with real PaddleOCR and the LLM pointed at a closed port) and 4 field-parsing tests.
- They cover: health (plus `check_llm` when unreachable), the field catalog, missing/empty/too-large uploads, invalid image, unsupported type by name/MIME and by content, fields reaching the pipeline in order (comma labels, JSON array, repeated parts), defaults when omitted, custom fields, invalid field lists, the result returned unchanged, DSPy errors keeping OCR (200), OCR errors → 500, unexpected exceptions → 500 without traceback or internal text, malformed multipart → 400 in the same shape, CORS preflight and simple requests (allowed and disallowed origins), and `/docs`, `/redoc`, `/openapi.json`.
- The 71 Phase 4 tests still pass (one Phase 4 field test was rewritten for `parse_fields_input`).

## Phase 6: Browser frontend (2026-10-06)

**Frontend:** `frontend/index.html`, `style.css`, `app.js` (plain HTML/CSS/vanilla JS, no build), served by FastAPI at `http://127.0.0.1:8000/` on the same origin as the API. **Browser check:** `scripts/ui_check.py`, which drives the installed Google Chrome headless over the DevTools protocol; no new packages. Only responses the UI cannot trigger by itself (400, 413, 500, network failure, timeout, malformed body) are simulated by replacing `fetch` for `/api/extract`. Everything else hits the real server.

### Results

| Check | Result |
|-------|--------|
| `ui_check.py ui` (Ollama stopped) | **40/40**: page load; 10 fields + defaults from `/api/fields`; image select/preview/remove; client rejects `.txt` and > 10 MB; check/uncheck; custom add (button and Enter), duplicate prevention, predefined-name detection, invalid name, remove; counter; 0 fields → button disabled and no request; multipart = `file` + `fields` JSON keys in display order + abort signal; HTTP 400/413/415/422/500 (internal and OCR) messages; network failure; 210 s timer armed + timeout message; malformed response; loading state (message, elapsed counter, controls locked) and its removal; **real** 422 (text named `.jpg`) and **real** 415 (GIF named `.png`); **real AI-unavailable**: OCR shown (21 lines, raw text) with "OCR completed, but AI extraction could not be completed because the local AI service is unavailable"; mobile 375 px without horizontal scroll; no JS exceptions |
| `ui_check.py real` (Ollama running) | **9/9**: `invoice_clear.png` with Invoice Number, GST Number, Total + custom "CGST Amount" → `INV-1001`, `29ABCDE1234F1Z5`, `619.50`, CGST Amount *Not found* ("3 of 4 requested fields found"). OCR 6.0 s + AI 8.4 s = **14.4 s** (14.6 s in the browser). 29 lines, average confidence 98.4%, boxes drawn on the canvas and aligned with the text (screenshot checked). A repeat replaced the results (DSPy cache: 6.0 s); removing the image cleared them |
| `pytest` | **110 passed**: 105 earlier + 2 static-serving tests + 3 threading regression tests |

### Bugs found and fixed during Phase 6

1. **PaddleOCR failed when called from different threads (backend, latent since Phase 4).** In the real server, OCR began failing with `RuntimeError: std::exception` (inside the text-line orientation model) and then failed on **every** request until restart.
   - Reproduced outside the server: 4/15 failures with a new thread per call, 0/15 with all calls on one thread.
   - Phase 4/5 tests missed it because pytest and the CLI run everything on one thread.
   - **Fix:** all PaddleOCR work (load + predict) runs on one dedicated worker thread, and a failed engine is discarded and reloaded on the next request.
   - Regression tests: one fast (thread identity, engine discarded) and one slow (real OCR from 8 different threads, 0 failures).
   - After the fix, the server log showed 0 OCR failures across all browser runs.
2. **The `hidden` attribute was overridden by CSS** (`.loading { display: flex }`), so the "Processing image…" card stayed visible after completion. Fixed with `[hidden] { display: none !important; }`. Found in a screenshot; the check now tests computed visibility.
3. Custom fields were shown with the backend-derived label ("Cgst Amount"). Now the label the user typed is shown ("CGST Amount").

### Limitations

- The 210 s limit is enforced in the browser only. A server-side OCR/LLM run keeps going after the browser gives up, until the LLM's own 180 s timeout.
- The custom "CGST Amount" field was not found by qwen2.5:3b in the 4-field request, although it found `14.75` when all fields were requested (Phase 4). Custom fields with abbreviations depend on the small model.
- Opening `index.html` from disk (`file://`) is not supported; use the FastAPI URL.
- The browser checks ran in headless Chrome only (no Firefox/Safari run). The mobile check was an emulated 375 px viewport.

## Phase 7: End-to-end integration (2026-10-06)

### Canonical request format

- **Before Phase 7:** the page sent **one** `fields` part holding a JSON array of keys. The backend accepted repeated parts, comma-separated labels/keys and JSON arrays.
- **Audit finding:** with labels, a custom field such as `Address, Line 1` would be split by the backend into `address` and `line_1`.
- **Now, the one browser format:** `file` + **one `fields` part per selected key, repeated, in display order** (`buildExtractForm()` in `frontend/app.js`). Keys never contain commas. The backend's other forms remain only as curl conveniences.
- **Locked by tests:**
  - `test_canonical_browser_multipart_request`: a byte-level, Chrome-shaped multipart body
  - `test_frontend_builds_canonical_request`: checks `app.js` itself
  - `test_all_predefined_and_custom_fields_roundtrip`: all 10 predefined + 4 custom fields through HTTP and the real pipeline logic
  - `test_comma_in_label_is_why_keys_are_canonical`
  - `test_no_fields_parts_means_defaults_but_ui_never_sends_that`
- **Browser check at network level:** Chrome's request body had parts `file, fields, fields, fields` = `invoice_number, total, pan_no`.

### Browser checks (`scripts/ui_check.py`, headless Chrome)

| Mode | Result | Highlights |
|------|--------|-----------|
| `ui` (Ollama stopped) | **53/53** | everything from Phase 6, plus: canonical parts on the wire; **real 210 s AbortController timer** (fast-forwarded) → message, controls restored, retry works; **all 10 predefined + 4 custom fields** (Vendor Address, Payment Reference, CGST Amount, Purchase Order Number) through the real server: one POST, wire = FastAPI = 14 keys in order, 14 rows in order with typed labels, "Not extracted" + OCR kept because the AI is off; mobile 375 px |
| `integration` (Ollama running) | **73/73** | `/health` (no LLM probe) and `?check_llm=true`; status "OCR ready · AI service ready"; **real** 400 (invalid field), 400 (empty file) and 413 bodies rendered by the page code; **6 sequential extractions** (below); box checks; memory; server log clean |

**Sequential runs** (one UI session; first run of these field selections = real LLM calls):

| # | Image | Fields sent (= FastAPI received) | Result | OCR | AI | Total |
|---|-------|----------------------------------|--------|-----|----|-------|
| 1 | invoice_clear.png | invoice_number, gst_number, total | `INV-1001`, `29ABCDE1234F1Z5`, `619.50` ✔ | 7.2 s | 74.7 s | 82.0 s |
| 2 | receipt_mono.png | vendor_name, date, total, bill_no (custom "Bill No") | Not found, `2026-09-28`, `413.00`, `58213` | 5.9 s | 38.4 s | 44.4 s |
| 3 | invoice_large.jpg (4960×5200 → 2000 px) | invoice_number, total | `INV-1001`, `619.50` ✔ | 12.1 s | 78.6 s | 91.1 s |
| 4 | invoice_blurry.jpg | invoice_number, customer_name, subtotal | `INV-1001`, `ABC Stores`, **`14.75` ✘** (see below) | 5.5 s | 23.4 s | 28.9 s |
| 5 | blank.png | date, total | "No text found", both Not found, AI not called | 1.1 s | 0 | 1.1 s |
| 6 | invoice_clear.png | invoice_number, gst_number, total, cgst_amount | ✔ values + CGST Amount Not found | 5.1 s | (cache) | 5.1 s |

- **For every run:**
  - exactly one POST per click
  - the wire `fields` parts = FastAPI's keys = the checked keys, in order
  - the table mirrors the selection
  - the raw OCR text belongs to the current image (no stale text)
  - the lines table matches `line_count`
  - controls restored
- A repeat of the whole sequence (served from DSPy's cache, AI ≈ 3 ms) also passed 73/73.
- The high AI times on runs 1 and 3 come from a cold model load plus swapping: about 1.3 GB was available, with the user's desktop apps running.

### Bounding boxes

- **Drawn at the scaled position:** for every line on every image, the polygon's first corner is painted on the canvas.
- **Every box covers ink** in the original image: clear 29/29, receipt 21/21, large 29/29, blurry 29/29.
- **Downscaled image:** large-image boxes ÷ 4 match the clear-image boxes, with a **max difference of 3.5 px** over 27 matched lines. The same check is a slow pytest, `test_downscaled_image_boxes_map_to_original_coordinates`.
- **A heuristic I discarded:** a first "darker inside than around" tightness check failed on the receipt and the blurred invoice. Rendered overlays (`outputs/ui_check/diag_*.png`) showed the boxes were correct; the heuristic breaks on tightly spaced and rotated lines. It was replaced by the checks above. A first version of the geometric check paired repeated texts (two `₹14.75`) wrongly; they are now paired top to bottom.

### Issues found

1. **Wrong value on the rotated (blurred) invoice: `Subtotal = 14.75` instead of `590.00`.** The cause is not the LLM. The minimal row grouping (PLAN.md §14) shifts the amount column up one row on the 1.5°-rotated image, so the text sent to the AI reads `Subtotal    14.75`, `SGST 2.5%    619.50`, `Total` (alone). This is the documented skew limitation, now observed end to end. **Not fixed in Phase 7** (an OCR post-processing change, outside integration scope). Proposed fix: skew-aware row grouping (estimate the page angle from the polygons, group on de-rotated y).
2. **qwen2.5:3b misses:** "Vendor Name" on the receipt (`FRESH MART SUPERMARKET` is in the text) and custom "CGST Amount". Both returned `null`; nothing was invented. These are known model limitations, out of scope.

### Resources

- One uvicorn worker, one `ollama serve` (the model loads on demand and unloads when idle), and no headless Chrome left after any check.
- **uvicorn footprint (RSS + swap) across 6 runs:** 1184 → 1233 → 1199 → 1263 → 1282 → 1282 → 1270 MB, so no growth beyond warm-up. RSS alone was misleading: it dropped to 313 MB while Ollama pushed the process into swap.
- Server log: 0 PaddleOCR failures and 0 tracebacks across all Phase 7 runs.

### Tests

- `pytest -m "not slow"`: **106 passed in 4.1 s**.
- `pytest`: **116 passed** (110 from Phase 6 + 5 API + 1 slow geometry test).
