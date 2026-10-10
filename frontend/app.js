"use strict";

/* ------------------------------------------------------------------ config */

// The one place the backend URL is defined. When the page is served by FastAPI
// (http://127.0.0.1:8000/) the API is on the same origin; if the file is opened
// some other way, fall back to the local default.
const API_BASE_URL = window.location.protocol.startsWith("http")
  ? window.location.origin
  : "http://127.0.0.1:8000";

const REQUEST_TIMEOUT_MS = 210_000; // end-to-end target (OCR + up to 180 s LLM)
const MAX_UPLOAD_MB = 10;           // same limit as the server (MAX_UPLOAD_MB)
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/bmp", "image/webp"];
const ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"];
const KEY_PATTERN = /^[a-z][a-z0-9_]{0,39}$/; // replaced by the server's pattern from /api/fields

/* ------------------------------------------------------------------- state */

const state = {
  file: null,
  objectUrl: null,
  predefined: [],      // [{key, label, multi, custom, default}] from GET /api/fields
  custom: [],          // [{key, label}] added by the user
  selected: new Set(), // keys
  maxFields: 20,
  keyPattern: KEY_PATTERN,
  busy: false,
  result: null,        // last PipelineResult with OCR data (for the box overlay)
};

const $ = (id) => document.getElementById(id);
const el = {};

/* --------------------------------------------------------------- helpers */

/** Same rule as backend/fields.py key_from_label: "PAN No." -> "pan_no". */
function keyFromLabel(name) {
  return name.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function seconds(ms) {
  return typeof ms === "number" ? `${(ms / 1000).toFixed(1)} s` : "–";
}

function make(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "className") node.className = v;
    else if (k === "text") node.textContent = v; // never innerHTML: OCR text is untrusted
    else node.setAttribute(k, v);
  }
  for (const child of children) if (child != null) node.append(child);
  return node;
}

/** Fields in display order: predefined (catalog order), then custom (added order). */
function orderedSelectedKeys() {
  const all = [...state.predefined.map((f) => f.key), ...state.custom.map((f) => f.key)];
  return all.filter((k) => state.selected.has(k));
}

function fetchWithTimeout(url, options = {}, timeoutMs = 10_000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...options, signal: controller.signal }).finally(() => clearTimeout(timer));
}

/* ------------------------------------------------------------ service info */

async function checkService() {
  try {
    const res = await fetchWithTimeout(`${API_BASE_URL}/health?check_llm=true`, {}, 8000);
    const body = await res.json();
    const ocr = body.ocr_error ? "OCR engine failed to load" : "OCR ready";
    const ai = body.llm_reachable
      ? (body.llm_model_available ? "AI service ready" : "AI model not installed in Ollama")
      : "AI service offline (OCR still works)";
    const allGood = !body.ocr_error && body.llm_reachable && body.llm_model_available;
    el.serviceStatus.textContent = `${ocr} · ${ai}`;
    el.serviceStatus.className = `service-status ${allGood ? "ok" : "warn"}`;
  } catch {
    el.serviceStatus.textContent = `Cannot reach the OCR server at ${API_BASE_URL}. Start the backend and reload.`;
    el.serviceStatus.className = "service-status warn";
  }
}

/* ------------------------------------------------------------------ fields */

async function loadFields() {
  el.fieldsError.hidden = true;
  try {
    const res = await fetchWithTimeout(`${API_BASE_URL}/api/fields`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const body = await res.json();
    if (!Array.isArray(body.fields)) throw new Error("bad response");
    state.predefined = body.fields;
    state.maxFields = Number(body.max_fields) || state.maxFields;
    try { state.keyPattern = new RegExp(body.custom_key_pattern); } catch { /* keep default */ }
    for (const f of state.predefined) if (f.default) state.selected.add(f.key);
    renderPredefined();
  } catch {
    el.predefinedList.replaceChildren();
    el.fieldsError.textContent = "Could not load the predefined fields from the server. "
      + "You can still add custom fields, or reload the page once the backend is running.";
    el.fieldsError.hidden = false;
  }
  updateControls();
}

function renderPredefined() {
  el.predefinedList.replaceChildren(...state.predefined.map((f) => {
    const id = `field-${f.key}`;
    const box = make("input", { type: "checkbox", id, value: f.key });
    box.checked = state.selected.has(f.key);
    box.addEventListener("change", () => toggleField(f.key, box.checked, box));
    return make("label", { className: "check", for: id }, box,
      make("span", { text: f.label }),
      f.multi ? make("span", { className: "badge", text: "list", title: "Can return several values" }) : null);
  }));
}

function renderCustom() {
  el.customList.replaceChildren(...state.custom.map((f) => {
    const id = `custom-${f.key}`;
    const box = make("input", { type: "checkbox", id, value: f.key });
    box.checked = state.selected.has(f.key);
    box.addEventListener("change", () => toggleField(f.key, box.checked, box));
    const remove = make("button", { type: "button", className: "btn btn-secondary btn-small",
      "aria-label": `Remove custom field ${f.label}`, text: "Remove" });
    remove.addEventListener("click", () => removeCustom(f.key));
    return make("li", {},
      make("label", { className: "check", for: id }, box, make("span", { text: f.label }),
        make("span", { className: "badge", text: "custom" })),
      remove);
  }));
}

function toggleField(key, checked, box) {
  if (checked && !state.selected.has(key) && state.selected.size >= state.maxFields) {
    box.checked = false;
    setCustomMessage(`You can select at most ${state.maxFields} fields.`, true);
    return;
  }
  if (checked) state.selected.add(key); else state.selected.delete(key);
  updateControls();
}

function setCustomMessage(text, isError = false) {
  el.customMessage.textContent = text;
  el.customMessage.classList.toggle("is-error", isError);
}

function addCustom() {
  const label = el.customInput.value.trim().replace(/\s+/g, " ");
  if (!label) {
    setCustomMessage("Type a field name first.", true);
    return;
  }
  const key = keyFromLabel(label);
  if (!state.keyPattern.test(key)) {
    setCustomMessage("Use letters, digits and spaces, starting with a letter (max 40 characters).", true);
    return;
  }
  const predefined = state.predefined.find((f) => f.key === key);
  if (predefined) {
    if (!state.selected.has(key) && state.selected.size >= state.maxFields) {
      setCustomMessage(`You can select at most ${state.maxFields} fields.`, true);
      return;
    }
    state.selected.add(key);
    renderPredefined();
    setCustomMessage(`"${predefined.label}" is already a predefined field, so it has been selected.`);
    el.customInput.value = "";
    updateControls();
    return;
  }
  if (state.custom.some((f) => f.key === key)) {
    setCustomMessage(`"${label}" has already been added.`, true);
    return;
  }
  if (state.selected.size >= state.maxFields) {
    setCustomMessage(`You can select at most ${state.maxFields} fields.`, true);
    return;
  }
  state.custom.push({ key, label });
  state.selected.add(key);
  renderCustom();
  setCustomMessage(`Added "${label}".`);
  el.customInput.value = "";
  el.customInput.focus();
  updateControls();
}

function removeCustom(key) {
  const field = state.custom.find((f) => f.key === key);
  state.custom = state.custom.filter((f) => f.key !== key);
  state.selected.delete(key);
  renderCustom();
  setCustomMessage(field ? `Removed "${field.label}".` : "");
  el.customInput.focus();
  updateControls();
}

function setAll(checked) {
  for (const f of state.predefined) {
    if (checked && !state.selected.has(f.key) && state.selected.size >= state.maxFields) break;
    if (checked) state.selected.add(f.key); else state.selected.delete(f.key);
  }
  renderPredefined();
  updateControls();
}

/* ------------------------------------------------------------------- image */

function validateFile(file) {
  const name = file.name.toLowerCase();
  const extOk = ALLOWED_EXTENSIONS.some((ext) => name.endsWith(ext));
  const typeOk = !file.type || ALLOWED_TYPES.includes(file.type);
  if (!extOk || !typeOk) return "This file type is not supported. Please choose a JPG, PNG, BMP or WEBP image.";
  if (file.size === 0) return "This file is empty.";
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return `This image is ${formatBytes(file.size)}; the maximum is ${MAX_UPLOAD_MB} MB.`;
  return null;
}

function selectFile(file) {
  if (state.busy || !file) return;
  const problem = validateFile(file);
  if (problem) {
    el.fileError.textContent = problem;
    el.fileError.hidden = false;
    el.fileInput.value = "";
    return; // keep any previously selected valid image
  }
  el.fileError.hidden = true;
  clearResults();
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.file = file;
  state.objectUrl = URL.createObjectURL(file);
  el.previewImg.src = state.objectUrl;
  el.fileName.textContent = file.name;
  el.fileSize.textContent = formatBytes(file.size);
  el.preview.hidden = false;
  updateControls();
}

function removeFile() {
  if (state.busy) return;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.file = null;
  state.objectUrl = null;
  el.fileInput.value = "";
  el.previewImg.removeAttribute("src");
  el.preview.hidden = true;
  el.fileError.hidden = true;
  clearResults();
  updateControls();
  el.fileInput.focus();
}

/* ---------------------------------------------------------------- controls */

function updateControls() {
  const count = state.selected.size;
  el.selectedCount.textContent = `Selected fields: ${count}`;
  let hint = "";
  if (state.busy) hint = "Extraction in progress…";
  else if (!state.file && count === 0) hint = "Choose an image and select at least one field.";
  else if (!state.file) hint = "Choose an image first.";
  else if (count === 0) hint = "Select at least one field to extract.";
  else hint = `Ready: ${count} field${count === 1 ? "" : "s"} will be extracted.`;
  el.extractHint.textContent = hint;
  el.extractBtn.disabled = state.busy || !state.file || count === 0;

  for (const node of document.querySelectorAll(
    "#predefined-list input, #custom-list input, #custom-list button, #custom-input, #custom-add, #select-all, #clear-all, #file-input, #remove-file")) {
    node.disabled = state.busy;
  }
  el.dropZone.classList.toggle("is-disabled", state.busy);
}

function setBusy(busy) {
  state.busy = busy;
  el.extractBtn.textContent = busy ? "Extracting…" : "Extract Selected Fields";
  el.extractBtn.setAttribute("aria-busy", String(busy));
  updateControls();
}

/* ----------------------------------------------------------------- extract */

let elapsedTimer = null;

function startLoading() {
  const started = Date.now();
  el.elapsed.textContent = "0";
  el.loading.hidden = false;
  elapsedTimer = setInterval(() => {
    el.elapsed.textContent = String(Math.floor((Date.now() - started) / 1000));
  }, 1000);
}

function stopLoading() {
  clearInterval(elapsedTimer);
  el.loading.hidden = true;
}

/**
 * The canonical browser request for POST /api/extract (PLAN.md §13):
 *   file   = the image
 *   fields = one part per selected field KEY, in display order (repeated `fields` parts)
 * Keys, not labels: a label may contain a comma ("Address, Line 1"), which the backend
 * would split into two fields; keys ("address_line_1") never contain commas.
 */
function buildExtractForm(file, keys) {
  const form = new FormData();
  form.append("file", file, file.name);
  for (const key of keys) form.append("fields", key);
  return form;
}

async function extract() {
  // 1-2. Validate (the button is normally disabled in these cases; never send a useless request).
  const keys = orderedSelectedKeys();
  if (!state.file) { showBanner("error", "No image selected", "Choose an image to extract fields from."); return; }
  if (keys.length === 0) { showBanner("error", "No fields selected", "Select at least one field to extract."); return; }

  // 3-4. Busy + loading state.
  clearResults();
  setBusy(true);
  startLoading();

  // 5-7. Multipart request in the canonical format (see buildExtractForm).
  const form = buildExtractForm(state.file, keys);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/extract`, { method: "POST", body: form, signal: controller.signal });
  } catch (err) {
    clearTimeout(timeout);
    stopLoading();
    setBusy(false);
    if (err && err.name === "AbortError") {
      showBanner("error", "Processing took too long",
        "Processing took too long. Please try again with a smaller or clearer image or fewer fields.");
    } else {
      showBanner("error", "Cannot reach the server",
        `The OCR server at ${API_BASE_URL} did not respond. Make sure the backend is running, then try again.`);
    }
    return;
  }

  // 8. Read JSON (the body is always a PipelineResult, also for errors).
  let body = null;
  try {
    body = await response.json();
  } catch { /* handled below */ }
  clearTimeout(timeout);
  stopLoading();
  setBusy(false);

  if (!body || typeof body !== "object" || typeof body.success !== "boolean") {
    showBanner("error", "Unexpected server response",
      `The server sent a response that could not be read (HTTP ${response.status}). Please try again.`);
    return;
  }
  renderResponse(response.status, body);
}

/* ------------------------------------------------------------------ render */

const HTTP_MESSAGES = {
  400: ["The request was not accepted", "Please check the image and the selected fields."],
  413: ["The image is too large", `The maximum upload size is ${MAX_UPLOAD_MB} MB. Please use a smaller image.`],
  415: ["Unsupported file type", "Please upload a JPG, PNG, BMP or WEBP image."],
  422: ["The image could not be read", "The file may be corrupt or not a real image. Please try another image."],
  500: ["Server error", "Something went wrong on the server. Please try again."],
};

const AI_FAILURE_TEXT = {
  unavailable: "OCR completed, but AI extraction could not be completed because the local AI service is unavailable. Start Ollama and try again.",
  timeout: "OCR completed, but AI extraction took too long (over 3 minutes). Try fewer fields or try again.",
  parse: "OCR completed, but the AI returned an answer that could not be read. Please try again.",
};

function renderResponse(status, body) {
  const err = body.error;

  if (status === 200 && body.success) {
    const found = Object.values(body.extracted || {}).filter((v) => v !== null && !(Array.isArray(v) && !v.length)).length;
    const total = (body.requested_fields || []).length;
    const noText = (body.warnings || []).includes("No text detected in image");
    if (noText) showBanner("warning", "No text found", "OCR did not detect any text in this image, so no fields could be extracted.");
    else showBanner("success", "Extraction complete", `${found} of ${total} requested field${total === 1 ? "" : "s"} found.`);
    renderFields(body);
    renderOcr(body);
    return;
  }

  if (err && err.stage === "dspy") {
    // OCR succeeded; only AI extraction failed. Keep and show the OCR result.
    showBanner("warning", "AI extraction failed",
      AI_FAILURE_TEXT[err.kind] || "OCR completed, but AI field extraction could not be completed. Please try again.");
    renderFields(body, true);
    renderOcr(body);
    return;
  }

  const [title, fallback] = HTTP_MESSAGES[status] || ["Request failed", `The server returned HTTP ${status}.`];
  let detail = fallback;
  if (err && err.stage === "ocr") detail = "Text recognition failed. Please try a clearer image.";
  else if (err && err.stage === "internal") detail = "Something went wrong on the server. Please try again.";
  else if (err && typeof err.message === "string" && status < 500) detail = `${err.message} ${fallback}`; // server messages for 4xx are user-safe
  showBanner("error", title, detail);
  if (body.ocr) renderOcr(body);
}

function showBanner(kind, title, text) {
  const icon = { success: "✓", warning: "⚠", error: "✕" }[kind];
  el.banner.className = `banner ${kind}`;
  el.banner.setAttribute("role", kind === "error" ? "alert" : "status");
  el.bannerTitle.textContent = `${icon} ${title}`;
  el.bannerText.textContent = text;
  el.banner.hidden = false;
  // Single-column layout: results sit below the button, so bring the outcome into view.
  if (window.matchMedia("(max-width: 900px)").matches) el.banner.scrollIntoView({ block: "start" });
}

function renderFields(body, aiFailed = false) {
  const labels = new Map((body.fields || []).map((f) => [f.key, f]));
  // Show custom fields with the label the user typed ("CGST Amount", not the derived "Cgst Amount").
  for (const f of state.custom) if (labels.has(f.key)) labels.set(f.key, { ...labels.get(f.key), label: f.label });
  const extracted = body.extracted || {};
  const rows = (body.requested_fields || []).map((key) => {
    const info = labels.get(key) || { label: key, custom: false };
    const dt = make("dt", {}, info.label, info.custom ? make("span", { className: "badge", text: "custom" }) : null);
    let dd;
    const value = aiFailed ? undefined : extracted[key];
    if (Array.isArray(value) && value.length) {
      dd = make("dd", {}, make("ul", {}, ...value.map((v) => make("li", { text: String(v) }))));
    } else if (typeof value === "string" && value !== "") {
      dd = make("dd", { text: value });
    } else {
      dd = make("dd", {}, make("span", { className: "not-found", text: aiFailed ? "Not extracted" : "Not found" }));
    }
    return make("div", { className: "result-row" }, dt, dd);
  });
  el.resultsList.replaceChildren(...rows);

  const adjustments = (body.value_adjustments || []).map(
    (a) => `${(labels.get(a.field) || { label: a.field }).label}: formatting restored from OCR text "${a.ocr_token}"`);
  el.resultsNote.textContent = aiFailed
    ? "Values could not be extracted; the OCR text below is still available."
    : [...adjustments, "Amounts are shown without currency symbols; see the raw OCR text for the original."].join(" · ");
  el.copyJson.hidden = aiFailed;
  el.results.hidden = false;
}

function renderOcr(body) {
  const ocr = body.ocr;
  if (!ocr) return;
  state.result = body;

  const items = [];
  const add = (label, value) => items.push(make("li", {}, `${label}: `, make("strong", { text: value })));
  add("Detected lines", String(ocr.line_count ?? (ocr.lines || []).length));
  if (typeof ocr.average_confidence === "number") add("Average confidence", `${(ocr.average_confidence * 100).toFixed(1)}%`);
  if (body.image) {
    add("Image", `${body.image.original_width} × ${body.image.original_height}${body.image.resized
      ? ` (OCR at ${body.image.processed_width} × ${body.image.processed_height})` : ""}`);
  }
  const t = body.timings_ms || {};
  if ("ocr" in t) add("OCR time", seconds(t.ocr));
  if ("extraction" in t) add("AI time", seconds(t.extraction));
  if ("total" in t) add("Total", seconds(t.total));
  el.ocrSummary.replaceChildren(...items);

  el.warnings.replaceChildren(...(body.warnings || []).map((w) => make("li", { text: w })));
  el.rawText.textContent = ocr.raw_text || "(no text detected)";
  el.readingText.textContent = ocr.text || "(no text detected)";

  el.linesBody.replaceChildren(...(ocr.lines || []).map((line, i) => {
    const pct = line.confidence * 100;
    const cls = pct >= 90 ? "conf-high" : pct >= 70 ? "conf-mid" : "conf-low";
    return make("tr", {},
      make("td", { text: String(i + 1) }),
      make("td", { text: line.text }),
      make("td", { className: cls, text: `${pct.toFixed(1)}%${pct < 70 ? " (low)" : ""}` }),
      make("td", { text: `[${(line.box || []).join(", ")}]` }));
  }));
  el.ocrSection.hidden = false;

  el.boxesToggleWrap.hidden = !(ocr.lines && ocr.lines.length);
  drawBoxes();
}

/** Draw OCR polygons over the preview. Coordinates are original-image pixels (backend
 *  scales them back after downscaling), so scale by displayed size / original size. */
function drawBoxes() {
  const canvas = el.boxCanvas;
  const ctx = canvas.getContext("2d");
  const img = el.previewImg;
  const result = state.result;
  const width = img.clientWidth;
  const height = img.clientHeight;
  canvas.width = Math.round(width * window.devicePixelRatio);
  canvas.height = Math.round(height * window.devicePixelRatio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (!result || !result.ocr || !el.boxesToggle.checked || !img.naturalWidth || !width) return;

  const origW = (result.image && result.image.original_width) || img.naturalWidth;
  const origH = (result.image && result.image.original_height) || img.naturalHeight;
  // The <img> uses object-fit: contain, so letterboxing may add offsets.
  const scale = Math.min(width / origW, height / origH);
  const offX = (width - origW * scale) / 2;
  const offY = (height - origH * scale) / 2;

  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(28, 124, 74, 0.9)";
  ctx.fillStyle = "rgba(28, 124, 74, 0.10)";
  for (const line of result.ocr.lines) {
    const pts = line.polygon && line.polygon.length >= 3 ? line.polygon
      : [[line.box[0], line.box[1]], [line.box[2], line.box[1]], [line.box[2], line.box[3]], [line.box[0], line.box[3]]];
    ctx.beginPath();
    pts.forEach(([x, y], i) => {
      const px = offX + x * scale;
      const py = offY + y * scale;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
}

function clearResults() {
  state.result = null;
  el.banner.hidden = true;
  el.results.hidden = true;
  el.ocrSection.hidden = true;
  el.resultsList.replaceChildren();
  el.linesBody.replaceChildren();
  el.boxesToggleWrap.hidden = true;
  if (el.boxCanvas.width) drawBoxes();
}

async function copyJson() {
  const result = state.result;
  if (!result) return;
  const data = Object.fromEntries((result.requested_fields || []).map((k) => [k, (result.extracted || {})[k] ?? null]));
  try {
    await navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    el.copyJson.textContent = "Copied";
  } catch {
    el.copyJson.textContent = "Copy failed";
  }
  setTimeout(() => { el.copyJson.textContent = "Copy as JSON"; }, 1500);
}

/* -------------------------------------------------------------------- init */

function init() {
  for (const [name, id] of Object.entries({
    serviceStatus: "service-status", dropZone: "drop-zone", fileInput: "file-input", fileError: "file-error",
    preview: "preview", fileName: "file-name", fileSize: "file-size", removeFile: "remove-file",
    previewImg: "preview-img", boxCanvas: "box-canvas", boxesToggle: "boxes-toggle", boxesToggleWrap: "boxes-toggle-wrap",
    predefinedList: "predefined-list", fieldsError: "fields-error", selectAll: "select-all", clearAll: "clear-all",
    customInput: "custom-input", customAdd: "custom-add", customMessage: "custom-message", customList: "custom-list",
    selectedCount: "selected-count", extractBtn: "extract-btn", extractHint: "extract-hint",
    loading: "loading", elapsed: "elapsed", banner: "banner", bannerTitle: "banner-title", bannerText: "banner-text",
    results: "results", resultsList: "results-list", resultsNote: "results-note", copyJson: "copy-json",
    ocrSection: "ocr-section", ocrSummary: "ocr-summary", warnings: "warnings", rawText: "raw-text",
    readingText: "reading-text", linesBody: "lines-body",
  })) el[name] = $(id);

  el.fileInput.addEventListener("change", () => selectFile(el.fileInput.files[0]));
  el.removeFile.addEventListener("click", removeFile);
  el.dropZone.addEventListener("dragover", (e) => { e.preventDefault(); if (!state.busy) el.dropZone.classList.add("is-dragover"); });
  el.dropZone.addEventListener("dragleave", () => el.dropZone.classList.remove("is-dragover"));
  el.dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    el.dropZone.classList.remove("is-dragover");
    if (e.dataTransfer && e.dataTransfer.files.length) selectFile(e.dataTransfer.files[0]);
  });

  el.selectAll.addEventListener("click", () => setAll(true));
  el.clearAll.addEventListener("click", () => setAll(false));
  el.customAdd.addEventListener("click", addCustom);
  el.customInput.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); addCustom(); } });
  el.extractBtn.addEventListener("click", extract);
  el.copyJson.addEventListener("click", copyJson);

  el.boxesToggle.addEventListener("change", drawBoxes);
  el.previewImg.addEventListener("load", drawBoxes);
  new ResizeObserver(drawBoxes).observe(el.previewImg);

  updateControls();
  loadFields();
  checkService();
}

document.addEventListener("DOMContentLoaded", init);
