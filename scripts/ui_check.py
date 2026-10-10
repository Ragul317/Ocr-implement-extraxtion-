"""Browser check of the one-page frontend (headless Chrome, DevTools protocol).

No test framework or extra packages: uses the installed Google Chrome and the
`websockets` package that uvicorn[standard] already brings in.

Start the server first:  uvicorn backend.main:app --host 127.0.0.1 --port 8000

Usage:
    python scripts/ui_check.py ui      # UI behaviour + error handling (Ollama should be STOPPED,
                                       # so the real "AI unavailable" path is exercised)
    python scripts/ui_check.py real    # one real PaddleOCR + Ollama extraction (Ollama RUNNING)
    python scripts/ui_check.py smoke   # final smoke test: invoice_clear + 3 fields (Ollama RUNNING)
    python scripts/ui_check.py integration
                                       # Phase 7: 6 sequential real extractions on all samples, wire-level
                                       # request checks, box alignment, real error bodies (Ollama RUNNING)

For request checks against what FastAPI received, start uvicorn with its output in
outputs/uvicorn.log (`uvicorn backend.main:app ... > outputs/uvicorn.log 2>&1`).

Screenshots and temporary files go to outputs/ui_check/ (git-ignored).
Server errors other than 400/413/415/422/500 that the UI cannot trigger by itself, and
network failure / timeout / malformed responses, are simulated by replacing window.fetch
for /api/extract only; everything else hits the real server.
"""

import asyncio
import base64
import itertools
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

BASE = "http://127.0.0.1:8000/"
ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "tests" / "samples"
OUT = ROOT / "outputs" / "ui_check"
DEBUG_PORT = 9333
SERVER_LOG = ROOT / "outputs" / "uvicorn.log"  # optional: start uvicorn with `> outputs/uvicorn.log 2>&1`
results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""), flush=True)


class CDP:
    def __init__(self, ws):
        self.ws, self.ids, self.pending, self.events = ws, itertools.count(1), {}, []

    async def reader(self):
        async for msg in self.ws:
            data = json.loads(msg)
            if "id" in data and data["id"] in self.pending:
                self.pending.pop(data["id"]).set_result(data)
            else:
                self.events.append(data)

    async def send(self, method, **params):
        i = next(self.ids)
        fut = asyncio.get_running_loop().create_future()
        self.pending[i] = fut
        await self.ws.send(json.dumps({"id": i, "method": method, "params": params}))
        data = await asyncio.wait_for(fut, 300)
        if "error" in data:
            raise RuntimeError(f"{method}: {data['error']}")
        return data.get("result", {})

    async def js(self, expr):
        r = await self.send("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"JS error: {r['exceptionDetails'].get('exception', {}).get('description')}")
        return r["result"].get("value")

    async def wait(self, expr, timeout=30, interval=0.25):
        end = time.time() + timeout
        while time.time() < end:
            if await self.js(expr):
                return True
            await asyncio.sleep(interval)
        return False

    async def set_file(self, path):
        doc = await self.send("DOM.getDocument")
        node = await self.send("DOM.querySelector", nodeId=doc["root"]["nodeId"], selector="#file-input")
        await self.send("DOM.setFileInputFiles", nodeId=node["nodeId"], files=[str(path)])

    async def shot(self, name):
        r = await self.send("Page.captureScreenshot", format="png", captureBeyondViewport=True)
        (OUT / f"{name}.png").write_bytes(base64.b64decode(r["data"]))

    async def reload(self):
        await self.send("Page.navigate", url=BASE)
        await asyncio.sleep(0.5)
        await self.wait("document.readyState === 'complete' && "
                        "document.querySelectorAll('#predefined-list input').length > 0", 20)

    async def click(self, sel):
        await self.js(f"document.querySelector('{sel}').click(); true")

    async def done(self, timeout):
        """Wait until a request finished: banner shown and button usable again."""
        return await self.wait(f"{SHOWN('#banner')} && !document.querySelector('#extract-btn').disabled", timeout, 0.5)


def TEXT(sel):
    return f"(document.querySelector('{sel}')?.textContent || '').trim()"


def SHOWN(sel):
    """Really rendered (not just the `hidden` property)."""
    return f"(getComputedStyle(document.querySelector('{sel}')).display !== 'none')"


# Replaces window.fetch: records multipart bodies; answers /api/extract from window.__mock if set.
MOCK_FETCH = """
window.__calls = [];
window.__realFetch = window.__realFetch || window.fetch;
window.__mock = null;
window.fetch = async (url, opts = {}) => {
  if (opts.body instanceof FormData) {
    window.__calls.push({url: String(url), method: opts.method, hasSignal: !!opts.signal,
      entries: [...opts.body.entries()].map(([k, v]) => [k, v instanceof File ? {name: v.name, size: v.size, type: v.type} : v])});
  }
  const m = window.__mock;
  if (m && String(url).includes('/api/extract')) {
    if (m.delay) await new Promise(r => setTimeout(r, m.delay));
    if (m.hang) return new Promise((_, reject) => opts.signal.addEventListener('abort',
      () => reject(new DOMException('The operation was aborted.', 'AbortError'))));
    if (m.reject === 'network') throw new TypeError('Failed to fetch');
    if (m.reject === 'abort') throw new DOMException('The operation was aborted.', 'AbortError');
    return new Response(m.body, {status: m.status, headers: {'Content-Type': m.type || 'application/json'}});
  }
  return window.__realFetch(url, opts);
};
true;
"""


def mock(status, success, stage=None, kind=None, message="msg", delay=0, with_ocr=False):
    """A PipelineResult-shaped mocked response (same schema as the real API).
    with_ocr=True makes it realistic for success: the real API always includes OCR when success is true."""
    body = {"success": success, "requested_fields": ["total"],
            "fields": [{"key": "total", "label": "Total", "multi": False, "custom": False}],
            "extracted": {"total": "619.50"} if success else {}, "value_adjustments": [], "warnings": [],
            "timings_ms": {}, "image": None, "ocr": None,
            "error": {"stage": stage, "kind": kind, "message": message} if stage else None}
    if with_ocr:
        line = {"text": "Total    619.50", "confidence": 0.99, "polygon": [[10, 10], [110, 10], [110, 30], [10, 30]],
                "box": [10, 10, 110, 30]}
        body["ocr"] = {"text": line["text"], "raw_text": line["text"], "lines": [line], "line_count": 1,
                       "average_confidence": 0.99, "low_confidence_dropped": 0}
        body["image"] = {"original_width": 1240, "original_height": 1300, "processed_width": 1240,
                         "processed_height": 1300, "resized": False, "format": "PNG"}
    return json.dumps({"status": status, "body": json.dumps(body), "delay": delay})


async def select_only(c, keys):
    await c.click("#clear-all")
    await c.js(f"{json.dumps(keys)}.forEach(k => document.querySelector('#field-' + k).click()); true")


async def add_custom(c, label):
    await c.js(f"var i = document.querySelector('#custom-input'); i.value = {json.dumps(label)}; "
               "document.querySelector('#custom-add').click(); true")


# ------------------------------------------------------------------ Phase 7 helpers

def key_from_label(name: str) -> str:
    """Same rule as backend/fields.py and frontend/app.js."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def server_log() -> list[str]:
    return SERVER_LOG.read_text(errors="replace").splitlines() if SERVER_LOG.exists() else []


def server_fields_since(n: int) -> list[list[str]]:
    """Field keys FastAPI received, from its per-request log lines (backend/main.py)."""
    out = []
    for line in server_log()[n:]:
        m = re.search(r"backend\.api: extract .*?fields=(\[.*?\]) ->", line)
        if m:
            out.append(json.loads(m.group(1).replace("'", '"')))
    return out


def extract_posts(c: "CDP", since: int) -> list[dict]:
    return [e["params"] for e in c.events[since:] if e.get("method") == "Network.requestWillBeSent"
            and e["params"]["request"]["url"].endswith("/api/extract") and e["params"]["request"]["method"] == "POST"]


async def posted_fields(c: "CDP", event: dict) -> list[str] | None:
    """`fields` parts of a captured request body (None if Chrome did not keep the body)."""
    data = None
    try:
        data = (await c.send("Network.getRequestPostData", requestId=event["requestId"]))["postData"]
    except RuntimeError:
        entries = event["request"].get("postDataEntries") or []
        data = "".join(base64.b64decode(x.get("bytes", "")).decode("latin-1") for x in entries) or None
    return re.findall(r'name="fields"\r\n\r\n(.*?)\r\n', data) if data else None


ALIGNMENT_JS = """(() => {
  const r = state.result, img = document.querySelector('#preview-img'), cv = document.querySelector('#box-canvas');
  if (!r || !r.ocr || !r.ocr.lines.length) return null;
  const ctx = cv.getContext('2d'), dpr = window.devicePixelRatio, W = img.clientWidth, H = img.clientHeight;
  const oW = r.image.original_width, oH = r.image.original_height, s = Math.min(W / oW, H / oH);
  const ox = (W - oW * s) / 2, oy = (H - oH * s) / 2;
  const tmp = document.createElement('canvas'), t = tmp.getContext('2d', {willReadFrequently: true});
  const darkRatio = (x, y, w, h) => {
    w = Math.max(1, Math.round(w)); h = Math.max(1, Math.round(h)); tmp.width = w; tmp.height = h;
    t.clearRect(0, 0, w, h); t.drawImage(img, x, y, w, h, 0, 0, w, h);
    const d = t.getImageData(0, 0, w, h).data; let n = 0;
    for (let i = 0; i < d.length; i += 4) if (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2] < 170) n++;
    return n / (w * h);
  };
  let drawn = 0, onText = 0;
  for (const ln of r.ocr.lines) {
    const [x1, y1, x2, y2] = ln.box, [px, py] = ln.polygon[0];
    const a = ctx.getImageData(Math.round((ox + px * s) * dpr) - 2, Math.round((oy + py * s) * dpr) - 2, 5, 5).data;
    for (let i = 3; i < a.length; i += 4) if (a[i] > 0) { drawn++; break; }
    if (darkRatio(x1, y1, x2 - x1, y2 - y1) > 0.03) onText++;  // the box is on ink, not on blank paper
  }
  return {lines: r.ocr.lines.length, drawn, onText, natural: [img.naturalWidth, img.naturalHeight],
          original: [oW, oH], displayed: [W, H], scale: +s.toFixed(4)};
})()"""


async def run_extraction(c: "CDP", image: str, predefined: list[str], custom: list[str], *, marker: str | None,
                         not_marker: str | None = None, timeout: int = 230, expect: dict | None = None) -> dict:
    """One extraction through the UI, verified at browser-network, server and DOM level."""
    tag = image.split(".")[0]
    await c.set_file(SAMPLES / image)
    await c.wait("document.querySelector('#preview-img').naturalWidth > 0", 15)
    record(f"[{tag}] new image clears previous results/boxes",
           not await c.js(SHOWN("#results")) and not await c.js(SHOWN("#ocr-section"))
           and not await c.js(SHOWN("#banner")) and await c.js("state.result === null"))
    await select_only(c, predefined)
    while await c.js("document.querySelectorAll('#custom-list li').length"):
        await c.click("#custom-list li button")
    for label in custom:
        await add_custom(c, label)
    keys = await c.js("orderedSelectedKeys()")
    labels = await c.js("orderedSelectedKeys().map(k => (state.predefined.find(f => f.key === k) "
                        "|| state.custom.find(f => f.key === k)).label)")
    catalog = await c.js("state.predefined.map(f => f.key)")
    record(f"[{tag}] selection = predefined (catalog order) + custom (added order)",
           keys == [k for k in catalog if k in predefined] + [key_from_label(x) for x in custom], str(keys))

    since, log_n, t0 = len(c.events), len(server_log()), time.time()
    await c.click("#extract-btn")
    ok = await c.done(timeout)
    wall = time.time() - t0
    await asyncio.sleep(0.3)

    posts = extract_posts(c, since)
    sent = await posted_fields(c, posts[0]) if len(posts) == 1 else None
    record(f"[{tag}] exactly one POST /api/extract on the wire", len(posts) == 1, f"{len(posts)} request(s)")
    if sent is not None:
        record(f"[{tag}] wire body: `fields` parts == checked keys, in order", sent == keys, str(sent))
    received = server_fields_since(log_n)
    if SERVER_LOG.exists():
        record(f"[{tag}] FastAPI received the same keys, in order", received == [keys], str(received))

    page = await c.js("""({
      title: document.querySelector('#banner-title').textContent.trim(),
      text: document.querySelector('#banner-text').textContent.trim(),
      rows: [...document.querySelectorAll('#results-list .result-row')].map(r =>
             [r.querySelector('dt').firstChild.textContent.trim(), r.querySelector('dd').innerText.trim()]),
      raw: document.querySelector('#raw-text').textContent,
      tableRows: document.querySelectorAll('#lines-body tr').length,
      lineCount: state.result ? state.result.ocr.line_count : null,
      requested: state.result ? state.result.requested_fields : null,
      extracted: state.result ? state.result.extracted : null,
      success: state.result ? state.result.success : null,
      timings: state.result ? state.result.timings_ms : null,
      controls: !document.querySelector('#extract-btn').disabled && !document.querySelector('#file-input').disabled,
    })""")
    record(f"[{tag}] request finished and controls restored", ok and page["controls"] and not await c.js(SHOWN("#loading")),
           f"{wall:.1f}s | {page['title']} | {page['text']}")
    record(f"[{tag}] result table mirrors the selection (labels as shown, same order)",
           [r[0] for r in page["rows"]] == labels and page["requested"] == keys
           and (not page["success"] or list((page["extracted"] or {}).keys()) == keys),
           " · ".join(f"{a}={b}" for a, b in page["rows"]))
    if marker:
        record(f"[{tag}] raw OCR text belongs to this image (no stale text)",
               marker in page["raw"] and (not_marker is None or not_marker not in page["raw"])
               and page["tableRows"] == page["lineCount"], f"{page['lineCount']} lines")
        align = await c.js(ALIGNMENT_JS)
        record(f"[{tag}] boxes drawn at the scaled positions", align and align["drawn"] == align["lines"], json.dumps(align))
        record(f"[{tag}] every box covers ink in the original image", align and align["onText"] == align["lines"],
               f"{align and align['onText']}/{align and align['lines']}")
    if expect:
        got = {k: (page["extracted"] or {}).get(k) for k in expect}
        record(f"[{tag}] expected values", got == expect, json.dumps(got, ensure_ascii=False))
    lines = await c.js("state.result && state.result.ocr ? state.result.ocr.lines.map(l => [l.text, l.box]) : []")
    return {"image": image, "keys": keys, "wall": round(wall, 1), "timings": page["timings"], "rows": page["rows"],
            "title": page["title"], "success": page["success"], "lines": lines}


async def check_ui(c: CDP):
    await c.reload()
    record("page loads (title + header text)", await c.js("document.title") == "OCR Extractor"
           and "Upload an invoice" in await c.js(TEXT(".lead")))
    keys = await c.js("[...document.querySelectorAll('#predefined-list input')].map(i => i.value)")
    record("fields loaded from GET /api/fields (10, server order)", len(keys) == 10 and keys[0] == "invoice_number", str(keys))
    checked = await c.js("[...document.querySelectorAll('#predefined-list input:checked')].map(i => i.value)")
    record("default-checked fields come from the API", checked == ["invoice_number", "vendor_name", "date", "total"], str(checked))
    record("selected count shows defaults", await c.js(TEXT("#selected-count")) == "Selected fields: 4")
    record("extract button disabled without image", await c.js("document.querySelector('#extract-btn').disabled"),
           await c.js(TEXT("#extract-hint")))
    await c.wait(f"!{TEXT('#service-status')}.startsWith('Checking')", 15)
    record("service status line shown", True, await c.js(TEXT("#service-status")))

    # Image select / preview / remove
    await c.set_file(SAMPLES / "invoice_clear.png")
    ok = await c.wait("document.querySelector('#preview-img').naturalWidth > 0", 10)
    record("image selection + preview + name/size", ok and await c.js(TEXT("#file-name")) == "invoice_clear.png",
           f"{await c.js(TEXT('#file-name'))} / {await c.js(TEXT('#file-size'))}")
    record("extract button enabled with image + fields", not await c.js("document.querySelector('#extract-btn').disabled"))
    await c.click("#remove-file")
    record("image removal", await c.js(f"!{SHOWN('#preview')} && !document.querySelector('#preview-img').getAttribute('src') "
                                       "&& document.querySelector('#extract-btn').disabled"))
    dropped = await c.js("""(async () => {
      const cv = document.createElement('canvas'); cv.width = 120; cv.height = 60;
      const blob = await new Promise(r => cv.toBlob(r, 'image/png'));
      const dt = new DataTransfer(); dt.items.add(new File([blob], 'dropped.png', {type: 'image/png'}));
      const zone = document.querySelector('#drop-zone');
      zone.dispatchEvent(new DragEvent('dragover', {dataTransfer: dt, bubbles: true, cancelable: true}));
      const highlighted = zone.classList.contains('is-dragover');
      zone.dispatchEvent(new DragEvent('drop', {dataTransfer: dt, bubbles: true, cancelable: true}));
      await new Promise(r => setTimeout(r, 300));
      return {highlighted, name: document.querySelector('#file-name').textContent,
              preview: getComputedStyle(document.querySelector('#preview')).display !== 'none'};
    })()""")
    record("drag and drop: highlight + file accepted + preview", dropped["highlighted"] and dropped["preview"]
           and dropped["name"] == "dropped.png", json.dumps(dropped))
    await c.click("#remove-file")

    # Client-side file validation (no request is sent)
    tmp = OUT / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "notes.txt").write_text("hello")
    await c.set_file(tmp / "notes.txt")
    record("client rejects .txt", await c.js(SHOWN("#file-error")) and not await c.js(SHOWN("#preview")),
           await c.js(TEXT("#file-error")))
    with open(tmp / "big.png", "wb") as f:
        f.truncate(11 * 1024 * 1024)
    await c.set_file(tmp / "big.png")
    record("client rejects > 10 MB", "maximum is 10 MB" in await c.js(TEXT("#file-error")), await c.js(TEXT("#file-error")))

    # Checkboxes, custom fields, duplicates, count
    await c.click("#field-vendor_name")
    record("uncheck predefined -> count 3", await c.js(TEXT("#selected-count")) == "Selected fields: 3")
    await c.click("#field-gst_number")
    record("check predefined -> count 4", await c.js(TEXT("#selected-count")) == "Selected fields: 4")
    await add_custom(c, "Vendor Address")
    record("custom field added and checked", await c.js("!!document.querySelector('#custom-vendor_address')?.checked")
           and await c.js(TEXT("#selected-count")) == "Selected fields: 5", await c.js(TEXT("#custom-message")))
    await c.js("var i = document.querySelector('#custom-input'); i.value = '  vendor   address '; "
               "i.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter'})); true")
    record("duplicate custom prevented (via Enter)", await c.js("document.querySelectorAll('#custom-list li').length") == 1,
           await c.js(TEXT("#custom-message")))
    await add_custom(c, "Customer Name")
    record("custom matching a predefined field selects it instead",
           await c.js("document.querySelector('#field-customer_name').checked")
           and await c.js("document.querySelectorAll('#custom-list li').length") == 1, await c.js(TEXT("#custom-message")))
    await add_custom(c, "###")
    record("invalid custom name rejected", await c.js("document.querySelector('#custom-message').classList.contains('is-error')"),
           await c.js(TEXT("#custom-message")))
    await c.click("#custom-list li button")
    record("custom field removed", await c.js("document.querySelectorAll('#custom-list li').length") == 0
           and await c.js(TEXT("#selected-count")) == "Selected fields: 5")
    await c.click("#clear-all")
    record("clear all -> 0", await c.js(TEXT("#selected-count")) == "Selected fields: 0")
    await c.set_file(SAMPLES / "invoice_clear.png")
    await c.wait("document.querySelector('#preview-img').naturalWidth > 0", 10)
    await c.js(MOCK_FETCH)
    record("0 fields: button disabled + hint", await c.js("document.querySelector('#extract-btn').disabled")
           and "at least one field" in await c.js(TEXT("#extract-hint")), await c.js(TEXT("#extract-hint")))
    await c.js("extract(); true")  # call the handler directly: it must refuse
    record("0 fields: no request sent", await c.js("window.__calls.length") == 0, await c.js(TEXT("#banner-title")))
    await c.click("#select-all")
    record("select all -> 10", await c.js(TEXT("#selected-count")) == "Selected fields: 10")

    # Multipart request (mocked response so no model runs)
    await select_only(c, ["total", "invoice_number"])
    await add_custom(c, "PAN No.")
    await c.js(f"window.__mock = {mock(200, True, with_ocr=True)}; true")
    await c.click("#extract-btn")
    await c.wait("window.__calls.length === 1", 5)
    call = await c.js("window.__calls[0]")
    names = [k for k, _ in call["entries"]]
    fields = [v for k, v in call["entries"] if k == "fields"]
    file_entry = next(v for k, v in call["entries"] if k == "file")
    record("canonical multipart: file + one `fields` part per key (display order) + abort signal",
           call["url"].endswith("/api/extract") and call["method"] == "POST" and call["hasSignal"]
           and names == ["file", "fields", "fields", "fields"] and fields == ["invoice_number", "total", "pan_no"]
           and file_entry["name"] == "invoice_clear.png", f"parts={names} fields={fields} file={file_entry}")
    await c.done(5)
    await c.js("window.__copied = null; navigator.clipboard.writeText = async t => { window.__copied = t; }; true")
    await c.click("#copy-json")
    await c.wait("window.__copied !== null", 3)
    copied = await c.js("window.__copied")
    record("Copy as JSON: requested fields only, values as shown", json.loads(copied or "null") == {"total": "619.50"}
           and await c.js(TEXT("#copy-json")) == "Copied", str(copied))

    # HTTP error statuses (mocked PipelineResult bodies)
    for name, body, title in [
        ("400", mock(400, False, "fields", "INVALID_FIELDS", "Invalid field name 'x'."), "not accepted"),
        ("413", mock(413, False, "request", "FILE_TOO_LARGE", "File exceeds 10 MB."), "too large"),
        ("415", mock(415, False, "request", "UNSUPPORTED_FILE_TYPE", "Only JPG, JPEG, PNG, BMP and WEBP images are supported."), "Unsupported"),
        ("422", mock(422, False, "preprocess", "invalid_image", "The file is not a valid image."), "could not be read"),
        ("500 internal", mock(500, False, "internal", "INTERNAL_ERROR", "Unexpected server error."), "Server error"),
        ("500 OCR failure", mock(500, False, "ocr", "ocr_failed", "Text recognition failed."), "Server error"),
    ]:
        await c.js(f"window.__mock = {body}; true")
        await c.click("#extract-btn")
        await c.done(5)
        t, txt = await c.js(TEXT("#banner-title")), await c.js(TEXT("#banner-text"))
        record(f"HTTP {name} shown safely", "error" in await c.js("document.querySelector('#banner').className")
               and title in t and "Traceback" not in txt, f"{t} | {txt}")

    await c.js("window.__mock = {reject: 'network'}; true")
    await c.click("#extract-btn")
    await c.done(5)
    record("network failure", "Cannot reach" in await c.js(TEXT("#banner-title")), await c.js(TEXT("#banner-text")))

    await c.js("window.__timeouts = []; window.__st = window.__st || window.setTimeout; "
               "window.setTimeout = (f, d, ...a) => { window.__timeouts.push(d); "
               "return window.__st(f, d === 210000 ? 1500 : d, ...a); }; true")
    await c.js("window.__mock = {hang: true}; true")
    t0 = time.time()
    await c.click("#extract-btn")
    await c.done(10)
    waited = time.time() - t0
    timers = await c.js("window.__timeouts")
    await c.js("window.setTimeout = window.__st; true")
    record("timeout: real 210 s AbortController timer fires (fast-forwarded) + friendly message",
           "took too long" in await c.js(TEXT("#banner-text")) and 210000 in timers and 1.0 < waited < 8,
           f"{await c.js(TEXT('#banner-text'))} | armed={timers} | aborted after {waited:.1f}s")
    record("timeout: controls restored, loading hidden",
           not await c.js("document.querySelector('#extract-btn').disabled")
           and not await c.js("document.querySelector('#field-total').disabled") and not await c.js(SHOWN("#loading")))
    await c.js(f"window.__mock = {mock(200, True)}; true")
    await c.click("#extract-btn")
    await c.done(5)
    record("timeout: user can try again", "Extraction complete" in await c.js(TEXT("#banner-title")))

    await c.js("window.__mock = {status: 502, body: '<html>Bad gateway</html>', type: 'text/html'}; true")
    await c.click("#extract-btn")
    await c.done(5)
    record("malformed (non-JSON) response", "Unexpected server response" in await c.js(TEXT("#banner-title")),
           await c.js(TEXT("#banner-text")))

    # Loading state (mock that answers after 3 s)
    await c.js(f"window.__mock = {mock(200, True, delay=3000)}; true")
    await c.click("#extract-btn")
    await asyncio.sleep(1.2)
    loading = await c.js(f"({{shown: {SHOWN('#loading')}, btn: document.querySelector('#extract-btn').disabled, "
                         "label: document.querySelector('#extract-btn').textContent.trim(), "
                         "locked: document.querySelector('#field-total').disabled, text: document.querySelector('#loading').innerText})")
    await c.shot("ui_loading")
    record("loading state: message + button/inputs disabled",
           loading["shown"] and loading["btn"] and loading["locked"] and loading["label"] == "Extracting…"
           and "up to a few minutes" in loading["text"], json.dumps(loading, ensure_ascii=False)[:180])
    await c.done(10)
    record("after response: loading hidden + controls re-enabled",
           not await c.js(SHOWN("#loading")) and not await c.js("document.querySelector('#field-total').disabled"))

    # Real server errors (no model needed)
    await c.reload()
    (tmp / "fake.jpg").write_text("this is not an image")
    await c.set_file(tmp / "fake.jpg")
    await asyncio.sleep(0.5)
    await c.click("#extract-btn")
    await c.done(30)
    record("REAL 422 (text file named .jpg)", "could not be read" in await c.js(TEXT("#banner-title")),
           await c.js(TEXT("#banner-text")))
    subprocess.run([sys.executable, "-c", "import sys; from PIL import Image; "
                    "Image.new('RGB', (40, 40), 'white').save(sys.argv[1], format='GIF')", str(tmp / "disguised.png")], check=True)
    await c.set_file(tmp / "disguised.png")
    await asyncio.sleep(0.5)
    await c.click("#extract-btn")
    await c.done(30)
    record("REAL 415 (GIF bytes named .png)", "Unsupported" in await c.js(TEXT("#banner-title")), await c.js(TEXT("#banner-text")))

    # Real OCR with Ollama stopped: OCR must still be shown
    await c.set_file(SAMPLES / "receipt_mono.png")
    await c.wait("document.querySelector('#preview-img').naturalWidth > 0", 10)
    await select_only(c, ["vendor_name", "total"])
    await c.click("#extract-btn")
    await c.done(180)
    info = await c.js("({title: document.querySelector('#banner-title').textContent, "
                      "text: document.querySelector('#banner-text').textContent, "
                      "raw: document.querySelector('#raw-text').textContent.slice(0, 40), "
                      "rows: document.querySelectorAll('#lines-body tr').length})")
    await c.shot("ui_ollama_down")
    record("REAL AI unavailable: OCR shown + clear message",
           "AI extraction failed" in info["title"] and "local AI service is unavailable" in info["text"]
           and await c.js(SHOWN("#ocr-section")) and info["rows"] >= 15 and "FRESH MART" in info["raw"],
           json.dumps(info, ensure_ascii=False)[:250])
    record("box toggle shown when OCR lines exist", await c.js(SHOWN("#boxes-toggle-wrap")))

    # All 10 predefined + 4 custom fields through the real server (AI is off: plumbing, not extraction)
    await c.send("Emulation.setDeviceMetricsOverride", width=1280, height=1000, deviceScaleFactor=1, mobile=False)
    catalog = await c.js("state.predefined.map(f => f.key)")
    res = await run_extraction(c, "invoice_clear.png", catalog,
                               ["Vendor Address", "Payment Reference", "CGST Amount", "Purchase Order Number"],
                               marker="INV-1001", not_marker="FRESH MART", timeout=120)
    record("14 fields with AI off: every row 'Not extracted', OCR kept",
           "AI extraction failed" in res["title"] and len(res["rows"]) == 14
           and all(v == "Not extracted" for _, v in res["rows"]), str(res["rows"][-4:]))
    painted = ("(() => { const c = document.querySelector('#box-canvas'); "
               "const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0; "
               "for (let i = 3; i < d.length; i += 4) if (d[i]) n++; return n; })()")
    on = await c.js(painted)
    await c.click("#boxes-toggle")
    off = await c.js(painted)
    await c.click("#boxes-toggle")
    again = await c.js(painted)
    record("bounding-box toggle: off clears the overlay, on redraws it", on > 1000 and off == 0 and again == on,
           f"painted px on/off/on = {on}/{off}/{again}")
    record("OCR details: raw text, AI text and confidence values shown",
           await c.js("document.querySelector('#raw-text').textContent.includes('29ABCDE1234F1Z5') && "
                      "document.querySelector('#reading-text').textContent.includes('Total') && "
                      "[...document.querySelectorAll('#lines-body tr td:nth-child(3)')].every(td => /\\d+\\.\\d%/.test(td.textContent))"),
           await c.js(TEXT("#ocr-summary")))

    # Mobile width
    await c.send("Emulation.setDeviceMetricsOverride", width=375, height=800, deviceScaleFactor=2, mobile=True)
    await asyncio.sleep(0.5)
    overflow = await c.js("document.documentElement.scrollWidth - window.innerWidth")
    await c.shot("ui_mobile")
    record("mobile 375 px: no horizontal page scroll", overflow <= 0, f"overflow={overflow}px")


async def check_smoke(c: CDP):
    """Final smoke test: invoice_clear.png + Invoice Number, GST Number, Total (Ollama RUNNING)."""
    await c.reload()
    await c.wait("document.querySelector('#service-status').classList.contains('ok')", 15)
    record("service status: OCR + AI ready", "AI service ready" in await c.js(TEXT("#service-status")),
           await c.js(TEXT("#service-status")))
    res = await run_extraction(c, "invoice_clear.png", ["invoice_number", "gst_number", "total"], [],
                               marker="INV-1001", expect={"invoice_number": "INV-1001",
                                                          "gst_number": "29ABCDE1234F1Z5", "total": "619.50"})
    await c.shot("smoke_result")
    print(f"\nsmoke: {res['wall']}s in browser, timings={res['timings']}, rows={res['rows']}")


async def check_real(c: CDP):
    await c.reload()
    await c.wait("document.querySelector('#service-status').classList.contains('ok')", 15)
    record("service status: OCR + AI ready", "AI service ready" in await c.js(TEXT("#service-status")),
           await c.js(TEXT("#service-status")))
    await c.set_file(SAMPLES / "invoice_clear.png")
    await c.wait("document.querySelector('#preview-img').naturalWidth > 0", 10)
    await select_only(c, ["invoice_number", "gst_number", "total"])
    await add_custom(c, "CGST Amount")
    t0 = time.time()
    await c.click("#extract-btn")
    await asyncio.sleep(2)
    record("REAL loading state visible", await c.js(SHOWN("#loading")), f"{await c.js(TEXT('#elapsed'))} s elapsed")
    ok = await c.done(230)
    wall = time.time() - t0
    rows = await c.js("[...document.querySelectorAll('#results-list .result-row')].map(r => "
                      "[r.querySelector('dt').innerText.trim(), r.querySelector('dd').innerText.trim()])")
    summary = await c.js("document.querySelector('#ocr-summary').innerText")
    canvas = await c.js("(() => { const c = document.querySelector('#box-canvas'); "
                        "const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0; "
                        "for (let i = 3; i < d.length; i += 4) if (d[i]) n++; return {w: c.width, h: c.height, painted: n}; })()")
    await c.shot("real_result")
    record("REAL extraction renders results", ok and "Extraction complete" in await c.js(TEXT("#banner-title")),
           f"{wall:.1f}s in browser | {rows} | {await c.js(TEXT('#banner-text'))}")
    record("raw OCR visible + lines table", await c.js("document.querySelector('#raw-text').textContent.includes('INV-1001')")
           and await c.js("document.querySelectorAll('#lines-body tr').length") >= 25, summary.replace("\n", " | "))
    record("bounding boxes drawn on the image", canvas["painted"] > 1000, json.dumps(canvas))
    record("loading hidden after real extraction", not await c.js(SHOWN("#loading")))
    # Same request again (DSPy cache -> quick): results are replaced, not duplicated.
    await c.click("#extract-btn")
    await asyncio.sleep(0.5)
    ok2 = await c.done(230)
    rows2 = await c.js("document.querySelectorAll('#results-list .result-row').length")
    record("second extraction replaces results", ok2 and rows2 == len(rows), f"rows={rows2}")
    await c.click("#remove-file")
    record("removing the image clears results",
           await c.js(f"!{SHOWN('#results')} && !{SHOWN('#ocr-section')} && !{SHOWN('#boxes-toggle-wrap')}"))


async def check_integration(c: CDP):
    """Phase 7: sequential real extractions (Ollama RUNNING), service status, real error rendering, resources."""
    await c.reload()
    health = await c.js("fetch(API_BASE_URL + '/health').then(r => r.json())")
    health_llm = await c.js("fetch(API_BASE_URL + '/health?check_llm=true').then(r => r.json())")
    record("GET /health: API + OCR up, no LLM probe", health["status"] == "ok" and health["ocr_loaded"]
           and "llm_reachable" not in health, json.dumps(health))
    record("GET /health?check_llm=true: Ollama reachable + model pulled",
           health_llm["llm_reachable"] and health_llm["llm_model_available"], json.dumps(health_llm))
    await c.wait("document.querySelector('#service-status').classList.contains('ok')", 15)
    record("service status line: OCR ready + AI ready", await c.js(TEXT("#service-status")) == "OCR ready · AI service ready",
           await c.js(TEXT("#service-status")))

    # Real server error bodies rendered by the real UI code (requests the UI itself would never send).
    for name, build, title in [
        ("400 invalid field", "fd.append('file', new File([png], 'a.png', {type: 'image/png'})); fd.append('fields', '###');", "not accepted"),
        ("400 empty file", "fd.append('file', new File([], 'a.png', {type: 'image/png'})); fd.append('fields', 'total');", "not accepted"),
        ("413 too large", "fd.append('file', new File([new Uint8Array(11 * 1024 * 1024)], 'a.png', {type: 'image/png'})); fd.append('fields', 'total');", "too large"),
    ]:
        out = await c.js(f"""(async () => {{
          const png = await (await fetch(API_BASE_URL + '/health')).blob();  // any bytes; only used for the 400 case
          const fd = new FormData(); {build}
          const r = await fetch(API_BASE_URL + '/api/extract', {{method: 'POST', body: fd}});
          renderResponse(r.status, await r.json());
          return {{status: r.status, title: document.querySelector('#banner-title').textContent,
                   text: document.querySelector('#banner-text').textContent}};
        }})()""")
        record(f"REAL {name}: rendered safely", title in out["title"] and "Traceback" not in out["text"],
               f"HTTP {out['status']} | {out['title']} | {out['text']}")
    await c.js("clearResults(); true")

    pid = subprocess.run(["pgrep", "-f", f"^{ROOT}/.venv/bin/python .*uvicorn"], capture_output=True, text=True).stdout.split()
    def rss_mb():
        try:
            status = Path(f"/proc/{pid[0]}/status").read_text().splitlines()
            kb = lambda key: int(next(l for l in status if l.startswith(key)).split()[1])  # noqa: E731
            return (kb("VmRSS") + kb("VmSwap")) // 1024  # resident + swapped out = real footprint
        except (IndexError, StopIteration, OSError):
            return None
    rss = [rss_mb()]
    runs = [
        ("invoice_clear.png", ["invoice_number", "gst_number", "total"], [], "INV-1001", "FRESH MART",
         {"invoice_number": "INV-1001", "gst_number": "29ABCDE1234F1Z5", "total": "619.50"}),
        ("receipt_mono.png", ["vendor_name", "date", "total"], ["Bill No"], "FRESH MART", "INV-1001", {"total": "413.00"}),
        ("invoice_large.jpg", ["invoice_number", "total"], [], "INV-1001", "FRESH MART",
         {"invoice_number": "INV-1001", "total": "619.50"}),
        ("invoice_blurry.jpg", ["invoice_number", "customer_name", "subtotal"], [], "INV-1001", "FRESH MART", None),
        ("blank.png", ["total", "date"], [], None, None, {"total": None, "date": None}),
        ("invoice_clear.png", ["invoice_number", "gst_number", "total"], ["CGST Amount"], "INV-1001", "FRESH MART",
         {"invoice_number": "INV-1001", "gst_number": "29ABCDE1234F1Z5", "total": "619.50"}),
    ]
    summary = []
    for image, predefined, custom, marker, not_marker, expect in runs:
        res = await run_extraction(c, image, predefined, custom, marker=marker, not_marker=not_marker, expect=expect)
        rss.append(rss_mb())
        summary.append(res)
        if image == "invoice_large.jpg":
            await c.shot("integration_large_boxes")
        if image == "blank.png":
            record("[blank] 'No text found', no boxes, AI not needed",
                   "No text found" in res["title"] and not await c.js(SHOWN("#boxes-toggle-wrap")), res["title"])
    clear = next(r for r in summary if r["image"] == "invoice_clear.png")
    large = next(r for r in summary if r["image"] == "invoice_large.jpg")
    def by_text(lines):  # repeated texts (two "₹14.75", two "1") are paired by order, top to bottom
        groups = {}
        for t, box in sorted(lines, key=lambda x: x[1][1]):
            groups.setdefault(t, []).append(box)
        return groups
    cb, lb = by_text(clear["lines"]), by_text(large["lines"])
    diffs = [max(abs(a / 4 - b) for a, b in zip(l_box, c_box)) for t, boxes in lb.items()
             if len(boxes) == len(cb.get(t, [])) for l_box, c_box in zip(boxes, cb[t])]
    record("downscaled image: large-image boxes / 4 == clear-image boxes (<= 8 px)",
           len(diffs) >= 25 and max(diffs) <= 8, f"{len(diffs)} matched lines, max diff {max(diffs, default=None):.1f} px")
    await c.shot("integration_final")
    print("\nsequential runs:")
    for r in summary:
        print(f"  {r['image']:<20} {r['wall']:>6}s  timings={r['timings']}  rows={r['rows']}")
    print(f"  uvicorn RSS+swap (MB) before/after each run: {rss}")
    record("uvicorn memory (RSS+swap) stable across sequential runs (no growth > 300 MB after warm-up)",
           None not in rss and max(rss[2:]) - rss[1] < 300, f"{rss} MB")
    if SERVER_LOG.exists():
        record("server log: no PaddleOCR failures / tracebacks",
               not any("PaddleOCR failed" in l or "Traceback" in l for l in server_log()))


async def main(mode: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="chrome-profile-", dir=OUT))
    chrome = subprocess.Popen(
        ["google-chrome", "--headless=new", f"--remote-debugging-port={DEBUG_PORT}", f"--user-data-dir={profile}",
         "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--window-size=1280,1000", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        page = None
        for _ in range(50):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json/list"))
                page = next(t for t in targets if t["type"] == "page")
                break
            except Exception:  # noqa: BLE001 - Chrome still starting
                time.sleep(0.2)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=50_000_000) as ws:
            c = CDP(ws)
            reader = asyncio.create_task(c.reader())
            for domain in ("Page", "Runtime", "DOM", "Network"):
                await c.send(f"{domain}.enable")
            await c.send("Emulation.setDeviceMetricsOverride", width=1280, height=1000, deviceScaleFactor=1, mobile=False)
            await {"ui": check_ui, "real": check_real, "integration": check_integration, "smoke": check_smoke}[mode](c)
            errors = [e for e in c.events if e.get("method") == "Runtime.exceptionThrown"]
            record("no uncaught JavaScript exceptions", not errors, str(errors[:1])[:300])
            reader.cancel()
    finally:
        chrome.terminate()
        chrome.wait(10)
        shutil.rmtree(profile, ignore_errors=True)
        shutil.rmtree(OUT / "tmp", ignore_errors=True)
    passed = sum(ok for _, ok, _ in results)
    print(f"\n{passed}/{len(results)} checks passed · screenshots in {OUT.relative_to(ROOT)}/")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("ui", "real", "integration", "smoke"):
        sys.exit(__doc__)
    sys.exit(asyncio.run(main(sys.argv[1])))
