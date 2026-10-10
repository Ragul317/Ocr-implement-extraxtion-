"""HTTP API tests (fast). The pipeline is replaced by a stub via dependency override, except
where the real pipeline rejects input before OCR (invalid/unsupported images: no model load).
A real PaddleOCR + unreachable-LLM end-to-end test is marked slow.
"""

import json

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app, get_pipeline
from backend.schemas import PipelineError, PipelineResult
from tests.conftest import SAMPLES, image_bytes

PNG = image_bytes((120, 60))
DEV_ORIGIN = "http://localhost:8000"


class StubPipeline:
    def __init__(self, result=None, error=None):
        self.calls, self.result, self.error = [], result, error

    def __call__(self, data, keys):
        self.calls.append((data, keys))
        if self.error:
            raise self.error
        return self.result or PipelineResult(success=True, requested_fields=keys or [],
                                             extracted={k: None for k in keys or []})


def make_client(stub=None, **app_kwargs):
    app = create_app(preload_ocr=False, **app_kwargs)
    if stub is not None:
        app.dependency_overrides[get_pipeline] = lambda: stub
    return TestClient(app, raise_server_exceptions=False)


def upload(client, data=PNG, name="invoice.png", mime="image/png", fields=None):
    form = None if fields is None else {"fields": fields}
    files = None if data is None else {"file": (name, data, mime)}
    return client.post("/api/extract", files=files, data=form)


# 1
def test_health():
    r = make_client().get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["ocr_loaded"] in (True, False)
    assert "llm_reachable" not in body  # Ollama is not contacted by default


def test_health_check_llm_unreachable(monkeypatch):
    import backend.main as main

    def boom(*a, **k):
        raise main.httpx.ConnectError("refused")

    monkeypatch.setattr(main.httpx, "get", boom)
    body = make_client().get("/health", params={"check_llm": "true"}).json()
    assert body["llm_reachable"] is False and body["status"] == "ok"


def test_fields_catalog():
    body = make_client().get("/api/fields").json()
    assert [f["key"] for f in body["fields"]][:2] == ["invoice_number", "vendor_name"]
    assert len(body["fields"]) == 10 and body["max_fields"] >= 10
    assert "hint" not in body["fields"][0]


# 2
def test_missing_file():
    stub = StubPipeline()
    r = upload(make_client(stub), data=None, fields="Total")
    assert r.status_code == 400
    assert r.json()["error"] == {"stage": "request", "kind": "NO_FILE", "message": "No image uploaded (form field 'file')."}
    assert stub.calls == []


# 3
def test_empty_file():
    stub = StubPipeline()
    r = upload(make_client(stub), data=b"")
    assert r.status_code == 400 and r.json()["error"]["kind"] == "EMPTY_FILE"
    assert stub.calls == []


def test_too_large():
    stub = StubPipeline()
    r = upload(make_client(stub, max_upload_bytes=1000), data=b"\x00" * 1001)
    assert r.status_code == 413 and r.json()["error"]["kind"] == "FILE_TOO_LARGE"
    assert stub.calls == []


# 4 (real pipeline: preprocessing rejects it before OCR, so no model is loaded)
def test_invalid_image():
    r = upload(make_client(), data=b"this is not an image", name="fake.jpg", mime="image/jpeg", fields="Total")
    assert r.status_code == 422
    body = r.json()
    assert body["success"] is False and body["error"]["stage"] == "preprocess"
    assert body["error"]["kind"] == "invalid_image" and body["requested_fields"] == ["total"]


# 5
@pytest.mark.parametrize("name, mime", [("notes.txt", "text/plain"), ("doc.pdf", "application/pdf"),
                                        ("anim.gif", "image/gif")])
def test_unsupported_type_by_name_or_mime(name, mime):
    stub = StubPipeline()
    r = upload(make_client(stub), name=name, mime=mime)
    assert r.status_code == 415 and r.json()["error"]["kind"] == "UNSUPPORTED_FILE_TYPE"
    assert stub.calls == []


def test_unsupported_format_detected_from_content():
    # GIF bytes disguised as .png: passes the name/MIME check, rejected by real preprocessing.
    r = upload(make_client(), data=image_bytes(fmt="GIF"), name="disguised.png")
    assert r.status_code == 415 and r.json()["error"]["kind"] == "unsupported_format"


# 6
def test_valid_image_reaches_pipeline_with_fields():
    stub = StubPipeline()
    r = upload(make_client(stub), fields="Invoice Number")
    assert r.status_code == 200
    (data, keys), = stub.calls
    assert data == PNG and keys == ["invoice_number"]


def test_fields_omitted_means_defaults():
    stub = StubPipeline()
    upload(make_client(stub))
    assert stub.calls[0][1] is None  # pipeline applies the default-checked fields


# 7
@pytest.mark.parametrize(
    "fields, expected",
    [
        ("Invoice Number,Customer Name,GST Number,Total", ["invoice_number", "customer_name", "gst_number", "total"]),
        ('["total", "invoice_number"]', ["total", "invoice_number"]),
        (["Total", "Date"], ["total", "date"]),  # repeated form parts
    ],
)
def test_multiple_fields_in_order(fields, expected):
    stub = StubPipeline()
    assert upload(make_client(stub), fields=fields).status_code == 200
    assert stub.calls[0][1] == expected


# 8
def test_custom_fields_accepted():
    stub = StubPipeline()
    upload(make_client(stub), fields="Invoice Number, Due Date, PAN No.")
    assert stub.calls[0][1] == ["invoice_number", "due_date", "pan_no"]


@pytest.mark.parametrize("fields, kind", [("[]", "NO_FIELDS_SELECTED"), ("Total, ###", "INVALID_FIELDS"),
                                          ("[not json", "INVALID_FIELDS")])
def test_invalid_fields(fields, kind):
    r = upload(make_client(), fields=fields)  # real pipeline: rejected before any image work
    assert r.status_code == 400 and r.json()["error"]["kind"] == kind


# 9
def test_pipeline_result_returned_unchanged(recorded_ocr_lines):
    result = PipelineResult(
        success=True,
        ocr={"text": "Total 619.50", "raw_text": "Total\n619.50", "lines": recorded_ocr_lines[:2],
             "line_count": 2, "average_confidence": 0.99},
        requested_fields=["total"],
        fields=[{"key": "total", "label": "Total", "multi": False, "custom": False}],
        extracted={"total": "619.50"},
        timings_ms={"ocr": 1, "extraction": 2, "total": 3},
    )
    r = upload(make_client(StubPipeline(result)), fields="Total")
    assert r.status_code == 200
    assert r.json() == json.loads(json.dumps(result.model_dump(mode="json")))


# 10
def test_dspy_error_keeps_ocr_and_is_safe(recorded_ocr_lines):
    result = PipelineResult(
        success=False,
        ocr={"text": "INV-1001", "raw_text": "INV-1001", "lines": recorded_ocr_lines[:1], "line_count": 1,
             "average_confidence": 1.0},
        requested_fields=["invoice_number"],
        extracted={},
        error=PipelineError(stage="dspy", kind="unavailable",
                            message="LLM unavailable: could not connect to Ollama at http://localhost:11434."),
    )
    r = upload(make_client(StubPipeline(result)), fields="Invoice Number")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False and body["error"]["stage"] == "dspy"
    assert body["ocr"]["line_count"] == 1 and body["ocr"]["lines"][0]["box"]


def test_ocr_error_maps_to_500():
    result = PipelineResult(success=False, error=PipelineError(stage="ocr", kind="ocr_failed", message="x"))
    assert upload(make_client(StubPipeline(result))).status_code == 500


# 11
def test_unexpected_exception_hides_traceback():
    stub = StubPipeline(error=RuntimeError("secret internal detail /home/user/x.py line 42"))
    r = upload(make_client(stub), fields="Total")
    assert r.status_code == 500
    text = r.text
    assert "Traceback" not in text and "secret internal detail" not in text
    assert r.json()["error"] == {"stage": "internal", "kind": "INTERNAL_ERROR",
                                 "message": "Unexpected server error. Please try again."}


def test_malformed_request_uses_same_shape():
    r = make_client(StubPipeline()).post("/api/extract", data={"file": "not-a-file"})
    assert r.status_code == 400 and r.json()["error"]["stage"] in ("request",)


# 12
def test_cors_preflight_allowed_origin():
    r = make_client().options("/api/extract", headers={
        "Origin": DEV_ORIGIN, "Access-Control-Request-Method": "POST"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == DEV_ORIGIN


def test_cors_simple_request_and_disallowed_origin():
    client = make_client()
    assert client.get("/health", headers={"Origin": DEV_ORIGIN}).headers["access-control-allow-origin"] == DEV_ORIGIN
    r = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_openapi_docs_available():
    client = make_client()
    schema = client.get("/openapi.json").json()
    assert "/api/extract" in schema["paths"] and "/health" in schema["paths"]
    assert client.get("/docs").status_code == 200 and client.get("/redoc").status_code == 200


@pytest.mark.slow
def test_real_ocr_with_llm_unreachable_over_http():
    """Real PaddleOCR through HTTP; LLM pointed at a closed port -> OCR preserved, 200, success=false."""
    from backend.dspy_service import Extractor
    from backend.pipeline import run_pipeline

    dead = Extractor(api_base="http://127.0.0.1:9", timeout=5, cache=False)
    stub_dep = lambda data, keys: run_pipeline(data, keys, extractor=dead)  # noqa: E731
    r = upload(make_client(stub_dep), data=(SAMPLES / "invoice_clear.png").read_bytes(), fields="Invoice Number, Total")
    assert r.status_code == 200
    body = r.json()
    assert body["error"]["stage"] == "dspy" and body["error"]["kind"] == "unavailable"
    assert body["ocr"]["line_count"] >= 25 and "INV-1001" in body["ocr"]["text"]


# Phase 6: the frontend is served by FastAPI on the same origin as the API.
def test_frontend_served_at_root():
    client = make_client()
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "OCR Extractor" in page.text and 'src="app.js"' in page.text
    js = client.get("/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert "/api/extract" in js.text and "REQUEST_TIMEOUT_MS = 210_000" in js.text
    assert client.get("/style.css").status_code == 200


def test_api_routes_win_over_static_mount():
    client = make_client()
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/api/fields").status_code == 200
    assert client.get("/docs").status_code == 200


# ---------------------------------------------------------------- Phase 7: canonical browser request
# The web page sends: one `file` part + one `fields` part per selected field KEY, in display order.

CUSTOM_LABELS = ["Vendor Address", "Payment Reference", "CGST Amount", "Purchase Order Number"]
CUSTOM_KEYS = ["vendor_address", "payment_reference", "cgst_amount", "purchase_order_number"]


def browser_multipart(file_name: str, file_bytes: bytes, mime: str, keys: list[str]) -> tuple[bytes, str]:
    """Byte-for-byte the multipart shape Chrome produces for buildExtractForm() in frontend/app.js."""
    boundary = "----WebKitFormBoundaryP7canonical"
    parts = [(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{file_name}"\r\n'
              f"Content-Type: {mime}\r\n\r\n").encode() + file_bytes + b"\r\n"]
    parts += [f'--{boundary}\r\nContent-Disposition: form-data; name="fields"\r\n\r\n{k}\r\n'.encode() for k in keys]
    return b"".join(parts) + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def test_canonical_browser_multipart_request():
    stub = StubPipeline()
    keys = ["invoice_number", "gst_number", "total", *CUSTOM_KEYS]
    image = (SAMPLES / "invoice_clear.png").read_bytes()
    body, content_type = browser_multipart("invoice_clear.png", image, "image/png", keys)
    r = make_client(stub).post("/api/extract", content=body, headers={"Content-Type": content_type})
    assert r.status_code == 200
    (data, received), = stub.calls
    assert data == image                      # file bytes unchanged
    assert received == keys                   # every key, in order, custom ones included
    assert r.json()["requested_fields"] == keys


def test_frontend_builds_canonical_request():
    """Locks the browser side: app.js must send repeated `fields` parts (keys), not JSON or labels."""
    import re
    from pathlib import Path

    js = (Path(__file__).resolve().parent.parent / "frontend" / "app.js").read_text(encoding="utf-8")
    fn = re.search(r"function buildExtractForm\(file, keys\) \{(.*?)\n\}", js, re.S)
    assert fn, "buildExtractForm(file, keys) not found in frontend/app.js"
    body = fn.group(1)
    assert 'form.append("file", file, file.name);' in body
    assert 'for (const key of keys) form.append("fields", key);' in body
    assert "JSON.stringify" not in body
    assert "buildExtractForm(state.file, keys)" in js


def test_all_predefined_and_custom_fields_roundtrip(recorded_ocr_lines):
    """All 10 predefined + 4 custom fields through HTTP and the REAL pipeline logic (field resolution,
    order, schema); only the OCR engine and the LLM are stubbed."""
    from backend.fields import PREDEFINED_FIELDS
    from backend.pipeline import run_pipeline
    from tests.test_pipeline_unit import StubExtractor, StubOcr

    predefined = [f.key for f in PREDEFINED_FIELDS]
    keys = predefined + CUSTOM_KEYS
    ext = StubExtractor({"invoice_number": "INV-1001", "total": "619.5", "cgst_amount": ""})
    pipeline = lambda data, k: run_pipeline(data, k, extractor=ext, run_ocr=StubOcr(recorded_ocr_lines))  # noqa: E731
    body, content_type = browser_multipart("invoice_clear.png", (SAMPLES / "invoice_clear.png").read_bytes(),
                                           "image/png", keys)
    r = make_client(pipeline).post("/api/extract", content=body, headers={"Content-Type": content_type})
    assert r.status_code == 200
    res = r.json()
    assert res["success"] is True and res["error"] is None
    assert res["requested_fields"] == keys                                   # order preserved
    assert ext.calls[0][1] == keys                                           # reached the extractor in order
    assert list(res["extracted"]) == keys                                    # one value per field, same order
    # Backend labels are derived from keys ("Cgst Amount"); the page shows the typed label instead.
    assert [f["label"] for f in res["fields"][-4:]] == ["Vendor Address", "Payment Reference", "Cgst Amount",
                                                        "Purchase Order Number"]
    assert [f["custom"] for f in res["fields"]] == [False] * 10 + [True] * 4
    assert res["extracted"]["invoice_number"] == "INV-1001"
    assert res["extracted"]["total"] == "619.50"                             # OCR formatting restored
    assert res["extracted"]["cgst_amount"] is None and res["extracted"]["vendor_address"] is None
    assert res["ocr"]["line_count"] == 29 and res["ocr"]["lines"][0]["box"] == recorded_ocr_lines[0]["box"]
    assert res["ocr"]["lines"][0]["confidence"] == recorded_ocr_lines[0]["confidence"]
    assert "INV-1001" in res["ocr"]["raw_text"]


def test_no_fields_parts_means_defaults_but_ui_never_sends_that():
    """With the repeated-parts format, zero parts = 'not sent' = default fields (documented)."""
    stub = StubPipeline()
    body, content_type = browser_multipart("a.png", PNG, "image/png", [])
    make_client(stub).post("/api/extract", content=body, headers={"Content-Type": content_type})
    assert stub.calls[0][1] is None


def test_comma_in_label_is_why_keys_are_canonical():
    """A label with a comma is split by the backend; the key the browser sends is not."""
    stub = StubPipeline()
    upload(make_client(stub), fields="Address, Line 1")
    assert stub.calls[0][1] == ["address", "line_1"]
    upload(make_client(stub), fields="address_line_1")
    assert stub.calls[1][1] == ["address_line_1"]
