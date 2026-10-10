"""FastAPI HTTP layer (PLAN.md §13). Thin: request checks + the existing pipeline.

Run:  uvicorn backend.main:app --host 127.0.0.1 --port 8000
Docs: /docs (Swagger UI), /redoc, /openapi.json
"""

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import httpx
from fastapi import Depends, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import ocr_service
from backend.config import PROJECT_ROOT, settings
from backend.fields import KEY_PATTERN, PREDEFINED_FIELDS, FieldValidationError, parse_fields_input
from backend.pipeline import run_pipeline
from backend.schemas import PipelineError, PipelineResult

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backend.api")

FRONTEND_DIR = PROJECT_ROOT / "frontend"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_MIME = {"image/jpeg", "image/png", "image/bmp", "image/x-ms-bmp", "image/webp"}
GENERIC_MIME = {"", "application/octet-stream"}  # some clients send these for any file

PipelineFn = Callable[[bytes, list[str] | None], PipelineResult]


def get_pipeline() -> PipelineFn:
    """Dependency: the processing pipeline (overridden with stubs in API tests)."""
    return run_pipeline


def error_result(stage: str, kind: str, message: str, requested_fields: list[str] | None = None) -> PipelineResult:
    """HTTP-layer errors use the same shape as pipeline results, so clients handle one format."""
    return PipelineResult(success=False, requested_fields=requested_fields or [],
                          error=PipelineError(stage=stage, kind=kind, message=message))


def respond(result: PipelineResult, status: int) -> JSONResponse:
    return JSONResponse(status_code=status, content=result.model_dump(mode="json"))


def status_for(result: PipelineResult) -> int:
    """Map the pipeline outcome to an HTTP status. LLM failures keep OCR output: 200 + success=false."""
    if result.error is None:
        return 200
    stage, kind = result.error.stage, result.error.kind
    if stage == "fields":
        return 400
    if stage == "preprocess":
        return 415 if kind == "unsupported_format" else 422
    if stage == "dspy":
        return 200  # partial success: OCR result is in the body
    return 500


def save_output(filename: str, result: PipelineResult) -> None:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename or "upload").name)[:80]
    path = PROJECT_ROOT / "outputs" / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe}.json"
    try:
        path.write_text(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("Could not save output to %s", path, exc_info=True)


def create_app(preload_ocr: bool | None = None, max_upload_bytes: int | None = None) -> FastAPI:
    preload = settings.ocr_preload if preload_ocr is None else preload_ocr
    upload_limit = max_upload_bytes or settings.max_upload_bytes

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ocr_error = None
        if preload:
            try:
                await asyncio.to_thread(ocr_service.get_engine)
            except ocr_service.OcrError:
                # Keep serving: /health reports it and requests return a clean OCR error.
                logger.exception("PaddleOCR could not be loaded at startup")
                app.state.ocr_error = "PaddleOCR could not be loaded (see server log)"
        yield

    app = FastAPI(
        title="Simple OCR API",
        version="0.5.0",
        description=(
            "Local OCR + field extraction. An uploaded image is preprocessed, read by **PaddleOCR** "
            "(PP-OCRv6_small, CPU) and the requested fields are extracted from the OCR text by **DSPy** "
            "using a local **Ollama** model (qwen2.5:3b). Extraction can take up to ~3 minutes on CPU."
        ),
        lifespan=lifespan,
    )
    app.state.ocr_error = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        problems = "; ".join(f"{'.'.join(str(p) for p in e.get('loc', [])[1:])}: {e.get('msg')}" for e in exc.errors())
        return respond(error_result("request", "INVALID_REQUEST", f"Invalid request: {problems}"[:300]), 400)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return respond(error_result("internal", "INTERNAL_ERROR", "Unexpected server error. Please try again."), 500)

    @app.get("/health", summary="Is the API running?", tags=["system"])
    def health(
        check_llm: bool = Query(False, description="Also check that Ollama answers and the model is pulled "
                                                   "(no model call, ~1 s)."),
    ):
        """Cheap liveness check. Does not run OCR and, by default, does not contact Ollama."""
        body = {
            "status": "ok",
            "ocr_model": settings.ocr_det_model,
            "ocr_loaded": ocr_service._engine is not None,
            "ocr_error": app.state.ocr_error,
            "llm_model": settings.llm_model,
        }
        if check_llm:
            model = settings.llm_model.split("/", 1)[-1]
            try:
                tags = httpx.get(settings.llm_api_base.rstrip("/") + "/api/tags", timeout=2).json()
                body["llm_reachable"] = True
                body["llm_model_available"] = any(m.get("name") == model for m in tags.get("models", []))
            except (httpx.HTTPError, ValueError):
                body["llm_reachable"] = False
                body["llm_model_available"] = False
        return body

    @app.get("/api/fields", summary="Predefined fields for the field selector", tags=["extraction"])
    def list_fields():
        """The predefined field catalog (PLAN.md §6). Any other valid name is accepted as a custom field."""
        return {
            "max_fields": settings.max_fields,
            "custom_key_pattern": KEY_PATTERN.pattern,
            "fields": [{**f.public(), "default": f.default} for f in PREDEFINED_FIELDS],
        }

    @app.post(
        "/api/extract",
        summary="OCR an image and extract the requested fields",
        tags=["extraction"],
        response_model=PipelineResult,
        responses={
            200: {"description": "Processed. `success=false` with `error.stage='dspy'` means OCR succeeded "
                                 "but LLM extraction failed; the OCR result is still included."},
            400: {"model": PipelineResult, "description": "Missing/empty file, or invalid field list"},
            413: {"model": PipelineResult, "description": "Upload larger than MAX_UPLOAD_MB"},
            415: {"model": PipelineResult, "description": "Not a JPG/PNG/BMP/WEBP image"},
            422: {"model": PipelineResult, "description": "Corrupt or undecodable image"},
            500: {"model": PipelineResult, "description": "OCR failure or unexpected server error"},
        },
    )
    def extract(
        file: UploadFile | None = File(None, description="Image: JPG, JPEG, PNG, BMP or WEBP"),
        fields: list[str] | None = Form(
            None,
            description="Fields to extract, in order. **Canonical (used by the web page): one `fields` part "
                        "per field key, repeated** (`fields=invoice_number`, `fields=total`, `fields=pan_no`). "
                        "Also accepted for curl convenience: comma-separated labels/keys "
                        "(`Invoice Number, GST Number, Total`) or a JSON array. Unknown names become custom fields. "
                        "Omit for the default fields.",
        ),
        pipeline: PipelineFn = Depends(get_pipeline),
    ):
        """Runs: validate upload → preprocess (downscale to 2000 px) → PaddleOCR → DSPy/Ollama → result.

        The body is always a `PipelineResult` (also for errors). Sync handler: FastAPI runs it in a
        worker thread, so slow OCR/LLM work does not block other requests such as /health.
        """
        try:
            keys = parse_fields_input(fields)
        except FieldValidationError as exc:
            return respond(error_result("fields", exc.code, exc.message), 400)

        if file is None:
            return respond(error_result("request", "NO_FILE", "No image uploaded (form field 'file').", keys), 400)

        suffix = Path(file.filename or "").suffix.lower()
        mime = (file.content_type or "").split(";")[0].strip().lower()
        if (suffix and suffix not in ALLOWED_EXTENSIONS) or (mime not in ALLOWED_MIME | GENERIC_MIME):
            return respond(error_result("request", "UNSUPPORTED_FILE_TYPE",
                                        "Only JPG, JPEG, PNG, BMP and WEBP images are supported.", keys), 415)

        data = file.file.read(upload_limit + 1)
        if not data:
            return respond(error_result("request", "EMPTY_FILE", "The uploaded file is empty.", keys), 400)
        if len(data) > upload_limit:
            return respond(error_result("request", "FILE_TOO_LARGE",
                                        f"File exceeds {upload_limit // (1024 * 1024)} MB.", keys), 413)

        start = time.perf_counter()
        result = pipeline(data, keys)
        del data
        status = status_for(result)
        logger.info("extract %r: %d bytes, fields=%s -> %d (%s) in %.1fs", Path(file.filename or "").name,
                    file.size or 0, result.requested_fields, status,
                    result.error.stage if result.error else "ok", time.perf_counter() - start)
        if settings.save_outputs:
            save_output(file.filename, result)
        return respond(result, status)

    # The one-page frontend (Phase 6), same origin as the API. Mounted last so API routes win.
    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    return app


app = create_app()
