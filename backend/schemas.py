"""Pydantic models for the pipeline result (PLAN.md §19 Phase 4); reused by the API in Phase 5."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OcrLine(BaseModel):
    text: str = Field(description="Text exactly as PaddleOCR recognised it")
    confidence: float
    polygon: list[list[int]] = Field(description="4 points [x, y] in original-image pixels")
    box: list[int] = Field(description="[x_min, y_min, x_max, y_max] in original-image pixels")


class OcrResult(BaseModel):
    text: str = Field(description="Reading-order text (rows grouped), as sent to DSPy")
    raw_text: str = Field(description="PaddleOCR output joined in detection order, unmodified")
    lines: list[OcrLine]
    line_count: int
    average_confidence: float | None
    low_confidence_dropped: int = 0


class ImageInfo(BaseModel):
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    resized: bool
    format: str


class FieldInfo(BaseModel):
    key: str
    label: str
    multi: bool
    custom: bool


class ValueAdjustment(BaseModel):
    field: str
    llm_value: str
    ocr_token: str
    value: str


class PipelineError(BaseModel):
    # request/internal are produced by the HTTP layer (backend/main.py), the rest by the pipeline.
    stage: Literal["request", "fields", "preprocess", "ocr", "dspy", "internal"]
    kind: str
    message: str


class PipelineResult(BaseModel):
    # The pipeline fills this step by step; validate each assignment (dicts -> models).
    model_config = ConfigDict(validate_assignment=True)

    success: bool
    image: ImageInfo | None = None
    ocr: OcrResult | None = None
    requested_fields: list[str] = []
    fields: list[FieldInfo] = []
    extracted: dict[str, str | list[str] | None] = {}
    value_adjustments: list[ValueAdjustment] = []
    warnings: list[str] = []
    timings_ms: dict[str, int] = {}
    error: PipelineError | None = None
