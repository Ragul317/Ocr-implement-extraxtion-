"""Application settings loaded from the project's .env file.

All configuration (including any API key) comes from environment variables.
Nothing secret is hardcoded here.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Real environment variables win over values in .env.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _get_int(name: str, default: int) -> int:
    value = _get_str(name)
    return int(value) if value else default


def _get_float(name: str, default: float) -> float:
    value = _get_str(name)
    return float(value) if value else default


def _get_bool(name: str, default: bool) -> bool:
    value = _get_str(name).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # LLM (DSPy via LiteLLM)
    llm_model: str
    llm_api_base: str
    llm_api_key: str
    llm_timeout_seconds: int
    llm_num_retries: int
    llm_cache: bool

    # OCR
    ocr_lang: str
    ocr_device: str
    ocr_det_model: str | None
    ocr_rec_model: str | None
    ocr_min_confidence: float
    ocr_max_side: int

    # Upload / request limits
    max_upload_mb: int
    max_image_pixels: int
    request_timeout_seconds: int

    # Field selection
    max_fields: int

    # HTTP API
    cors_allow_origins: list[str]
    ocr_preload: bool

    # Misc
    save_outputs: bool
    log_level: str

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def __repr__(self) -> str:
        # Never print the API key.
        masked = "***" if self.llm_api_key else "(empty)"
        return f"Settings(llm_model={self.llm_model!r}, llm_api_base={self.llm_api_base!r}, llm_api_key={masked})"


def load_settings() -> Settings:
    return Settings(
        llm_model=_get_str("LLM_MODEL", "ollama_chat/qwen2.5:3b"),
        llm_api_base=_get_str("LLM_API_BASE", "http://localhost:11434"),
        llm_api_key=_get_str("LLM_API_KEY"),
        llm_timeout_seconds=_get_int("LLM_TIMEOUT_SECONDS", 180),
        llm_num_retries=_get_int("LLM_NUM_RETRIES", 0),
        llm_cache=_get_bool("LLM_CACHE", True),
        ocr_lang=_get_str("OCR_LANG", "en"),
        ocr_device=_get_str("OCR_DEVICE", "cpu"),
        # Empty means "let PaddleOCR choose"; the default is the Phase 2 choice.
        ocr_det_model=_get_str("OCR_DET_MODEL", "PP-OCRv6_small_det") or None,
        ocr_rec_model=_get_str("OCR_REC_MODEL", "PP-OCRv6_small_rec") or None,
        ocr_min_confidence=_get_float("OCR_MIN_CONFIDENCE", 0.5),
        ocr_max_side=_get_int("OCR_MAX_SIDE", 2000),
        max_upload_mb=_get_int("MAX_UPLOAD_MB", 10),
        max_image_pixels=_get_int("MAX_IMAGE_PIXELS", 40_000_000),
        request_timeout_seconds=_get_int("REQUEST_TIMEOUT_SECONDS", 210),
        max_fields=_get_int("MAX_FIELDS", 20),
        cors_allow_origins=[o.strip() for o in _get_str(
            "CORS_ALLOW_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if o.strip()],
        ocr_preload=_get_bool("OCR_PRELOAD", True),
        save_outputs=_get_bool("SAVE_OUTPUTS", False),
        log_level=_get_str("LOG_LEVEL", "INFO").upper(),
    )


settings = load_settings()
