"""DSPy field extraction over OCR text (PLAN.md §15).

OCR text + requested fields -> dspy.Predict(FieldExtractionSignature) -> local Ollama
(qwen2.5:3b) -> normalized values for exactly the requested fields.
"""

import logging
import re
import threading
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

import dspy

from backend.config import settings
from backend.fields import FieldSpec

logger = logging.getLogger(__name__)

EMPTY_MARKERS = {"", "n/a", "na", "null", "none", "not found", "not present", "not available", "-", "unknown"}
MULTI_SEPARATOR = "|"

# A value the LLM returned that is just a number (candidate for format restoration).
PLAIN_NUMBER = re.compile(r"^\d[\d,]*(?:\.\d+)?$")
# Number tokens in OCR text, optionally prefixed by a currency marker. Tokens glued to
# letters/digits (GSTIN "29ABCDE...", "1L", "5kg") are excluded by the look-arounds.
OCR_NUMBER = re.compile(r"(?<![\w.,])(₹|Rs\.?|INR|\$|€|£)?\s?(\d[\d,]*(?:\.\d+)?)(?![\w])")


class FieldExtractionSignature(dspy.Signature):
    """Extract the requested fields from noisy OCR text of an invoice, bill, receipt or document.
    Return a value for EVERY key in requested_fields and for NO other keys.
    Use only information present in the text. If a field is not present, use an empty string.
    Do not calculate or invent values. Copy values exactly as written, except amounts:
    remove the currency symbol but keep every digit and decimal exactly as written
    (e.g. '₹724.50' -> '724.50'). Currency symbols such as ₹ may be misread by OCR
    as 'R', 'Rs', 'Z' or '?'. For list-type fields, separate values with ' | '."""

    ocr_text: str = dspy.InputField(desc="Raw OCR text, one detected line per row")
    requested_fields: dict[str, str] = dspy.InputField(desc="Fields to extract: key -> what the field means")
    # Not `values`: Prediction is dict-like (.values is a method) and the 3B model echoed
    # the inputs with that name, breaking ChatAdapter parsing (Phase 3 finding).
    extracted_values: dict[str, str] = dspy.OutputField(
        desc="Same keys as requested_fields -> extracted value, or empty string if absent"
    )


class ExtractionError(RuntimeError):
    """LLM extraction failed. kind: unavailable | timeout | parse | unknown."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


@dataclass
class ExtractionResult:
    values: dict                                   # key -> str | list[str] | None, request order
    raw: dict                                      # the LLM's dict, as returned
    adjustments: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- normalization


def _clean(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        value = f" {MULTI_SEPARATOR} ".join(str(v) for v in value)
    text = str(value).strip()
    return None if text.lower() in EMPTY_MARKERS else text


def _as_decimal(number: str) -> Decimal | None:
    try:
        return Decimal(number.replace(",", ""))
    except InvalidOperation:
        return None


def restore_number_format(value: str, ocr_text: str) -> tuple[str, str | None, str | None]:
    """Copy the OCR formatting of a plain-number value, if that is unambiguous.

    Returns (value, ocr_token_used or None, warning or None). Only formatting is copied
    from the OCR text (e.g. '724.5' -> '724.50'); values are never computed or changed.
    """
    if not PLAIN_NUMBER.match(value):
        return value, None, None
    target = _as_decimal(value)
    if target is None:
        return value, None, None

    matches: dict[str, str] = {}  # number formatting -> full OCR token (with currency)
    for m in OCR_NUMBER.finditer(ocr_text):
        number = m.group(2)
        if _as_decimal(number) == target:
            matches.setdefault(number, m.group(0).strip())

    if len(matches) == 1:
        number, token = next(iter(matches.items()))
        return number, (token if number != value else None), None
    if not matches:
        return value, None, f"value {value!r} not found in OCR text"
    return value, None, f"value {value!r} matches several formats in OCR text {sorted(matches)}; kept as returned"


def _appears_in(value: str, ocr_text: str) -> bool:
    squash = lambda s: re.sub(r"\s+", "", s).lower()  # noqa: E731
    return squash(value) in squash(ocr_text)


def normalize_values(raw: dict | None, fields: list[FieldSpec], ocr_text: str) -> ExtractionResult:
    """Enforce 'requested fields only', in request order, and restore number formatting."""
    raw = dict(raw or {})
    by_lower = {str(k).strip().lower(): v for k, v in raw.items()}
    requested = {f.key for f in fields}
    result = ExtractionResult(values={}, raw=raw)

    extra = [k for k in raw if str(k).strip().lower() not in requested]
    if extra:
        logger.debug("LLM returned unrequested keys (dropped): %s", extra)

    for spec in fields:
        value = _clean(raw[spec.key] if spec.key in raw else by_lower.get(spec.key))
        if value is None:
            result.values[spec.key] = None
            continue

        items = [v.strip() for v in value.split(MULTI_SEPARATOR) if v.strip()] if spec.multi else [value]
        fixed_items = []
        for item in items:
            fixed, token, warning = restore_number_format(item, ocr_text)
            if token:
                result.adjustments.append({"field": spec.key, "llm_value": item, "ocr_token": token, "value": fixed})
            if warning:
                result.warnings.append(f"{spec.key}: {warning}")
            elif not PLAIN_NUMBER.match(fixed) and not _appears_in(fixed, ocr_text):
                result.warnings.append(f"{spec.key}: value {fixed!r} not found verbatim in OCR text")
            fixed_items.append(fixed)

        if spec.multi:
            result.values[spec.key] = fixed_items or None
        else:
            result.values[spec.key] = fixed_items[0]
    return result


# ---------------------------------------------------------------- LLM call


def _classify(exc: BaseException) -> tuple[str, str]:
    chain, e = [], exc
    while e is not None and len(chain) < 6:
        chain.append(e)
        e = e.__cause__ or e.__context__
    names = " ".join(type(x).__name__ for x in chain)
    text = " ".join(str(x) for x in chain).lower()
    if "timeout" in names.lower() or "timed out" in text:
        return "timeout", f"LLM timed out after {settings.llm_timeout_seconds} s"
    if "notfound" in names.lower() and "model" in text:
        return "unavailable", f"LLM model not available in Ollama ({settings.llm_model}). Run: ollama pull"
    if any(n in names for n in ("LMTransportError", "APIConnectionError", "ConnectError", "ConnectionError")):
        return "unavailable", f"LLM unavailable: could not connect to Ollama at {settings.llm_api_base}. Is `ollama serve` running?"
    if "AdapterParseError" in names:
        return "parse", "Could not parse the LLM output"
    return "unknown", "LLM extraction failed (details in the server log)"


class Extractor:
    """Holds one dspy.LM and one dspy.Predict module; safe to share across threads."""

    def __init__(self, api_base: str | None = None, timeout: int | None = None, cache: bool | None = None):
        self.lm = dspy.LM(
            settings.llm_model,
            api_base=api_base or settings.llm_api_base,
            api_key=settings.llm_api_key or "",
            temperature=0.0,
            max_tokens=1000,
            timeout=timeout or settings.llm_timeout_seconds,
            num_retries=settings.llm_num_retries,
            cache=settings.llm_cache if cache is None else cache,
        )
        self.predict = dspy.Predict(FieldExtractionSignature)

    def extract(self, ocr_text: str, fields: list[FieldSpec]) -> ExtractionResult:
        requested = {f.key: f.hint + (" (list every value)" if f.multi else "") for f in fields}
        try:
            # dspy.context is per-thread, unlike changing global dspy.configure from worker threads.
            with dspy.context(lm=self.lm):
                prediction = self.predict(ocr_text=ocr_text, requested_fields=requested)
            raw = prediction.extracted_values
        except Exception as exc:  # noqa: BLE001 - classified and re-raised as ExtractionError
            kind, message = _classify(exc)
            logger.warning("DSPy extraction failed (%s): %s", kind, exc)
            raise ExtractionError(kind, message) from exc
        if not isinstance(raw, dict):
            raise ExtractionError("parse", "LLM output was not a key/value object")
        return normalize_values(raw, fields, ocr_text)


_extractor: Extractor | None = None
_extractor_lock = threading.Lock()


def get_extractor() -> Extractor:
    global _extractor
    if _extractor is None:
        with _extractor_lock:
            if _extractor is None:
                _extractor = Extractor()
    return _extractor
