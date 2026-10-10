"""Field catalog and requested-field validation (Dynamic Field Selection, PLAN.md §6).

The predefined catalog lives only here; the API serves it to the frontend and the
DSPy service uses the hints. Any valid key that is not predefined is a custom field.
"""

import json
import re
from dataclasses import asdict, dataclass

from backend.config import settings

KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,39}$")


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    hint: str
    multi: bool = False
    custom: bool = False
    default: bool = False

    def public(self) -> dict:
        """What callers outside the backend see (the LLM hint stays server-side)."""
        data = asdict(self)
        data.pop("hint")
        data.pop("default")
        return data


PREDEFINED_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("invoice_number", "Invoice Number", "Invoice / bill / receipt number", default=True),
    FieldSpec("vendor_name", "Vendor Name", "Seller, shop or company issuing the document", default=True),
    FieldSpec("customer_name", "Customer Name", "Buyer / billed-to name"),
    FieldSpec("date", "Date", "Invoice/bill date, exactly as written", default=True),
    FieldSpec("gst_number", "GST Number", "GSTIN, a 15-character alphanumeric tax ID (e.g. 29ABCDE1234F1Z5)"),
    FieldSpec("phone_number", "Phone Number", "Phone/mobile number as written"),
    FieldSpec("product", "Product", "Names of purchased items, one per line item", multi=True),
    FieldSpec("quantity", "Quantity", "Quantity for each line item, same order as Product", multi=True),
    FieldSpec("subtotal", "Subtotal", "Amount before tax, without currency symbol"),
    FieldSpec("total", "Total", "Grand total amount, without currency symbol", default=True),
)
PREDEFINED_BY_KEY = {f.key: f for f in PREDEFINED_FIELDS}
DEFAULT_FIELD_KEYS = [f.key for f in PREDEFINED_FIELDS if f.default]


class FieldValidationError(ValueError):
    """Raised for an unusable field list. `code` maps to the API error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def label_for_key(key: str) -> str:
    """`due_date` -> `Due Date`."""
    return " ".join(part.capitalize() for part in key.split("_") if part)


def custom_field(key: str) -> FieldSpec:
    label = label_for_key(key)
    return FieldSpec(key, label, f"Value of '{label}' if present in the document", custom=True)


def resolve_fields(keys: list[str] | None, max_fields: int | None = None) -> list[FieldSpec]:
    """Validate requested keys and resolve them to FieldSpecs, keeping order.

    None -> the default-checked fields. Duplicates are dropped (first wins).
    Unknown but valid keys become custom fields.
    """
    if keys is None:
        keys = DEFAULT_FIELD_KEYS
    if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
        raise FieldValidationError("INVALID_FIELDS", "Fields must be a list of field names.")

    unique = list(dict.fromkeys(k.strip() for k in keys))
    if not unique:
        raise FieldValidationError("NO_FIELDS_SELECTED", "Select at least one field to extract.")

    limit = max_fields if max_fields is not None else settings.max_fields
    if len(unique) > limit:
        raise FieldValidationError("INVALID_FIELDS", f"Too many fields: {len(unique)} (max {limit}).")

    for key in unique:
        if not KEY_PATTERN.match(key):
            raise FieldValidationError(
                "INVALID_FIELDS",
                f"Invalid field name {key[:50]!r}. Use lowercase letters, digits and underscores, starting with a letter.",
            )
    return [PREDEFINED_BY_KEY.get(key) or custom_field(key) for key in unique]


def key_from_label(name: str) -> str:
    """'Invoice Number' -> 'invoice_number', 'PAN No.' -> 'pan_no' (same rule as the frontend, PLAN.md §6)."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def parse_fields_input(values: list[str] | None) -> list[str] | None:
    """Turn the HTTP `fields` form value(s) into field keys, keeping order.

    Accepted (also mixed, and as repeated `fields` form parts):
      - JSON array:      ["invoice_number", "Customer Name"]
      - comma-separated: Invoice Number, Customer Name, gst_number, Total
    Labels and keys are both accepted ("GST Number" == "gst_number").
    None (no `fields` part sent) -> None, i.e. the default fields.
    """
    if values is None:
        return None
    names: list[str] = []
    for raw in values:
        raw = (raw or "").strip()
        if raw.startswith("["):
            try:
                items = json.loads(raw)
            except json.JSONDecodeError:
                raise FieldValidationError("INVALID_FIELDS", "Fields must be a JSON array of field names.") from None
            if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
                raise FieldValidationError("INVALID_FIELDS", "Fields must be a JSON array of field names.")
            names.extend(items)
        else:
            names.extend(raw.split(","))

    keys = []
    for name in names:
        if not name.strip():
            continue
        key = key_from_label(name)
        if not KEY_PATTERN.match(key):
            raise FieldValidationError(
                "INVALID_FIELDS",
                f"Invalid field name {name.strip()[:50]!r}. Use letters, digits and spaces, starting with a letter.",
            )
        keys.append(key)
    return keys
