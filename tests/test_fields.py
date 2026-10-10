import pytest

from backend.fields import (
    DEFAULT_FIELD_KEYS,
    PREDEFINED_FIELDS,
    FieldValidationError,
    label_for_key,
    parse_fields_input,
    resolve_fields,
)


def test_one_field():
    fields = resolve_fields(["total"])
    assert [f.key for f in fields] == ["total"]
    assert fields[0].custom is False


def test_order_preserved_and_duplicates_removed():
    assert [f.key for f in resolve_fields(["total", "date", "total"])] == ["total", "date"]


def test_all_predefined_fields_allowed():
    keys = [f.key for f in PREDEFINED_FIELDS]
    assert len(keys) == 10
    assert [f.key for f in resolve_fields(keys)] == keys


def test_custom_field_resolution():
    (field,) = resolve_fields(["due_date"])
    assert field.custom and field.label == "Due Date" and not field.multi
    assert "Due Date" in field.hint


def test_multi_valued_predefined():
    assert {f.key for f in resolve_fields(["product", "quantity"]) if f.multi} == {"product", "quantity"}


def test_none_means_defaults():
    assert [f.key for f in resolve_fields(None)] == DEFAULT_FIELD_KEYS


@pytest.mark.parametrize(
    "keys, code",
    [
        ([], "NO_FIELDS_SELECTED"),
        (["Bad Key!"], "INVALID_FIELDS"),
        (["1abc"], "INVALID_FIELDS"),
        ([1, 2], "INVALID_FIELDS"),
        ("total", "INVALID_FIELDS"),
        ([f"f{i}" for i in range(21)], "INVALID_FIELDS"),
    ],
)
def test_invalid_field_lists(keys, code):
    with pytest.raises(FieldValidationError) as exc:
        resolve_fields(keys, max_fields=20)
    assert exc.value.code == code


def test_parse_fields_input_formats():
    assert parse_fields_input(['["invoice_number","total"]']) == ["invoice_number", "total"]
    assert parse_fields_input(["Invoice Number,Customer Name, GST Number ,Total"]) == [
        "invoice_number", "customer_name", "gst_number", "total"]
    assert parse_fields_input(["Total", "invoice_number"]) == ["total", "invoice_number"]  # repeated parts
    assert parse_fields_input(["PAN No., Due Date"]) == ["pan_no", "due_date"]
    assert parse_fields_input(None) is None
    assert parse_fields_input([""]) == [] and parse_fields_input(["[]"]) == []


@pytest.mark.parametrize("value", ["[not json", '["a", 1]', "123abc", "Total, ###"])
def test_parse_fields_input_invalid(value):
    with pytest.raises(FieldValidationError):
        parse_fields_input([value])


def test_parsed_labels_resolve_to_predefined():
    keys = parse_fields_input(["Invoice Number, GST Number, Due Date"])
    specs = resolve_fields(keys)
    assert [(f.key, f.custom) for f in specs] == [("invoice_number", False), ("gst_number", False), ("due_date", True)]
    assert DEFAULT_FIELD_KEYS == [f.key for f in resolve_fields(None)]


def test_public_view_hides_hint():
    assert set(resolve_fields(["total"])[0].public()) == {"key", "label", "multi", "custom"}


def test_label_for_key():
    assert label_for_key("pan_no") == "Pan No"
