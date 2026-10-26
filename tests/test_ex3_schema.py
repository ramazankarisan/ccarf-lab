"""The invoice schema: required-but-nullable fields, Decimal money, the category rule."""

import json
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from ex3_extraction.schema import Category, Invoice, LineItem

NULLABLE = (
    "vendor_tax_id",
    "due_date",
    "category_detail",
    "stated_subtotal",
    "stated_tax",
    "stated_total",
)


def invoice_data(**overrides: Any) -> dict[str, Any]:
    """A valid extraction as JSON-like data; overrides replace single fields."""
    data: dict[str, Any] = {
        "vendor_name": "Nordlicht Office Supplies GmbH",
        "invoice_number": "NL-2026-0042",
        "invoice_date": "2026-10-19",
        "vendor_tax_id": "DE000000000",
        "due_date": "2026-11-18",
        "currency": "EUR",
        "category": "hardware",
        "category_detail": None,
        "line_items": [
            {"description": "Desk lamp", "quantity": 2, "unit_price": "24.50", "amount": "49.00"}
        ],
        "stated_subtotal": "49.00",
        "stated_tax": "9.31",
        "stated_total": "58.31",
    }
    return data | overrides


def test_a_complete_invoice_parses_into_typed_values():
    invoice = Invoice.model_validate(invoice_data())

    assert invoice.invoice_date == date(2026, 10, 19)
    assert invoice.currency == "EUR"
    assert invoice.category is Category.HARDWARE
    assert invoice.line_items[0].amount == Decimal("49.00")


@pytest.mark.parametrize("field", NULLABLE)
def test_nullable_fields_accept_null(field: str):
    invoice = Invoice.model_validate(invoice_data(**{field: None}))

    assert getattr(invoice, field) is None


@pytest.mark.parametrize("field", [*NULLABLE, "vendor_name", "line_items"])
def test_every_field_is_required_even_when_null_is_allowed(field: str):
    data = invoice_data()
    del data[field]

    with pytest.raises(ValidationError, match=field):
        Invoice.model_validate(data)


def test_the_json_schema_marks_nullable_fields_required():
    """Step 2 sends this schema as the tool's input_schema, so the model sees the same rule."""
    schema = Invoice.model_json_schema()

    assert set(schema["required"]) == set(Invoice.model_fields)
    assert {"type": "null"} in schema["properties"]["vendor_tax_id"]["anyOf"]


def test_money_stays_exact():
    item = LineItem.model_validate(
        {"description": "Headphones", "quantity": 1, "unit_price": 129.99, "amount": 129.99}
    )

    assert item.amount == Decimal("129.99")
    assert str(item.amount) == "129.99"


@pytest.mark.parametrize("value", ["NaN", "Infinity", float("nan")])
def test_money_rejects_nan_and_infinity(value: Any):
    with pytest.raises(ValidationError):
        LineItem.model_validate(
            {"description": "x", "quantity": 1, "unit_price": value, "amount": "1.00"}
        )


def test_dates_must_already_be_iso():
    """Turning 19.10.2026 into 2026-10-19 is the extractor's job, not the schema's."""
    with pytest.raises(ValidationError, match="invoice_date"):
        Invoice.model_validate(invoice_data(invoice_date="19.10.2026"))


def test_output_json_uses_iso_dates_and_exact_money():
    dumped = json.loads(Invoice.model_validate(invoice_data()).model_dump_json())

    assert dumped["invoice_date"] == "2026-10-19"
    assert dumped["stated_total"] == "58.31"


def test_a_category_outside_the_enum_is_rejected():
    with pytest.raises(ValidationError, match="category"):
        Invoice.model_validate(invoice_data(category="catering"))


def test_any_iso_currency_code_is_accepted():
    """No enum: a CHF invoice stays CHF instead of being forced into EUR."""
    assert Invoice.model_validate(invoice_data(currency="CHF")).currency == "CHF"


@pytest.mark.parametrize("code", ["eur", "€", "EURO", "US", ""])
def test_currency_must_be_an_iso_code(code: str):
    with pytest.raises(ValidationError, match="currency"):
        Invoice.model_validate(invoice_data(currency=code))


def test_other_needs_a_detail():
    invoice = Invoice.model_validate(invoice_data(category="other", category_detail="catering"))

    assert invoice.category_detail == "catering"


@pytest.mark.parametrize("detail", [None, "", "   "])
def test_other_without_a_detail_is_rejected(detail: str | None):
    with pytest.raises(ValidationError, match="needs a category_detail"):
        Invoice.model_validate(invoice_data(category="other", category_detail=detail))


def test_a_detail_without_other_is_rejected():
    with pytest.raises(ValidationError, match="must be null"):
        Invoice.model_validate(invoice_data(category="hardware", category_detail="lamps"))


def test_line_items_may_be_empty():
    assert Invoice.model_validate(invoice_data(line_items=[])).line_items == []


def test_unknown_keys_are_rejected():
    """A typo like vendor_taxid must fail, not look like a missing tax ID."""
    with pytest.raises(ValidationError, match="Extra inputs"):
        Invoice.model_validate(invoice_data(vendor_taxid="DE000000000"))


# --- Money: the same format rule in the schema and in the parser --------------------

MONEY_FIELDS = ("stated_subtotal", "stated_tax", "stated_total")
ITEM_FIELDS = ("quantity", "unit_price", "amount")
NUMBER_OR_PLAIN_STRING = [
    {"type": "number"},
    {"type": "string", "pattern": r"^-?\d+(\.\d+)?$"},
]


@pytest.mark.parametrize("field", MONEY_FIELDS)
def test_invoice_money_schema_is_a_number_or_a_plain_decimal_string_or_null(field: str):
    schema = Invoice.model_json_schema()["properties"][field]

    assert schema["anyOf"] == [*NUMBER_OR_PLAIN_STRING, {"type": "null"}]


@pytest.mark.parametrize("field", ITEM_FIELDS)
def test_line_item_numbers_have_the_same_schema_without_null(field: str):
    schema = Invoice.model_json_schema()["$defs"]["LineItem"]["properties"][field]

    assert schema["anyOf"] == NUMBER_OR_PLAIN_STRING


@pytest.mark.parametrize("value", ["1.037,58", "null", "1E3", " 12.5", "12.", "€12.50", ""])
def test_strings_outside_the_pattern_are_rejected(value: str):
    """What the earlier baseline runs sent ("1.037,58", "null") and other near misses."""
    with pytest.raises(ValidationError, match="plain decimal string"):
        Invoice.model_validate(invoice_data(stated_total=value))


@pytest.mark.parametrize("value", ["1234.56", "-5.00", "12", 1234.56, 12])
def test_numbers_and_plain_decimal_strings_are_accepted(value: str | float):
    invoice = Invoice.model_validate(invoice_data(stated_total=value))

    assert invoice.stated_total == Decimal(str(value))


def test_line_item_quantities_follow_the_same_rule():
    item = {"description": "Taxi ride", "quantity": "null", "unit_price": 1, "amount": 1}

    with pytest.raises(ValidationError, match=r"line_items\.0\.quantity"):
        Invoice.model_validate(invoice_data(line_items=[item]))
