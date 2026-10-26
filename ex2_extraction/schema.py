"""The invoice schema: what a correct extraction looks like, before any model call.

Required is not the same as non-null. A field like vendor_tax_id has no default,
so an extraction must always contain it, but null is a valid value. The model has
to address every field, and "not in the document" has an honest answer (null)
instead of a made-up tax ID.

Money is Decimal, never float: 129.99 as a float is 129.98999999999999488...
Its JSON schema is a number or a plain decimal string ("1234.56"), so a German
"1.234,56" or the word "null" break the schema the model sees, not only the parser.

Subtotal, tax and total are extracted separately, exactly as printed. The model
never adds them up or "fixes" them: Step 3's code compares them, so an invoice
with VAT on top is not mistaken for one whose numbers don't add up.
"""

import re
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, WithJsonSchema, model_validator

DECIMAL_PATTERN = r"^-?\d+(\.\d+)?$"


def plain_decimal_string(value: Any) -> Any:
    """Hold strings to the same pattern the schema shows the model. Without this,
    Pydantic would still accept "1E3" or " 12.5" that the schema forbids."""
    if isinstance(value, str) and not re.fullmatch(DECIMAL_PATTERN, value):
        raise ValueError(f"must be a number or a plain decimal string like 1234.56, got {value!r}")
    return value


# Every amount and quantity. The tool schema tells the model the format up front;
# the validator enforces the same rule on whatever comes back.
Money = Annotated[
    Decimal,
    BeforeValidator(plain_decimal_string),
    WithJsonSchema({"anyOf": [{"type": "number"}, {"type": "string", "pattern": DECIMAL_PATTERN}]}),
]

# ISO 4217: a closed format, not a business taxonomy. A pattern lets a CHF invoice
# through as CHF; an enum would force it into EUR or reject it.
CurrencyCode = Annotated[str, Field(pattern=r"^[A-Z]{3}$")]


class Category(StrEnum):
    HARDWARE = "hardware"
    SOFTWARE = "software"
    SERVICES = "services"
    TRAVEL = "travel"
    OTHER = "other"


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(
        description="The item text as printed, or a short summary when the document "
        "describes the item in prose."
    )
    quantity: Money = Field(  # Decimal, not int: 1.5 hours of work is a quantity too
        description="Number of units, hours or days, such as 3 or 1.5."
    )
    unit_price: Money = Field(
        description="Price per unit as printed, without currency symbols or thousands "
        "separators, such as 1234.56."
    )
    amount: Money = Field(description="The line total as printed for this item.")


class Invoice(BaseModel):
    # An unknown key is an error, not silently dropped: a typo like "vendor_taxid"
    # would otherwise look like a missing tax ID.
    model_config = ConfigDict(extra="forbid")

    # Descriptions are the interface the model reads for every field: they end up in
    # the tool's input_schema. Variant A of Step 4 strips them (see json_schema()).
    # None of them may use a vendor, number, product or tax ID from the corpus.
    vendor_name: str = Field(
        description="The seller's name as printed. Not the buyer, guest or bill-to party."
    )
    invoice_number: str = Field(
        description="The seller's invoice or receipt number as printed. Not an order, "
        "booking, purchase order or customer number."
    )
    invoice_date: date = Field(  # ISO 8601 out, whatever format the document used
        description="The date the invoice was issued, as YYYY-MM-DD. When day and month "
        "could be swapped, decide from the document's cues: a US address or US-style "
        "terms mean month first, a European address or language means day first."
    )
    vendor_tax_id: str | None = Field(  # required but nullable: the field that gets fabricated
        description="The seller's tax ID only, without its label (EIN, VAT No, USt-IdNr., "
        "Steuernummer). Never the buyer's or customer's. Null if the seller's is not printed."
    )
    due_date: date | None = Field(
        description="The payment due date as YYYY-MM-DD, only if one is printed. A date "
        "the invoice was paid or settled is not a due date. Null if none is printed."
    )
    currency: CurrencyCode = Field(
        description="The ISO 4217 code of the invoice currency, such as EUR, USD or GBP; "
        "a symbol like €, $ or £ tells which."
    )
    category: Category = Field(
        description="What was bought. hardware: physical equipment such as computers, "
        "devices, network gear and accessories. software: licenses, subscriptions and "
        "hosted software. services: professional work such as consulting, design, "
        "translation, maintenance or support. travel: transport and accommodation, "
        "including meals billed with a stay. other: anything not listed, such as food and "
        "drink for events, venue or equipment hire, or office supplies."
    )
    category_detail: str | None = Field(  # what "other" means; null for every other category
        description="Only when category is other: a short name for the kind of expense, "
        "such as venue hire. Null for every other category."
    )
    line_items: list[LineItem] = Field(  # may be empty: a terse receipt lists no items
        description="Only rows the document prints as separate items with a quantity or "
        "price. A receipt that shows only a total has no line items: use []."
    )
    stated_subtotal: Money | None = Field(
        description="The net amount before tax, only if a line for it is printed. A "
        "document with a single total has a null subtotal. Copy it; never compute it."
    )
    stated_tax: Money | None = Field(
        description="The tax amount (VAT, sales tax, USt.), only if a line for it is "
        "printed, including tax shown as included in the total. Copy it; never compute it."
    )
    stated_total: Money | None = Field(
        description="The final amount including tax, only if printed. When both a net and "
        "a gross total are printed, this is the gross one. Null if no total is printed; "
        "never add up the items."
    )

    @model_validator(mode="after")
    def detail_only_for_other(self) -> Self:
        if self.category is Category.OTHER and not (self.category_detail or "").strip():
            raise ValueError("category 'other' needs a category_detail")
        if self.category is not Category.OTHER and self.category_detail is not None:
            raise ValueError("category_detail must be null unless category is 'other'")
        return self


def strip_descriptions(schema: Any) -> Any:
    """A copy of a JSON schema without "description" annotations. Property names are
    kept: LineItem has a field called description."""
    if isinstance(schema, list):
        return [strip_descriptions(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    stripped = {}
    for key, value in schema.items():
        if key == "properties":
            stripped[key] = {name: strip_descriptions(prop) for name, prop in value.items()}
        elif key != "description":
            stripped[key] = strip_descriptions(value)
    return stripped


def json_schema(*, with_descriptions: bool) -> dict[str, Any]:
    """The Invoice schema for the tool. Without descriptions it is exactly the Step 3
    schema, so Step 4's variant A reproduces the earlier runs."""
    schema = Invoice.model_json_schema()
    return schema if with_descriptions else strip_descriptions(schema)
