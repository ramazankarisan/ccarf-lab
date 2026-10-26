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

    description: str
    quantity: Money  # Decimal, not int: 1.5 hours of work is a quantity too
    unit_price: Money
    amount: Money


class Invoice(BaseModel):
    # An unknown key is an error, not silently dropped: a typo like "vendor_taxid"
    # would otherwise look like a missing tax ID.
    model_config = ConfigDict(extra="forbid")

    vendor_name: str
    invoice_number: str
    invoice_date: date  # ISO 8601 out, whatever format the document used
    vendor_tax_id: str | None  # required but nullable: the field that gets fabricated
    due_date: date | None  # required but nullable
    currency: CurrencyCode
    category: Category
    category_detail: str | None  # what "other" means; null for every other category
    line_items: list[LineItem]  # may be empty: a terse receipt lists no items
    stated_subtotal: Money | None  # net amount as printed; null if none printed
    stated_tax: Money | None  # tax amount as printed; null if none printed
    stated_total: Money | None  # the total as printed, not a sum; null if none printed

    @model_validator(mode="after")
    def detail_only_for_other(self) -> Self:
        if self.category is Category.OTHER and not (self.category_detail or "").strip():
            raise ValueError("category 'other' needs a category_detail")
        if self.category is not Category.OTHER and self.category_detail is not None:
            raise ValueError("category_detail must be null unless category is 'other'")
        return self
