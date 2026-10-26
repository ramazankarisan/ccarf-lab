"""The invoice schema: what a correct extraction looks like, before any model call.

Required is not the same as non-null. A field like vendor_tax_id has no default,
so an extraction must always contain it, but null is a valid value. The model has
to address every field, and "not in the document" has an honest answer (null)
instead of a made-up tax ID.

Money is Decimal, never float: 129.99 as a float is 129.98999999999999488...

Subtotal, tax and total are extracted separately, exactly as printed. The model
never adds them up or "fixes" them: Step 3's code compares them, so an invoice
with VAT on top is not mistaken for one whose numbers don't add up.
"""

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    quantity: Decimal  # Decimal, not int: 1.5 hours of work is a quantity too
    unit_price: Decimal
    amount: Decimal


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
    stated_subtotal: Decimal | None  # net amount as printed; null if none printed
    stated_tax: Decimal | None  # tax amount as printed; null if none printed
    stated_total: Decimal | None  # the total as printed, not a sum; null if none printed

    @model_validator(mode="after")
    def detail_only_for_other(self) -> Self:
        if self.category is Category.OTHER and not (self.category_detail or "").strip():
            raise ValueError("category 'other' needs a category_detail")
        if self.category is not Category.OTHER and self.category_detail is not None:
            raise ValueError("category_detail must be null unless category is 'other'")
        return self
