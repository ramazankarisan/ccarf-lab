"""Semantic checks: do the printed numbers agree with each other?

The schema checks types and Pydantic checks cross-field rules; these checks compare
values that are each valid on their own. All money is compared in cents, with no
tolerance beyond that. A null never triggers a check: "not printed" is a valid
answer, so a check whose inputs are missing is skipped, not failed.

A failed check does not say the extraction is wrong. It may be a misread, or the
document itself may not add up (inv_07). The pipeline tells the two apart.
"""

from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

from pydantic import BaseModel

from ex3_extraction.schema import Invoice

CENT = Decimal("0.01")


class IssueCode(StrEnum):
    ITEM_ARITHMETIC = "item_arithmetic"  # quantity x unit_price != amount
    ITEMS_VS_BASE = "items_vs_base"  # sum of amounts != subtotal (or total)
    SUBTOTAL_TAX_TOTAL = "subtotal_tax_total"  # subtotal + tax != total


class Issue(BaseModel):
    code: IssueCode
    field: str  # the field to re-read, e.g. "line_items[1].amount" or "stated_total"
    message: str  # specific enough to send back to the model


def cents(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def check_items(invoice: Invoice) -> list[Issue]:
    issues = []
    for index, item in enumerate(invoice.line_items):
        expected = cents(item.quantity * item.unit_price)
        if expected != cents(item.amount):
            issues.append(
                Issue(
                    code=IssueCode.ITEM_ARITHMETIC,
                    field=f"line_items[{index}].amount",
                    message=(
                        f"line item {index + 1}: {item.quantity} x {item.unit_price} = "
                        f"{expected}, but amount is {item.amount}"
                    ),
                )
            )
    return issues


def check_items_vs_base(invoice: Invoice) -> list[Issue]:
    """The items add up to the subtotal, or to the total when no subtotal is printed."""
    if not invoice.line_items:
        return []
    if invoice.stated_subtotal is not None:
        field, base = "stated_subtotal", invoice.stated_subtotal
    elif invoice.stated_total is not None:
        field, base = "stated_total", invoice.stated_total
    else:
        return []
    items_sum = cents(sum((item.amount for item in invoice.line_items), Decimal(0)))
    if items_sum == cents(base):
        return []
    return [
        Issue(
            code=IssueCode.ITEMS_VS_BASE,
            field=field,
            message=f"line items sum to {items_sum}, but {field} is {base}",
        )
    ]


def check_subtotal_tax_total(invoice: Invoice) -> list[Issue]:
    subtotal, tax, total = invoice.stated_subtotal, invoice.stated_tax, invoice.stated_total
    if subtotal is None or tax is None or total is None:
        return []
    if cents(subtotal + tax) == cents(total):
        return []
    return [
        Issue(
            code=IssueCode.SUBTOTAL_TAX_TOTAL,
            field="stated_total",
            message=f"stated_subtotal {subtotal} + stated_tax {tax} = "
            f"{cents(subtotal + tax)}, but stated_total is {total}",
        )
    ]


def check(invoice: Invoice) -> list[Issue]:
    """Every semantic issue in one extraction; an empty list means the numbers agree."""
    return check_items(invoice) + check_items_vs_base(invoice) + check_subtotal_tax_total(invoice)
