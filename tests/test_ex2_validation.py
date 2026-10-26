"""Semantic checks: item arithmetic, items vs subtotal or total, subtotal + tax = total."""

from decimal import Decimal
from typing import Any

import pytest

from ex2_extraction.ground_truth import load_ground_truth
from ex2_extraction.schema import Invoice, LineItem
from ex2_extraction.validation import Issue, IssueCode, cents, check

TRUTH = load_ground_truth()


def invoice(doc: str, **changes: Any) -> Invoice:
    return TRUTH[doc].invoice.model_copy(update=changes)


def item(quantity: str, unit_price: str, amount: str) -> LineItem:
    return LineItem(
        description="Item",
        quantity=Decimal(quantity),
        unit_price=Decimal(unit_price),
        amount=Decimal(amount),
    )


def codes(issues: list[Issue]) -> list[tuple[IssueCode, str]]:
    return [(issue.code, issue.field) for issue in issues]


@pytest.mark.parametrize("doc", sorted(TRUTH))
def test_the_ground_truth_passes_except_the_invoice_that_does_not_add_up(doc: str):
    """The baseline for the checker: only inv_07 (edge case 7) fails, and VAT on top
    (inv_05, inv_08, inv_09) is not a false positive."""
    issues = check(TRUTH[doc].invoice)

    if 7 in TRUTH[doc].edge_cases:
        assert codes(issues) == [(IssueCode.ITEMS_VS_BASE, "stated_total")]
        assert issues[0].message == "line items sum to 1163.00, but stated_total is 1203.00"
    else:
        assert issues == []


# --- item_arithmetic ---------------------------------------------------------------


def test_an_item_whose_amount_is_not_quantity_times_price_is_flagged():
    items = [item("6", "149.00", "894.00"), item("1", "219.00", "210.00")]

    issues = check(invoice("inv_07.txt", line_items=items, stated_total=None))

    assert codes(issues) == [(IssueCode.ITEM_ARITHMETIC, "line_items[1].amount")]
    assert issues[0].message == "line item 2: 1 x 219.00 = 219.00, but amount is 210.00"


def test_item_arithmetic_is_compared_in_cents():
    """3 x 0.333 = 0.999, which is 1.00 in cents: no issue. 1.01 would be one."""
    in_cents = invoice("inv_04.txt", line_items=[item("3", "0.333", "1.00")], stated_total=None)
    off_by_one = invoice("inv_04.txt", line_items=[item("3", "0.333", "1.01")], stated_total=None)

    assert check(in_cents) == []
    assert codes(check(off_by_one)) == [(IssueCode.ITEM_ARITHMETIC, "line_items[0].amount")]


# --- items_vs_base -------------------------------------------------------------------


def test_items_are_compared_with_the_subtotal_when_one_is_printed():
    issues = check(invoice("inv_09.txt", stated_subtotal=Decimal("1100.00"), stated_tax=None))

    assert codes(issues) == [(IssueCode.ITEMS_VS_BASE, "stated_subtotal")]


def test_items_are_compared_with_the_total_when_no_subtotal_is_printed():
    """inv_09 without its subtotal: the items (1200) no longer match the gross 1440."""
    issues = check(invoice("inv_09.txt", stated_subtotal=None))

    assert codes(issues) == [(IssueCode.ITEMS_VS_BASE, "stated_total")]


def test_no_items_or_no_base_means_nothing_to_compare():
    """inv_03 has no items; inv_10 prints no total. Absent is not inconsistent."""
    assert check(invoice("inv_03.txt")) == []
    assert check(invoice("inv_10.txt")) == []
    assert check(invoice("inv_07.txt", stated_total=None)) == []


# --- subtotal_tax_total ---------------------------------------------------------------


def test_subtotal_plus_tax_must_equal_the_total():
    issues = check(invoice("inv_08.txt", stated_tax=Decimal("197.41")))

    assert codes(issues) == [(IssueCode.SUBTOTAL_TAX_TOTAL, "stated_total")]
    assert issues[0].message == (
        "stated_subtotal 1037.58 + stated_tax 197.41 = 1234.99, but stated_total is 1234.72"
    )


@pytest.mark.parametrize("missing", ["stated_subtotal", "stated_tax", "stated_total"])
def test_a_null_never_triggers_the_sum_check(missing: str):
    """inv_12 prints tax but no subtotal: that is a valid answer, not an issue."""
    changes: dict[str, Any] = {missing: None, "line_items": []}

    assert check(invoice("inv_08.txt", **changes)) == []


def test_cents_rounds_half_up():
    assert cents(Decimal("0.125")) == Decimal("0.13")
    assert cents(Decimal("197.1402")) == Decimal("197.14")
