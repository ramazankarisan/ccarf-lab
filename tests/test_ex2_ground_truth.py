"""The ground truth: complete, valid against the schema, and true to the documents."""

import re

import pytest

from ex2_extraction.ground_truth import (
    CORPUS_DIR,
    EDGE_CASES,
    GroundTruth,
    load_document,
    load_ground_truth,
)
from ex2_extraction.validation import check

TRUTH = load_ground_truth()
DOCS = sorted(TRUTH)


def test_every_document_has_ground_truth_and_vice_versa():
    files = sorted(path.name for path in CORPUS_DIR.glob("*.txt"))

    assert files == DOCS
    assert len(DOCS) == 12


def test_every_edge_case_appears_at_least_once():
    covered = {case for entry in TRUTH.values() for case in entry.edge_cases}

    assert covered == set(EDGE_CASES)


def test_edge_cases_are_known_numbers():
    assert all(set(entry.edge_cases) <= set(EDGE_CASES) for entry in TRUTH.values())


@pytest.mark.parametrize("name", DOCS)
def test_identifiers_are_copied_from_the_document(name: str):
    """Ground truth records what the document says: nothing in it is made up."""
    text = load_document(name)
    invoice = TRUTH[name].invoice

    assert invoice.vendor_name.casefold() in text.casefold()
    assert invoice.invoice_number in text
    if invoice.vendor_tax_id is not None:
        assert invoice.vendor_tax_id in text


@pytest.mark.parametrize("name", DOCS)
def test_line_amounts_are_quantity_times_unit_price(name: str):
    """Catches typos in the ground truth itself."""
    for item in TRUTH[name].invoice.line_items:
        assert item.quantity * item.unit_price == item.amount


def entries_with(case: int) -> list[GroundTruth]:
    return [entry for entry in TRUTH.values() if case in entry.edge_cases]


def test_absent_fields_are_null():
    assert all(entry.invoice.vendor_tax_id is None for entry in entries_with(4))
    assert all(entry.invoice.due_date is None for entry in entries_with(5))
    assert all(entry.invoice.stated_total is None for entry in entries_with(10))


@pytest.mark.parametrize("name", DOCS)
def test_only_the_broken_invoice_fails_the_number_checks(name: str):
    """VAT on top is not a discrepancy: with subtotal and tax extracted separately,
    the Step 3 checks flag case 7 and nothing else. No false positives."""
    assert (check(TRUTH[name].invoice) == []) is (7 not in TRUTH[name].edge_cases)


def test_the_mismatch_keeps_the_printed_total():
    """Case 7: the items add up to 1163.00, but the document prints 1203.00."""
    invoice = TRUTH["inv_07.txt"].invoice
    items_sum = sum(item.amount for item in invoice.line_items)

    assert str(items_sum) == "1163.00"
    assert (invoice.stated_subtotal, invoice.stated_tax) == (None, None)
    assert str(invoice.stated_total) == "1203.00"


@pytest.mark.parametrize(
    ("name", "subtotal", "tax", "total"),
    [
        ("inv_08.txt", "1037.58", "197.14", "1234.72"),
        ("inv_09.txt", "1200.00", "240.00", "1440.00"),
    ],
)
def test_net_tax_and_gross_are_each_recorded_as_printed(
    name: str, subtotal: str, tax: str, total: str
):
    """Case 9: the net amount is printed as a 'TOTAL' too; it belongs in stated_subtotal."""
    invoice = TRUTH[name].invoice

    assert (str(invoice.stated_subtotal), str(invoice.stated_tax)) == (subtotal, tax)
    assert str(invoice.stated_total) == total


def test_tax_included_in_the_total_has_no_subtotal():
    """inv_12 prints only the VAT included in the total, no net amount."""
    invoice = TRUTH["inv_12.txt"].invoice

    assert invoice.stated_subtotal is None
    assert str(invoice.stated_tax) == "81.50"


def test_the_ambiguous_date_is_read_the_us_way():
    """Case 11: 03/04/2026 next to a Texas address and Net 30 means March 4."""
    invoice = TRUTH["inv_11.txt"].invoice

    assert invoice.invoice_date.isoformat() == "2026-03-04"
    assert invoice.due_date is not None
    assert (invoice.due_date - invoice.invoice_date).days == 30


def test_the_buyers_vat_number_is_not_the_vendors():
    """Case 12: the only VAT number printed belongs to the guest's company."""
    assert "GB000000012" in load_document("inv_12.txt")
    assert TRUTH["inv_12.txt"].invoice.vendor_tax_id is None


def test_the_corpus_uses_only_reserved_email_domains():
    for name in DOCS:
        for domain in re.findall(r"@([\w.-]+)", load_document(name)):
            assert domain in {"example.com", "example.org"}
