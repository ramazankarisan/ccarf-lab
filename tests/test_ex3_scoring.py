"""Scoring: exact field comparison, mismatch classes, the report and the corpus run."""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from anthropic import Anthropic
from anthropic.types import Message

from ex3_extraction.extractor import ExtractionResult, TokenUsage, run_corpus
from ex3_extraction.ground_truth import load_ground_truth
from ex3_extraction.schema import Category, LineItem
from ex3_extraction.scoring import (
    FIELDS,
    DocScore,
    MismatchKind,
    field_accuracy,
    format_report,
    score_document,
)

TRUTH = load_ground_truth()
USAGE = TokenUsage(input_tokens=100, output_tokens=50)


def extracted(doc: str, **changes: Any) -> ExtractionResult:
    """The ground truth for one document as if extracted, with some fields changed."""
    invoice = TRUTH[doc].invoice.model_copy(update=changes)
    return ExtractionResult(
        doc=doc, invoice=invoice, raw_input={}, validation_error=None, usage=USAGE
    )


def score(doc: str, **changes: Any) -> DocScore:
    return score_document(TRUTH[doc].invoice, extracted(doc, **changes))


def kinds(result: DocScore) -> dict[str, MismatchKind]:
    return {m.field: m.kind for m in result.mismatches}


def test_twelve_fields_are_scored():
    assert len(FIELDS) == 12


@pytest.mark.parametrize("doc", sorted(TRUTH))
def test_the_ground_truth_scores_perfectly_against_itself(doc: str):
    result = score(doc)

    assert (result.valid, result.correct, result.mismatches) == (True, 12, [])


def test_decimals_compare_by_value():
    assert score("inv_07.txt", stated_total=Decimal("1203.0")).correct == 12


def test_strings_are_trimmed():
    assert score("inv_01.txt", invoice_number="  BWD-10482 ").correct == 12


def test_a_value_where_the_truth_is_null_is_a_fabrication():
    """inv_12: copying the buyer's VAT number is the classic fabrication."""
    result = score("inv_12.txt", vendor_tax_id="GB000000012")

    assert kinds(result) == {"vendor_tax_id": MismatchKind.FABRICATION}
    assert (result.mismatches[0].truth, result.mismatches[0].extracted) == (None, "GB000000012")


def test_a_missed_value_is_an_omission():
    assert kinds(score("inv_07.txt", stated_total=None)) == {"stated_total": MismatchKind.OMISSION}


def test_a_different_value_is_wrong():
    """inv_09: taking the net 'TOTAL' instead of the gross."""
    assert kinds(score("inv_09.txt", stated_total=Decimal("1200.00"))) == {
        "stated_total": MismatchKind.WRONG
    }


def item(description: str = "Taxi ride", amount: str = "23.40") -> LineItem:
    return LineItem(
        description=description,
        quantity=Decimal(1),
        unit_price=Decimal(amount),
        amount=Decimal(amount),
    )


def test_line_items_invented_for_a_receipt_without_them_are_a_fabrication():
    assert kinds(score("inv_03.txt", line_items=[item()])) == {
        "line_items": MismatchKind.FABRICATION
    }


def test_line_item_counts_are_shown_in_reports():
    result = score("inv_03.txt", line_items=[item()])

    assert (result.mismatches[0].truth, result.mismatches[0].extracted) == ("0 items", "1 item")


def test_line_items_missed_are_an_omission():
    assert kinds(score("inv_09.txt", line_items=[])) == {"line_items": MismatchKind.OMISSION}


def test_line_items_need_the_same_count_and_equal_numbers():
    truth_items = TRUTH["inv_09.txt"].invoice.line_items
    changed_amount = [truth_items[0], truth_items[1].model_copy(update={"amount": Decimal(1)})]
    changed_quantity = [truth_items[0].model_copy(update={"quantity": Decimal(5)}), truth_items[1]]

    for items in (truth_items[:1], changed_amount, changed_quantity):
        assert kinds(score("inv_09.txt", line_items=items)) == {"line_items": MismatchKind.WRONG}


def test_line_item_descriptions_are_not_scored():
    """inv_02 is an email: it prints no item description, so any wording is legitimate."""
    [truth_item] = TRUTH["inv_02.txt"].invoice.line_items
    reworded = truth_item.model_copy(
        update={"description": "Consulting services, 5 to 7 October 2026 (three days)"}
    )

    assert score("inv_02.txt", line_items=[reworded]).correct == 12


# --- Free-text fields: the metric ignores harmless variation ------------------------


@pytest.mark.parametrize(
    ("doc", "printed"),
    [("inv_07.txt", "KESTREL NETWORKS CORP."), ("inv_09.txt", "ASHBY COMPONENTS LTD")],
)
def test_vendor_names_compare_without_case(doc: str, printed: str):
    """Copying the name in capitals, as printed, is not an error."""
    assert score(doc, vendor_name=f" {printed} ").correct == 12


def test_a_different_vendor_name_is_still_wrong():
    assert kinds(score("inv_07.txt", vendor_name="Kestrel Networks")) == {
        "vendor_name": MismatchKind.WRONG
    }


def test_category_detail_counts_presence_not_wording():
    assert score("inv_06.txt", category_detail="Catering services").correct == 12


def test_a_missing_or_invented_category_detail_still_counts():
    """A valid invoice pairs the detail with the category, so these come in pairs,
    like the baseline's catering invoice filed under services."""
    assert kinds(score("inv_06.txt", category=Category.SERVICES, category_detail=None)) == {
        "category": MismatchKind.WRONG,
        "category_detail": MismatchKind.OMISSION,
    }
    assert kinds(score("inv_01.txt", category=Category.OTHER, category_detail="docks")) == {
        "category": MismatchKind.WRONG,
        "category_detail": MismatchKind.FABRICATION,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("invoice_number", "bwd-10482"),
        ("vendor_tax_id", "00-0000002"),
        ("invoice_date", date(2026, 2, 10)),
        ("currency", "EUR"),
        ("category", Category.SERVICES),
        ("stated_total", Decimal("1101.51")),
    ],
)
def test_identifiers_dates_codes_and_amounts_stay_exact(field: str, value: Any):
    assert kinds(score("inv_01.txt", **{field: value})) == {field: MismatchKind.WRONG}


def test_an_invalid_extraction_fails_every_field():
    result = ExtractionResult(
        doc="inv_06.txt",
        invoice=None,
        raw_input={"category": "hardware"},
        validation_error="category_detail must be null\nmore detail",
        usage=USAGE,
    )

    scored = score_document(TRUTH["inv_06.txt"].invoice, result)

    assert (scored.valid, scored.correct) == (False, 0)
    assert {m.kind for m in scored.mismatches} == {MismatchKind.INVALID}
    assert len(scored.mismatches) == 12
    assert scored.validation_error == "category_detail must be null\nmore detail"


def test_field_accuracy_is_the_share_of_documents_with_that_field_right():
    scores = [score("inv_07.txt", stated_total=None), score("inv_09.txt")]

    accuracy = field_accuracy(scores)

    assert accuracy["stated_total"] == 0.5
    assert accuracy["vendor_name"] == 1.0


def test_the_report_shows_documents_fields_and_totals():
    invalid = ExtractionResult(
        doc="inv_06.txt",
        invoice=None,
        raw_input={},
        validation_error="category_detail must be null\nmore detail",
        usage=USAGE,
    )
    scores = [
        score("inv_12.txt", vendor_tax_id="GB000000012"),
        score("inv_01.txt"),
        score_document(TRUTH["inv_06.txt"].invoice, invalid),
    ]

    report = format_report(scores)

    assert "inv_12.txt   1/1    11/12    vendor_tax_id (fabrication) x1" in report
    assert "inv_01.txt   1/1    12/12    -" in report
    assert "inv_06.txt   0/1    0/12     invalid: category_detail must be null" in report
    assert "vendor_tax_id       33%" in report  # wrong in inv_12, invalid in inv_06
    assert "vendor_name         67%" in report
    assert "overall: 23/36 fields correct (fabrication: 1, omission: 0, wrong: 0, invalid: 12)" in (
        report
    )


def test_the_report_sums_each_document_over_its_runs():
    """Three runs of one document: which errors repeat, and how often it was valid."""
    scores = [
        score("inv_02.txt", stated_subtotal=Decimal("2550")),
        score("inv_02.txt"),
        score("inv_02.txt", stated_subtotal=Decimal("2550")),
    ]

    report = format_report(scores)

    assert "inv_02.txt   3/3    34/36    stated_subtotal (fabrication) x2" in report
    assert "stated_subtotal     33%" in report


# --- The corpus run, with a fake model that returns the ground truth ---------------


class ReplayMessages:
    """Answers each call with the ground truth of the next document, in corpus order,
    starting over for each run."""

    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs: Any) -> Message:
        docs = sorted(TRUTH)
        tool_input = TRUTH[docs[self.calls % len(docs)]].invoice.model_dump(mode="json")
        self.calls += 1
        return Message.model_validate(
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": kwargs["model"],
                "content": [
                    {"type": "tool_use", "id": "t", "name": "record_invoice", "input": tool_input}
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 1000, "output_tokens": 400},
            }
        )


class ReplayClient:
    def __init__(self) -> None:
        self.messages = ReplayMessages()


def test_run_corpus_scores_every_run_and_saves_them_all(tmp_path: Path):
    out_file = tmp_path / "runs" / "baseline.json"

    report = run_corpus(cast(Anthropic, ReplayClient()), "some-model", out_file, runs=3)

    assert report.startswith("3 runs x 12 documents")
    assert "inv_01.txt   3/3    36/36    -" in report
    assert "overall: 432/432 fields correct" in report
    assert "tokens: 36000 in, 14400 out" in report
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["model"] == "some-model"
    assert len(saved["runs"]) == 3
    for run in saved["runs"]:
        assert [r["doc"] for r in run["results"]] == sorted(TRUTH)
        assert all(s["correct"] == 12 for s in run["scores"])
