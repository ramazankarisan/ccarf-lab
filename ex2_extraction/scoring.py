"""Field-by-field scoring of extractions against the ground truth.

Every mismatch gets a class, because each points to a different fix:
fabrication (truth is null, a value was extracted) is the dangerous one,
omission (a value was missed) and wrong (a different value).
The metric matches what downstream needs, so harmless variation is not scored as
an error. Identifiers, dates, codes, categories and every number compare exactly
(Decimals by value, strings trimmed); there is no fuzzy matching. Three free-text
fields are looser on purpose:
- vendor_name ignores case: company identity does not depend on it.
- line item descriptions are left out: a narrative document prints none, so any
  wording is legitimate. Count, quantity, unit price and amount must match.
- category_detail counts presence only: "catering" and "Catering services" both say
  what "other" means. A null where a value belongs is still an omission.
An invalid extraction fails every field.
"""

from collections import Counter
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from ex2_extraction.extractor import ExtractionResult
from ex2_extraction.schema import Invoice, LineItem

FIELDS = (
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "vendor_tax_id",
    "due_date",
    "currency",
    "category",
    "category_detail",
    "stated_subtotal",
    "stated_tax",
    "stated_total",
    "line_items",  # one field: same count, every item equal
)


class MismatchKind(StrEnum):
    FABRICATION = "fabrication"  # truth null, extracted a value
    OMISSION = "omission"  # truth a value, extracted null
    WRONG = "wrong"  # truth a value, extracted a different value
    INVALID = "invalid"  # the extraction failed validation, so nothing counts


class Mismatch(BaseModel):
    field: str
    kind: MismatchKind
    truth: str | None
    extracted: str | None


class DocScore(BaseModel):
    doc: str
    valid: bool
    correct: int
    mismatches: list[Mismatch]
    validation_error: str | None


def is_absent(value: Any) -> bool:
    """Null, or an empty line-item list: the document has nothing for this field."""
    return value is None or value == []


def same_item(truth: LineItem, got: LineItem) -> bool:
    """The numbers only: a description is free text (see the module docstring)."""
    return (
        truth.quantity == got.quantity
        and truth.unit_price == got.unit_price
        and truth.amount == got.amount
    )


def same(field: str, truth: Any, got: Any) -> bool:
    """Both values are present here; absent values are handled by classify()."""
    if field == "line_items":
        return len(truth) == len(got) and all(map(same_item, truth, got))
    if field == "vendor_name":
        return bool(truth.strip().casefold() == got.strip().casefold())
    if field == "category_detail":
        return True  # presence is what counts
    if isinstance(truth, str) and isinstance(got, str):
        return truth.strip() == got.strip()
    return bool(truth == got)  # Decimal("1.0") == Decimal("1.00"), dates, enums


def show(value: Any) -> str | None:
    """A short JSON-friendly form for reports: 1203.00, 2026-03-04, "3 items"."""
    if value is None:
        return None
    if isinstance(value, list):
        return f"{len(value)} item" if len(value) == 1 else f"{len(value)} items"
    return str(value)


def classify(field: str, truth: Any, got: Any) -> MismatchKind | None:
    """None when the field is correct, otherwise the kind of mismatch."""
    if is_absent(truth) and is_absent(got):
        return None
    if is_absent(truth):
        return MismatchKind.FABRICATION
    if is_absent(got):
        return MismatchKind.OMISSION
    return None if same(field, truth, got) else MismatchKind.WRONG


def score_document(truth: Invoice, result: ExtractionResult) -> DocScore:
    if result.invoice is None:
        failed = [
            Mismatch(
                field=field,
                kind=MismatchKind.INVALID,
                truth=show(getattr(truth, field)),
                extracted=None,
            )
            for field in FIELDS
        ]
        return DocScore(
            doc=result.doc,
            valid=False,
            correct=0,
            mismatches=failed,
            validation_error=result.validation_error,
        )
    mismatches = []
    for field in FIELDS:
        expected, got = getattr(truth, field), getattr(result.invoice, field)
        kind = classify(field, expected, got)
        if kind is not None:
            mismatches.append(
                Mismatch(field=field, kind=kind, truth=show(expected), extracted=show(got))
            )
    return DocScore(
        doc=result.doc,
        valid=True,
        correct=len(FIELDS) - len(mismatches),
        mismatches=mismatches,
        validation_error=None,
    )


def field_accuracy(scores: list[DocScore]) -> dict[str, float]:
    """Per field: the share of documents where it was correct."""
    failed = {field: 0 for field in FIELDS}
    for score in scores:
        for mismatch in score.mismatches:
            failed[mismatch.field] += 1
    return {field: 1 - failed[field] / len(scores) for field in FIELDS}


def format_report(scores: list[DocScore]) -> str:
    """One row per document, summed over every run that scored it, then the field
    accuracy over all extractions. Several runs show which errors repeat."""
    by_doc: dict[str, list[DocScore]] = {}
    for score in scores:
        by_doc.setdefault(score.doc, []).append(score)

    lines = [f"{'doc':<12} {'valid':<6} {'correct':<8} mismatches"]
    for doc, runs in by_doc.items():
        counted = Counter(
            f"{m.field} ({m.kind})"
            for run in runs
            for m in run.mismatches
            if m.kind is not MismatchKind.INVALID
        )
        found = [f"{label} x{n}" for label, n in counted.items()]
        found += [
            f"invalid: {(run.validation_error or '').splitlines()[0]}"
            for run in runs
            if not run.valid
        ]
        valid = f"{sum(run.valid for run in runs)}/{len(runs)}"
        correct = f"{sum(run.correct for run in runs)}/{len(runs) * len(FIELDS)}"
        lines.append(f"{doc:<12} {valid:<6} {correct:<8} {', '.join(found) or '-'}")

    lines += ["", f"{'field':<16} accuracy"]
    for field, accuracy in field_accuracy(scores).items():
        lines.append(f"{field:<16} {accuracy:>6.0%}")

    total = sum(score.correct for score in scores)
    kinds = [m.kind for score in scores for m in score.mismatches]
    counts = ", ".join(f"{kind}: {kinds.count(kind)}" for kind in MismatchKind)
    lines += [
        "",
        f"overall: {total}/{len(scores) * len(FIELDS)} fields correct ({counts})",
    ]
    return "\n".join(lines)
