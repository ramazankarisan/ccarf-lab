"""Extraction with retries, but only where a retry can help.

Three kinds of problem, three responses:
- invalid (the schema or Pydantic rejects the input): a misread or a format slip.
  Retry up to MAX_INVALID_RETRIES times, sending the specific errors back.
- inconsistent (check() finds printed numbers that disagree): maybe a misread,
  maybe a document that does not add up. Re-check once, telling the model to copy
  the values as printed. If they still disagree, the status is conflict: keep the
  extraction, record the conflict for a human and stop.
- absent (null): "not printed" is a valid answer. Never retried.

A value that changes on the re-check is not trusted either way. It may be a fixed
misread (inv_09: 1400 re-read as 1440) or an invented value that makes the numbers
agree (inv_07: 1203 "repaired" to 1163). Only the document can tell, so the pipeline
records which fields changed (changed_on_recheck) for a human to audit.

Each retry continues one conversation, as in Ex 1: the failed tool_use, then a
tool_result with is_error true listing the errors, then the next call. The model
sees the document, its own output and the precise errors together.
"""

import json
import os
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message, MessageParam, ToolUseBlock
from pydantic import BaseModel

from ex3_extraction.extractor import (
    TOOL_NAME,
    TokenUsage,
    call_model,
    document_message,
    parse_invoice,
)
from ex3_extraction.ground_truth import load_document, load_ground_truth
from ex3_extraction.schema import Invoice
from ex3_extraction.validation import Issue, check

MAX_INVALID_RETRIES = 2  # after the first call: at most 3 calls for an invalid input

FIX_INVALID = (
    "The record_invoice input was invalid:\n{errors}\n"
    "Fix these fields and call record_invoice again with the complete invoice."
)

# Without the second sentence the model "repairs" inv_07's printed total to the sum
# of its items: a fabrication that passes validation.
RECHECK = (
    "These printed values do not agree with each other:\n{issues}\n"
    "Re-read these fields in the document and copy them exactly as printed. "
    "Do not change values to make them agree. The document itself may be inconsistent. "
    "Call record_invoice again with the complete invoice."
)


class Status(StrEnum):
    OK = "ok"  # valid, and the printed numbers agree
    CONFLICT = "conflict"  # valid, but the document's own numbers disagree: a human decides
    FAILED = "failed"  # still invalid after the retries: no extraction


class Attempt(BaseModel):
    """One model call and what was wrong with its answer."""

    raw_input: dict[str, Any]
    validation_error: str | None
    issues: list[Issue]
    usage: TokenUsage


class Outcome(BaseModel):
    doc: str
    status: Status
    invoice: Invoice | None  # kept for ok and conflict, None for failed
    attempts: list[Attempt]
    conflicts: list[Issue]
    # Fields whose value differs before and after the re-check: a review signal,
    # because a corrected misread and a fabricated "fix" look the same to code.
    changed_on_recheck: list[str]


def changed_fields(before: Invoice, after: Invoice) -> list[str]:
    return [
        field for field in Invoice.model_fields if getattr(before, field) != getattr(after, field)
    ]


def tool_call(response: Message) -> ToolUseBlock | None:
    return next((block for block in response.content if block.type == "tool_use"), None)


def feedback(response: Message, tool_use: ToolUseBlock | None, text: str) -> list[MessageParam]:
    """The model's turn as it was, then the errors as an is_error tool_result."""
    if tool_use is None:  # no call to answer: say what is missing in plain text
        return [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": text},
        ]
    return [
        {"role": "assistant", "content": response.content},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "is_error": True,
                    "content": text,
                }
            ],
        },
    ]


def bullets(lines: list[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def extract_with_retries(client: Anthropic, model: str, doc: str) -> Outcome:
    messages: list[MessageParam] = [document_message(load_document(doc))]
    attempts: list[Attempt] = []
    invalid_retries = 0
    rechecked = False
    last_valid: tuple[Invoice, list[Issue]] | None = None  # kept if the re-check breaks

    def done(status: Status, invoice: Invoice | None, conflicts: list[Issue]) -> Outcome:
        before = last_valid[0] if last_valid is not None else None
        changed = changed_fields(before, invoice) if before and invoice else []
        return Outcome(
            doc=doc,
            status=status,
            invoice=invoice,
            attempts=attempts,
            conflicts=conflicts,
            changed_on_recheck=changed,
        )

    while True:
        response = call_model(client, model, messages)
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        tool_use = tool_call(response)
        if tool_use is None:
            raw: dict[str, Any] = {}
            invoice: Invoice | None = None
            error: str | None = f"no {TOOL_NAME} call (stop_reason: {response.stop_reason})"
        else:
            raw = tool_use.input
            invoice, error = parse_invoice(raw)
        issues = check(invoice) if invoice is not None else []
        attempts.append(Attempt(raw_input=raw, validation_error=error, issues=issues, usage=usage))

        if invoice is None:
            if invalid_retries == MAX_INVALID_RETRIES:
                if last_valid is not None:  # a malformed re-check answer loses nothing
                    return done(Status.CONFLICT, *last_valid)
                return done(Status.FAILED, None, [])
            invalid_retries += 1
            text = FIX_INVALID.format(errors=bullets((error or "").splitlines()))
            messages += feedback(response, tool_use, text)
            continue

        if not issues:
            return done(Status.OK, invoice, [])

        if rechecked:  # the values were re-read and still disagree: the document does
            return done(Status.CONFLICT, invoice, issues)
        rechecked = True
        last_valid = (invoice, issues)
        text = RECHECK.format(issues=bullets([issue.message for issue in issues]))
        messages += feedback(response, tool_use, text)


PIPELINE_FILE = Path("runs") / "pipeline.json"  # runs/ is gitignored


def run_pipeline(client: Anthropic, model: str, out_file: Path, runs: int = 3) -> str:
    """Every corpus document `runs` times through the pipeline. Per document: the
    statuses, the calls, the fields changed on re-check, the conflicts, and the score
    of the final extraction against the ground truth. Saves every outcome."""
    from ex3_extraction.extractor import ExtractionResult
    from ex3_extraction.scoring import score_document

    truth = load_ground_truth()
    by_doc: dict[str, list[tuple[Outcome, int]]] = {doc: [] for doc in sorted(truth)}
    for _ in range(runs):
        for doc in by_doc:
            outcome = extract_with_retries(client, model, doc)
            last = outcome.attempts[-1]
            result = ExtractionResult(
                doc=doc,
                invoice=outcome.invoice,
                raw_input=last.raw_input,
                validation_error=last.validation_error,
                usage=last.usage,
            )
            by_doc[doc].append((outcome, score_document(truth[doc].invoice, result).correct))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model": model,
        "outcomes": [o.model_dump(mode="json") for runs_ in by_doc.values() for o, _ in runs_],
    }
    out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    def counted(labels: list[str]) -> str:
        return ", ".join(f"{label} x{n}" for label, n in Counter(labels).items()) or "-"

    def row(*cells: str) -> str:
        widths = (12, 20, 8, 8, 22)
        padded = [cell.ljust(width) for cell, width in zip(cells, widths, strict=False)]
        return " ".join([*padded, *cells[len(widths) :]]).rstrip()

    lines = [
        f"{runs} runs x {len(by_doc)} documents",
        row("doc", "status", "calls", "correct", "changed_on_recheck", "conflicts"),
    ]
    total = 0
    for doc, results in by_doc.items():
        outcomes = [o for o, _ in results]
        correct = sum(c for _, c in results)
        total += correct
        status = counted([o.status.value for o in outcomes])
        calls = ",".join(str(len(o.attempts)) for o in outcomes)
        changed = counted([f for o in outcomes for f in o.changed_on_recheck])
        conflicts = counted([i.code.value for o in outcomes for i in o.conflicts])
        lines.append(row(doc, status, calls, f"{correct}/{runs * 12}", changed, conflicts))
    lines.append(f"\noverall: {total}/{runs * len(by_doc) * 12} fields correct")
    lines.append(f"saved: {out_file}")
    return "\n".join(lines)


if __name__ == "__main__":
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    print(run_pipeline(Anthropic(), os.environ["MODEL"], PIPELINE_FILE))
