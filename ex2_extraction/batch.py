"""Step 5: the corpus through the Message Batches API, with variant B.

A batch runs asynchronously at half the price, but it has no conversation loop:
each request gets one answer. So the Step 3 retries become a second batch. Round 1
sends every document; round 2 resubmits only what needs it:
- errored requests (here one is broken on purpose: its tool_choice names a tool that
  does not exist), fixed and sent fresh;
- invalid extractions, sent with the document, the failed output and the specific
  errors, as in Step 3;
- inconsistent extractions, sent once with the re-check instruction, as in Step 3.

Results come back in any order, so they are matched by custom_id, never by position.
"""

import json
import os
import re
import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

from anthropic import Anthropic
from anthropic.types import Message, MessageParam, ToolChoiceToolParam
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse
from anthropic.types.messages.batch_create_params import Request
from pydantic import BaseModel

from ex2_extraction.extractor import (
    FORCE_RECORD_INVOICE,
    MAX_TOKENS,
    ExtractionResult,
    TokenUsage,
    document_message,
    parse_invoice,
)
from ex2_extraction.ground_truth import load_document, load_ground_truth
from ex2_extraction.pipeline import (
    FIX_INVALID,
    RECHECK,
    bullets,
    changed_fields,
    feedback,
    tool_call,
)
from ex2_extraction.schema import Invoice
from ex2_extraction.scoring import score_document
from ex2_extraction.validation import Issue, check
from ex2_extraction.variants import B, Variant

CUSTOM_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")  # the API's rule: no "." allowed
DOCS = [f"inv_{n:02d}.txt" for n in range(1, 11)]
BROKEN_DOC = "inv_11.txt"  # the eleventh is sent broken in round 1
BROKEN_TOOL_CHOICE: ToolChoiceToolParam = {"type": "tool", "name": "record_receipt"}
POLL_SECONDS = 15
BATCH_FILE = Path("runs") / "batch.json"


class Status(StrEnum):
    OK = "ok"  # valid, and the printed numbers agree
    INVALID = "invalid"  # failed validation: resubmit with the errors
    INCONSISTENT = "inconsistent"  # valid, numbers disagree: resubmit once to re-check
    CONFLICT = "conflict"  # still disagrees after the re-check: a human decides
    FAILED = "failed"  # still invalid after the resubmit
    ERRORED = "errored"  # the request itself failed: fix it and resubmit


class Round(BaseModel):
    """What one batch round produced for one document."""

    batch_id: str
    result_type: str  # succeeded, errored, canceled or expired
    status: Status
    error: str | None  # the API error, or the validation errors
    issues: list[Issue]
    usage: TokenUsage | None  # None when the request errored


class DocRecord(BaseModel):
    custom_id: str
    doc: str
    rounds: list[Round]
    invoice: Invoice | None  # the last valid extraction
    changed_on_recheck: list[str]


def custom_id(doc: str) -> str:
    """inv_01.txt -> inv_01: the document name, without the "." the API rejects."""
    name = doc.removesuffix(".txt")
    if not CUSTOM_ID.fullmatch(name):
        raise ValueError(f"not a valid custom_id: {name!r}")
    return name


def request(
    doc: str,
    model: str,
    messages: list[MessageParam],
    variant: Variant = B,
    tool_choice: ToolChoiceToolParam = FORCE_RECORD_INVOICE,
) -> Request:
    params: MessageCreateParamsNonStreaming = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": variant.system_prompt,
        "tools": [variant.tool],
        "tool_choice": tool_choice,
        "messages": messages,
    }
    return {"custom_id": custom_id(doc), "params": params}


def wait(
    client: Anthropic,
    batch_id: str,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> MessageBatch:
    """Poll until the batch has ended. Most batches end within an hour, the limit is 24."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return batch
        counts = batch.request_counts
        log(f"  {batch_id}: {batch.processing_status}, {counts.processing} processing")
        sleep(POLL_SECONDS)


def run_batch(
    client: Anthropic,
    requests: list[Request],
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> tuple[str, dict[str, MessageBatchIndividualResponse]]:
    """Submit, wait, and return the results keyed by custom_id."""
    batch = client.messages.batches.create(requests=requests)
    log(f"submitted {batch.id} with {len(requests)} requests")
    wait(client, batch.id, sleep, log)
    results = {result.custom_id: result for result in client.messages.batches.results(batch.id)}
    return batch.id, results


def judge(
    batch_id: str, result: MessageBatchIndividualResponse
) -> tuple[Round, Message | None, Invoice | None]:
    """Classify one result: errored, invalid, inconsistent or ok."""
    outcome = result.result
    if outcome.type != "succeeded":
        api_error = outcome.error.error.message if outcome.type == "errored" else outcome.type
        round_ = Round(
            batch_id=batch_id,
            result_type=outcome.type,
            status=Status.ERRORED,
            error=api_error,
            issues=[],
            usage=None,
        )
        return round_, None, None
    message = outcome.message
    usage = TokenUsage(
        input_tokens=message.usage.input_tokens, output_tokens=message.usage.output_tokens
    )
    tool_use = tool_call(message)
    invoice, error = parse_invoice(tool_use.input) if tool_use else (None, "no record_invoice call")
    issues = check(invoice) if invoice is not None else []
    if invoice is None:
        status = Status.INVALID
    elif issues:
        status = Status.INCONSISTENT
    else:
        status = Status.OK
    round_ = Round(
        batch_id=batch_id,
        result_type="succeeded",
        status=status,
        error=error,
        issues=issues,
        usage=usage,
    )
    return round_, message, invoice


def follow_up(doc: str, model: str, round_: Round, message: Message | None) -> Request:
    """Round 2's request for one document, by what went wrong in round 1."""
    first = document_message(load_document(doc))
    if round_.status is Status.ERRORED or message is None:
        return request(doc, model, [first])  # the fixed request: the forced, existing tool
    if round_.status is Status.INVALID:
        text = FIX_INVALID.format(errors=bullets((round_.error or "").splitlines()))
    else:
        text = RECHECK.format(issues=bullets([issue.message for issue in round_.issues]))
    return request(doc, model, [first, *feedback(message, tool_call(message), text)])


def second_round_status(first: Round, second: Round) -> Status:
    """A re-check that still disagrees is a conflict; an invalid answer after the
    errors were sent back is a failure."""
    if second.status is Status.INCONSISTENT and first.status is Status.INCONSISTENT:
        return Status.CONFLICT
    if second.status is Status.INVALID:
        return Status.FAILED
    return second.status


def run(
    client: Anthropic,
    model: str,
    out_file: Path,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> str:
    docs = [*DOCS, BROKEN_DOC]
    first_requests = [request(doc, model, [document_message(load_document(doc))]) for doc in DOCS]
    first_requests.append(
        request(
            BROKEN_DOC,
            model,
            [document_message(load_document(BROKEN_DOC))],
            tool_choice=BROKEN_TOOL_CHOICE,
        )
    )
    batch_1, results_1 = run_batch(client, first_requests, sleep, log)

    records: dict[str, DocRecord] = {}
    retry: list[Request] = []
    for doc in docs:
        cid = custom_id(doc)
        round_, message, invoice = judge(batch_1, results_1[cid])
        records[cid] = DocRecord(
            custom_id=cid, doc=doc, rounds=[round_], invoice=invoice, changed_on_recheck=[]
        )
        if round_.status is not Status.OK:
            retry.append(follow_up(doc, model, round_, message))

    batch_2 = None
    if retry:
        batch_2, results_2 = run_batch(client, retry, sleep, log)
        for request_ in retry:
            record = records[request_["custom_id"]]
            round_, _, invoice = judge(batch_2, results_2[record.custom_id])
            round_.status = second_round_status(record.rounds[0], round_)
            record.rounds.append(round_)
            if invoice is not None:
                if record.invoice is not None:
                    record.changed_on_recheck = changed_fields(record.invoice, invoice)
                record.invoice = invoice

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(
            {
                "batches": [batch_1, batch_2],
                "records": [r.model_dump(mode="json") for r in records.values()],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return report(batch_1, batch_2, list(records.values()))


def report(batch_1: str, batch_2: str | None, records: list[DocRecord]) -> str:
    truth = load_ground_truth()
    lines = [
        f"batch 1: {batch_1}",
        f"batch 2: {batch_2 or '- (nothing to resubmit)'}",
        "",
        f"{'custom_id':<10} {'round 1':<13} {'round 2':<13} {'correct':<8} notes",
    ]
    total_in = total_out = correct = 0
    for record in records:
        first = record.rounds[0]
        second = record.rounds[1] if len(record.rounds) > 1 else None
        for round_ in record.rounds:
            if round_.usage is not None:
                total_in += round_.usage.input_tokens
                total_out += round_.usage.output_tokens
        result = ExtractionResult(
            doc=record.doc,
            invoice=record.invoice,
            raw_input={},
            validation_error=None if record.invoice else "no valid extraction",
            usage=TokenUsage(input_tokens=0, output_tokens=0),
        )
        score = score_document(truth[record.doc].invoice, result).correct
        correct += score
        notes: list[str] = []
        if first.error:
            notes.append(first.error.splitlines()[0])
        notes += [issue.message for issue in first.issues]
        if record.changed_on_recheck:
            notes.append(f"changed: {', '.join(record.changed_on_recheck)}")
        lines.append(
            f"{record.custom_id:<10} {first.status:<13} {(second.status if second else '-'):<13} "
            f"{score:>2}/12    {'; '.join(notes) or '-'}"
        )
    lines += [
        "",
        f"resubmitted: {', '.join(r.custom_id for r in records if len(r.rounds) > 1) or 'nothing'}",
        f"final: {correct}/{len(records) * 12} fields correct",
        f"tokens: {total_in} in, {total_out} out (batch price: half the standard rate)",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    print(run(Anthropic(), os.environ["MODEL"], BATCH_FILE))
