"""Structured extraction through tool use: the Invoice schema is the tool's input_schema.

tool_choice forces the one tool, so every document produces an extraction and the
model cannot answer in prose. The tool forces the shape, not the meaning: JSON Schema
cannot say "category_detail only when category is other", so the input is parsed
with Invoice.model_validate as well. Schema-compliant is not the same as valid.
"""

import json
import os
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message, ToolChoiceToolParam, ToolParam
from pydantic import BaseModel, ValidationError

from ex3_extraction.ground_truth import load_document, load_ground_truth
from ex3_extraction.schema import Invoice

TOOL_NAME = "record_invoice"
BASELINE_FILE = Path("runs") / "baseline.json"  # runs/ is gitignored
MAX_TOKENS = 4096  # one invoice needs about 500 output tokens

# Zero-shot on purpose: this is the baseline, and Step 4 measures what examples add.
SYSTEM_PROMPT = """\
You extract data from invoices and record it with the record_invoice tool.

<rules>
- Extract only what the document prints. If a value is absent, use null. Never guess.
- Copy the subtotal, tax and total exactly as printed. Do not compute, add up or \
correct them, even if the numbers do not match.
- Write dates as ISO 8601 (YYYY-MM-DD). When a date like 07/08 could be read either \
way, decide day and month from the document's own cues, such as its address, \
language and currency.
- vendor_tax_id is the seller's tax ID. A tax or VAT number of the buyer or customer \
is never the vendor's.
- Choose the category from the list. If none fits, use "other" and name the kind of \
expense in category_detail. For every other category, category_detail is null.
</rules>"""

RECORD_INVOICE: ToolParam = {
    "name": TOOL_NAME,
    "description": (
        "Record the fields of one invoice exactly as the document states them. "
        "Use null for any field the document does not contain; never guess or compute "
        "a value. Call this once per document."
    ),
    "input_schema": Invoice.model_json_schema(),
}

# One schema and every document must yield an extraction, so the named tool is forced.
# "any" would fit several schemas (one per document type); "auto" allows a prose answer.
FORCE_RECORD_INVOICE: ToolChoiceToolParam = {"type": "tool", "name": TOOL_NAME}


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class ExtractionResult(BaseModel):
    """One document's extraction, kept whole so a run can be scored and inspected later."""

    doc: str  # corpus file name, e.g. "inv_07.txt"
    invoice: Invoice | None  # None when the tool input failed validation
    raw_input: dict[str, Any]  # exactly what the model sent, valid or not
    validation_error: str | None
    usage: TokenUsage


def parse_invoice(tool_input: dict[str, Any]) -> tuple[Invoice | None, str | None]:
    """The validated invoice, or None and the validation error. No retry here (Step 3)."""
    try:
        return Invoice.model_validate(tool_input), None
    except ValidationError as exc:
        return None, str(exc)


def extract(client: Anthropic, model: str, document: str) -> Message:
    """One extraction call. The document goes in tags, so its text is data, not
    instructions."""
    return client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=[RECORD_INVOICE],
        tool_choice=FORCE_RECORD_INVOICE,
        messages=[{"role": "user", "content": f"<document>\n{document}\n</document>"}],
    )


def to_result(doc: str, response: Message) -> ExtractionResult:
    """Turn one response into a result. A missing tool call is recorded, not raised."""
    usage = TokenUsage(
        input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens
    )
    tool_use = next((block for block in response.content if block.type == "tool_use"), None)
    if tool_use is None:
        missing = f"no {TOOL_NAME} call (stop_reason: {response.stop_reason})"
        return ExtractionResult(
            doc=doc, invoice=None, raw_input={}, validation_error=missing, usage=usage
        )
    invoice, error = parse_invoice(tool_use.input)
    return ExtractionResult(
        doc=doc, invoice=invoice, raw_input=tool_use.input, validation_error=error, usage=usage
    )


def extract_document(client: Anthropic, model: str, doc: str) -> ExtractionResult:
    """Extract one corpus document by file name."""
    return to_result(doc, extract(client, model, load_document(doc)))


def run_corpus(client: Anthropic, model: str, out_file: Path, runs: int = 3) -> str:
    """Extract every corpus document `runs` times, score, save everything, return the
    report. One run of 12 documents is noisy: the same prompt fails different documents
    from run to run, so a comparison needs several runs."""
    from ex3_extraction import scoring  # here, not at the top: scoring imports this module

    truth = load_ground_truth()
    record: dict[str, Any] = {"model": model, "runs": []}
    all_results: list[ExtractionResult] = []
    all_scores = []
    for _ in range(runs):
        results = [extract_document(client, model, doc) for doc in sorted(truth)]
        scores = [scoring.score_document(truth[r.doc].invoice, r) for r in results]
        record["runs"].append(
            {
                "results": [result.model_dump(mode="json") for result in results],
                "scores": [score.model_dump(mode="json") for score in scores],
            }
        )
        all_results += results
        all_scores += scores

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    tokens_in = sum(result.usage.input_tokens for result in all_results)
    tokens_out = sum(result.usage.output_tokens for result in all_results)
    return (
        f"{runs} runs x {len(truth)} documents\n"
        f"{scoring.format_report(all_scores)}\n"
        f"tokens: {tokens_in} in, {tokens_out} out\n"
        f"saved: {out_file}"
    )


if __name__ == "__main__":
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    print(run_corpus(Anthropic(), os.environ["MODEL"], BASELINE_FILE))
