"""The retry pipeline: invalid is retried, inconsistent is re-checked once, absent never."""

import json
from pathlib import Path
from typing import Any, cast

import pytest
from anthropic import Anthropic
from anthropic.types import Message

from ex3_extraction.extractor import TOOL_NAME, ExtractionResult
from ex3_extraction.ground_truth import load_ground_truth
from ex3_extraction.pipeline import (
    MAX_INVALID_RETRIES,
    Status,
    extract_with_retries,
    run_pipeline,
)
from ex3_extraction.scoring import MismatchKind, score_document
from ex3_extraction.validation import IssueCode

TRUTH = load_ground_truth()


def truth_input(doc: str, **changes: Any) -> dict[str, Any]:
    """The correct tool input for a document, as JSON, with some fields changed."""
    return TRUTH[doc].invoice.model_dump(mode="json") | changes


def reply(tool_input: dict[str, Any] | None, n: int = 0) -> Message:
    """A model turn: a record_invoice call, or plain text when tool_input is None."""
    content: list[dict[str, Any]] = (
        [{"type": "text", "text": "I can't."}]
        if tool_input is None
        else [{"type": "tool_use", "id": f"toolu_{n}", "name": TOOL_NAME, "input": tool_input}]
    )
    return Message.model_validate(
        {
            "id": f"msg_{n}",
            "type": "message",
            "role": "assistant",
            "model": "some-model",
            "content": content,
            "stop_reason": "end_turn" if tool_input is None else "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 400},
        }
    )


class ScriptedMessages:
    """Answers each call with the next scripted reply and keeps what was sent."""

    def __init__(self, replies: list[dict[str, Any] | None]) -> None:
        self.replies = [reply(tool_input, n) for n, tool_input in enumerate(replies)]
        self.sent: list[list[Any]] = []

    def create(self, **kwargs: Any) -> Message:
        self.sent.append(list(kwargs["messages"]))  # a copy: the pipeline appends to it
        return self.replies[len(self.sent) - 1]


class ScriptedClient:
    def __init__(self, replies: list[dict[str, Any] | None]) -> None:
        self.messages = ScriptedMessages(replies)


def run(doc: str, replies: list[dict[str, Any] | None]):
    client = ScriptedClient(replies)
    outcome = extract_with_retries(cast(Anthropic, client), "some-model", doc)
    return outcome, client.messages.sent


def last_feedback(sent: list[list[Any]]) -> dict[str, Any]:
    """The tool_result sent with the latest call."""
    block: dict[str, Any] = sent[-1][-1]["content"][0]
    return block


# --- ok and absent ----------------------------------------------------------------


def test_a_valid_consistent_extraction_needs_one_call():
    outcome, sent = run("inv_09.txt", [truth_input("inv_09.txt")])

    assert (outcome.status, len(sent)) == (Status.OK, 1)
    assert outcome.invoice == TRUTH["inv_09.txt"].invoice
    assert outcome.conflicts == []


def test_an_absent_value_is_never_retried():
    """inv_10 prints no total: null is the answer, not a problem to fix."""
    outcome, sent = run("inv_10.txt", [truth_input("inv_10.txt")])

    assert (outcome.status, len(sent)) == (Status.OK, 1)
    assert outcome.invoice is not None
    assert outcome.invoice.stated_total is None


# --- invalid: retry with the errors ---------------------------------------------------


def test_an_invalid_input_is_retried_with_the_specific_errors():
    bad_item = {
        "description": "Taxi ride",
        "quantity": "null",
        "unit_price": "null",
        "amount": 23.4,
    }
    outcome, sent = run(
        "inv_03.txt",
        [truth_input("inv_03.txt", line_items=[bad_item]), truth_input("inv_03.txt")],
    )

    assert (outcome.status, len(sent)) == (Status.OK, 2)
    assert outcome.attempts[0].validation_error is not None
    assert outcome.attempts[1].validation_error is None
    # The second call carries the document, the failed call and the errors as one conversation.
    document, failed_turn, feedback = sent[1]
    assert "<document>" in document["content"]
    assert failed_turn["role"] == "assistant"
    result = feedback["content"][0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "toolu_0"
    assert result["is_error"] is True
    assert "- line_items.0.quantity: Value error, must be a number" in result["content"]
    assert "- line_items.0.unit_price:" in result["content"]


def test_invalid_input_fails_after_two_retries():
    bad = truth_input("inv_06.txt", category="hardware")  # a detail without "other"
    outcome, sent = run("inv_06.txt", [bad] * (MAX_INVALID_RETRIES + 1))

    assert (outcome.status, len(sent)) == (Status.FAILED, 3)
    assert outcome.invoice is None
    assert all(a.validation_error for a in outcome.attempts)


def test_a_reply_without_the_tool_call_is_answered_in_text_and_retried():
    outcome, sent = run("inv_01.txt", [None, truth_input("inv_01.txt")])

    assert (outcome.status, len(sent)) == (Status.OK, 2)
    assert outcome.attempts[0].validation_error == "no record_invoice call (stop_reason: end_turn)"
    assert "no record_invoice call" in sent[1][-1]["content"]


# --- inconsistent: one re-check, then ok or conflict -----------------------------------


def test_a_misread_number_is_fixed_by_the_recheck():
    """inv_09: the total was misread as 1400; re-reading it gives the printed 1440."""
    misread = truth_input("inv_09.txt", stated_total="1400.00")
    outcome, sent = run("inv_09.txt", [misread, truth_input("inv_09.txt")])

    assert (outcome.status, len(sent)) == (Status.OK, 2)
    assert outcome.changed_on_recheck == ["stated_total"]  # plausible, but unproven
    assert [i.code for i in outcome.attempts[0].issues] == [IssueCode.SUBTOTAL_TAX_TOTAL]
    recheck = last_feedback(sent)
    assert recheck["is_error"] is True
    assert (
        "1200.00 + stated_tax 240.00 = 1440.00, but stated_total is 1400.00" in recheck["content"]
    )


def test_a_repair_that_ignores_the_instruction_passes_the_gate_but_stays_visible():
    """The risk this field exists for: on inv_07 the model changes the printed 1203.00
    to the item sum 1163.00. The numbers now agree, so the status is ok, and the
    fabrication would pass unseen. changed_on_recheck flags it, and the scorer shows
    it is wrong."""
    repaired = truth_input("inv_07.txt", stated_total="1163.00")
    outcome, _ = run("inv_07.txt", [truth_input("inv_07.txt"), repaired])

    assert outcome.status is Status.OK
    assert outcome.changed_on_recheck == ["stated_total"]
    result = ExtractionResult(
        doc="inv_07.txt",
        invoice=outcome.invoice,
        raw_input=repaired,
        validation_error=None,
        usage=outcome.attempts[-1].usage,
    )
    scored = score_document(TRUTH["inv_07.txt"].invoice, result)
    assert [(m.field, m.kind) for m in scored.mismatches] == [("stated_total", MismatchKind.WRONG)]


def test_without_a_recheck_nothing_is_marked_changed():
    outcome, _ = run("inv_09.txt", [truth_input("inv_09.txt")])

    assert outcome.changed_on_recheck == []


def test_the_recheck_forbids_repairing_values():
    _, sent = run("inv_07.txt", [truth_input("inv_07.txt")] * 2)

    text = last_feedback(sent)["content"]
    assert "copy them exactly as printed" in text
    assert "Do not change values to make them agree" in text
    assert "The document itself may be inconsistent" in text


def test_a_document_that_does_not_add_up_ends_as_a_conflict():
    """inv_07: the same printed values after the re-check. Keep them, record the
    conflict, stop. No third call."""
    outcome, sent = run("inv_07.txt", [truth_input("inv_07.txt")] * 2)

    assert (outcome.status, len(sent)) == (Status.CONFLICT, 2)
    assert outcome.invoice == TRUTH["inv_07.txt"].invoice  # printed total 1203.00 kept
    assert outcome.changed_on_recheck == []
    assert [(i.code, i.field) for i in outcome.conflicts] == [
        (IssueCode.ITEMS_VS_BASE, "stated_total")
    ]


def test_the_recheck_happens_only_once():
    """An invalid answer to the re-check is retried, but it starts no second re-check."""
    consistent_but_wrong = truth_input("inv_07.txt")
    invalid = truth_input("inv_07.txt", stated_total="1.203,00")
    outcome, sent = run("inv_07.txt", [consistent_but_wrong, invalid, consistent_but_wrong])

    assert (outcome.status, len(sent)) == (Status.CONFLICT, 3)
    assert outcome.attempts[1].validation_error is not None


def test_invalid_and_inconsistent_have_separate_budgets():
    """Two invalid retries do not use up the re-check, and the re-check does not use up
    an invalid retry."""
    invalid = truth_input("inv_07.txt", stated_total="null")
    printed = truth_input("inv_07.txt")
    outcome, sent = run("inv_07.txt", [invalid, invalid, printed, printed])

    assert (outcome.status, len(sent)) == (Status.CONFLICT, 4)


def test_a_broken_recheck_keeps_the_last_valid_extraction():
    """The re-check answers are invalid until the retries run out: the earlier valid
    extraction is kept as a conflict instead of being thrown away."""
    printed = truth_input("inv_07.txt")
    invalid = truth_input("inv_07.txt", stated_total="1.203,00")
    outcome, sent = run("inv_07.txt", [printed, invalid, invalid, invalid])

    assert (outcome.status, len(sent)) == (Status.CONFLICT, 4)
    assert outcome.invoice == TRUTH["inv_07.txt"].invoice
    assert outcome.changed_on_recheck == []
    assert [i.code for i in outcome.conflicts] == [IssueCode.ITEMS_VS_BASE]


def test_attempts_record_usage_per_call():
    outcome, _ = run("inv_07.txt", [truth_input("inv_07.txt")] * 2)

    assert [a.usage.input_tokens for a in outcome.attempts] == [1000, 1000]


# --- the corpus run ---------------------------------------------------------------------


class ReplayTruth:
    """Answers every call with the ground truth of the document in the first message."""

    def create(self, **kwargs: Any) -> Message:
        text = kwargs["messages"][0]["content"]
        doc = next(d for d in sorted(TRUTH) if TRUTH[d].invoice.invoice_number in text)
        return reply(truth_input(doc))


@pytest.fixture
def replay_client() -> Anthropic:
    client = type("Client", (), {"messages": ReplayTruth()})()
    return cast(Anthropic, client)


def test_run_pipeline_reports_each_document_over_its_runs(replay_client: Anthropic, tmp_path: Path):
    out_file = tmp_path / "runs" / "pipeline.json"

    report = run_pipeline(replay_client, "some-model", out_file, runs=2)

    assert report.startswith("2 runs x 12 documents")
    assert "inv_01.txt   ok x2                1,1      24/24    -" in report
    assert (
        "inv_07.txt   conflict x2          2,2      24/24    -                      "
        "items_vs_base x2"
    ) in report
    assert "overall: 288/288 fields correct" in report
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(saved["outcomes"]) == 24
    assert saved["outcomes"][0]["changed_on_recheck"] == []
