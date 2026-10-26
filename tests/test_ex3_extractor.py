"""The extraction tool, the zero-shot prompt and the call."""

from typing import Any, cast

import pytest
from anthropic import Anthropic
from anthropic.types import Message

from ex3_extraction.extractor import (
    FORCE_RECORD_INVOICE,
    RECORD_INVOICE,
    SYSTEM_PROMPT,
    TOOL_NAME,
    ExtractionResult,
    extract,
    extract_document,
    parse_invoice,
    to_result,
)
from ex3_extraction.ground_truth import load_ground_truth
from ex3_extraction.schema import Invoice


def valid_input() -> dict[str, Any]:
    """A valid tool input as plain JSON data."""
    return load_ground_truth()["inv_06.txt"].invoice.model_dump(mode="json")


def test_the_tool_schema_is_the_invoice_schema():
    """$defs and $ref stay in: the API accepts them (checked live with one call)."""
    schema = Invoice.model_json_schema()

    assert RECORD_INVOICE["input_schema"] == schema
    assert {"LineItem", "Category"} <= set(schema["$defs"])


def test_the_named_tool_is_forced():
    assert FORCE_RECORD_INVOICE == {"type": "tool", "name": TOOL_NAME}
    assert RECORD_INVOICE["name"] == TOOL_NAME


def test_a_valid_input_parses():
    invoice, error = parse_invoice(valid_input())

    assert error is None
    assert invoice is not None
    assert invoice.category_detail == "catering"


def test_a_schema_compliant_input_can_still_be_invalid():
    """JSON Schema allows a detail on any category; the cross-field rule does not."""
    tool_input = valid_input() | {"category": "hardware"}

    invoice, error = parse_invoice(tool_input)

    assert invoice is None
    assert error is not None
    assert "category_detail must be null" in error


def test_numbers_from_the_model_become_exact_decimals():
    tool_input = valid_input() | {"stated_total": 982.5}

    invoice, _ = parse_invoice(tool_input)

    assert invoice is not None
    assert str(invoice.stated_total) == "982.5"


# --- The prompt and the call ------------------------------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        "If a value is absent, use null. Never guess.",
        "Do not compute, add up or correct them",
        "ISO 8601",
        "document's own cues",
        "is never the vendor's",
        'use "other" and name the kind of expense in category_detail',
    ],
)
def test_the_prompt_keeps_each_rule(rule: str):
    assert rule in SYSTEM_PROMPT


def test_the_baseline_prompt_has_no_examples():
    """Step 4 measures what few-shot adds, so the baseline must not have any."""
    assert "<example" not in SYSTEM_PROMPT


def test_the_prompt_shares_nothing_with_the_corpus():
    """A value from the corpus in the prompt would leak answers into the score."""
    for entry in load_ground_truth().values():
        invoice = entry.invoice
        for value in (invoice.vendor_name, invoice.invoice_number, invoice.vendor_tax_id):
            assert value is None or value not in SYSTEM_PROMPT
    for date_hint in ("03/04", "19.10"):
        assert date_hint not in SYSTEM_PROMPT


def response(content: list[dict[str, Any]], stop_reason: str = "tool_use") -> Message:
    """A real SDK Message, as the API would return it."""
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "some-model",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1500, "output_tokens": 480},
        }
    )


def tool_call(tool_input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": "toolu_test", "name": TOOL_NAME, "input": tool_input}


class FakeMessages:
    def __init__(self, reply: object = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.reply = object() if reply is None else reply

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self.reply


class FakeClient:
    def __init__(self, reply: object = None) -> None:
        self.messages = FakeMessages(reply)


def test_extract_forces_the_tool_and_wraps_the_document():
    fake = FakeClient()

    response = extract(cast(Anthropic, fake), "some-model", "INVOICE 1")

    assert response is fake.messages.reply
    call = fake.messages.calls[0]
    assert call["model"] == "some-model"
    assert call["system"] == SYSTEM_PROMPT
    assert call["tools"] == [RECORD_INVOICE]
    assert call["tool_choice"] == FORCE_RECORD_INVOICE
    assert call["messages"] == [{"role": "user", "content": "<document>\nINVOICE 1\n</document>"}]


# --- Results ----------------------------------------------------------------------


def test_a_valid_call_becomes_a_result_with_the_invoice():
    result = to_result("inv_06.txt", response([tool_call(valid_input())]))

    assert result.invoice is not None
    assert result.validation_error is None
    assert result.raw_input == valid_input()
    assert (result.usage.input_tokens, result.usage.output_tokens) == (1500, 480)


def test_an_invalid_call_keeps_the_raw_input_and_the_error():
    """The failure is recorded for scoring; nothing is retried or repaired (Step 3)."""
    bad_input = valid_input() | {"category": "hardware"}

    result = to_result("inv_06.txt", response([tool_call(bad_input)]))

    assert result.invoice is None
    assert result.raw_input == bad_input
    assert result.validation_error is not None
    assert "category_detail must be null" in result.validation_error


def test_a_response_without_the_tool_call_is_recorded_not_raised():
    result = to_result(
        "inv_06.txt", response([{"type": "text", "text": "Sorry"}], stop_reason="end_turn")
    )

    assert result.invoice is None
    assert result.raw_input == {}
    assert result.validation_error == "no record_invoice call (stop_reason: end_turn)"


def test_a_result_survives_a_round_trip_through_json():
    """runs/baseline.json stores results; loading one back must give the same result."""
    result = to_result("inv_06.txt", response([tool_call(valid_input())]))

    assert ExtractionResult.model_validate_json(result.model_dump_json()) == result


def test_extract_document_sends_the_corpus_text():
    fake = FakeClient(response([tool_call(valid_input())]))

    result = extract_document(cast(Anthropic, fake), "some-model", "inv_06.txt")

    assert result.doc == "inv_06.txt"
    assert "Café Lindgren Catering" in fake.messages.calls[0]["messages"][0]["content"]
