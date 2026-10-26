"""The batch flow: custom_ids, polling, matching by custom_id, and the second batch."""

import json
from pathlib import Path
from typing import Any, cast

import pytest
from anthropic import Anthropic
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse

from ex2_extraction.batch import (
    BROKEN_DOC,
    BROKEN_TOOL_CHOICE,
    DOCS,
    Status,
    custom_id,
    request,
    run,
    wait,
)
from ex2_extraction.extractor import FORCE_RECORD_INVOICE, TOOL_NAME
from ex2_extraction.ground_truth import load_ground_truth
from ex2_extraction.variants import B

TRUTH = load_ground_truth()
NULL_ITEM = {"description": "Taxi ride", "quantity": "null", "unit_price": "null", "amount": 23.4}


def truth_input(doc: str, **changes: Any) -> dict[str, Any]:
    return TRUTH[doc].invoice.model_dump(mode="json") | changes


# First answers that need a second round; every other first answer is the ground truth.
FIRST_ANSWERS = {
    "inv_03.txt": truth_input("inv_03.txt", line_items=[NULL_ITEM]),  # invalid
    "inv_07.txt": truth_input("inv_07.txt"),  # inconsistent, and stays so: conflict
    "inv_09.txt": truth_input("inv_09.txt", stated_total="1400.00"),  # a misread
}


def batch(batch_id: str, status: str) -> MessageBatch:
    counts = {"processing": 1 if status != "ended" else 0, "succeeded": 0}
    counts |= {"errored": 0, "canceled": 0, "expired": 0}
    return MessageBatch.model_validate(
        {
            "id": batch_id,
            "type": "message_batch",
            "processing_status": status,
            "request_counts": counts,
            "created_at": "2026-10-26T12:00:00Z",
            "expires_at": "2026-10-27T12:00:00Z",
            "ended_at": None,
            "archived_at": None,
            "cancel_initiated_at": None,
            "results_url": None,
        }
    )


class FakeBatches:
    """Answers like the Batches API: one 'in_progress' poll, then 'ended', and the
    results in reverse order, so matching by position would go wrong."""

    def __init__(self, lenient: bool = False, still_invalid: frozenset[str] = frozenset()) -> None:
        self.submitted: list[list[Any]] = []
        self.polls: dict[str, int] = {}
        self.lenient = lenient  # accept any tool_choice and answer every first round right
        self.still_invalid = still_invalid  # documents that stay invalid in round 2

    def create(self, requests: list[Any]) -> MessageBatch:
        self.submitted.append(requests)
        batch_id = f"msgbatch_{len(self.submitted)}"
        self.polls[batch_id] = 0
        return batch(batch_id, "in_progress")

    def retrieve(self, batch_id: str) -> MessageBatch:
        self.polls[batch_id] += 1
        return batch(batch_id, "ended" if self.polls[batch_id] > 1 else "in_progress")

    def results(self, batch_id: str) -> list[MessageBatchIndividualResponse]:
        requests = self.submitted[int(batch_id.split("_")[1]) - 1]
        return [self.answer(r) for r in reversed(requests)]

    def answer(self, req: Any) -> MessageBatchIndividualResponse:
        cid, params = req["custom_id"], req["params"]
        if params["tool_choice"]["name"] != TOOL_NAME and not self.lenient:
            error = {
                "type": "invalid_request_error",
                "message": "tool_choice names an unknown tool",
            }
            outcome: dict[str, Any] = {
                "type": "errored",
                "error": {"type": "error", "error": error, "request_id": None},
            }
        else:
            doc = f"{cid}.txt"
            first_round = len(params["messages"]) == 1
            if self.lenient and first_round:
                # every answer valid and consistent, inv_07 included: nothing to resubmit
                tool_input = truth_input(doc) | (
                    {"stated_total": "1163.00"} if doc == "inv_07.txt" else {}
                )
            elif doc in self.still_invalid and not first_round:
                tool_input = FIRST_ANSWERS[doc]
            else:
                tool_input = (
                    FIRST_ANSWERS.get(doc, truth_input(doc)) if first_round else truth_input(doc)
                )
            content = [
                {"type": "tool_use", "id": f"t_{cid}", "name": TOOL_NAME, "input": tool_input}
            ]
            message = {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "model": params["model"],
                "content": content,
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 3000, "output_tokens": 450},
            }
            outcome = {"type": "succeeded", "message": message}
        return MessageBatchIndividualResponse.model_validate({"custom_id": cid, "result": outcome})


@pytest.fixture
def fake() -> FakeBatches:
    return FakeBatches()


@pytest.fixture
def report(fake: FakeBatches, tmp_path: Path) -> str:
    client = type("Client", (), {"messages": type("M", (), {"batches": fake})()})()
    return run(
        cast(Anthropic, client),
        "some-model",
        tmp_path / "batch.json",
        sleep=lambda _: None,
        log=lambda _: None,
    )


def test_custom_ids_are_document_names_without_the_dot():
    assert custom_id("inv_01.txt") == "inv_01"
    with pytest.raises(ValueError, match="not a valid custom_id"):
        custom_id("inv 01.txt")


def test_requests_use_variant_b_and_the_forced_tool():
    req = request("inv_01.txt", "some-model", [{"role": "user", "content": "x"}])

    assert req["custom_id"] == "inv_01"
    assert req["params"]["tools"] == [B.tool]
    assert req["params"]["system"] == B.system_prompt
    assert req["params"]["tool_choice"] == FORCE_RECORD_INVOICE


def test_wait_polls_until_the_batch_has_ended(fake: FakeBatches):
    fake.create([])
    slept: list[float] = []
    client = type("Client", (), {"messages": type("M", (), {"batches": fake})()})()

    ended = wait(cast(Anthropic, client), "msgbatch_1", sleep=slept.append, log=lambda _: None)

    assert ended.processing_status == "ended"
    assert slept == [15]


def test_round_one_sends_ten_documents_and_one_broken_request(fake: FakeBatches, report: str):
    first = fake.submitted[0]

    assert [r["custom_id"] for r in first] == [*(d.removesuffix(".txt") for d in DOCS), "inv_11"]
    assert first[-1]["params"]["tool_choice"] == BROKEN_TOOL_CHOICE
    assert all(r["params"]["tool_choice"] == FORCE_RECORD_INVOICE for r in first[:-1])


def test_only_what_needs_it_is_resubmitted(fake: FakeBatches, report: str):
    second = {r["custom_id"]: r["params"] for r in fake.submitted[1]}

    assert sorted(second) == ["inv_03", "inv_07", "inv_09", "inv_11"]
    assert second["inv_11"]["tool_choice"] == FORCE_RECORD_INVOICE  # fixed
    assert len(second["inv_11"]["messages"]) == 1  # sent fresh


def test_an_invalid_extraction_is_resubmitted_with_its_errors(fake: FakeBatches, report: str):
    messages = {r["custom_id"]: r["params"]["messages"] for r in fake.submitted[1]}["inv_03"]
    document, failed, feedback = messages

    assert "<document>" in document["content"]
    assert failed["role"] == "assistant"
    result = feedback["content"][0]
    assert (result["type"], result["is_error"], result["tool_use_id"]) == (
        "tool_result",
        True,
        "t_inv_03",
    )
    assert "line_items.0.quantity" in result["content"]


def test_an_inconsistent_extraction_is_resubmitted_with_the_recheck(fake: FakeBatches, report: str):
    messages = {r["custom_id"]: r["params"]["messages"] for r in fake.submitted[1]}["inv_07"]

    assert "Do not change values to make them agree" in messages[-1]["content"][0]["content"]


def test_results_are_matched_by_custom_id_and_reported(report: str, tmp_path: Path):
    """The fake returns results in reverse order; every row still belongs to its document."""
    assert "batch 1: msgbatch_1" in report
    assert "batch 2: msgbatch_2" in report
    assert "inv_01     ok            -             12/12    -" in report
    assert "inv_03     invalid       ok            12/12" in report
    assert "inv_07     inconsistent  conflict      12/12" in report
    assert "inv_09     inconsistent  ok            12/12" in report
    assert "changed: stated_total" in report
    assert (
        "inv_11     errored       ok            12/12    tool_choice names an unknown tool"
        in report
    )
    assert "resubmitted: inv_03, inv_07, inv_09, inv_11" in report
    assert "final: 132/132 fields correct" in report
    # 10 first-round answers + 4 second-round answers; the errored request used no tokens
    assert "tokens: 42000 in, 6300 out" in report

    saved = json.loads((tmp_path / "batch.json").read_text(encoding="utf-8"))
    assert saved["batches"] == ["msgbatch_1", "msgbatch_2"]
    statuses = {r["custom_id"]: [x["status"] for x in r["rounds"]] for r in saved["records"]}
    assert statuses["inv_07"] == [Status.INCONSISTENT, Status.CONFLICT]
    assert statuses["inv_11"] == [Status.ERRORED, Status.OK]


def test_the_broken_document_is_the_eleventh():
    assert BROKEN_DOC not in DOCS
    assert len(DOCS) == 10


def run_with(fake: FakeBatches, tmp_path: Path) -> str:
    client = type("Client", (), {"messages": type("M", (), {"batches": fake})()})()
    return run(
        cast(Anthropic, client),
        "some-model",
        tmp_path / "batch.json",
        sleep=lambda _: None,
        log=lambda _: None,
    )


def test_an_extraction_still_invalid_after_the_resubmit_is_a_failure(tmp_path: Path):
    report = run_with(FakeBatches(still_invalid=frozenset({"inv_03.txt"})), tmp_path)

    assert "inv_03     invalid       failed         0/12" in report


def test_when_nothing_needs_it_there_is_no_second_batch(tmp_path: Path):
    fake = FakeBatches(lenient=True)

    report = run_with(fake, tmp_path)

    assert len(fake.submitted) == 1
    assert "batch 2: - (nothing to resubmit)" in report
    assert "resubmitted: nothing" in report
