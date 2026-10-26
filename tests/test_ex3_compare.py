"""Comparing variants from their saved runs, scored by the same code."""

from pathlib import Path
from typing import Any, cast

import pytest
from anthropic import Anthropic
from anthropic.types import Message

from ex3_extraction.compare import compare, load_runs
from ex3_extraction.extractor import TOOL_NAME
from ex3_extraction.ground_truth import load_ground_truth
from ex3_extraction.pipeline import run_pipeline
from ex3_extraction.variants import VARIANTS

TRUTH = load_ground_truth()


class Replay:
    """Answers with the ground truth, except catering filed as services when told to,
    and charges a fixed number of input tokens per call."""

    def __init__(self, misfile_catering: bool, input_tokens: int) -> None:
        self.misfile_catering = misfile_catering
        self.input_tokens = input_tokens

    def create(self, **kwargs: Any) -> Message:
        text = kwargs["messages"][0]["content"]
        doc = next(d for d in sorted(TRUTH) if TRUTH[d].invoice.invoice_number in text)
        tool_input = TRUTH[doc].invoice.model_dump(mode="json")
        if self.misfile_catering and doc == "inv_06.txt":
            tool_input |= {"category": "services", "category_detail": None}
        return Message.model_validate(
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "model": "some-model",
                "content": [
                    {"type": "tool_use", "id": "t", "name": TOOL_NAME, "input": tool_input}
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": self.input_tokens, "output_tokens": 400},
            }
        )


def save_runs(runs_dir: Path, name: str, misfile_catering: bool, input_tokens: int) -> None:
    client = type("Client", (), {"messages": Replay(misfile_catering, input_tokens)})()
    run_pipeline(cast(Anthropic, client), "some-model", runs_dir, VARIANTS[name], runs=3)


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    save_runs(tmp_path, "A", misfile_catering=True, input_tokens=2000)
    save_runs(tmp_path, "C", misfile_catering=False, input_tokens=2500)
    return tmp_path


def test_load_runs_reads_every_run_in_order(runs_dir: Path):
    runs = load_runs(runs_dir, "A")

    assert len(runs) == 3
    assert [o.doc for o in runs[0]] == sorted(TRUTH)


def test_load_runs_without_files_is_an_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no runs for variant B"):
        load_runs(tmp_path, "B")


def test_compare_shows_documents_fields_pipeline_and_cost(runs_dir: Path):
    report = compare(runs_dir, ["A", "C"])

    assert (
        "inv_06.txt         30/36       36/36  "
        "A: category (wrong) 3/3, category_detail (omission) 3/3"
    ) in report
    assert "inv_01.txt         36/36       36/36  -" in report
    assert "overall          426/432     432/432" in report
    assert "category                 92%        100%" in report
    assert "status conflict                      3           3" in report
    assert "changed_on_recheck                   0           0" in report
    # inv_07 takes 2 calls a run: (11 x 1 + 2) / 12
    assert "calls per document                1.08        1.08" in report
    assert "input tokens per document         2167        2708" in report
    assert "runs                                 3           3" in report
