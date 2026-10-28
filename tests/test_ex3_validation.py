import json
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import SubagentStopHookInput

from ex3_research.validation import (
    MAX_ERRORS,
    HandbackLog,
    correction_request,
    handback_errors,
    make_handback_validator,
)
from tests.test_ex3_findings import DOC_FINDING, NO_SOURCES, PARTIAL, WEB_FINDING

EMPTY: dict[str, Any] = {"status": "complete", "findings": []}
VALID_WEB = json.dumps({"status": "complete", "findings": [WEB_FINDING, NO_SOURCES]})
# The shape Step 3's archive-down re-run handed back (#2 web_researcher): a partial
# result without suggested_alternatives. Its raw text was not saved, so this is
# rebuilt from the error that run printed.
NO_ALTERNATIVES = json.dumps({k: v for k, v in PARTIAL.items() if k != "suggested_alternatives"})


def stop(
    agent_type: str, message: str, *, active: bool = False, agent_id: str = "agent-1"
) -> SubagentStopHookInput:
    return {
        "hook_event_name": "SubagentStop",
        "session_id": "session-test",
        "transcript_path": "/dev/null",
        "cwd": "/project",
        "stop_hook_active": active,
        "agent_id": agent_id,
        "agent_transcript_path": "/dev/null",
        "agent_type": agent_type,
        "last_assistant_message": message,
    }


# --- handback_errors -----------------------------------------------------------


@pytest.mark.parametrize(
    ("agent_type", "text"),
    [
        ("web_researcher", VALID_WEB),
        ("web_researcher", json.dumps(PARTIAL)),
        ("doc_analyst", json.dumps({"status": "complete", "findings": [DOC_FINDING]})),
        ("doc_analyst", f"Here it is:\n```json\n{json.dumps(EMPTY)}\n```"),
    ],
)
def test_a_valid_hand_back_has_no_errors(agent_type, text):
    assert handback_errors(agent_type, text) == []


@pytest.mark.parametrize(
    ("agent_type", "text", "errors"),
    [
        (
            "web_researcher",
            json.dumps([WEB_FINDING]),
            ['(whole hand-back): no JSON object starting with {"status": ...} was found'],
        ),
        ("web_researcher", NO_ALTERNATIVES, ["partial.suggested_alternatives: Field required"]),
        (
            "web_researcher",
            json.dumps({**EMPTY, "findings": [{**WEB_FINDING, "evidence_excerpt": ""}]}),
            ["complete.findings.0.web.evidence_excerpt: String should have at least 1 character"],
        ),
        (
            "doc_analyst",
            json.dumps({"status": "done"}),
            [
                "(top level): Input tag 'done' found using 'status' does not match any of the "
                "expected tags: 'complete', 'partial'"
            ],
        ),
        (
            "web_researcher",
            json.dumps({**EMPTY, "findings": [DOC_FINDING]}),
            [
                "nordwell-eu-outlook-2026: source_type is 'doc', but web_researcher can only "
                "report 'web' sources; mention other sources inside a claim instead"
            ],
        ),
    ],
)
def test_an_unusable_hand_back_lists_each_field_and_problem(agent_type, text, errors):
    assert handback_errors(agent_type, text) == errors


def test_every_error_is_listed_not_only_the_first():
    broken = {**PARTIAL, "failure_type": "network", "attempted_queries": []}
    del broken["suggested_alternatives"]
    errors = handback_errors("web_researcher", json.dumps(broken))
    assert [e.split(":")[0] for e in errors] == [
        "partial.failure_type",
        "partial.attempted_queries",
        "partial.suggested_alternatives",
    ]


def test_an_undecodable_object_is_skipped_for_the_next_one():
    text = '{"status": broken} then ' + json.dumps(EMPTY)
    assert handback_errors("doc_analyst", text) == []


def test_correction_request_lists_errors_and_forbids_new_tool_calls():
    request = correction_request(["partial.suggested_alternatives: Field required"])
    assert request.startswith(
        "Your hand-back does not match the result format. Errors:\n"
        "- partial.suggested_alternatives: Field required\n"
    )
    assert "keeping every finding and every query you tried" in request
    assert request.endswith("Do not call any more tools.")


def test_correction_request_counts_errors_beyond_the_limit():
    request = correction_request([f"field{n}: bad" for n in range(MAX_ERRORS + 3)])
    assert f"- field{MAX_ERRORS - 1}: bad" in request
    assert f"field{MAX_ERRORS}: bad" not in request
    assert "- and 3 more" in request


# --- the hook -------------------------------------------------------------------


async def run_hook(log: HandbackLog, hook_input: SubagentStopHookInput) -> Any:
    return await make_handback_validator(log)(hook_input, None, {"signal": None})


@pytest.mark.anyio
async def test_an_invalid_hand_back_is_blocked_once_with_its_errors():
    log = HandbackLog()
    output = await run_hook(log, stop("web_researcher", NO_ALTERNATIVES))
    assert output["decision"] == "block"
    assert "- partial.suggested_alternatives: Field required" in output["reason"]
    [correction] = log.corrections
    assert (correction.agent_type, correction.agent_id) == ("web_researcher", "agent-1")
    assert correction.handback == NO_ALTERNATIVES
    assert correction.outcome == "pending"


@pytest.mark.anyio
async def test_a_corrected_second_hand_back_passes():
    log = HandbackLog()
    await run_hook(log, stop("web_researcher", NO_ALTERNATIVES))
    assert await run_hook(log, stop("web_researcher", VALID_WEB, active=True)) == {}
    [correction] = log.corrections
    assert correction.outcome == "passed"
    assert correction.final_errors == []


@pytest.mark.anyio
async def test_a_second_invalid_hand_back_is_let_through_as_format_failed():
    log = HandbackLog()
    await run_hook(log, stop("doc_analyst", "[]"))
    assert await run_hook(log, stop("doc_analyst", "still []", active=True)) == {}
    [correction] = log.corrections
    assert correction.outcome == "format_failed"
    assert correction.final_errors == [
        '(whole hand-back): no JSON object starting with {"status": ...} was found'
    ]


@pytest.mark.anyio
async def test_the_second_stop_counts_even_without_stop_hook_active():
    """Pairing by agent_id: never a second block for the same subagent."""
    log = HandbackLog()
    await run_hook(log, stop("doc_analyst", "[]"))
    assert await run_hook(log, stop("doc_analyst", "[]")) == {}
    assert log.corrections[0].outcome == "format_failed"


@pytest.mark.anyio
async def test_parallel_subagents_of_one_type_are_tracked_apart():
    log = HandbackLog()
    await run_hook(log, stop("web_researcher", NO_ALTERNATIVES, agent_id="a"))
    await run_hook(log, stop("web_researcher", "[]", agent_id="b"))
    await run_hook(log, stop("web_researcher", VALID_WEB, active=True, agent_id="b"))
    assert [(c.agent_id, c.outcome) for c in log.corrections] == [("a", "pending"), ("b", "passed")]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "hook_input",
    [
        stop("web_researcher", VALID_WEB),
        stop("web_researcher", VALID_WEB, active=True),  # another hook blocked it: no record
        stop("synthesizer", "# Report\n\n## Established\n- ..."),  # Markdown, not a result
        stop("general-purpose", "anything"),
    ],
)
async def test_valid_researchers_and_other_agents_may_stop(hook_input):
    log = HandbackLog()
    assert await run_hook(log, hook_input) == {}
    assert log.corrections == []


@pytest.mark.anyio
async def test_a_stop_without_a_last_message_is_blocked():
    hook_input = stop("web_researcher", "")
    del hook_input["last_assistant_message"]
    output = await run_hook(HandbackLog(), hook_input)
    assert output["decision"] == "block"


# --- hand-backs captured from a live run ----------------------------------------

CAPTURED = json.loads(
    (Path(__file__).parent / "fixtures" / "ex3_broken_handbacks.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("captured", CAPTURED, ids=[c["agent_type"] for c in CAPTURED])
def test_captured_broken_hand_backs_get_the_errors_the_live_hook_sent(captured):
    """A web_researcher's partial result without suggested_alternatives (the same gap as
    Step 3's archive-down run), and a doc_analyst that returned the coordinator's own
    verification fields instead of a result object."""
    assert handback_errors(captured["agent_type"], captured["handback"]) == captured["errors"]


@pytest.mark.anyio
async def test_a_captured_broken_hand_back_is_blocked_with_its_errors():
    captured = CAPTURED[0]
    output = await run_hook(HandbackLog(), stop(captured["agent_type"], captured["handback"]))
    assert output["decision"] == "block"
    assert "- partial.suggested_alternatives: Field required" in output["reason"]
