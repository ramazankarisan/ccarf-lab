import json
from typing import Any

import pytest
from claude_agent_sdk import PostToolUseHookInput, PreToolUseHookInput

from ex4_research.completeness import (
    MAX_LISTED,
    FindingsLedger,
    denial_reason,
    make_completeness_hooks,
    response_text,
)
from ex4_research.findings import Finding, NoSources
from ex4_research.timeline import SPAWN_TOOL, SYNTHESIZER
from tests.test_ex4_findings import DOC_FINDING, NO_SOURCES, WEB_FINDING

WEB = Finding.model_validate(WEB_FINDING)
DOC = Finding.model_validate(DOC_FINDING)


def post_spawn(subagent: str, tool_response: object) -> PostToolUseHookInput:
    return {
        "hook_event_name": "PostToolUse",
        "session_id": "session-test",
        "transcript_path": "/dev/null",
        "cwd": "/project",
        "tool_name": SPAWN_TOOL,
        "tool_input": {"subagent_type": subagent, "prompt": "Research it."},
        "tool_response": tool_response,
        "tool_use_id": "toolu_1",
    }


def pre_spawn(subagent: str, prompt: str) -> PreToolUseHookInput:
    return {
        "hook_event_name": "PreToolUse",
        "session_id": "session-test",
        "transcript_path": "/dev/null",
        "cwd": "/project",
        "tool_name": SPAWN_TOOL,
        "tool_input": {"subagent_type": subagent, "prompt": prompt},
        "tool_use_id": "toolu_2",
    }


def as_json(*findings: Finding) -> str:
    return json.dumps([f.model_dump(mode="json") for f in findings])


# --- response_text ------------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "text"),
    [
        ("plain", "plain"),
        (
            {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            "text\na\ntext\nb",
        ),
        ({"status": "completed", "totalDurationMs": 12}, "completed\n"),
        (None, ""),
    ],
)
def test_response_text_collects_every_string(response, text):
    assert response_text(response) == text


# --- ledger -------------------------------------------------------------------


def test_ledger_records_each_finding_once():
    ledger = FindingsLedger()
    ledger.record([WEB, DOC])
    ledger.record([WEB])
    assert ledger.received == [WEB, DOC]


def test_nothing_is_missing_when_every_finding_is_in_the_prompt():
    ledger = FindingsLedger(received=[WEB, DOC])
    assert ledger.missing_from(f"Question?\n{as_json(WEB)}\nand\n{as_json(DOC)}") == []


def test_a_dropped_or_edited_finding_is_missing():
    ledger = FindingsLedger(received=[WEB, DOC])
    edited = DOC.model_copy(update={"claim": "About EUR 2 billion."})
    assert ledger.missing_from(as_json(WEB, edited)) == [DOC]


def test_an_invalid_finding_in_the_prompt_does_not_hide_the_valid_ones():
    """Run 5 of the live timing runs: one malformed finding made all 29 count as missing."""
    ledger = FindingsLedger(received=[WEB, DOC])
    broken = {**DOC_FINDING, "source_type": "report"}
    prompt = json.dumps([WEB.model_dump(mode="json"), broken, DOC.model_dump(mode="json")])
    assert ledger.missing_from(prompt) == []


def test_findings_count_however_the_coordinator_nests_them():
    """Two live runs wrapped the result objects in an outer array; nothing was missing."""
    ledger = FindingsLedger(received=[WEB, DOC])
    wrapped = [
        {
            "subagent": "web_researcher",
            "result": {"status": "complete", "findings": [WEB.model_dump(mode="json")]},
        },
        {"status": "complete", "findings": [DOC.model_dump(mode="json")]},
    ]
    assert ledger.missing_from(f"Findings:\n{json.dumps(wrapped, indent=2)}") == []


def test_a_prompt_without_findings_misses_them_all():
    ledger = FindingsLedger(received=[WEB, DOC])
    assert ledger.missing_from("Summarize what you know.") == [WEB, DOC]


def test_denial_reason_names_the_missing_findings_and_counts_the_rest():
    reason = denial_reason([DOC] * (MAX_LISTED + 2), total=12, read=2)
    assert reason.startswith(f"Synthesizer prompt is missing {MAX_LISTED + 2} of the 12 findings")
    assert "(it contains 2 valid findings)" in reason
    assert "[nordwell-eu-outlook-2026] EU-27 EV recycling revenue" in reason
    assert reason.endswith("; and 2 more.")
    assert denial_reason([WEB], total=1).endswith("in 2025.")


# --- hooks --------------------------------------------------------------------


@pytest.mark.anyio
async def test_recorder_keeps_findings_from_researcher_hand_backs():
    ledger = FindingsLedger()
    record, _ = make_completeness_hooks(ledger)
    envelope = json.dumps({"status": "complete", "findings": [WEB.model_dump(mode="json")]})
    response = {"content": [{"type": "text", "text": envelope}]}
    assert await record(post_spawn("web_researcher", response), "t", {"signal": None}) == {}
    partial = {
        "status": "partial",
        "failure_type": "timeout",
        "attempted_queries": ["q"],
        "partial_findings": [DOC.model_dump(mode="json"), NO_SOURCES],
        "suggested_alternatives": [],
    }
    await record(post_spawn("doc_analyst", json.dumps(partial)), "t", {"signal": None})
    assert ledger.received == [WEB, DOC, NoSources.model_validate(NO_SOURCES)]


@pytest.mark.anyio
async def test_a_no_sources_finding_must_reach_the_synthesizer_too():
    empty = NoSources.model_validate(NO_SOURCES)
    ledger = FindingsLedger(received=[WEB, empty])
    _, guard = make_completeness_hooks(ledger)
    output = await guard(pre_spawn(SYNTHESIZER, as_json(WEB)), "t", {"signal": None})
    reason = output["hookSpecificOutput"]["permissionDecisionReason"]  # type: ignore[typeddict-item]
    assert "[search: EU battery tonnage recycled 2023] No sources found" in reason


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("subagent", "response"),
    [
        (SYNTHESIZER, "A report that quotes " + as_json(WEB)),  # not a researcher
        ("web_researcher", "I found nothing usable."),  # no valid findings
    ],
)
async def test_recorder_ignores_what_is_not_researcher_findings(subagent, response):
    ledger = FindingsLedger()
    record, _ = make_completeness_hooks(ledger)
    await record(post_spawn(subagent, response), "t", {"signal": None})
    assert ledger.received == []


@pytest.mark.anyio
async def test_guard_denies_a_synthesizer_prompt_with_missing_findings():
    ledger = FindingsLedger(received=[WEB, DOC])
    _, guard = make_completeness_hooks(ledger)
    output = await guard(pre_spawn(SYNTHESIZER, as_json(WEB)), "t", {"signal": None})
    specific: Any = output["hookSpecificOutput"]  # type: ignore[typeddict-item]
    assert specific["permissionDecision"] == "deny"
    assert "missing 1 of the 2 findings" in specific["permissionDecisionReason"]
    assert ledger.denials == [specific["permissionDecisionReason"]]


@pytest.mark.anyio
async def test_guard_allows_a_complete_prompt_and_other_spawns():
    ledger = FindingsLedger(received=[WEB, DOC])
    _, guard = make_completeness_hooks(ledger)
    assert await guard(pre_spawn(SYNTHESIZER, as_json(WEB, DOC)), "t", {"signal": None}) == {}
    assert await guard(pre_spawn("doc_analyst", "Check it."), "t", {"signal": None}) == {}
    assert ledger.denials == []


@pytest.mark.anyio
async def test_guard_ignores_a_spawn_without_a_dict_input():
    ledger = FindingsLedger(received=[WEB])
    _, guard = make_completeness_hooks(ledger)
    hook_input = pre_spawn(SYNTHESIZER, "")
    hook_input["tool_input"] = "not a dict"  # type: ignore[typeddict-item]
    assert await guard(hook_input, "t", {"signal": None}) == {}
