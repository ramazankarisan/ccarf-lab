import json
from typing import Any

import pytest
from claude_agent_sdk import PostToolUseHookInput, PreToolUseHookInput

from ex4_research.completeness import (
    FINDINGS_HEADER,
    FindingsLedger,
    make_completeness_hooks,
    response_text,
)
from ex4_research.findings import Finding, NoSources, valid_findings
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


def test_attach_to_appends_every_finding_after_the_coordinators_text():
    empty = NoSources.model_validate(NO_SOURCES)
    ledger = FindingsLedger(received=[WEB, DOC, empty])
    prompt = ledger.attach_to("Question: how big?\nProcess notes: the archive timed out.\n")
    assert prompt.startswith("Question: how big?\nProcess notes: the archive timed out.\n\n")
    assert FINDINGS_HEADER in prompt
    assert valid_findings(prompt) == [WEB, DOC, empty]  # exactly, nothing retyped


def test_attach_to_with_no_findings_says_so_in_json():
    assert FindingsLedger().attach_to("Q").endswith(f"{FINDINGS_HEADER}\n[]")


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


async def attach(ledger: FindingsLedger, hook_input: PreToolUseHookInput) -> Any:
    _, attach_findings = make_completeness_hooks(ledger)
    return await attach_findings(hook_input, "t", {"signal": None})


@pytest.mark.anyio
async def test_the_synthesizer_prompt_gets_every_finding_from_code():
    """Live runs: a coordinator copying 50 findings changed one each time."""
    ledger = FindingsLedger(received=[WEB, DOC])
    output = await attach(ledger, pre_spawn(SYNTHESIZER, "Question: how big? Notes: none."))
    specific = output["hookSpecificOutput"]
    assert specific["permissionDecision"] == "allow"
    updated = specific["updatedInput"]
    assert updated["subagent_type"] == SYNTHESIZER
    assert updated["prompt"].startswith("Question: how big? Notes: none.\n\n")
    assert valid_findings(updated["prompt"]) == [WEB, DOC]
    assert updated["run_in_background"] is False
    assert ledger.attached == [2]


@pytest.mark.anyio
async def test_findings_the_coordinator_dropped_or_edited_are_still_attached():
    ledger = FindingsLedger(received=[WEB, DOC])
    edited = DOC.model_copy(update={"claim": "About EUR 2 billion."})
    output = await attach(ledger, pre_spawn(SYNTHESIZER, f"Findings: {as_json(edited)}"))
    attached = valid_findings(output["hookSpecificOutput"]["updatedInput"]["prompt"])
    assert WEB in attached
    assert DOC in attached


@pytest.mark.anyio
async def test_other_spawns_are_left_alone():
    ledger = FindingsLedger(received=[WEB])
    assert await attach(ledger, pre_spawn("doc_analyst", "Check it.")) == {}
    hook_input = pre_spawn(SYNTHESIZER, "")
    hook_input["tool_input"] = "not a dict"  # type: ignore[typeddict-item]
    assert await attach(ledger, hook_input) == {}
    assert ledger.attached == []
