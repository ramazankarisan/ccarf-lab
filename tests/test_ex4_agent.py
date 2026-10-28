import json
from typing import Any

import pytest
from claude_agent_sdk import PreToolUseHookInput
from mcp import Client

from ex4_research.agent import (
    COORDINATOR_MAX_TURNS,
    FIXED_FORMAT,
    LOAD_DOCUMENT,
    MODE_RULES,
    MODES,
    SERVER_NAME,
    SPAWN_MATCHER,
    SYNTHESIZER_PROMPT,
    WEB_SEARCH,
    allow_owner_only,
    build_agents,
    build_options,
    coordinator_prompt,
    keep_in_foreground,
    mcp_tool_name,
    tool_decision,
)
from ex4_research.completeness import FindingsLedger
from ex4_research.findings import RESULT_FORMAT, Finding, valid_findings
from ex4_research.server import mcp
from ex4_research.timeline import REPORT_SECTIONS, SPAWN_TOOL, SYNTHESIZER
from ex4_research.validation import HandbackLog
from tests.test_ex4_completeness import post_spawn, pre_spawn
from tests.test_ex4_findings import DOC_FINDING
from tests.test_ex4_validation import stop

# Offline only: these build options and prompts, never start an agent.


def pre_tool_use(tool_name: str, agent_type: str | None) -> PreToolUseHookInput:
    data: PreToolUseHookInput = {
        "hook_event_name": "PreToolUse",
        "session_id": "session-test",
        "transcript_path": "/dev/null",
        "cwd": "/project",
        "tool_name": tool_name,
        "tool_input": {},
        "tool_use_id": "toolu_1",
    }
    if agent_type is not None:
        data["agent_id"] = "agent-1"
        data["agent_type"] = agent_type
    return data


# --- Tool restrictions --------------------------------------------------------


@pytest.mark.anyio
async def test_tool_names_match_the_mcp_server():
    async with Client(mcp) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
    assert {WEB_SEARCH, LOAD_DOCUMENT} == {mcp_tool_name(name) for name in names}
    assert f"mcp__{SERVER_NAME}__web_search" == WEB_SEARCH


def test_there_are_two_researchers_and_a_synthesizer():
    assert set(build_agents()) == {"web_researcher", "doc_analyst", SYNTHESIZER}


def test_each_researcher_has_exactly_one_tool_and_the_synthesizer_none():
    agents = build_agents()
    assert agents["web_researcher"].tools == [WEB_SEARCH]
    assert agents["doc_analyst"].tools == [LOAD_DOCUMENT]
    assert agents[SYNTHESIZER].tools == []
    assert agents[SYNTHESIZER].mcpServers is None


def test_subagents_cannot_spawn_subagents():
    for agent in build_agents().values():
        assert agent.tools is not None
        assert SPAWN_TOOL not in agent.tools
        assert "Task" not in agent.tools


def test_researchers_ask_for_structured_findings():
    agents = build_agents()
    for name in ("web_researcher", "doc_analyst"):
        assert agents[name].description
        assert RESULT_FORMAT in agents[name].prompt


@pytest.mark.parametrize(
    ("agent_name", "tool"), [("web_researcher", "web_search"), ("doc_analyst", "load_document")]
)
def test_each_researcher_gets_a_server_with_only_its_own_tool(agent_name, tool):
    """AgentDefinition.tools did not hide the other tool live, so the server must not have it."""
    agent = build_agents()[agent_name]
    assert agent.mcpServers is not None
    [server] = agent.mcpServers
    assert isinstance(server, dict)
    assert server[SERVER_NAME]["args"] == ["run", "python", "-m", "ex4_research.server", tool]


@pytest.mark.parametrize("mode", MODES)
def test_coordinator_can_only_spawn_subagents(mode):
    options = build_options(model="test-model", mode=mode)
    assert options.tools == [SPAWN_TOOL]
    assert options.allowed_tools == [SPAWN_TOOL]
    assert options.mcp_servers == {}  # the research tools are attached to subagents only
    assert options.agents == build_agents()
    assert options.max_turns == COORDINATOR_MAX_TURNS


def test_coordinator_ignores_the_developers_local_claude_config():
    options = build_options(model="test-model")
    assert options.setting_sources == []
    assert options.strict_mcp_config is True
    assert options.model == "test-model"


def test_coordinator_runs_the_owner_check_and_the_foreground_rule():
    owner, spawn = build_options(model="test-model").hooks["PreToolUse"]  # type: ignore[index]
    assert owner.matcher == f"{WEB_SEARCH}|{LOAD_DOCUMENT}"
    assert owner.hooks == [allow_owner_only]
    assert spawn.matcher == SPAWN_MATCHER == "Agent|Task"
    assert spawn.hooks[0] is keep_in_foreground
    assert len(spawn.hooks) == 2  # then the findings attacher


@pytest.mark.anyio
async def test_findings_recorded_after_a_spawn_are_attached_for_the_synthesizer():
    """The recorder and the attacher share the ledger passed to build_options."""
    ledger = FindingsLedger()
    hooks = build_options(model="m", ledger=ledger).hooks
    assert hooks is not None
    [recorder_matcher] = hooks["PostToolUse"]
    assert recorder_matcher.matcher == SPAWN_MATCHER
    [record] = recorder_matcher.hooks
    attach = hooks["PreToolUse"][1].hooks[1]
    finding = Finding.model_validate(DOC_FINDING)
    envelope = json.dumps({"status": "complete", "findings": [DOC_FINDING]})
    await record(post_spawn("doc_analyst", envelope), "t", {"signal": None})
    assert ledger.received == [finding]
    output: Any = await attach(pre_spawn(SYNTHESIZER, "No findings here."), "t", {"signal": None})
    assert valid_findings(output["hookSpecificOutput"]["updatedInput"]["prompt"]) == [finding]


def test_build_options_without_a_ledger_still_attaches():
    assert len(build_options(model="m").hooks["PreToolUse"][1].hooks) == 2  # type: ignore[index]


def test_no_subagent_is_defined_as_a_background_agent():
    assert all(agent.background is False for agent in build_agents().values())


def spawn_input(tool_input: Any) -> PreToolUseHookInput:
    data = pre_tool_use(SPAWN_TOOL, None)
    data["tool_input"] = tool_input
    return data


@pytest.mark.anyio
async def test_a_background_spawn_is_rewritten_to_run_in_the_foreground():
    tool_input = {"subagent_type": "web_researcher", "prompt": "p", "run_in_background": True}
    output = await keep_in_foreground(spawn_input(tool_input), "t", {"signal": None})
    specific: Any = output["hookSpecificOutput"]  # type: ignore[typeddict-item]
    assert specific["permissionDecision"] == "allow"
    assert specific["updatedInput"] == {**tool_input, "run_in_background": False}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "tool_input",
    [{"subagent_type": "web_researcher"}, {"run_in_background": False}, "not a dict"],
)
async def test_a_foreground_spawn_is_left_alone(tool_input):
    assert await keep_in_foreground(spawn_input(tool_input), "t", {"signal": None}) == {}


@pytest.mark.parametrize(
    ("tool", "agent_type", "allowed"),
    [
        (WEB_SEARCH, "web_researcher", True),
        (LOAD_DOCUMENT, "doc_analyst", True),
        (WEB_SEARCH, "doc_analyst", False),
        (LOAD_DOCUMENT, "web_researcher", False),
        (WEB_SEARCH, SYNTHESIZER, False),
        (LOAD_DOCUMENT, SYNTHESIZER, False),
        (WEB_SEARCH, None, False),  # the coordinator itself
        (LOAD_DOCUMENT, None, False),
        ("mcp__research__fetch_url", "web_researcher", False),
    ],
)
def test_only_the_owning_subagent_may_call_a_research_tool(tool, agent_type, allowed):
    assert tool_decision(tool, agent_type)[0] is allowed


@pytest.mark.anyio
async def test_hook_allows_the_owner():
    hook_input = pre_tool_use(WEB_SEARCH, "web_researcher")
    output = await allow_owner_only(hook_input, "t", {"signal": None})
    assert output["hookSpecificOutput"]["permissionDecision"] == "allow"  # type: ignore[typeddict-item]


@pytest.mark.anyio
async def test_hook_denies_anyone_else_with_a_reason():
    output = await allow_owner_only(pre_tool_use(LOAD_DOCUMENT, None), "t", {"signal": None})
    specific: Any = output["hookSpecificOutput"]  # type: ignore[typeddict-item]
    assert specific["permissionDecision"] == "deny"
    reason = f"Only the doc_analyst subagent may call {LOAD_DOCUMENT}."
    assert specific["permissionDecisionReason"] == reason


# --- Prompts ------------------------------------------------------------------


def test_verification_is_a_quality_criterion_not_a_fixed_step():
    prompt = coordinator_prompt("parallel")
    assert (
        "Headline figures must be verified against the primary report whenever a finding\n"
        "  cites a doc_id; delegate that to doc_analyst."
    ) in prompt
    assert "always" not in prompt.lower()


@pytest.mark.parametrize("mode", MODES)
def test_each_mode_gets_only_its_own_scheduling_rule(mode):
    prompt = coordinator_prompt(mode)
    assert MODE_RULES[mode] in prompt
    other = next(m for m in MODES if m != mode)
    assert MODE_RULES[other] not in prompt


def test_parallel_mode_asks_for_several_spawn_calls_in_one_response():
    assert "same response, as several Agent tool calls together" in MODE_RULES["parallel"]
    assert "one Agent tool call per response" in MODE_RULES["sequential"]


def test_build_options_uses_the_modes_prompt():
    for mode in MODES:
        assert build_options(model="m", mode=mode).system_prompt == coordinator_prompt(mode)


def test_the_format_rule_lives_in_the_subagent_definitions():
    """Live coordinators dictated their own fields; the rule now sits with each subagent."""
    agents = build_agents()
    for name in ("web_researcher", "doc_analyst"):
        assert agents[name].description.endswith(FIXED_FORMAT)
        assert "This format is fixed." in agents[name].prompt
    assert "Established, Contested, Coverage" in agents[SYNTHESIZER].description
    assert "format" not in coordinator_prompt("parallel").lower()


def test_findings_reach_the_synthesizer_through_code_not_the_coordinator():
    """Hub and spoke: no subagent talks to another; code attaches the findings."""
    prompt = coordinator_prompt("parallel")
    assert "Subagents never talk to each" in prompt
    assert "Code attaches\n  every finding the researchers returned to its prompt" in prompt
    assert "do not copy findings" in prompt
    assert "Code attaches every finding" in build_agents()[SYNTHESIZER].description


def test_synthesizer_must_cite_every_claim_and_add_nothing():
    assert "Cite every claim with its finding's source_id in square brackets" in SYNTHESIZER_PROMPT
    assert "Add\n  no facts, numbers or sources of your own." in SYNTHESIZER_PROMPT


# --- Step 3: failures -----------------------------------------------------------


def test_researchers_retry_a_transient_error_once_then_hand_back_partial():
    for name in ("web_researcher", "doc_analyst"):
        prompt = build_agents()[name].prompt
        assert "isRetryable true is transient: retry the same call once" in prompt
        assert "return a partial\n  result" in prompt
        assert "record it as a no-sources\n  finding" in prompt


def test_coordinator_continues_with_partial_results_and_says_what_failed():
    prompt = coordinator_prompt("parallel")
    assert "never abort\n  the run" in prompt
    assert "at most one extra spawn per gap" in prompt
    assert "never let a failure pass silently" in prompt


def test_synthesizer_ends_with_a_coverage_section():
    assert '"## Coverage" last' in SYNTHESIZER_PROMPT
    assert '"Gaps": what is missing or weak, each with its cause' in SYNTHESIZER_PROMPT


def test_fail_reaches_only_the_web_search_servers():
    agents = build_options(model="m", fail="empty_once").agents
    assert agents is not None
    [web] = agents["web_researcher"].mcpServers or []
    [doc] = agents["doc_analyst"].mcpServers or []
    assert isinstance(web, dict)
    assert isinstance(doc, dict)
    assert web[SERVER_NAME]["env"] == {"EX4_FAIL": "empty_once"}
    assert "env" not in doc[SERVER_NAME]


def test_no_fail_means_no_env():
    [web] = build_agents()["web_researcher"].mcpServers or []
    assert isinstance(web, dict)
    assert "env" not in web[SERVER_NAME]


def test_a_mistyped_fail_value_is_rejected_before_any_run():
    with pytest.raises(ValueError, match="Unknown EX4_FAIL case"):
        build_options(model="m", fail="timeout-once")


def test_cli_fail_cases_are_valid_plans():
    from ex4_research.agent import FAIL_CASES
    from ex4_research.faults import parse_plan

    assert {parse_plan(value).case for value in FAIL_CASES.values()} == {
        "timeout_once",
        "category_timeout",
        "empty_once",
    }


# --- Step 4: hand-back validation, established vs contested -------------------------


def test_report_has_established_then_contested_then_coverage():
    sections = [SYNTHESIZER_PROMPT.index(f'"## {name}"') for name in REPORT_SECTIONS]
    assert sections == sorted(sections)


def test_established_needs_two_independent_sources_or_a_primary_report():
    assert "supported by at least two independent sources, or by a\n  primary report" in (
        SYNTHESIZER_PROMPT
    )
    assert "A press page and the report it summarizes are one source." in SYNTHESIZER_PROMPT


def test_contested_figures_stand_side_by_side_and_are_never_resolved():
    assert "side by side" in SYNTHESIZER_PROMPT
    assert "each figure with its source, date and\n  scope, then the likely reason" in (
        SYNTHESIZER_PROMPT
    )
    assert "Never\n  average them, never pick one, never drop one." in SYNTHESIZER_PROMPT


@pytest.mark.anyio
async def test_the_handback_validator_runs_when_any_subagent_stops():
    log = HandbackLog()
    hooks = build_options(model="m", handbacks=log).hooks
    assert hooks is not None
    [matcher] = hooks["SubagentStop"]
    assert matcher.matcher is None  # every subagent; the hook skips non-researchers
    [validate] = matcher.hooks
    output: Any = await validate(stop("web_researcher", "not json"), None, {"signal": None})
    assert output["decision"] == "block"
    assert len(log.corrections) == 1


def test_build_options_without_a_handback_log_still_validates():
    hooks = build_options(model="m").hooks
    assert hooks is not None
    assert len(hooks["SubagentStop"][0].hooks) == 1


@pytest.mark.anyio
async def test_the_foreground_rule_leaves_the_synthesizer_to_the_attacher():
    """Only one hook may return updatedInput for a call; attach_findings owns this one."""
    tool_input = {"subagent_type": SYNTHESIZER, "prompt": "p", "run_in_background": True}
    assert await keep_in_foreground(spawn_input(tool_input), "t", {"signal": None}) == {}
