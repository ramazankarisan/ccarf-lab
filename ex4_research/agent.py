"""A coordinator with three subagents on the Claude Agent SDK.

The coordinator has no research tools of its own: it can only spawn subagents.
Two researchers each see one MCP tool and return structured findings; a
synthesizer with no tools writes the report from the findings the coordinator
hands it. Subagents never talk to each other: everything goes through the
coordinator (hub and spoke).

Run against the real API (both modes, one after the other):
    uv run python -m ex4_research.agent "How big was the EU EV battery recycling market in 2025?"
    uv run python -m ex4_research.agent --mode parallel --verbose "..."
"""

import argparse
import json
import logging
import os
import sys
from collections.abc import AsyncIterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Literal, get_args

import anyio
from claude_agent_sdk import (
    AgentDefinition,
    ClaudeAgentOptions,
    ClaudeSDKError,
    HookContext,
    HookInput,
    HookJSONOutput,
    HookMatcher,
    Message,
    query,
)
from claude_agent_sdk.types import McpStdioServerConfig  # not re-exported at the top level
from dotenv import load_dotenv

from ex4_research.completeness import FindingsLedger, make_completeness_hooks
from ex4_research.faults import FAIL_ENV, parse_plan
from ex4_research.findings import RESULT_FORMAT
from ex4_research.timeline import (
    SPAWN_TOOL,
    SPAWN_TOOL_NAMES,
    SYNTHESIZER,
    ResearchRun,
    collect,
    format_phase_table,
    format_run,
    format_timings,
    gather_timed,
)
from ex4_research.validation import HandbackLog, make_handback_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_NAME = "research"

# Sequential mode needs a turn per spawn: 2 web + 2 doc + synthesizer + the answer.
COORDINATOR_MAX_TURNS = 12
SUBAGENT_MAX_TURNS = 8

Mode = Literal["parallel", "sequential"]
MODES: tuple[Mode, ...] = get_args(Mode)


def mcp_tool_name(tool: str) -> str:
    """The name the Agent SDK gives one of our MCP tools, e.g. mcp__research__web_search."""
    return f"mcp__{SERVER_NAME}__{tool}"


WEB_SEARCH = mcp_tool_name("web_search")
LOAD_DOCUMENT = mcp_tool_name("load_document")
SPAWN_MATCHER = "|".join(sorted(SPAWN_TOOL_NAMES))  # "Agent|Task"


# --- Prompts ------------------------------------------------------------------

# How to schedule spawns. Only this line differs between the modes.
MODE_RULES: Mapping[Mode, str] = {
    "parallel": "Subtasks that do not depend on each other's results must be spawned in "
    "the same response, as several Agent tool calls together. Wait only for results "
    "that a later subtask actually needs.",
    "sequential": "Spawn one subagent at a time: one Agent tool call per response, and "
    "wait for its findings before you spawn the next.",
}

COORDINATOR_PROMPT = """\
You coordinate research on market questions. You cannot search, read sources or
write the report yourself; you delegate to subagents. Subagents never talk to each
other: whatever one needs from another, you pass on in its prompt.

Goals and quality criteria:
- Cover the question from both web pages and the primary reports.
- Give each subagent everything it needs in its prompt: it does not see this
  conversation. Name the topic, the time period, the exact facts wanted and why.
- Headline figures must be verified against the primary report whenever a finding
  cites a doc_id; delegate that to doc_analyst.
- {mode_rule}
- A partial result is still a result: continue with what it found, and never abort
  the run because a tool failed. For a gap, you may try one alternative from its
  suggested_alternatives: at most one extra spawn per gap.
- When the research is complete, delegate the report to synthesizer. Code attaches
  every finding the researchers returned to its prompt, so do not copy findings
  into it. Write the question, what matters most, and process notes: every tool
  failure, partial result, empty search, skipped step and alternative you tried,
  each with its cause.
- Your final answer is the synthesizer's report, unchanged. If the synthesizer
  could not run, or a step failed in a way its report does not show, say so
  explicitly at the top of your answer: never let a failure pass silently."""

# When tools fail. Shared by both researchers.
ERROR_RULES = """\
- An error with isRetryable true is transient: retry the same call once. If it
  fails again, stop retrying it: finish with what you have and return a partial
  result with every query you tried and alternatives the coordinator could try.
- An error with isRetryable false will fail the same way again: fix the input if the
  error says how, otherwise leave that path and report it.
- A call that works but finds nothing is not an error: record it as a no-sources
  finding, then try other keywords if it matters.
"""

WEB_RESEARCHER_PROMPT = f"""\
You research a topic on the web and report what credible sources say.

Goals and quality criteria:
- Find the pages that answer the question you were given, not just the first hit.
- Prefer recent sources from identifiable publishers (research firms, industry
  bodies, regulators). Note when a page is old, promotional or unsourced, and
  rate it low rather than dropping it silently.
- Keep each figure's scope: geography, year, and what is counted.
- Report only what a page says. No estimates or conclusions of your own.
- If a page refers to a longer report by doc_id, name that doc_id in the finding's
  claim, so the coordinator can have the figure checked against it.
- Search the "news archive" category for history and earlier forecasts, and the
  default "current" category for recent figures.
{ERROR_RULES}
{RESULT_FORMAT}"""

DOC_ANALYST_PROMPT = f"""\
You read research reports in depth and report what they say.

Goals and quality criteria:
- Read the reports that bear on the question you were given. If you were not
  given doc_ids, an unknown doc_id returns the list of available reports.
- When asked to verify a figure, say whether the report states it, and with which
  scope: confirm, correct or qualify it with the report's own words.
- Capture definitions and method along with numbers: a figure without its scope
  (geography, years, what is counted, nominal or real) is not useful.
- Report forecasts as forecasts, with the year they refer to.
- Report only what a report says. No estimates or conclusions of your own.
{ERROR_RULES}
{RESULT_FORMAT}"""

SYNTHESIZER_PROMPT = """\
You write a research report from findings you are given. You have no tools and no
other information.

Goals and quality criteria:
- Answer the question you were given, using only the findings in your prompt. Add
  no facts, numbers or sources of your own.
- Cite every claim with its finding's source_id in square brackets, for example
  [acme-market-report-2024] or [https://example.com/page]. A sentence with a figure
  and no citation is a defect.
- Prefer a figure confirmed by a report (source_type doc) and say when a headline
  figure has only web support.
- Structure the report as a short answer, then these sections in this order:
- "## Established": claims supported by at least two independent sources, or by a
  primary report (source_type doc), that no credible source contradicts. Cite every
  supporting source. A press page and the report it summarizes are one source.
- "## Contested": every point where credible sources give different figures or
  conclusions, side by side (a table works): each figure with its source, date and
  scope, then the likely reason they differ, from what the findings say about
  scope, method or date. If the findings do not show a reason, say so. Never
  average them, never pick one, never drop one. A claim made only by a
  low-confidence source (promotional, unsourced, outdated) is not a contest: name
  it under Coverage as excluded. If nothing is contested, say so.
- "## Coverage" last, with two lists. "Well supported": the
  conclusions backed by a report or by several credible sources, with citations.
  "Gaps": what is missing or weak, each with its cause, taken from the findings and
  process notes (a search that timed out, an unavailable source category, a search
  that found nothing, a report not read). If nothing failed, say so.

Return the report as Markdown, nothing else."""


# --- Options ------------------------------------------------------------------


def stdio_server(tool: str, fail: str | None = None) -> McpStdioServerConfig:
    """The research server as a subprocess over stdio, with only this one tool registered.

    fail sets EX4_FAIL in that process, to inject a failure (see faults.py).
    """
    config: McpStdioServerConfig = {
        "type": "stdio",
        "command": "uv",
        "args": ["run", "python", "-m", "ex4_research.server", tool],
    }
    if fail:
        config["env"] = {FAIL_ENV: fail}
    return config


# Part of each researcher's description, which is what the coordinator reads about
# it. A live coordinator still dictated its own fields (claim_checked, verified, ...)
# to doc_analyst, so the format rule lives with the subagent, not in the
# coordinator's prompt.
FIXED_FORMAT = (
    "It returns findings in its own fixed JSON result format, which code checks: ask it "
    "for facts, never for a format, fields or a table."
)


def build_agents(fail: str | None = None) -> dict[str, AgentDefinition]:
    """The three subagents. Each researcher gets its own server that has only its tool.

    AgentDefinition.tools alone is not enough: in a live run (SDK 0.2.172),
    web_researcher still saw and called load_document from a server that had
    both tools. A tool the server never registers cannot leak. The servers are
    attached to the subagents only, so the coordinator never sees research tools.
    """
    return {
        "web_researcher": AgentDefinition(
            description="Searches the web for current pages, news and press releases on "
            "a topic. Use it for recent figures, announcements, and to find which "
            f"longer reports exist. {FIXED_FORMAT}",
            prompt=WEB_RESEARCHER_PROMPT,
            tools=[WEB_SEARCH],
            mcpServers=[{SERVER_NAME: stdio_server("web_search", fail)}],
            model="inherit",
            maxTurns=SUBAGENT_MAX_TURNS,
            background=False,
        ),
        "doc_analyst": AgentDefinition(
            description="Reads full research reports from the report library by doc_id. "
            "Use it to verify figures against the primary report, and for definitions, "
            f"methods, detailed figures and forecasts. {FIXED_FORMAT}",
            prompt=DOC_ANALYST_PROMPT,
            tools=[LOAD_DOCUMENT],
            mcpServers=[{SERVER_NAME: stdio_server("load_document")}],
            model="inherit",
            maxTurns=SUBAGENT_MAX_TURNS,
            background=False,
        ),
        SYNTHESIZER: AgentDefinition(
            description="Writes the final report, with a citation on every claim, in its "
            "own fixed structure (Established, Contested, Coverage). Code attaches every "
            "finding to its prompt: give it the question, what matters most and your "
            "process notes. Use it once, after research is complete.",
            prompt=SYNTHESIZER_PROMPT,
            tools=[],  # writes from its prompt only
            model="inherit",
            maxTurns=1,
            background=False,
        ),
    }


# Which subagent may call which tool. The per-subagent servers decide what a
# subagent can see; this hook is the second check on what it may run.
TOOL_OWNERS: Mapping[str, str] = {WEB_SEARCH: "web_researcher", LOAD_DOCUMENT: "doc_analyst"}


def tool_decision(tool_name: str, agent_type: str | None) -> tuple[bool, str]:
    """(allowed, reason) for agent_type calling tool_name. None means the coordinator."""
    owner = TOOL_OWNERS.get(tool_name)
    if owner is None:
        return False, f"{tool_name} is not a research tool."
    if agent_type != owner:
        return False, f"Only the {owner} subagent may call {tool_name}."
    return True, f"{tool_name} is the {owner} subagent's own tool."


async def allow_owner_only(
    input_data: HookInput, tool_use_id: str | None, context: HookContext
) -> HookJSONOutput:
    """PreToolUse: approve a research tool for its owning subagent, deny it for anyone else.

    allowed_tools lists only the spawn tool, so without this hook the subagents'
    MCP calls would need a permission prompt that a headless run cannot answer.
    """
    agent_type = input_data.get("agent_type")  # absent on the coordinator's own calls
    allowed, reason = tool_decision(
        str(input_data.get("tool_name", "")), agent_type if isinstance(agent_type, str) else None
    )
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow" if allowed else "deny",
            "permissionDecisionReason": reason,
        }
    }


# --- PreToolUse: subagents run in the foreground ----------------------------------
# The spawn tool can start a subagent in the background (run_in_background). Its
# result is then only "Async agent launched", and the findings arrive later as a
# notification. A live sequential run did exactly that, which breaks the explicit
# hand-off and makes "one at a time" meaningless. The hook rewrites the input.

BACKGROUND_FLAG = "run_in_background"
FOREGROUND_REASON = (
    "Subagents run in the foreground here, so their findings come back as the tool result."
)


async def keep_in_foreground(
    input_data: HookInput, tool_use_id: str | None, context: HookContext
) -> HookJSONOutput:
    """PreToolUse on the spawn tool: turn a background spawn into a foreground one."""
    tool_input = input_data.get("tool_input")
    if not isinstance(tool_input, dict) or not tool_input.get(BACKGROUND_FLAG):
        return {}  # no opinion: Agent is in allowed_tools anyway
    if tool_input.get("subagent_type") == SYNTHESIZER:
        return {}  # attach_findings rewrites the synthesizer's input, foreground included
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": FOREGROUND_REASON,
            "updatedInput": {**tool_input, BACKGROUND_FLAG: False},
        }
    }


def coordinator_prompt(mode: Mode) -> str:
    return COORDINATOR_PROMPT.format(mode_rule=MODE_RULES[mode])


def build_options(
    model: str,
    mode: Mode = "parallel",
    ledger: FindingsLedger | None = None,
    fail: str | None = None,
    handbacks: HandbackLog | None = None,
) -> ClaudeAgentOptions:
    """The coordinator: the spawn tool and nothing else, plus the three subagents.

    ledger collects this run's findings for the completeness check on the
    synthesizer's prompt; without one, a fresh ledger is used.
    fail is an EX4_FAIL value for the web_search servers; it is checked here, so
    a typo fails fast instead of testing nothing. handbacks logs the corrections
    the SubagentStop hook asked researchers for.
    """
    parse_plan(fail)
    record_findings, attach_findings = make_completeness_hooks(
        ledger if ledger is not None else FindingsLedger()
    )
    return ClaudeAgentOptions(
        model=model,
        system_prompt=coordinator_prompt(mode),  # a plain string replaces Claude Code's prompt
        max_turns=COORDINATOR_MAX_TURNS,
        cwd=PROJECT_ROOT,  # so `uv run` finds this project when it starts the server
        tools=[SPAWN_TOOL],  # the only built-in tool: no Bash, Read, WebSearch ...
        allowed_tools=[SPAWN_TOOL],
        agents=build_agents(fail),
        hooks={
            "PreToolUse": [
                HookMatcher(matcher=f"{WEB_SEARCH}|{LOAD_DOCUMENT}", hooks=[allow_owner_only]),
                HookMatcher(matcher=SPAWN_MATCHER, hooks=[keep_in_foreground, attach_findings]),
            ],
            "PostToolUse": [HookMatcher(matcher=SPAWN_MATCHER, hooks=[record_findings])],
            "SubagentStop": [
                HookMatcher(
                    hooks=[make_handback_validator(handbacks if handbacks else HandbackLog())]
                )
            ],
        },
        # Isolation: ignore the developer's own ~/.claude settings, agents, hooks
        # and MCP servers, so every machine runs the same coordinator.
        setting_sources=[],
        strict_mcp_config=True,
    )


# --- CLI ----------------------------------------------------------------------

# --fail names and the EX4_FAIL value each one sets (see faults.py).
FAIL_CASES: Mapping[str, str] = {
    "timeout-once": "timeout_once",
    "archive-down": "category_timeout:news archive",
    "empty-once": "empty_once",
}


async def run_mode(  # pragma: no cover - live API call
    question: str, mode: Mode, fail: str | None = None
) -> ResearchRun:
    ledger, handbacks = FindingsLedger(), HandbackLog()
    options = build_options(
        model=os.environ["MODEL"], mode=mode, ledger=ledger, fail=fail, handbacks=handbacks
    )
    messages: AsyncIterable[Message] = query(prompt=question, options=options)
    run = collect(await gather_timed(messages), mode=mode)
    run.attached = ledger.attached
    run.corrections = handbacks.corrections
    return run


async def main(  # pragma: no cover - live API call
    question: str, modes: tuple[Mode, ...], runs: int, fail: str | None = None
) -> list[ResearchRun]:
    load_dotenv()
    # One after the other, never at the same time, so the timings don't compete.
    # Modes alternate (p, s, p, s ...) so a slow minute of the API hits both.
    return [await run_mode(question, mode, fail) for _ in range(runs) for mode in modes]


def cli() -> None:  # pragma: no cover - live API call
    parser = argparse.ArgumentParser(description="Run the research coordinator.")
    parser.add_argument("question", nargs="+", help="the research question")
    parser.add_argument(
        "--mode",
        choices=[*MODES, "both"],
        default="both",
        help="how the coordinator schedules subagents (default: both, to compare timings)",
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="runs per mode; above 1, reports are left out"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="also print every subagent's prompt and hand-back"
    )
    parser.add_argument(
        "--fail",
        choices=sorted(FAIL_CASES),
        help="inject a web_search failure and print errors, hand-backs and coverage",
    )
    parser.add_argument(
        "--save-handbacks",
        type=Path,
        help="write every hand-back the validation hook sent back, as JSON, to this file",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, force=True)
    modes = MODES if args.mode == "both" else (args.mode,)

    try:
        fail = FAIL_CASES[args.fail] if args.fail else None
        runs = anyio.run(main, " ".join(args.question), modes, args.runs, fail)
    except ClaudeSDKError as error:
        print(f"Agent stopped: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
    for number, run in enumerate(runs, start=1):
        print(f"{'#' * 78}\n# Run {number}: {run.mode}\n{'#' * 78}\n")
        texts = args.runs == 1
        print(format_run(run, verbose=args.verbose, texts=texts, failures=bool(args.fail)))
        print()
    if args.save_handbacks:
        saved = [asdict(c) for run in runs for c in run.corrections]
        args.save_handbacks.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    print(format_timings(runs))
    if args.runs > 1:
        print("\n" + format_phase_table(runs))
    if any(run.result is None or run.result.is_error for run in runs):
        sys.exit(1)


if __name__ == "__main__":
    cli()
