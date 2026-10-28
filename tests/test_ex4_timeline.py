import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    Message,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from ex4_research.agent import LOAD_DOCUMENT, WEB_SEARCH
from ex4_research.timeline import (
    SPAWN_TOOL,
    SYNTHESIZER,
    SYNTHESIZER_PROMPT_PREVIEW,
    PhaseTimes,
    ResearchRun,
    Spawn,
    Timed,
    collect,
    coverage_section,
    findings_status,
    format_attachments,
    format_corrections,
    format_degradation,
    format_failures,
    format_phase_table,
    format_report_structure,
    format_run,
    format_synthesis,
    format_timeline,
    format_timings,
    gather_timed,
    max_concurrency,
    phase_times,
    report_section,
    run_issues,
    span,
    summarize,
    union_length,
)
from tests.test_ex4_findings import DOC_FINDING, PARTIAL, WEB_FINDING

# Offline only: scripted message streams, never a live agent.

WEB_JSON = json.dumps({"status": "complete", "findings": [WEB_FINDING]})
DOC_JSON = json.dumps({"status": "complete", "findings": [DOC_FINDING]})
PARTIAL_JSON = json.dumps(PARTIAL)
REPORT = (
    "EUR 1.9 billion [nordwell-eu-outlook-2026], or EUR 3.4 billion on a wider scope "
    f"[{WEB_FINDING['source_id']}]."
)


def spawn_call(block_id: str, subagent: str, prompt: str = "Do the research.") -> ToolUseBlock:
    return ToolUseBlock(
        id=block_id,
        name=SPAWN_TOOL,
        input={"subagent_type": subagent, "description": f"{subagent} job", "prompt": prompt},
    )


def coordinator(message_id: str | None, *blocks: Any) -> AssistantMessage:
    return AssistantMessage(content=list(blocks), model="m", message_id=message_id)


def hand_back(block_id: str, text: str, is_error: bool = False) -> UserMessage:
    return UserMessage(
        content=[
            ToolResultBlock(
                tool_use_id=block_id, content=[{"type": "text", "text": text}], is_error=is_error
            )
        ]
    )


def result_message(report: str | None = REPORT, cost: float | None = 0.05) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1200,
        duration_api_ms=1000,
        is_error=False,
        num_turns=4,
        session_id="session-test",
        total_cost_usd=cost,
        result=report,
    )


def parallel_stream() -> list[Timed]:
    """Two web researchers in one response, then a doc analyst, then the synthesizer."""
    synth_prompt = f"Question: how big?\nFindings:\n{json.dumps([WEB_FINDING, DOC_FINDING])}"
    return [
        (1.0, coordinator("r1", TextBlock(text="Delegating."), spawn_call("w1", "web_researcher"))),
        (1.0, coordinator("r1", spawn_call("w2", "web_researcher"))),  # same response, own message
        (
            2.0,
            AssistantMessage(
                content=[ToolUseBlock(id="t1", name=WEB_SEARCH, input={"query": "EU market"})],
                model="m",
                parent_tool_use_id="w1",
            ),
        ),
        (
            2.5,
            AssistantMessage(
                content=[ToolUseBlock(id="t2", name=LOAD_DOCUMENT, input={"doc_id": "x"})],
                model="m",
                parent_tool_use_id="w1",
            ),
        ),
        (
            3.0,
            UserMessage(
                content=[
                    ToolResultBlock(tool_use_id="t1", content="{}"),
                    ToolResultBlock(
                        tool_use_id="t2", content="Only the doc_analyst ...", is_error=True
                    ),
                    TextBlock(text="skipped"),
                    ToolResultBlock(tool_use_id="unknown", content="x"),
                ],
                parent_tool_use_id="w1",
            ),
        ),
        (10.0, hand_back("w1", WEB_JSON)),
        (12.0, hand_back("w2", WEB_JSON)),
        (13.0, coordinator("r2", spawn_call("d1", "doc_analyst"))),
        (20.0, hand_back("d1", DOC_JSON)),
        (21.0, coordinator("r3", spawn_call("s1", SYNTHESIZER, synth_prompt))),
        (30.0, hand_back("s1", REPORT)),
        (30.5, UserMessage(content="a plain text user message is skipped")),
        (31.0, result_message()),
    ]


# --- collect --------------------------------------------------------------------


def test_collect_numbers_coordinator_responses_by_message_id():
    run = collect(parallel_stream(), mode="parallel")
    assert run.mode == "parallel"
    assert [(s.subagent_type, s.response) for s in run.spawns] == [
        ("web_researcher", 1),
        ("web_researcher", 1),
        ("doc_analyst", 2),
        (SYNTHESIZER, 3),
    ]


def test_collect_records_spawn_and_hand_back_times():
    run = collect(parallel_stream())
    first = run.spawns[0]
    assert (first.spawned_at, first.returned_at) == (1.0, 10.0)
    assert first.output == WEB_JSON
    assert run.finished_at == 31.0
    assert run.result is not None


def test_collect_records_each_subagents_tool_calls_and_outcomes():
    first = collect(parallel_stream()).spawns[0]
    search, denied = first.tool_calls
    assert (search.name, search.input, search.is_error) == (
        WEB_SEARCH,
        {"query": "EU market"},
        False,
    )
    assert (denied.name, denied.is_error) == (LOAD_DOCUMENT, True)


def test_spawns_without_a_message_id_each_count_as_their_own_response():
    stream: list[Timed] = [
        (1.0, coordinator(None, spawn_call("a", "web_researcher"))),
        (2.0, coordinator(None, spawn_call("b", "doc_analyst"))),
    ]
    assert [s.response for s in collect(stream).spawns] == [1, 2]


def test_collect_accepts_the_legacy_task_name_and_ignores_other_tools():
    stream: list[Timed] = [
        (
            1.0,
            coordinator(
                "r1",
                ToolUseBlock(id="s1", name="Task", input={"subagent_type": "doc_analyst"}),
                ToolUseBlock(id="x", name="Bash", input={}),
            ),
        )
    ]
    [spawn] = collect(stream).spawns
    assert spawn.subagent_type == "doc_analyst"
    assert spawn.returned_at is None


@pytest.mark.anyio
async def test_gather_timed_stamps_messages_since_the_start():
    ticks = iter([100.0, 101.5, 104.0])

    async def stream() -> AsyncIterator[Message]:
        yield coordinator("r1")
        yield result_message()

    timed = await gather_timed(stream(), clock=lambda: next(ticks))
    assert [at for at, _ in timed] == [1.5, 4.0]


# --- concurrency ------------------------------------------------------------------


def spawn_at(start: float, end: float | None) -> Spawn:
    return Spawn(
        "id", "web_researcher", "", "", False, response=1, spawned_at=start, returned_at=end
    )


@pytest.mark.parametrize(
    ("intervals", "peak"),
    [
        ([], 0),
        ([(0, 5)], 1),
        ([(0, 5), (1, 6)], 2),
        ([(0, 5), (5, 9)], 1),  # back to back is sequential, not overlap
        ([(0, 5), (6, 9), (7, 8)], 2),
        ([(0, None), (10, 12)], 2),  # never handed back: still running
    ],
)
def test_max_concurrency(intervals, peak):
    assert max_concurrency(spawn_at(start, end) for start, end in intervals) == peak


# --- formatting -------------------------------------------------------------------


def test_timeline_shows_spawns_hand_backs_and_overlap_in_time_order():
    text = format_timeline(collect(parallel_stream(), mode="parallel"))
    lines = text.splitlines()
    assert lines[0] == "Timeline (parallel):"
    assert lines[1].strip() == "0.0s  start"
    assert "1.0s  spawn #1 web_researcher" in lines[2]
    assert "response 1" in lines[2]
    assert "1.0s  spawn #2 web_researcher" in lines[3]
    assert lines[4].strip() == "1.0s  coordinator: Delegating."
    assert "10.0s  back  #1 web_researcher" in lines[5]
    assert "after 9.0s: complete: 1 findings" in lines[5]
    assert "31.0s  done" in text
    assert lines[-2] == "At most 2 subagent(s) ran at once; 4 spawns in 3 coordinator response(s)."
    # 9 + 11 + 7 + 9 seconds of work between the first spawn (1.0) and the last back (30.0)
    assert lines[-1] == "Subagent work: 36.0s within 29.0s, first spawn to last back."


def test_timeline_shows_a_spawn_that_asked_for_the_background():
    block = spawn_call("a", "web_researcher")
    block.input["run_in_background"] = True
    run = collect([(1.0, coordinator("r1", block))])
    assert run.spawns[0].background is True
    assert "response 1, asked for background" in format_timeline(run)


def test_timeline_lists_subagents_that_never_handed_back():
    stream: list[Timed] = [(1.0, coordinator("r1", spawn_call("a", "doc_analyst")))]
    text = format_timeline(collect(stream))
    assert "Timeline (run):" in text
    assert "never handed back: #1 doc_analyst" in text
    assert "done" not in text
    assert "Subagent work" not in text


def spawn_with(subagent: str, output: str | None, is_error: bool = False) -> Spawn:
    spawn = spawn_at(0, 1)
    spawn.subagent_type, spawn.output, spawn.is_error = subagent, output, is_error
    return spawn


@pytest.mark.parametrize(
    ("spawn", "status"),
    [
        (spawn_with("web_researcher", WEB_JSON), "complete: 1 findings"),
        (spawn_with("doc_analyst", DOC_JSON), "complete: 1 findings"),
        (
            spawn_with("web_researcher", DOC_JSON),
            "complete: 1 findings, 1 with source_type other than 'web'",
        ),
        (
            spawn_with("web_researcher", PARTIAL_JSON),
            "partial (timeout): 2 findings (1 no-sources)",
        ),
        (spawn_with("web_researcher", "no json here"), 'INVALID: no {"status": ...}'),
        (spawn_with("web_researcher", None), "no hand-back"),
        (spawn_with(SYNTHESIZER, "Denied:   missing 2", is_error=True), "ERROR: Denied: missing 2"),
        (spawn_with(SYNTHESIZER, REPORT), "report"),
        (spawn_with("other_agent", WEB_JSON), "complete: 1 findings"),
    ],
)
def test_findings_status(spawn, status):
    assert findings_status(spawn).startswith(status)


def test_synthesis_shows_the_shortened_prompt_the_report_and_its_citations():
    text = format_synthesis(collect(parallel_stream(), mode="parallel"))
    assert "=== Synthesizer's prompt (" in text
    assert "contains 2 valid findings as JSON" in text
    assert "Question: how big?" in text
    assert "=== Final report (parallel):" in text
    assert REPORT in text
    assert "--- citations: 2 distinct source_ids cited; not in the findings: none;" in text
    assert "findings' sources never cited: 0" in text
    assert "Result: success | turns: 4 | cost: $0.0500" in text


def test_a_long_synthesizer_prompt_is_cut_short():
    long_prompt = "x" * (SYNTHESIZER_PROMPT_PREVIEW + 50)
    stream: list[Timed] = [(1.0, coordinator("r1", spawn_call("s1", SYNTHESIZER, long_prompt)))]
    text = format_synthesis(collect(stream))
    assert "x" * SYNTHESIZER_PROMPT_PREVIEW + " ..." in text
    assert "x" * (SYNTHESIZER_PROMPT_PREVIEW + 1) not in text
    assert "its findings JSON is no JSON array" in text
    assert "not checked: no valid findings to check them against" in text
    assert "Result: (no result message)" in text


def test_without_a_synthesizer_citations_are_checked_against_all_findings():
    stream = [m for m in parallel_stream() if not _is_synth(m)]
    stream[-1] = (31.0, result_message(report="EUR 9 billion [invented-2025].", cost=None))
    stream.insert(0, (0.5, coordinator("r0", spawn_call("bad", "web_researcher"))))
    stream.insert(1, (0.6, hand_back("bad", "not json")))
    text = format_synthesis(collect(stream))
    assert "=== No synthesizer was spawned." in text
    assert "not in the findings: invented-2025" in text
    assert "findings' sources never cited: 2" in text
    assert "cost: n/a" in text


def _is_synth(timed: Timed) -> bool:
    _, message = timed
    if isinstance(message, AssistantMessage):
        return any(isinstance(b, ToolUseBlock) and b.id == "s1" for b in message.content)
    if isinstance(message, UserMessage) and isinstance(message.content, list):
        return any(
            isinstance(b, ToolResultBlock) and b.tool_use_id == "s1" for b in message.content
        )
    return False


def test_format_run_adds_details_only_when_verbose():
    run = collect(parallel_stream(), mode="parallel")
    short, verbose = format_run(run), format_run(run, verbose=True)
    assert "Timeline (parallel):" in short
    assert "=== Final report (parallel):" in short
    assert "--- prompt it received:" not in short
    assert "=== 1. web_researcher: web_researcher job" in verbose
    assert f'{WEB_SEARCH} {{"query": "EU market"}}  [ok]' in verbose
    assert "[ERROR: Only the doc_analyst ...]" in verbose
    assert "--- raw hand-back:" in verbose


def test_verbose_details_handle_missing_calls_and_hand_backs():
    stream: list[Timed] = [(1.0, coordinator("r1", spawn_call("a", "doc_analyst")))]
    text = format_run(collect(stream), verbose=True)
    assert "--- its tool calls:\n  (none)" in text
    assert "--- raw hand-back:\n(none)" in text


def test_verbose_details_mark_calls_without_a_result():
    stream: list[Timed] = [
        (1.0, coordinator("r1", spawn_call("a", "web_researcher"))),
        (
            2.0,
            AssistantMessage(
                content=[ToolUseBlock(id="t9", name=WEB_SEARCH, input={"query": "late"})],
                model="m",
                parent_tool_use_id="a",
            ),
        ),
    ]
    assert '{"query": "late"}  [no result]' in format_run(collect(stream), verbose=True)


def test_timings_compare_the_modes():
    fast, slow, broken = ResearchRun("parallel"), ResearchRun("sequential"), ResearchRun("x")
    fast.finished_at, slow.finished_at = 41.24, 77.9
    assert format_timings([fast, slow]) == "Wall-clock: parallel 41.2s | sequential 77.9s"
    assert format_timings([broken]) == "Wall-clock: x n/a"


# --- denials and brief output -----------------------------------------------------


def test_attachments_are_counted_per_synthesis():
    run = collect(parallel_stream())
    assert format_attachments(run) == (
        "=== Findings attached by code to the synthesizer: none (no synthesis)"
    )
    run.attached = [47]
    assert format_attachments(run) == "=== Findings attached by code to the synthesizer: 47"
    assert "=== Findings attached by code to the synthesizer: 47" in format_run(run)


def test_brief_output_keeps_the_checks_but_not_the_texts():
    text = format_run(collect(parallel_stream(), mode="parallel"), texts=False)
    assert "contains 2 valid findings as JSON" in text
    assert "Question: how big?" not in text
    assert "=== Final report" not in text
    assert "--- citations: 2 distinct source_ids cited" in text


# --- phase timings ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("found", "length"),
    [
        ([], 0.0),
        ([(0, 5)], 5.0),
        ([(0, 5), (3, 8)], 8.0),
        ([(0, 2), (4, 6)], 4.0),
        ([(0, 10), (2, 3)], 10.0),
    ],
)
def test_union_length_counts_overlap_once(found, length):
    assert union_length(found) == length


def test_span_of_nothing_is_none():
    assert span([]) is None


def test_phase_times_of_the_parallel_stream():
    times = phase_times(collect(parallel_stream()))
    # web 1.0-12.0, doc 13.0-20.0, synthesis 21.0-30.0; busy 11+7+9 of 31 seconds
    assert times == PhaseTimes(
        web=11.0, doc=7.0, synthesis=9.0, coordinator=4.0, total=31.0, web_spawns=2, doc_spawns=1
    )


def test_phase_times_of_an_unfinished_run():
    times = phase_times(collect([(1.0, coordinator("r1", spawn_call("a", "doc_analyst")))]))
    assert (times.doc, times.coordinator, times.total, times.doc_spawns) == (None, None, None, 1)


@pytest.mark.parametrize(
    ("values", "text"),
    [
        ([10.0, 20.0, 30.0], "20.0 (10.0-30.0)"),
        ([5.0], "5.0"),
        ([None, None], "n/a"),
        ([4.0, None, 8.0], "6.0 (4.0-8.0, 2 runs)"),
        ([None, 3.0], "3.0, 1 runs"),
    ],
)
def test_summarize(values, text):
    assert summarize(values) == text


def test_phase_table_compares_modes_row_by_row():
    runs = []
    for mode, shift in (("parallel", 0.0), ("sequential", 10.0), ("parallel", 2.0)):
        run = collect([(at + shift, m) for at, m in parallel_stream()], mode=mode)
        runs.append(run)
    table = format_phase_table(runs)
    lines = table.splitlines()
    assert lines[0] == "Per-phase timings: mean (min-max)"
    assert lines[2] == "| Phase (seconds) | parallel (2 runs) | sequential (1 runs) |"
    assert "| web research | 11.0 (11.0-11.0) | 11.0 |" in lines
    assert "| total wall-clock | 32.0 (31.0-33.0) | 41.0 |" in lines
    assert lines[-1] == "| spawns per run (web+doc) | 2+1, 2+1 | 2+1 |"


# --- Step 3: failures, degradation, coverage ---------------------------------------

TIMEOUT = json.dumps(
    {
        "errorCategory": "transient",
        "isRetryable": True,
        "description": "web_search timed out: the 'news archive' search backend did not respond.",
    }
)
COVERAGE = """## Coverage

### Well supported
- EUR 1.9 billion in 2025 [nordwell-eu-outlook-2026].

### Gaps
- Historical growth: the news archive search timed out twice."""
REPORT_WITH_COVERAGE = f"# Report\n\nEUR 1.9 billion [nordwell-eu-outlook-2026].\n\n{COVERAGE}\n"


def failure_stream(report: str = REPORT_WITH_COVERAGE) -> list[Timed]:
    """A web researcher times out twice and hands back partial; the first synthesis is denied."""

    def inner(block_id: str, at: float) -> Timed:
        return (
            at,
            AssistantMessage(
                content=[
                    ToolUseBlock(
                        id=block_id,
                        name=WEB_SEARCH,
                        input={"query": "q", "category": "news archive"},
                    )
                ],
                model="m",
                parent_tool_use_id="w1",
            ),
        )

    def error(block_id: str, at: float) -> Timed:
        return (
            at,
            UserMessage(
                content=[ToolResultBlock(tool_use_id=block_id, content=TIMEOUT, is_error=True)],
                parent_tool_use_id="w1",
            ),
        )

    return [
        (1.0, coordinator("r1", spawn_call("w1", "web_researcher"))),
        inner("t1", 2.0),
        error("t1", 2.1),
        inner("t2", 3.0),
        error("t2", 3.1),
        (8.0, hand_back("w1", PARTIAL_JSON)),
        (9.0, coordinator("r2", TextBlock(text="Partial result; trying one alternative."))),
        (10.0, coordinator("r3", spawn_call("s1", SYNTHESIZER, "Findings: []"))),
        (10.0, hand_back("s1", "PreToolUse:Agent hook error: prompt is missing 2", True)),
        (12.0, coordinator("r4", spawn_call("s2", SYNTHESIZER, f"Findings: {PARTIAL_JSON}"))),
        (20.0, hand_back("s2", report)),
        (21.0, result_message(report=report)),
    ]


def failure_run(report: str = REPORT_WITH_COVERAGE) -> ResearchRun:
    run = collect(failure_stream(report), mode="parallel")
    return run


def test_coverage_section_runs_to_the_next_heading_of_its_level():
    report = f"# R\n\n{COVERAGE}\n\n## Sources\n- x"
    assert coverage_section(report) == COVERAGE
    assert coverage_section(f"intro\n{COVERAGE}") == COVERAGE
    assert coverage_section("# Report\nNo such section.") is None


def test_the_coordinators_decision_is_on_the_timeline():
    text = format_timeline(failure_run())
    assert "coordinator: Partial result; trying one alternative." in text


def test_run_issues_lists_every_degradation():
    assert run_issues(failure_run()) == [
        "#1 web_researcher: 2 tool call(s) returned an error",
        "#1 web_researcher: partial (timeout): 2 findings (1 no-sources)",
        "#1 web_researcher: reported empty searches",
        "#2 synthesizer: spawn failed or was denied",
    ]


def test_run_issues_flags_missing_hand_backs_and_a_missing_synthesizer():
    stream: list[Timed] = [
        (1.0, coordinator("r1", spawn_call("a", "doc_analyst"))),
        (2.0, coordinator("r2", spawn_call("b", "web_researcher"))),
        (3.0, hand_back("b", "no json")),
    ]
    assert run_issues(collect(stream)) == [
        "#1 doc_analyst: never handed back",
        '#2 web_researcher: INVALID: no {"status": ...} result object in the text',
        "no synthesizer report",
    ]


def test_failures_show_each_tool_error_and_the_hand_back():
    text = format_failures(failure_run())
    assert "--- #1 web_researcher: partial (timeout): 2 findings (1 no-sources)" in text
    assert text.count("error from mcp__research__web_search") == 2
    assert '"errorCategory": "transient", "isRetryable": true' in text
    assert 'attempted_queries: ["battery recycling forecast 2022"' in text
    assert "suggested_alternatives: [\"search 'current' for 2025 growth figures\"]" in text
    assert "no-sources finding: No sources found for EU tonnage recycled in 2023." in text
    assert "--- #2 synthesizer: ERROR: PreToolUse:Agent hook error" in text
    assert "--- #3 synthesizer" not in text  # it worked


def clean_run() -> ResearchRun:
    """The parallel stream without its one denied tool call."""
    run = collect(parallel_stream())
    run.spawns[0].tool_calls = run.spawns[0].tool_calls[:1]
    return run


def test_failures_show_the_start_of_an_invalid_hand_back():
    stream: list[Timed] = [
        (1.0, coordinator("r1", spawn_call("a", "doc_analyst"))),
        (2.0, hand_back("a", '[{"claim": "a bare array"}]')),
    ]
    text = format_failures(collect(stream))
    assert '--- #1 doc_analyst: INVALID: no {"status": ...}' in text
    assert 'hand-back starts: [{"claim": "a bare array"}]' in text


def test_failures_without_trouble_say_none():
    assert format_failures(clean_run()) == "=== Tool errors and hand-backs:\n  none"


def test_degradation_shows_issues_and_the_coverage_section():
    text = format_degradation(failure_run())
    assert "  - #2 synthesizer: spawn failed or was denied" in text
    assert COVERAGE in text
    assert "!!!" not in text


def test_degradation_without_coverage_is_called_out():
    text = format_degradation(failure_run(report="# Report\nAll good."))
    assert "=== Coverage section of the final answer:\n(missing)" in text
    assert "!!! The run degraded but the answer has no Coverage section." in text


def test_a_clean_run_has_no_issues():
    text = format_degradation(clean_run())
    assert "=== Degradation, from the trace:\n  none" in text


def test_format_run_adds_the_failure_sections_only_when_asked():
    run = failure_run()
    assert "=== Tool errors and hand-backs:" not in format_run(run)
    full = format_run(run, failures=True)
    assert "=== Tool errors and hand-backs:" in full
    assert "=== Degradation, from the trace:" in full


def test_failures_list_empty_searches_from_a_complete_result():
    empty = {"claim": "No sources found for 2023 tonnage.", "source_type": "search", "query": "q"}
    complete = json.dumps({"status": "complete", "findings": [empty]})
    stream: list[Timed] = [
        (1.0, coordinator("r1", spawn_call("a", "web_researcher"))),
        (2.0, hand_back("a", complete)),
    ]
    text = format_failures(collect(stream))
    assert "--- #1 web_researcher: complete: 1 findings (1 no-sources)" in text
    assert "no-sources finding: No sources found for 2023 tonnage. (query: 'q')" in text
    assert "attempted_queries" not in text


# --- Step 4: corrections and report structure -------------------------------------

STRUCTURED = """# EU market

Short answer.

## Established
- EUR 1.9 billion in 2025 for EU-27 EV processing revenue [nordwell-eu-outlook-2026].

## Contested
### 2025 market size
- EUR 1.9 billion: EU-27, EV only, service revenue [nordwell-eu-outlook-2026].
- EUR 3.4 billion: EU-27 + EFTA + UK, all lithium-ion, metal value included
  [ebca-circularity-report-2025].
The difference is scope, as both reports state.

## Coverage
- Gaps: none failed.
"""


def structured_run(report: str = STRUCTURED) -> ResearchRun:
    return collect([(1.0, result_message(report=report))], mode="parallel")


def test_report_section_stops_at_the_next_section_but_keeps_subheadings():
    contested = report_section(STRUCTURED, "Contested")
    assert contested is not None
    assert contested.startswith("## Contested\n### 2025 market size")
    assert contested.endswith("as both reports state.")
    assert report_section(STRUCTURED, "Established") == (
        "## Established\n- EUR 1.9 billion in 2025 for EU-27 EV processing revenue "
        "[nordwell-eu-outlook-2026]."
    )


def test_report_structure_lists_each_section_and_shows_contested():
    text = format_report_structure(structured_run())
    assert text.startswith("=== Report sections: Established yes, Contested yes, Coverage yes")
    assert "### 2025 market size" in text
    assert "## Established\n- EUR 1.9 billion" in text
    assert "## Coverage\n- Gaps: none failed." in text
    assert "### 2025" not in format_report_structure(structured_run(), texts=False)


def test_report_structure_flags_missing_sections():
    run = structured_run("# Report\n\n## Coverage\n- x")
    text = format_report_structure(run, texts=False)
    assert text == "=== Report sections: Established MISSING, Contested MISSING, Coverage yes"


def test_corrections_show_the_errors_sent_back_and_the_second_try():
    from ex4_research.validation import Correction

    run = structured_run()
    assert format_corrections(run) == "=== Hand-back corrections (SubagentStop): none"
    run.corrections = [
        Correction(
            "web_researcher",
            "a",
            ["partial.suggested_alternatives: Field required"],
            "x",
            outcome="passed",
        ),
        Correction(
            "doc_analyst",
            "b",
            ["(whole hand-back): no JSON object"],
            "[]",
            outcome="format_failed",
            final_errors=["(whole hand-back): no JSON object"],
        ),
        Correction("doc_analyst", "c", ["e"], "[]"),
    ]
    assert format_corrections(run) == (
        "=== Hand-back corrections (SubagentStop):\n"
        "--- web_researcher (a): blocked, 1 error(s) sent:\n"
        "    - partial.suggested_alternatives: Field required\n"
        "  second hand-back: passed\n"
        "--- doc_analyst (b): blocked, 1 error(s) sent:\n"
        "    - (whole hand-back): no JSON object\n"
        "  second hand-back: format_failed (1 error(s)):\n"
        "    - (whole hand-back): no JSON object\n"
        "--- doc_analyst (c): blocked, 1 error(s) sent:\n"
        "    - e\n"
        "  second hand-back: none arrived"
    )
    assert run_issues(run)[-2:] == [
        "doc_analyst: hand-back format_failed after one correction",
        "doc_analyst: hand-back pending after one correction",
    ]


def test_format_run_shows_corrections_and_structure():
    run = structured_run()
    full, brief = format_run(run), format_run(run, texts=False)
    assert "=== Hand-back corrections (SubagentStop): none" in full
    assert full.count("### 2025 market size") == 1  # in the report, not printed twice
    assert brief.count("### 2025 market size") == 1  # report left out, Contested shown


def test_synthesis_says_code_attached_the_findings():
    """The trace shows the coordinator's prompt from before the hook rewrote it."""
    run = collect(parallel_stream(), mode="parallel")
    run.attached = [50]
    text = format_synthesis(run)
    assert "=== Synthesizer's prompt (" in text
    assert "the coordinator's text; code attached 50 findings):" in text
    assert "--- citations: 2 distinct source_ids cited; not in the findings: none;" in text
