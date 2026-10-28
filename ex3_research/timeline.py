"""Reading a research run from the SDK's message stream: who was spawned, when, and what came back.

Pure functions plus one async collector; no agent run happens in here.
"""

import json
import re
import statistics
import time
from collections.abc import AsyncIterable, Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import (
    AssistantMessage,
    Message,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from ex1_support_agent.agent import result_text
from ex3_research.findings import (
    Finding,
    NoSources,
    Partial,
    check_citations,
    findings_of,
    parse_findings,
    parse_research_result,
)

if TYPE_CHECKING:  # validation.py imports this module; import it for types only
    from ex3_research.validation import Correction

# The subagent-spawning tool. The exam (and older SDKs) call it "Task"; in
# claude-agent-sdk 0.2.172 (bundled CLI 2.1.308) it is "Agent". The CLI still
# accepts "Task" as an alias, but tool_use blocks report the new name.
SPAWN_TOOL = "Agent"
SPAWN_TOOL_NAMES = frozenset({"Agent", "Task"})

# The source_type each researcher must report; the synthesizer reports no findings.
EXPECTED_SOURCE_TYPE = {"web_researcher": "web", "doc_analyst": "doc"}
SYNTHESIZER = "synthesizer"
SYNTHESIZER_PROMPT_PREVIEW = 700  # characters of the synthesizer's prompt to print

# A message with the time it arrived, in seconds since the run started.
Timed = tuple[float, Message]


async def gather_timed(
    messages: AsyncIterable[Message], clock: Callable[[], float] = time.monotonic
) -> list[Timed]:
    """Read the whole stream, stamping each message with its arrival time."""
    start = clock()
    return [(clock() - start, message) async for message in messages]


@dataclass
class InnerCall:
    """A tool call a subagent made, and whether it worked."""

    name: str
    input: dict[str, Any]
    is_error: bool | None = None  # None until its result arrives
    output: str = ""


@dataclass
class Spawn:
    """One subagent the coordinator started: what it was told, when, and what it returned."""

    tool_use_id: str
    subagent_type: str
    description: str
    prompt: str
    background: bool  # asked for run_in_background (the hook turns it off)
    response: int  # which coordinator response spawned it, counting from 1
    spawned_at: float
    returned_at: float | None = None  # None if it never handed back
    tool_calls: list[InnerCall] = field(default_factory=list)
    output: str | None = None
    is_error: bool = False


@dataclass
class ResearchRun:
    mode: str = ""
    spawns: list[Spawn] = field(default_factory=list)
    result: ResultMessage | None = None
    finished_at: float | None = None  # when the ResultMessage arrived: the wall-clock time
    attached: list[int] = field(default_factory=list)  # findings code attached per synthesis
    notes: list[tuple[float, str]] = field(default_factory=list)  # the coordinator's own text
    # From the SubagentStop hook: every hand-back it sent back, and how the retry went.
    corrections: list["Correction"] = field(default_factory=list)


def spawn_from(block: ToolUseBlock, response: int, at: float) -> Spawn:
    return Spawn(
        tool_use_id=block.id,
        subagent_type=str(block.input.get("subagent_type", "")),
        description=str(block.input.get("description", "")),
        prompt=str(block.input.get("prompt", "")),
        background=bool(block.input.get("run_in_background")),
        response=response,
        spawned_at=at,
    )


def collect(messages: Iterable[Timed], mode: str = "") -> ResearchRun:
    """Group the timed message stream by subagent.

    The coordinator's own messages have parent_tool_use_id None; a subagent's
    messages carry the id of the spawn tool_use that started it. Several spawn
    calls in one coordinator response share its message_id (the SDK may deliver
    them as one AssistantMessage or several), so they get the same response number.
    """
    run = ResearchRun(mode=mode)
    spawns: dict[str, Spawn] = {}
    inner_calls: dict[str, InnerCall] = {}
    responses: dict[str, int] = {}
    for at, message in messages:
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock) and message.parent_tool_use_id is None:
                    run.notes.append((at, block.text))
                if not isinstance(block, ToolUseBlock):
                    continue
                if message.parent_tool_use_id in spawns:
                    call = InnerCall(block.name, block.input)
                    inner_calls[block.id] = call
                    spawns[message.parent_tool_use_id].tool_calls.append(call)
                elif message.parent_tool_use_id is None and block.name in SPAWN_TOOL_NAMES:
                    key = message.message_id or f"no-id-{id(message)}"
                    response = responses.setdefault(key, len(responses) + 1)
                    spawn = spawn_from(block, response, at)
                    spawns[block.id] = spawn
                    run.spawns.append(spawn)
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    continue
                if block.tool_use_id in spawns:
                    spawn = spawns[block.tool_use_id]
                    spawn.output = result_text(block.content)
                    spawn.is_error = bool(block.is_error)
                    spawn.returned_at = at
                elif block.tool_use_id in inner_calls:
                    call = inner_calls[block.tool_use_id]
                    call.output = result_text(block.content)
                    call.is_error = bool(block.is_error)
        elif isinstance(message, ResultMessage):
            run.result = message
            run.finished_at = at
    return run


def max_concurrency(spawns: Iterable[Spawn]) -> int:
    """The most subagents that were running at the same moment.

    A subagent that never handed back counts as running until the end. At equal
    times a hand-back is counted before a spawn, so back-to-back is not overlap.
    """
    events: list[tuple[float, int]] = []
    for spawn in spawns:
        events.append((spawn.spawned_at, 1))
        events.append((spawn.returned_at if spawn.returned_at is not None else float("inf"), -1))
    running = peak = 0
    for _, delta in sorted(events):
        running += delta
        peak = max(peak, running)
    return peak


# --- Formatting -------------------------------------------------------------------


def call_status(call: InnerCall) -> str:
    """[ok], [no result], or [ERROR: ...] with the start of the error, e.g. a denial reason."""
    if call.is_error is None:
        return "[no result]"
    if call.is_error:
        return f"[ERROR: {' '.join(call.output.split())[:160]}]"
    return "[ok]"


def one_line(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


def findings_status(spawn: Spawn) -> str:
    """What the hand-back contained: complete or partial, how many findings, or why invalid."""
    if spawn.output is None:
        return "no hand-back"
    if spawn.is_error:  # e.g. a hook denial: show its reason
        return f"ERROR: {one_line(spawn.output, 160)}"
    if spawn.subagent_type == SYNTHESIZER:
        return "report"
    try:
        result = parse_research_result(spawn.output)
    except ValueError as error:
        return f"INVALID: {one_line(str(error), 200)}"
    findings = findings_of(result)
    empty = sum(isinstance(f, NoSources) for f in findings)
    counts = f"{len(findings)} findings" + (f" ({empty} no-sources)" if empty else "")
    expected = EXPECTED_SOURCE_TYPE.get(spawn.subagent_type)
    wrong = [
        f for f in findings if isinstance(f, Finding) and expected and f.source_type != expected
    ]
    if wrong:
        counts += f", {len(wrong)} with source_type other than {expected!r}"
    if isinstance(result, Partial):
        return f"partial ({result.failure_type}): {counts}"
    return f"complete: {counts}"


def format_timeline(run: ResearchRun) -> str:
    """Every spawn and hand-back in time order, then how much ran at once."""
    events: list[tuple[float, int, str]] = []
    for number, spawn in enumerate(run.spawns, start=1):
        name = f"#{number} {spawn.subagent_type}"
        asked = ", asked for background" if spawn.background else ""
        text = f"spawn {name:<20} response {spawn.response}{asked}"
        events.append((spawn.spawned_at, 1, text))
        if spawn.returned_at is not None:
            took = spawn.returned_at - spawn.spawned_at
            text = f"back  {name:<20} after {took:.1f}s: {findings_status(spawn)}"
            events.append((spawn.returned_at, 0, text))
    for at, note in run.notes:
        events.append((at, 2, f"coordinator: {one_line(note, 160)}"))
    lines = [f"Timeline ({run.mode or 'run'}):", f"  {0:7.1f}s  start"]
    lines += [f"  {at:7.1f}s  {text}" for at, _, text in sorted(events)]
    lines += [
        f"  {'':>7}   never handed back: #{n} {s.subagent_type}"
        for n, s in enumerate(run.spawns, start=1)
        if s.returned_at is None
    ]
    if run.finished_at is not None:
        lines.append(f"  {run.finished_at:7.1f}s  done")
    responses = len({spawn.response for spawn in run.spawns})
    lines.append(
        f"At most {max_concurrency(run.spawns)} subagent(s) ran at once; "
        f"{len(run.spawns)} spawns in {responses} coordinator response(s)."
    )
    intervals = [(s.spawned_at, s.returned_at) for s in run.spawns if s.returned_at is not None]
    if intervals:
        # Busy time vs elapsed time: what overlap saved. Modes that split the work
        # differently do different amounts of it, so compare this, not only wall-clock.
        work = sum(back - spawned for spawned, back in intervals)
        span = max(back for _, back in intervals) - min(spawned for spawned, _ in intervals)
        lines.append(f"Subagent work: {work:.1f}s within {span:.1f}s, first spawn to last back.")
    return "\n".join(lines)


def format_spawn_details(run: ResearchRun) -> str:
    """Each subagent's prompt, tool calls and raw hand-back (the --verbose view)."""
    lines: list[str] = []
    for number, spawn in enumerate(run.spawns, start=1):
        lines += [
            f"=== {number}. {spawn.subagent_type}: {spawn.description}",
            "--- prompt it received:",
            spawn.prompt,
            "--- its tool calls:",
        ]
        lines += [
            f"  {call.name} {json.dumps(call.input)}  {call_status(call)}"
            for call in spawn.tool_calls
        ] or ["  (none)"]
        lines += ["--- raw hand-back:", spawn.output or "(none)", ""]
    return "\n".join(lines)


def synthesizer_spawn(run: ResearchRun) -> Spawn | None:
    """The last synthesizer spawn: the one whose report the coordinator should return."""
    synths = [spawn for spawn in run.spawns if spawn.subagent_type == SYNTHESIZER]
    return synths[-1] if synths else None


def researched_findings(run: ResearchRun) -> list[Finding | NoSources]:
    """Every finding in the researchers' valid hand-backs, in spawn order."""
    findings: list[Finding | NoSources] = []
    for spawn in run.spawns:
        if spawn.subagent_type in EXPECTED_SOURCE_TYPE and spawn.output and not spawn.is_error:
            try:
                findings += findings_of(parse_research_result(spawn.output))
            except ValueError:
                continue
    return findings


def format_synthesis(run: ResearchRun, *, texts: bool = True) -> str:
    """The synthesizer's prompt (shortened), the final report, and a citation check.

    texts=False keeps the headers and the check but leaves out the prompt and report.
    """
    synth = synthesizer_spawn(run)
    report = (run.result.result if run.result else None) or ""
    lines: list[str] = []
    if synth is None:
        lines.append("=== No synthesizer was spawned.")
        given = researched_findings(run)
    else:
        if run.attached:
            # The trace holds the coordinator's text from before the hook attached
            # the findings, so the findings are the ones the researchers handed back.
            given = researched_findings(run)
            contents = f"the coordinator's text; code attached {run.attached[-1]} findings"
        else:
            try:
                given = parse_findings(synth.prompt)
                contents = f"contains {len(given)} valid findings as JSON"
            except ValueError as error:
                given = researched_findings(run)
                contents = f"its findings JSON is {' '.join(str(error).split())[:200]}"
        shown = synth.prompt[:SYNTHESIZER_PROMPT_PREVIEW]
        more = " ..." if len(synth.prompt) > SYNTHESIZER_PROMPT_PREVIEW else ""
        lines.append(f"=== Synthesizer's prompt ({len(synth.prompt)} chars; {contents}):")
        if texts:
            lines.append(shown + more)
    if texts:
        lines += ["", f"=== Final report ({run.mode or 'run'}):", report or "(none)", ""]
    check = check_citations(report, given)
    if given:
        lines.append(
            f"--- citations: {len(check.cited)} distinct source_ids cited; "
            f"not in the findings: {', '.join(sorted(check.unknown)) or 'none'}; "
            f"findings' sources never cited: {len(check.uncited)}"
        )
    else:
        lines.append(
            f"--- citations: {len(check.cited)} distinct source_ids cited; "
            "not checked: no valid findings to check them against"
        )
    result = run.result
    if result is None:
        lines.append("Result: (no result message)")
    else:
        cost = f"${result.total_cost_usd:.4f}" if result.total_cost_usd is not None else "n/a"
        lines.append(f"Result: {result.subtype} | turns: {result.num_turns} | cost: {cost}")
    return "\n".join(lines)


# --- Failures and coverage (Step 3) --------------------------------------------------

REPORT_SECTIONS = ("Established", "Contested", "Coverage")


def report_section(report: str, name: str) -> str | None:
    """A report section: its heading up to the next heading of that level or higher."""
    heading = re.compile(rf"^#{{1,6}}\s*{re.escape(name)}\b.*$", re.MULTILINE | re.IGNORECASE)
    match = heading.search(report)
    if match is None:
        return None
    level = len(match.group(0)) - len(match.group(0).lstrip("#"))
    rest = report[match.end() :]
    following = re.search(rf"^#{{1,{level}}}\s", rest, re.MULTILINE)
    return (match.group(0) + (rest[: following.start()] if following else rest)).strip()


def coverage_section(report: str) -> str | None:
    return report_section(report, "Coverage")


def run_issues(run: ResearchRun) -> list[str]:
    """Every way this run degraded, read from the trace: what the answer must not hide."""
    issues: list[str] = []
    for number, spawn in enumerate(run.spawns, start=1):
        name = f"#{number} {spawn.subagent_type}"
        errors = [call for call in spawn.tool_calls if call.is_error]
        if errors:
            issues.append(f"{name}: {len(errors)} tool call(s) returned an error")
        if spawn.returned_at is None:
            issues.append(f"{name}: never handed back")
        elif spawn.is_error:
            issues.append(f"{name}: spawn failed or was denied")
        elif spawn.subagent_type in EXPECTED_SOURCE_TYPE:
            status = findings_status(spawn)
            if status.startswith(("partial", "INVALID")):
                issues.append(f"{name}: {status}")
            if "no-sources" in status:
                issues.append(f"{name}: reported empty searches")
    if synthesizer_spawn(run) is None:
        issues.append("no synthesizer report")
    issues += [
        f"{c.agent_type}: hand-back {c.outcome} after one correction"
        for c in run.corrections
        if c.outcome != "passed"
    ]
    return issues


def format_failures(run: ResearchRun) -> str:
    """Per subagent with trouble: each tool error as returned, and what it handed back."""
    lines = ["=== Tool errors and hand-backs:"]
    for number, spawn in enumerate(run.spawns, start=1):
        errors = [call for call in spawn.tool_calls if call.is_error]
        status = findings_status(spawn) if spawn.output is not None else "no hand-back"
        if (
            not errors
            and not status.startswith(("partial", "INVALID", "ERROR"))
            and ("no-sources" not in status)
        ):
            continue
        lines.append(f"--- #{number} {spawn.subagent_type}: {status}")
        for call in errors:
            lines.append(f"  error from {call.name} {json.dumps(call.input)}:")
            lines.append(f"    {one_line(call.output, 400)}")
        try:
            result = parse_research_result(spawn.output or "")
        except ValueError:
            if status.startswith("INVALID"):
                lines.append(f"  hand-back starts: {one_line(spawn.output or '', 300)}")
            continue
        if isinstance(result, Partial):
            lines.append(f"  attempted_queries: {json.dumps(result.attempted_queries)}")
            lines.append(f"  suggested_alternatives: {json.dumps(result.suggested_alternatives)}")
        for finding in findings_of(result):
            if isinstance(finding, NoSources):
                lines.append(f"  no-sources finding: {finding.claim} (query: {finding.query!r})")
    if len(lines) == 1:
        lines.append("  none")
    return "\n".join(lines)


def format_degradation(run: ResearchRun) -> str:
    """The run's issues from the trace, and whether the final answer shows them."""
    report = (run.result.result if run.result else None) or ""
    coverage = coverage_section(report)
    issues = run_issues(run)
    lines = ["=== Degradation, from the trace:"]
    lines += [f"  - {issue}" for issue in issues] or ["  none"]
    lines += ["", "=== Coverage section of the final answer:", coverage or "(missing)"]
    if issues and coverage is None:
        lines.append("!!! The run degraded but the answer has no Coverage section.")
    return "\n".join(lines)


def format_corrections(run: ResearchRun) -> str:
    """Every hand-back the SubagentStop hook sent back: the errors, and the second try."""
    if not run.corrections:
        return "=== Hand-back corrections (SubagentStop): none"
    lines = ["=== Hand-back corrections (SubagentStop):"]
    for c in run.corrections:
        lines.append(f"--- {c.agent_type} ({c.agent_id}): blocked, {len(c.errors)} error(s) sent:")
        lines += [f"    - {one_line(error, 200)}" for error in c.errors]
        if c.outcome == "passed":
            lines.append("  second hand-back: passed")
        elif c.outcome == "format_failed":
            lines.append(f"  second hand-back: format_failed ({len(c.final_errors)} error(s)):")
            lines += [f"    - {one_line(error, 200)}" for error in c.final_errors]
        else:
            lines.append("  second hand-back: none arrived")
    return "\n".join(lines)


def format_report_structure(run: ResearchRun, *, texts: bool = True) -> str:
    """Which required sections the final answer has, then (texts=True) each of them."""
    report = (run.result.result if run.result else None) or ""
    present = {name: report_section(report, name) is not None for name in REPORT_SECTIONS}
    lines = [
        "=== Report sections: "
        + ", ".join(f"{name} {'yes' if ok else 'MISSING'}" for name, ok in present.items())
    ]
    if texts:
        for name in REPORT_SECTIONS:
            section = report_section(report, name)
            if section:
                lines += ["", section]
    return "\n".join(lines)


def format_attachments(run: ResearchRun) -> str:
    if not run.attached:
        return "=== Findings attached by code to the synthesizer: none (no synthesis)"
    counts = ", ".join(str(n) for n in run.attached)
    return f"=== Findings attached by code to the synthesizer: {counts}"


def format_run(
    run: ResearchRun, *, verbose: bool = False, texts: bool = True, failures: bool = False
) -> str:
    """failures=True (Step 3) adds tool errors, hand-backs, degradation and coverage."""
    parts = [format_timeline(run), format_attachments(run), format_corrections(run)]
    if failures:
        parts.append(format_failures(run))
    if verbose:
        parts.append(format_spawn_details(run))
    parts.append(format_synthesis(run, texts=texts))
    # The sections are printed here only when the full report was not.
    parts.append(format_report_structure(run, texts=not texts))
    if failures:
        parts.append(format_degradation(run))
    return "\n\n".join(parts)


def format_timings(runs: Iterable[ResearchRun]) -> str:
    """Wall-clock time per mode, e.g. 'parallel 41.2s | sequential 77.9s'."""
    timings = [
        f"{run.mode} {run.finished_at:.1f}s" if run.finished_at is not None else f"{run.mode} n/a"
        for run in runs
    ]
    return "Wall-clock: " + " | ".join(timings)


# --- Phase timings ------------------------------------------------------------------

Interval = tuple[float, float]


def intervals(spawns: Iterable[Spawn]) -> list[Interval]:
    return [(s.spawned_at, s.returned_at) for s in spawns if s.returned_at is not None]


def span(found: list[Interval]) -> float | None:
    """First start to last end, or None if there is nothing to measure."""
    if not found:
        return None
    return max(end for _, end in found) - min(start for start, _ in found)


def union_length(found: list[Interval]) -> float:
    """Total time covered by at least one interval: overlaps count once."""
    total, reach = 0.0, float("-inf")
    for start, end in sorted(found):
        if end > reach:
            total += end - max(start, reach)
            reach = end
    return total


@dataclass(frozen=True)
class PhaseTimes:
    """Seconds per phase of one run. A phase that did not happen is None."""

    web: float | None
    doc: float | None
    synthesis: float | None
    coordinator: float | None  # time with no subagent running: the coordinator thinking
    total: float | None
    web_spawns: int
    doc_spawns: int


def phase_times(run: ResearchRun) -> PhaseTimes:
    """Phases from the timeline: each researcher type's first spawn to last hand-back,
    the synthesizer's spawns, and the rest of the wall-clock time."""
    by_type = {
        name: [s for s in run.spawns if s.subagent_type == name]
        for name in ("web_researcher", "doc_analyst", SYNTHESIZER)
    }
    everything = intervals(run.spawns)
    coordinator = (
        run.finished_at - union_length(everything) if run.finished_at is not None else None
    )
    return PhaseTimes(
        web=span(intervals(by_type["web_researcher"])),
        doc=span(intervals(by_type["doc_analyst"])),
        synthesis=span(intervals(by_type[SYNTHESIZER])),
        coordinator=coordinator,
        total=run.finished_at,
        web_spawns=len(by_type["web_researcher"]),
        doc_spawns=len(by_type["doc_analyst"]),
    )


def summarize(values: Iterable[float | None]) -> str:
    """'mean (min-max)' over the runs that had the phase, e.g. '18.3 (17.9-19.9)'."""
    values = list(values)
    present = [v for v in values if v is not None]
    if not present:
        return "n/a"
    missing = "" if len(present) == len(values) else f", {len(present)} runs"
    if len(present) == 1:
        return f"{present[0]:.1f}{missing}"
    return f"{statistics.mean(present):.1f} ({min(present):.1f}-{max(present):.1f}{missing})"


def format_phase_table(runs: Iterable[ResearchRun]) -> str:
    """Per-phase seconds for each mode over its runs: mean (min-max)."""
    by_mode: dict[str, list[PhaseTimes]] = {}
    for run in runs:
        by_mode.setdefault(run.mode, []).append(phase_times(run))
    rows: list[tuple[str, Callable[[PhaseTimes], float | None]]] = [
        ("web research", lambda t: t.web),
        ("doc verification", lambda t: t.doc),
        ("synthesis", lambda t: t.synthesis),
        ("coordinator only", lambda t: t.coordinator),
        ("total wall-clock", lambda t: t.total),
    ]
    modes = list(by_mode)
    counts = {mode: len(times) for mode, times in by_mode.items()}
    header = "| Phase (seconds) | " + " | ".join(f"{m} ({counts[m]} runs)" for m in modes) + " |"
    lines = [
        "Per-phase timings: mean (min-max)",
        "",
        header,
        "|---" * (len(modes) + 1) + "|",
    ]
    for label, pick in rows:
        cells = [summarize([pick(t) for t in by_mode[m]]) for m in modes]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    spawns = [", ".join(f"{t.web_spawns}+{t.doc_spawns}" for t in by_mode[m]) for m in modes]
    lines.append("| spawns per run (web+doc) | " + " | ".join(spawns) + " |")
    return "\n".join(lines)
