"""Hooks that make the hand-off to the synthesizer complete, not just requested.

The coordinator's prompt asks it to pass every finding to the synthesizer
unchanged; a live run passed them as separate arrays, and nothing stopped it from
dropping or editing some. These hooks check instead:

- PostToolUse on the spawn tool records every finding a researcher hands back.
- PreToolUse on the spawn tool denies a synthesizer spawn whose prompt is missing
  any of them, and says which, so the coordinator can retry with all of them.
"""

import contextlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import HookCallback, HookContext, HookInput, HookJSONOutput

from ex4_research.findings import (
    Finding,
    NoSources,
    findings_of,
    parse_research_result,
    valid_findings,
)
from ex4_research.timeline import EXPECTED_SOURCE_TYPE, SYNTHESIZER

logger = logging.getLogger(__name__)

RESEARCHERS = frozenset(EXPECTED_SOURCE_TYPE)
MAX_LISTED = 8  # missing findings named in a denial; the rest are counted


def response_text(value: object) -> str:
    """Every string inside a tool response, joined: the hand-back text wherever it sits."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(response_text(item) for item in value.values())
    if isinstance(value, list | tuple):
        return "\n".join(response_text(item) for item in value)
    return ""


AnyFinding = Finding | NoSources


def finding_key(finding: AnyFinding) -> str:
    """Identity of a finding: all of its fields, so an edited finding counts as missing."""
    return finding.model_dump_json()


@dataclass
class FindingsLedger:
    """Every finding researchers handed back in one run, and every denial issued."""

    received: list[AnyFinding] = field(default_factory=list)
    denials: list[str] = field(default_factory=list)

    def record(self, findings: Iterable[AnyFinding]) -> None:
        known = {finding_key(f) for f in self.received}
        for finding in findings:
            if finding_key(finding) not in known:
                known.add(finding_key(finding))
                self.received.append(finding)

    def missing_from(self, prompt: str) -> list[AnyFinding]:
        """The recorded findings that the prompt does not contain unchanged."""
        given = {finding_key(f) for f in valid_findings(prompt)}
        return [f for f in self.received if finding_key(f) not in given]


def label(finding: AnyFinding) -> str:
    source = finding.source_id if isinstance(finding, Finding) else f"search: {finding.query}"
    return f"[{source}] {finding.claim[:80].rstrip('.')}"


def denial_reason(missing: list[AnyFinding], total: int, read: int = 0) -> str:
    """read: how many valid findings the prompt did contain, so the coordinator can
    tell a dropped finding (read is high) from JSON it could not parse (read is 0)."""
    listed = [label(f) for f in missing[:MAX_LISTED]]
    more = len(missing) - len(listed)
    return (
        f"Synthesizer prompt is missing {len(missing)} of the {total} findings the "
        f"researchers returned (it contains {read} valid findings). Spawn it again with "
        "every finding as JSON, unchanged; any nesting works. Missing: "
        + "; ".join(listed)
        + (f"; and {more} more." if more else ".")
    )


def make_completeness_hooks(ledger: FindingsLedger) -> tuple[HookCallback, HookCallback]:
    """(PostToolUse recorder, PreToolUse guard) for the spawn tool, sharing one ledger."""

    async def record_findings(
        input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        tool_input: Any = input_data.get("tool_input") or {}
        if isinstance(tool_input, dict) and tool_input.get("subagent_type") in RESEARCHERS:
            # An invalid hand-back adds nothing to require; the timeline reports it.
            with contextlib.suppress(ValueError):
                result = parse_research_result(response_text(input_data.get("tool_response")))
                ledger.record(findings_of(result))
        return {}

    async def require_all_findings(
        input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        tool_input: Any = input_data.get("tool_input") or {}
        if not isinstance(tool_input, dict) or tool_input.get("subagent_type") != SYNTHESIZER:
            return {}
        prompt = str(tool_input.get("prompt", ""))
        missing = ledger.missing_from(prompt)
        if not missing:
            return {}
        reason = denial_reason(missing, len(ledger.received), len(valid_findings(prompt)))
        ledger.denials.append(reason)
        logger.warning("Denied synthesizer spawn: %s", reason)
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    return record_findings, require_all_findings
