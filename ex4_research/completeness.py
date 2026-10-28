"""Hooks that make the hand-off to the synthesizer complete by construction.

Step 2 asked the coordinator to copy every finding into the synthesizer's prompt
and denied a prompt that missed one. Live runs showed a model cannot copy 50
findings exactly: one excerpt always changed, and every synthesizer spawn was
denied. So code passes the data instead of the model:

- PostToolUse on the spawn tool records every finding a researcher hands back.
- PreToolUse on the spawn tool rewrites a synthesizer spawn's prompt: the
  coordinator's text (question, emphasis, process notes) plus every recorded
  finding as JSON, attached by code.
"""

import contextlib
import json
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
)
from ex4_research.timeline import EXPECTED_SOURCE_TYPE, SYNTHESIZER

logger = logging.getLogger(__name__)

RESEARCHERS = frozenset(EXPECTED_SOURCE_TYPE)


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
    """Identity of a finding: all of its fields, so a repeat is recorded once."""
    return finding.model_dump_json()


FINDINGS_HEADER = (
    "Findings, attached by code: every finding the researchers handed back, complete and unchanged."
)


@dataclass
class FindingsLedger:
    """Every finding researchers handed back in one run, and how many each synthesis got."""

    received: list[AnyFinding] = field(default_factory=list)
    attached: list[int] = field(default_factory=list)  # findings attached per synthesizer spawn

    def record(self, findings: Iterable[AnyFinding]) -> None:
        known = {finding_key(f) for f in self.received}
        for finding in findings:
            if finding_key(finding) not in known:
                known.add(finding_key(finding))
                self.received.append(finding)

    def attach_to(self, prompt: str) -> str:
        """The coordinator's prompt with every recorded finding appended as JSON."""
        data = json.dumps([f.model_dump(mode="json") for f in self.received], indent=1)
        return f"{prompt.rstrip()}\n\n{FINDINGS_HEADER}\n{data}"


def make_completeness_hooks(ledger: FindingsLedger) -> tuple[HookCallback, HookCallback]:
    """(PostToolUse recorder, PreToolUse attacher) for the spawn tool, sharing one ledger."""

    async def record_findings(
        input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        tool_input: Any = input_data.get("tool_input") or {}
        if isinstance(tool_input, dict) and tool_input.get("subagent_type") in RESEARCHERS:
            # An invalid hand-back adds nothing; the timeline reports it.
            with contextlib.suppress(ValueError):
                result = parse_research_result(response_text(input_data.get("tool_response")))
                ledger.record(findings_of(result))
        return {}

    async def attach_findings(
        input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        tool_input: Any = input_data.get("tool_input") or {}
        if not isinstance(tool_input, dict) or tool_input.get("subagent_type") != SYNTHESIZER:
            return {}
        ledger.attached.append(len(ledger.received))
        logger.info("Attached %d findings to the synthesizer prompt", len(ledger.received))
        updated = {
            **tool_input,
            "prompt": ledger.attach_to(str(tool_input.get("prompt", ""))),
            # This hook owns the synthesizer's input, so it applies the foreground
            # rule too: two hooks returning updatedInput for one call would compete.
            "run_in_background": False,
        }
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "permissionDecisionReason": "Findings attached by code.",
                "updatedInput": updated,
            }
        }

    return record_findings, attach_findings
