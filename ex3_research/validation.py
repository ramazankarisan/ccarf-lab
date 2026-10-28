"""Hand-back validation: a researcher may not stop with a result code cannot read.

The hook runs on the SubagentStop event (named SubagentStop in claude-agent-sdk
0.2.172, bundled CLI 2.1.308; its input carries agent_type, agent_id,
last_assistant_message and stop_hook_active).

Live runs showed researchers handing back results that code could not
read (a missing field, an empty excerpt, a finding without source_type), most
often after a tool failure. The coordinator still read the text, but the
completeness hook and the citation check silently lost those findings.

When a researcher stops, the hook validates its last message. If it is invalid,
the hook blocks the stop once and sends back every error as field and problem,
so the subagent corrects its own output while it still has the context. The
second hand-back is let through either way (stop_hook_active is then true): if
it is still invalid it is marked format_failed, so a subagent can never loop.
"""

import json
import logging
from dataclasses import dataclass, field

from claude_agent_sdk import HookCallback, HookContext, HookInput, HookJSONOutput
from pydantic import ValidationError

from ex3_research.findings import RESULT, RESULT_START, Finding, findings_of
from ex3_research.timeline import EXPECTED_SOURCE_TYPE

logger = logging.getLogger(__name__)

MAX_ERRORS = 10  # errors listed in one correction request; the rest are counted


def first_result_object(text: str) -> object | None:
    """The first decodable {"status": ...} object in text, or None."""
    decoder = json.JSONDecoder()
    for match in RESULT_START.finditer(text):
        try:
            value: object = decoder.raw_decode(text, match.start())[0]
        except json.JSONDecodeError:
            continue
        return value
    return None


def handback_errors(agent_type: str, text: str) -> list[str]:
    """Every problem with a researcher's hand-back, as 'field: problem'; empty if usable."""
    value = first_result_object(text)
    if value is None:
        return ['(whole hand-back): no JSON object starting with {"status": ...} was found']
    try:
        result = RESULT.validate_python(value)
    except ValidationError as error:
        return [
            f"{'.'.join(str(part) for part in e['loc']) or '(top level)'}: {e['msg']}"
            for e in error.errors()
        ]
    expected = EXPECTED_SOURCE_TYPE.get(agent_type)
    return [
        f"{f.source_id}: source_type is {f.source_type!r}, but {agent_type} can only report "
        f"{expected!r} sources; mention other sources inside a claim instead"
        for f in findings_of(result)
        if isinstance(f, Finding) and expected is not None and f.source_type != expected
    ]


def correction_request(errors: list[str]) -> str:
    listed = errors[:MAX_ERRORS]
    more = f"\n- and {len(errors) - len(listed)} more" if len(errors) > len(listed) else ""
    return (
        "Your hand-back does not match the result format. Errors:\n"
        + "\n".join(f"- {e}" for e in listed)
        + more
        + "\nReturn it again as exactly one JSON object in the result format from your "
        "instructions, keeping every finding and every query you tried. Do not call any "
        "more tools."
    )


@dataclass
class Correction:
    """One blocked stop: who, what was wrong, the hand-back, and how the retry went."""

    agent_type: str
    agent_id: str
    errors: list[str]
    handback: str
    outcome: str = "pending"  # "passed" or "format_failed" once the second hand-back arrives
    final_errors: list[str] = field(default_factory=list)


@dataclass
class HandbackLog:
    corrections: list[Correction] = field(default_factory=list)


def make_handback_validator(log: HandbackLog) -> HookCallback:
    async def validate_handback(
        input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        agent_type = str(input_data.get("agent_type", ""))
        if agent_type not in EXPECTED_SOURCE_TYPE:
            return {}  # the synthesizer writes Markdown, not a result object
        agent_id = str(input_data.get("agent_id", ""))
        text = str(input_data.get("last_assistant_message", ""))
        errors = handback_errors(agent_type, text)
        earlier = [c for c in log.corrections if c.agent_id == agent_id and c.outcome == "pending"]
        if input_data.get("stop_hook_active") or earlier:
            # The second hand-back: let it through, and record how it went.
            for correction in earlier:
                correction.outcome = "format_failed" if errors else "passed"
                correction.final_errors = errors
            if errors:
                logger.warning("%s format_failed after one correction: %s", agent_type, errors)
            return {}
        if not errors:
            return {}
        log.corrections.append(Correction(agent_type, agent_id, errors, text))
        return {"decision": "block", "reason": correction_request(errors)}

    return validate_handback
