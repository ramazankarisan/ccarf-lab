"""Hooks that make agent behavior deterministic instead of prompt-dependent.

Pure functions plus the SDK hook callbacks; no agent run happens in here.
"""

import json
import logging
import math
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, TypeGuard

from claude_agent_sdk import HookCallback, HookContext, HookInput, HookJSONOutput

logger = logging.getLogger(__name__)

# The name agent.py registers the MCP server under. The SDK exposes each tool
# as mcp__<server>__<tool>, so tool names and hook matchers are built from it.
SERVER_NAME = "support"


def mcp_tool_name(tool: str) -> str:
    """The name the Agent SDK gives one of our MCP tools, e.g. mcp__support__lookup_order."""
    return f"mcp__{SERVER_NAME}__{tool}"


# --- PreToolUse: refund limit -------------------------------------------------
# The limit lives here and nowhere else: not in the backend, not in the prompt.
# A prompt rule is a request the model may ignore; this hook is a guarantee.

REFUND_LIMIT = 500
# Built from constants: a matcher with a typo matches nothing and fails silently.
REFUND_MATCHER = mcp_tool_name("process_refund")

OVER_LIMIT_REASON = (
    f"Refunds over ${REFUND_LIMIT} need a human. Call escalate_to_human with the amount and order."
)
INVALID_AMOUNT_REASON = (
    "Refund blocked: amount must be a number in USD, for example 49.99. "
    "Ask the customer for the amount if it is unclear."
)


def check_refund(tool_input: Mapping[str, object]) -> str | None:
    """Return a deny reason for this process_refund input, or None to allow it.

    Fails closed: the hook sees the model's raw input before schema validation,
    so anything that is not a plain finite number (a missing amount, "650",
    True, NaN, inf) is denied rather than trusted to be caught later.
    """
    amount = tool_input.get("amount")
    # bool is a subclass of int in Python (True == 1), so it is excluded first.
    if isinstance(amount, bool) or not isinstance(amount, int | float):
        return INVALID_AMOUNT_REASON
    if not math.isfinite(amount):
        return INVALID_AMOUNT_REASON
    if amount > REFUND_LIMIT:
        return OVER_LIMIT_REASON
    return None


async def enforce_refund_limit(
    input_data: HookInput, tool_use_id: str | None, context: HookContext
) -> HookJSONOutput:
    """PreToolUse callback: deny process_refund calls that check_refund rejects."""
    if input_data["hook_event_name"] != "PreToolUse" or input_data["tool_name"] != REFUND_MATCHER:
        return {}  # not ours: no opinion, the normal permission rules apply
    reason = check_refund(input_data["tool_input"])
    if reason is None:
        return {}
    logger.warning("Denied %s %s: %s", REFUND_MATCHER, input_data["tool_input"], reason)
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,  # the model reads this as the tool's error
        }
    }


# --- PostToolUse: normalize and trim ------------------------------------------
# Deterministic transformations happen in code, before the model sees the data.
# The model once read delivered_at=1792368000 as "October 18" (it is the 19th).

# Key names decide what is a timestamp, never the size of a number.
TIMESTAMP_KEYS = frozenset({"delivered_at", "refunded_at", "created_at", "at"})

# Allow-lists, not block-lists: a noisy field added to the backend later is
# dropped automatically instead of leaking until someone notices.
ORDER_FIELDS = ("found", "order_id", "customer_id", "status", "total", "delivered_at", "items")
ITEM_FIELDS = ("name", "qty", "unit_price")

LOOKUP_ORDER = mcp_tool_name("lookup_order")
# "|" joins exact tool names in a matcher.
CLEAN_OUTPUT_MATCHER = f"{LOOKUP_ORDER}|{REFUND_MATCHER}"


def to_iso(timestamp: int) -> str:
    """Unix seconds to ISO 8601 UTC, e.g. 1792368000 -> "2026-10-19T00:00:00Z"."""
    return datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_unix_seconds(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def normalize_timestamps(obj: object) -> object:
    """Return a copy with every known timestamp key converted to ISO 8601 UTC.

    Walks nested dicts and lists. None stays None, and so does anything under
    a timestamp key that is not an int.
    """
    if isinstance(obj, Mapping):
        return {
            key: to_iso(value)
            if key in TIMESTAMP_KEYS and _is_unix_seconds(value)
            else normalize_timestamps(value)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [normalize_timestamps(item) for item in obj]
    return obj


def trim_order(order: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the order fields a support conversation needs.

    Drops warehouse_code, carrier_internal_ref, sku_audit_log and each item's sku.
    """
    trimmed = {key: order[key] for key in ORDER_FIELDS if key in order}
    if isinstance(trimmed.get("items"), list):
        trimmed["items"] = [
            {key: item[key] for key in ITEM_FIELDS if key in item}
            for item in trimmed["items"]
            if isinstance(item, Mapping)
        ]
    return trimmed


def clean_tool_output(tool_name: str, tool_response: object) -> str | None:
    """The tool output the model should see instead, or None to leave it unchanged.

    Only successful JSON results are transformed. Anything else (an error
    result, plain text, an unexpected shape) passes through untouched.
    """
    if not isinstance(tool_response, str):
        return None
    try:
        data = json.loads(tool_response)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "errorCategory" in data:
        return None
    cleaned = normalize_timestamps(data)
    if tool_name == LOOKUP_ORDER and isinstance(cleaned, dict):
        cleaned = trim_order(cleaned)
    return json.dumps(cleaned)


# --- Authorization: whose order is this? --------------------------------------
# Verifying WHO the customer is does not authorize them for EVERY order. If the
# model only saw the details when the prompt allowed it, one slip would leak
# another customer's order. So the details never reach the model at all.

# One reason per situation, each saying what to do next (the 2.2 lesson): a
# single "verify first" made the model re-verify, retry and even escalate when
# the order simply belonged to someone else.
NOT_VERIFIED_REASON = "Verify the customer with get_customer, then call lookup_order again."
NOT_ON_ACCOUNT_REASON = (
    "This order is not on the verified customer's account. Do not retry and do not "
    "escalate. Tell the customer you couldn't find it on their account and ask them "
    "to check the number."
)


def withhold_unowned_order(
    order: dict[str, Any], verified_customer_id: str | None
) -> dict[str, Any]:
    """Keep a found order's details only if it belongs to the verified customer.

    Otherwise the model learns just that the order exists, which is what it may
    tell an unverified customer anyway.
    """
    if not order.get("found"):
        return order
    if verified_customer_id is None:
        reason = NOT_VERIFIED_REASON
    elif order.get("customer_id") != verified_customer_id:
        reason = NOT_ON_ACCOUNT_REASON
    else:
        return order
    return {"found": True, "order_id": order.get("order_id"), "details_withheld": reason}


def make_output_cleaner(verified_customer_id: Callable[[], str | None]) -> HookCallback:
    """A PostToolUse callback that replaces the output the model sees.

    verified_customer_id is asked on every call, so it always reflects the
    session's latest verification. A function rather than a CaseFacts keeps
    this module independent of case_facts.py.

    updatedToolOutput REPLACES the output. additionalContext would only add to
    it, so the model would still read the raw data next to the clean copy.
    """

    async def replace_tool_output(
        input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        if input_data["hook_event_name"] != "PostToolUse":
            return {}
        tool_name = input_data["tool_name"]
        cleaned = clean_tool_output(tool_name, input_data["tool_response"])
        if cleaned is None:
            return {}
        if tool_name == LOOKUP_ORDER:
            order = withhold_unowned_order(json.loads(cleaned), verified_customer_id())
            cleaned = json.dumps(order)
        return {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "updatedToolOutput": cleaned,
            }
        }

    return replace_tool_output
