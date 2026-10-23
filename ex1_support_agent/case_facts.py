"""Case facts: exact values recorded by code from tool results.

In a long conversation the model summarizes, rounds and mixes things up:
"$129.99" becomes "about $130", or two order IDs get swapped. So the key facts
are not left to the model's memory. A PostToolUse hook records them from each
successful tool result, and render() turns them into a block the session puts
in front of the model again on every turn.
"""

import json
from typing import Any

from claude_agent_sdk import HookCallback, HookContext, HookInput, HookJSONOutput
from pydantic import BaseModel, Field

from ex1_support_agent.hooks import clean_tool_output, mcp_tool_name

TOOL_PREFIX = mcp_tool_name("")  # "mcp__support__"


class OrderFacts(BaseModel):
    customer_id: str  # the owner: render() shows details only to them
    status: str
    total: float
    delivered_at: str | None  # ISO 8601, as the PostToolUse hook produces it


class RefundFacts(BaseModel):
    customer_id: str
    refund_id: str
    order_id: str
    amount: float


class CaseFacts(BaseModel):
    verified_customer_id: str | None = None
    orders: dict[str, OrderFacts] = Field(default_factory=dict)
    refunds: list[RefundFacts] = Field(default_factory=list)
    escalation_ticket: str | None = None
    escalated_order_ids: set[str] = Field(default_factory=set)  # handed to a human


# --- Recording ----------------------------------------------------------------


def record(facts: CaseFacts, tool: str, tool_input: dict[str, Any], result: dict[str, Any]) -> None:
    """Update facts from one successful tool result. tool is the short name, e.g. lookup_order."""
    match tool:
        case "get_customer":
            # Verified only by a unique match on something the customer must know.
            # A name is not proof: anyone can type a name.
            identified_by_secret = bool(tool_input.get("email") or tool_input.get("customer_id"))
            # One customer per conversation: the first verification sticks.
            # Typing someone else's email later must not switch accounts.
            if identified_by_secret and result.get("count") == 1 and not facts.verified_customer_id:
                facts.verified_customer_id = result["matches"][0]["customer_id"]
        case "lookup_order":
            # Code records the full truth, including orders the customer is not
            # verified for. render() decides what the model may see.
            if result.get("found"):
                facts.orders[result["order_id"]] = OrderFacts(
                    customer_id=result["customer_id"],
                    status=result["status"],
                    total=result["total"],
                    delivered_at=result.get("delivered_at"),
                )
        case "process_refund":
            facts.refunds.append(
                RefundFacts(
                    customer_id=result["customer_id"],
                    refund_id=result["refund_id"],
                    order_id=result["order_id"],
                    amount=result["amount"],
                )
            )
            if order := facts.orders.get(result["order_id"]):
                order.status = result["status"]
        case "escalate_to_human":
            facts.escalation_ticket = result["ticket_id"]
            summary = tool_input.get("summary") or {}
            facts.escalated_order_ids.update(
                str(o).lstrip("#") for o in summary.get("order_ids", [])
            )


def make_fact_recorder(facts: CaseFacts) -> HookCallback:
    """A PostToolUse callback that records into this session's own CaseFacts.

    A closure instead of a module-level global: two sessions never share facts.
    It reads the same cleaned output the model sees (ISO dates, trimmed orders).
    """

    async def record_case_facts(
        input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        if input_data["hook_event_name"] != "PostToolUse":
            return {}
        tool_name = input_data["tool_name"]
        cleaned = clean_tool_output(tool_name, input_data["tool_response"])
        if cleaned is None:  # an error or an unexpected shape: nothing to record
            return {}
        tool = tool_name.removeprefix(TOOL_PREFIX)
        record(facts, tool, input_data["tool_input"], json.loads(cleaned))
        return {}  # observe only: the output itself is replaced by replace_tool_output

    return record_case_facts


# --- Rendering ----------------------------------------------------------------


# The block goes into the model's context every turn, so it is a path for data
# just like tool output. Redaction has to cover it too, and render() is its
# single gate: whatever the verification is NOW decides what is shown.
WITHHELD_NOT_VERIFIED = "details withheld (customer not verified)"
WITHHELD_NOT_ON_ACCOUNT = "details withheld (not on the verified customer's account)"


def _withheld(facts: CaseFacts, owner: str) -> str | None:
    """Why details for this owner's data are hidden right now, or None to show them."""
    if facts.verified_customer_id is None:
        return WITHHELD_NOT_VERIFIED
    if owner != facts.verified_customer_id:
        return WITHHELD_NOT_ON_ACCOUNT
    return None


def _money(amount: float) -> str:
    return f"{amount:.2f}"


def render(facts: CaseFacts) -> str:
    """A stable <case_facts> block with exact values, in a fixed order."""
    lines = [
        "<case_facts>",
        "Recorded by the system from tool results. These values are exact: use them "
        "instead of your memory of the conversation.",
        f"verified_customer_id: {facts.verified_customer_id or 'none (not verified)'}",
        "orders:",
    ]
    for order_id in sorted(facts.orders):
        order = facts.orders[order_id]
        if withheld := _withheld(facts, order.customer_id):
            lines.append(f"  - {order_id}: {withheld}")
        else:
            lines.append(
                f"  - {order_id}: status={order.status} total={_money(order.total)} "
                f"delivered_at={order.delivered_at or 'none'}"
            )
    if not facts.orders:
        lines.append("  (none)")
    lines.append("refunds:")
    for refund in facts.refunds:
        if withheld := _withheld(facts, refund.customer_id):
            lines.append(f"  - order {refund.order_id}: {withheld}")
        else:
            lines.append(
                f"  - {refund.refund_id}: order={refund.order_id} amount={_money(refund.amount)}"
            )
    if not facts.refunds:
        lines.append("  (none)")
    escalated = ", ".join(sorted(facts.escalated_order_ids)) or "none"
    lines += [
        f"escalation_ticket: {facts.escalation_ticket or 'none'}",
        f"escalated_orders: {escalated}",
        "</case_facts>",
    ]
    return "\n".join(lines)
