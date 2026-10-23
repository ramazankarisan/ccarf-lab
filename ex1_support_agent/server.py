"""MCP server exposing the fake support backend as tools.

Run over stdio:  uv run python -m ex1_support_agent.server

Uses the mcp 2.x SDK, where v1's FastMCP is called MCPServer.
"""

import functools
import inspect
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Self

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ex1_support_agent import backend

# Logs go to stderr. Never print to stdout: over stdio, stdout is the protocol channel.
logger = logging.getLogger(__name__)

mcp = MCPServer("support-backend")


# --- Structured errors (2.2) ----------------------------------------------------
# description is written for the agent (what happened, what to do next);
# customerMessage is written for the end customer, or null when the agent
# should not pass anything on.


@dataclass(frozen=True)
class ErrorKind:
    category: str
    retryable: bool
    next_step: str


ERROR_KINDS: dict[type[backend.BackendError], ErrorKind] = {
    backend.BackendTimeout: ErrorKind("transient", True, "Retrying the same call may succeed."),
    backend.InvalidInput: ErrorKind(
        "validation", False, "Fix the input, or ask the customer for the correct value."
    ),
    backend.PolicyViolation: ErrorKind(
        "business",
        False,
        "Do not retry. Explain the rule to the customer; escalate if they dispute it.",
    ),
    backend.NotAuthorized: ErrorKind(
        "permission",
        False,
        "Do not retry with another customer_id. Ask the customer to double-check "
        "the order number or confirm their identity.",
    ),
}

NOT_ON_ACCOUNT_MESSAGE = (
    "We couldn't find that order on your account. Please double-check the order number."
)
INTERNAL_ERROR_MESSAGE = "Something went wrong on our side. A support agent will look into it."


def error_result(exc: Exception) -> CallToolResult:
    """Turn any exception into an isError tool result the agent can act on."""
    if isinstance(exc, backend.BackendError) and (kind := ERROR_KINDS.get(type(exc))):
        category, retryable = kind.category, kind.retryable
        description = f"{exc.description} {kind.next_step}"
        customer_message: str | None = None
        if isinstance(exc, backend.PolicyViolation):
            customer_message = exc.description
        elif isinstance(exc, backend.NotAuthorized):
            customer_message = NOT_ON_ACCOUNT_MESSAGE
    else:
        # Unknown failure: log the details for operators, show nothing internal.
        logger.exception("Unexpected error in tool call")
        category, retryable = "internal", False
        description = "Unexpected internal error. Do not retry; escalate to a human."
        customer_message = INTERNAL_ERROR_MESSAGE

    payload = {
        "errorCategory": category,
        "isRetryable": retryable,
        "description": description,
        "customerMessage": customer_message,
    }
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))], is_error=True
    )


def tool_description(fn: Callable[..., Any]) -> str:
    """The docstring as the model should read it: no source indentation, no line wraps."""
    paragraphs = inspect.cleandoc(fn.__doc__ or "").split("\n\n")
    return "\n\n".join(" ".join(paragraph.split()) for paragraph in paragraphs)


def support_tool[**P](
    fn: Callable[P, dict[str, Any]],
) -> Callable[P, dict[str, Any] | CallToolResult]:
    """Register fn as an MCP tool with a cleaned description and structured errors.

    No retries here: the agent decides that from isRetryable.
    """

    @functools.wraps(fn)  # keeps name, docstring and signature for the input schema
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> dict[str, Any] | CallToolResult:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            return error_result(exc)

    mcp.tool(description=tool_description(fn))(wrapper)
    return wrapper


class HandoffSummary(BaseModel):
    """Everything a human agent needs to act without reading the transcript."""

    # Reject unknown fields instead of dropping them silently. The schema then
    # says additionalProperties: false, so the model also sees it up front.
    model_config = ConfigDict(extra="forbid")

    customer_id: str | None = Field(
        description="Customer ID from get_customer, e.g. 'C001'. null if not identified yet."
    )
    verified: bool = Field(
        description="true only if exactly one get_customer match came from the customer's "
        "email or customer ID. A name-only match is not verified."
    )
    order_ids: list[Annotated[str, Field(pattern=r"^\d{5}$")]] = Field(
        description="Every order involved, as 5-digit numbers without '#', e.g. ['45678']. "
        "Empty list if none."
    )
    issue: str = Field(description="What the customer wants, in one sentence.")
    root_cause: str = Field(description="What actually went wrong, as far as is known.")
    amount: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description="Money involved in USD, e.g. 650.0. null if no amount applies.",
    )
    actions_taken: list[str] = Field(
        description="What was already tried, in order, including blocked or failed attempts, "
        "e.g. ['Verified by email', 'process_refund for 650.0 blocked: needs a human']. "
        "Empty list if nothing was tried."
    )
    policy_applied: str | None = Field(
        default=None,
        description="The policy or system rule involved, e.g. '30-day refund window'. "
        "null if none.",
    )
    recommended_action: str = Field(description="What the human agent should do next.")

    @model_validator(mode="after")
    def verified_needs_a_customer_id(self) -> Self:
        if self.verified and self.customer_id is None:
            raise ValueError("verified is true but customer_id is null")
        return self


@support_tool
def get_customer(
    customer_id: Annotated[str | None, Field(description="Exact customer ID, e.g. 'C001'.")] = None,
    email: Annotated[str | None, Field(description="Email address, case-insensitive.")] = None,
    name: Annotated[
        str | None, Field(description="Full name, case-insensitive, e.g. 'Maria Schmidt'.")
    ] = None,
) -> dict[str, Any]:
    """Find customer records by customer ID, email or name.

    Pass at least one field, for example name="Maria Schmidt" or
    email="maria.schmidt@example.com"; matching ignores upper/lower case, and
    giving several fields narrows the search. The result is
    {"matches": [...], "count": n}: count 0 means the search worked and nobody
    matched, so ask the customer to check the spelling or give their email.
    count above 1 means several customers share those details: ask the customer
    for their email or customer ID, and never pick one of the matches yourself.
    Use this tool to identify the customer before a refund; if the message
    contains an order number, call lookup_order first instead, even when a name
    is also mentioned.
    """
    matches = backend.get_customer(customer_id=customer_id, email=email, name=name)
    return {"matches": matches, "count": len(matches)}


@support_tool
def lookup_order(
    order_id: Annotated[
        str, Field(description="5-digit order number, with or without '#': '12345' or '#12345'.")
    ],
) -> dict[str, Any]:
    """Get one order by its order number, with status, total, delivery date and items.

    order_id is a 5-digit number with or without a leading '#', for example
    "12345" or "#12345"; any other format is rejected as invalid input. An
    unknown order returns {"found": false, ...}: that is a normal answer, not an
    error, so ask the customer to double-check the number. Use this tool whenever
    the customer's message contains an order number, even if it also mentions a
    name; use get_customer only when there is no order number. Looking up an
    order does not verify who you are talking to: before a refund, identify the
    customer with get_customer and check that the order's customer_id matches.
    If the result has details_withheld, do exactly what its text says: it
    differs for an unverified customer and for an order on another account.
    """
    order = backend.lookup_order(order_id)
    if order is None:
        return {"found": False, "order_id": order_id}
    return {"found": True, **order}


@support_tool
def process_refund(
    order_id: Annotated[
        str, Field(description="5-digit order number, with or without '#': '12345' or '#12345'.")
    ],
    customer_id: Annotated[
        str, Field(description="Customer ID verified through get_customer, e.g. 'C001'.")
    ],
    amount: Annotated[
        float, Field(description="Refund amount in USD, greater than 0, e.g. 49.99.")
    ],
) -> dict[str, Any]:
    """Refund a delivered order and return a confirmation with a refund_id.

    Prerequisite: customer_id must come from a get_customer result that matches
    the person you are talking to, and the order must belong to that customer;
    never pass a customer_id you guessed or copied from the order alone.
    order_id uses the same format as lookup_order ("12345" or "#12345"), and
    amount is in USD and greater than 0, for example 49.99; it may be less than
    the order total for a partial refund. Only delivered orders can be refunded,
    once, within 30 days of delivery, and never for more than the order total; a
    refused refund returns an error that says which rule applied. If the
    customer wants something this tool cannot do, or a rule blocks the refund and
    the customer disputes it, use escalate_to_human instead of retrying.
    """
    return backend.process_refund(order_id, customer_id, amount)


@support_tool
def escalate_to_human(summary: HandoffSummary) -> dict[str, Any]:
    """Hand the case over to a human support agent and return a ticket_id.

    Use it immediately when the customer asks for a human, even before you know
    who they are (set customer_id to null and verified to false), and do not try
    to solve the problem first. Also use it for requests that the other tools or
    the written policy do not cover, for example a competitor price match, when a
    refund is blocked and the error says a human must decide, or when the
    customer disputes a rule you have already explained. The human must be able
    to act without reading the conversation, so pass ONE argument, summary, and
    put every field inside it: issue, root_cause, order_ids, amount,
    actions_taken (including blocked attempts), policy_applied and
    recommended_action, for example "Approve or deny the 650.0 refund for order
    45678 after checking the damage report". After
    escalating, tell the customer a human will follow up and share the ticket_id.
    """
    ticket = backend.create_escalation(summary.customer_id, summary.model_dump())
    return {"ticket_id": ticket["ticket_id"]}


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
