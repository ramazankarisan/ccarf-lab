import json
from typing import Any

import pytest
from mcp import Client
from mcp.types import TextContent

from ex1_support_agent import backend
from ex1_support_agent.server import (
    NOT_ON_ACCOUNT_MESSAGE,
    HandoffSummary,
    error_result,
    fail_next_from_env,
    mcp,
)

FULL_HANDOFF: dict[str, Any] = {
    "customer_id": "C004",
    "verified": True,
    "order_ids": ["45678"],
    "issue": "Wants a full refund for a monitor that arrived broken.",
    "root_cause": "Item damaged in delivery; refund blocked by the system rule for large amounts.",
    "amount": 650.0,
    "actions_taken": [
        "Verified the customer by email",
        "Looked up order 45678: delivered 5 days ago, total 650.0",
        "process_refund for 650.0 blocked: needs a human",
    ],
    "policy_applied": "Refunds above the automatic limit need a human",
    "recommended_action": "Approve or deny the 650.0 refund after checking the damage report.",
}

# Client(server) connects in-process: no subprocess, no network.
# (mcp 2.x replacement for v1's create_connected_server_and_client_session.)


async def call(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a tool, assert it succeeded, and return its JSON result."""
    async with Client(mcp) as client:
        result = await client.call_tool(tool, arguments)
    assert not result.is_error, result.content
    block = result.content[0]
    assert isinstance(block, TextContent)
    data: dict[str, Any] = json.loads(block.text)
    return data


async def call_error(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a tool, assert it failed, and return its structured error."""
    async with Client(mcp) as client:
        result = await client.call_tool(tool, arguments)
    assert result.is_error
    block = result.content[0]
    assert isinstance(block, TextContent)
    data: dict[str, Any] = json.loads(block.text)
    return data


# --- Skeleton -----------------------------------------------------------------


@pytest.mark.anyio
async def test_server_identifies_as_support_backend():
    async with Client(mcp) as client:
        assert client.server_info is not None
        assert client.server_info.name == "support-backend"


@pytest.mark.anyio
async def test_server_exposes_exactly_the_four_tools():
    async with Client(mcp) as client:
        tools = await client.list_tools()

    assert {t.name for t in tools.tools} == {
        "get_customer",
        "lookup_order",
        "process_refund",
        "escalate_to_human",
    }


# --- Success results ----------------------------------------------------------


@pytest.mark.anyio
async def test_get_customer_returns_matches_and_count():
    data = await call("get_customer", {"name": "Maria Schmidt"})

    assert data["count"] == 2
    assert {c["customer_id"] for c in data["matches"]} == {"C001", "C002"}


@pytest.mark.anyio
async def test_get_customer_with_no_match_is_a_success_with_count_zero():
    data = await call("get_customer", {"name": "Nobody Here"})

    assert data == {"matches": [], "count": 0}


@pytest.mark.anyio
async def test_lookup_order_returns_the_order():
    data = await call("lookup_order", {"order_id": "#12345"})

    assert data["found"] is True
    assert data["order_id"] == "12345"
    assert data["customer_id"] == "C001"


@pytest.mark.anyio
async def test_lookup_order_not_found_is_a_success():
    data = await call("lookup_order", {"order_id": "99999"})

    assert data == {"found": False, "order_id": "99999"}


@pytest.mark.anyio
async def test_process_refund_returns_the_confirmation():
    data = await call("process_refund", {"order_id": "12345", "customer_id": "C001", "amount": 50})

    assert data["refund_id"] == "RF-0001"
    assert data["status"] == "refunded"


@pytest.mark.anyio
async def test_escalate_to_human_returns_a_ticket_id():
    data = await call("escalate_to_human", {"summary": FULL_HANDOFF})

    assert data == {"ticket_id": "ESC-0001"}


# --- Descriptions (2.1) -------------------------------------------------------
# The description is how the model picks a tool, so its key rules are tested
# like code: removing one by accident fails the build.


async def tool_descriptions() -> dict[str, str]:
    async with Client(mcp) as client:
        tools = await client.list_tools()
    return {t.name: t.description or "" for t in tools.tools}


@pytest.mark.anyio
async def test_every_description_has_at_least_four_sentences_and_an_example():
    for name, description in (await tool_descriptions()).items():
        sentences = [s for s in description.replace("\n", " ").split(". ") if s.strip()]
        assert len(sentences) >= 4, name
        assert "for example" in description, name


@pytest.mark.anyio
async def test_descriptions_are_sent_without_source_indentation():
    for name, description in (await tool_descriptions()).items():
        assert "\n " not in description, name


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool", "required_phrases"),
    [
        # lookup_order vs get_customer: an order number decides.
        ("lookup_order", ["contains an order number", "get_customer"]),
        ("get_customer", ["call lookup_order first", "never pick one"]),
        # process_refund vs escalate_to_human: verified customer, then escalate.
        ("process_refund", ["Prerequisite", "get_customer", "escalate_to_human"]),
        ("escalate_to_human", ["asks for a human", "customer_id to null", "price match"]),
    ],
)
async def test_descriptions_state_the_routing_rules(tool, required_phrases):
    description = (await tool_descriptions())[tool]

    for phrase in required_phrases:
        assert phrase in description, f"{tool}: missing {phrase!r}"


@pytest.mark.anyio
async def test_no_description_mentions_the_refund_limit():
    # The $500 limit is enforced by the PreToolUse hook, not by guidance text.
    for name, description in (await tool_descriptions()).items():
        assert "500" not in description, name


# --- Structured errors (2.2) --------------------------------------------------


@pytest.mark.anyio
async def test_timeout_is_transient_and_retryable():
    backend.fail_next(1)

    error = await call_error("lookup_order", {"order_id": "12345"})

    assert error["errorCategory"] == "transient"
    assert error["isRetryable"] is True
    assert error["customerMessage"] is None


@pytest.mark.anyio
async def test_server_does_not_retry_on_its_own():
    # One pending failure: if the server retried internally, the first call
    # would succeed. The agent owns retries, based on isRetryable.
    backend.fail_next(1)

    await call_error("lookup_order", {"order_id": "12345"})
    data = await call("lookup_order", {"order_id": "12345"})

    assert data["found"] is True


@pytest.mark.anyio
async def test_support_fail_next_env_makes_the_first_calls_time_out():
    # The live suite sets this in the stdio server's env: its backend is in that process.
    fail_next_from_env({"SUPPORT_FAIL_NEXT": "1"})

    error = await call_error("lookup_order", {"order_id": "12345"})
    data = await call("lookup_order", {"order_id": "12345"})

    assert error["errorCategory"] == "transient"
    assert data["found"] is True


@pytest.mark.anyio
async def test_without_support_fail_next_nothing_fails():
    fail_next_from_env({})

    assert (await call("lookup_order", {"order_id": "12345"}))["found"] is True


@pytest.mark.anyio
async def test_malformed_order_id_is_a_validation_error_for_the_agent_only():
    error = await call_error("lookup_order", {"order_id": "12-345"})

    assert error["errorCategory"] == "validation"
    assert error["isRetryable"] is False
    assert "5 digits" in error["description"]
    assert error["customerMessage"] is None


@pytest.mark.anyio
async def test_policy_violation_explains_the_rule_to_the_customer():
    error = await call_error(
        "process_refund", {"order_id": "23456", "customer_id": "C002", "amount": 10}
    )

    assert error["errorCategory"] == "business"
    assert error["isRetryable"] is False
    assert "not delivered" in error["customerMessage"]
    assert "escalate" in error["description"]


@pytest.mark.anyio
@pytest.mark.parametrize("order_id", ["12345", "99999"])
async def test_wrong_or_unknown_order_gives_the_same_permission_error(order_id):
    # Someone else's order and a mistyped number look identical to the caller.
    error = await call_error(
        "process_refund", {"order_id": order_id, "customer_id": "C002", "amount": 10}
    )

    assert error["errorCategory"] == "permission"
    assert error["isRetryable"] is False
    assert error["customerMessage"] == NOT_ON_ACCOUNT_MESSAGE


@pytest.mark.anyio
async def test_unexpected_error_is_internal_and_leaks_nothing(monkeypatch):
    def broken_lookup(order_id: str) -> None:
        raise RuntimeError("db password=hunter2 at /srv/db.py line 42")

    monkeypatch.setattr(backend, "lookup_order", broken_lookup)

    error = await call_error("lookup_order", {"order_id": "12345"})

    assert error["errorCategory"] == "internal"
    assert error["isRetryable"] is False
    assert "hunter2" not in json.dumps(error)
    assert "Traceback" not in json.dumps(error)


@pytest.mark.parametrize(
    ("exc", "category", "retryable"),
    [
        (backend.BackendTimeout("t"), "transient", True),
        (backend.InvalidInput("i"), "validation", False),
        (backend.PolicyViolation("p"), "business", False),
        (backend.NotAuthorized("n"), "permission", False),
        (KeyError("k"), "internal", False),
    ],
)
def test_error_result_has_the_same_four_fields_for_every_category(exc, category, retryable):
    result = error_result(exc)

    assert result.is_error
    block = result.content[0]
    assert isinstance(block, TextContent)
    payload = json.loads(block.text)
    assert set(payload) == {"errorCategory", "isRetryable", "description", "customerMessage"}
    assert payload["errorCategory"] == category
    assert payload["isRetryable"] is retryable


# --- Schema validation: handled by the SDK, not by error_result ----------------
# Arguments that don't match the input schema are rejected by the framework
# before our wrapper runs. The agent then gets the SDK's plain-text error, not
# our 4-field JSON. These tests document that, so the agent loop knows
# it must handle both shapes.


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("arguments", "bad_field"),
    [
        ({"order_id": "12345", "amount": 10}, "customer_id"),  # missing field
        ({"order_id": "12345", "customer_id": "C001", "amount": "abc"}, "amount"),  # wrong type
    ],
)
async def test_schema_errors_are_plain_text_from_the_sdk(arguments, bad_field):
    async with Client(mcp) as client:
        result = await client.call_tool("process_refund", arguments)

    assert result.is_error
    block = result.content[0]
    assert isinstance(block, TextContent)
    assert bad_field in block.text
    assert "errorCategory" not in block.text  # not our structured format


# --- Handoff summary ----------------------------------------------------------
# A human must be able to act on the ticket without reading the transcript.


@pytest.mark.anyio
async def test_escalation_stores_every_handoff_field(monkeypatch):
    stored: dict[str, Any] = {}

    def fake_create_escalation(customer_id: str | None, summary: dict[str, Any]) -> dict[str, Any]:
        stored.update(customer_id=customer_id, summary=summary)
        return {"ticket_id": "ESC-0042"}

    monkeypatch.setattr(backend, "create_escalation", fake_create_escalation)

    data = await call("escalate_to_human", {"summary": FULL_HANDOFF})

    assert data == {"ticket_id": "ESC-0042"}
    assert stored == {"customer_id": "C004", "summary": FULL_HANDOFF}


def test_handoff_has_the_fields_a_human_needs():
    assert list(HandoffSummary.model_fields) == [
        "customer_id",
        "verified",
        "order_ids",
        "issue",
        "root_cause",
        "amount",
        "actions_taken",
        "policy_applied",
        "recommended_action",
    ]


def test_an_unidentified_customer_can_be_escalated_with_minimal_facts():
    summary = HandoffSummary(
        customer_id=None,
        verified=False,
        order_ids=[],
        issue="Asked for a human.",
        root_cause="Not investigated; escalated on request.",
        actions_taken=[],
        recommended_action="Call the customer back and verify them.",
    )

    assert summary.amount is None
    assert summary.policy_applied is None


@pytest.mark.parametrize(
    ("change", "bad_field"),
    [
        pytest.param({"verified": True, "customer_id": None}, "customer_id", id="verified-no-id"),
        pytest.param({"order_ids": ["#45678"]}, "order_ids", id="order-id-with-hash"),
        pytest.param({"order_ids": ["4567"]}, "order_ids", id="order-id-4-digits"),
        pytest.param({"amount": float("nan")}, "amount", id="amount-nan"),
    ],
)
def test_handoff_rejects_inconsistent_or_inexact_facts(change, bad_field):
    with pytest.raises(ValueError, match=bad_field):
        HandoffSummary(**(FULL_HANDOFF | change))


@pytest.mark.parametrize("missing", ["verified", "order_ids", "actions_taken", "issue"])
def test_handoff_requires_the_key_fields(missing):
    summary = {key: value for key, value in FULL_HANDOFF.items() if key != missing}

    with pytest.raises(ValueError, match=missing):
        HandoffSummary(**summary)


@pytest.mark.anyio
async def test_escalate_description_asks_for_every_field():
    description = (await tool_descriptions())["escalate_to_human"]

    for field in ("issue", "order_ids", "actions_taken", "policy_applied", "verified"):
        assert field in description, field
    # Without this the model also repeated every field at the top level, as strings.
    assert "ONE argument, summary" in description


def test_handoff_rejects_invented_fields():
    # The model once sent "order_ids_note"; by default Pydantic would drop it silently.
    with pytest.raises(ValueError, match="order_ids_note"):
        HandoffSummary(**(FULL_HANDOFF | {"order_ids_note": "45678"}))


@pytest.mark.anyio
async def test_handoff_schema_tells_the_model_no_other_fields_are_allowed():
    async with Client(mcp) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}

    schema = tools["escalate_to_human"].input_schema
    assert schema["$defs"]["HandoffSummary"]["additionalProperties"] is False


@pytest.mark.anyio
async def test_invented_field_comes_back_as_an_error_the_model_can_fix():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "escalate_to_human", {"summary": FULL_HANDOFF | {"order_ids_note": "45678"}}
        )

    assert result.is_error
    block = result.content[0]
    assert isinstance(block, TextContent)
    assert "order_ids_note" in block.text
