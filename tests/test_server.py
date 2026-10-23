import json
from typing import Any

import pytest
from mcp import Client
from mcp.types import TextContent

from ex1_support_agent import backend
from ex1_support_agent.server import NOT_ON_ACCOUNT_MESSAGE, error_result, mcp

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
    summary = {
        "customer_id": None,
        "reason": "Customer asked for a human.",
        "root_cause": "Not investigated; escalated on request.",
        "recommended_action": "Call the customer back.",
    }

    data = await call("escalate_to_human", {"summary": summary})

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
    # The $500 limit is enforced by the Step 5 hook, not by guidance text.
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
# our 4-field JSON. These tests document that, so the agent loop (Step 4) knows
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
