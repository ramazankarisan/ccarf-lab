import json
import logging
from typing import Any

import pytest
from claude_agent_sdk import HookInput, PostToolUseHookInput, PreToolUseHookInput

from ex1_support_agent import backend
from ex1_support_agent.agent import build_options
from ex1_support_agent.hooks import (
    CLEAN_OUTPUT_MATCHER,
    INVALID_AMOUNT_REASON,
    LOOKUP_ORDER,
    NOT_ON_ACCOUNT_REASON,
    NOT_VERIFIED_REASON,
    OVER_LIMIT_REASON,
    REFUND_LIMIT,
    REFUND_MATCHER,
    check_refund,
    clean_tool_output,
    enforce_refund_limit,
    make_output_cleaner,
    mcp_tool_name,
    normalize_timestamps,
    to_iso,
    trim_order,
    withhold_unowned_order,
)

# Most output tests run as if customer C001, the owner of order 12345, is verified.
replace_tool_output = make_output_cleaner(lambda: "C001")


def pre_tool_use(tool_name: str, tool_input: dict[str, Any]) -> PreToolUseHookInput:
    return {
        "hook_event_name": "PreToolUse",
        "session_id": "session-test",
        "transcript_path": "/dev/null",
        "cwd": "/project",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_use_id": "toolu_1",
    }


def post_tool_use(tool_name: str, tool_response: object) -> PostToolUseHookInput:
    return {
        "hook_event_name": "PostToolUse",
        "session_id": "session-test",
        "transcript_path": "/dev/null",
        "cwd": "/project",
        "tool_name": tool_name,
        "tool_input": {},
        "tool_response": tool_response,
        "tool_use_id": "toolu_1",
    }


def lookup_order_output(order_id: str) -> str:
    """What the MCP server returns for lookup_order, as the SDK hands it to the hook."""
    order = backend.lookup_order(order_id)
    assert order is not None
    return json.dumps({"found": True, **order})


def refund(amount: object) -> dict[str, Any]:
    return {"order_id": "45678", "customer_id": "C004", "amount": amount}


# --- check_refund (pure) ------------------------------------------------------


def test_refund_limit_is_500():
    assert REFUND_LIMIT == 500


@pytest.mark.parametrize("amount", [0.01, 49.99, 499.99, 500, 500.0])
def test_amounts_up_to_the_limit_are_allowed(amount):
    assert check_refund(refund(amount)) is None


@pytest.mark.parametrize("amount", [500.01, 650, 10_000])
def test_amounts_over_the_limit_are_redirected_to_a_human(amount):
    reason = check_refund(refund(amount))

    assert reason == OVER_LIMIT_REASON
    assert "escalate_to_human" in reason  # a redirect, not a dead end


@pytest.mark.parametrize(
    "amount",
    [
        pytest.param("650", id="numeric-string"),
        pytest.param("abc", id="text"),
        pytest.param(None, id="null"),
        pytest.param(True, id="bool-true"),  # True == 1 in Python
        pytest.param(False, id="bool-false"),
        pytest.param(float("nan"), id="nan"),  # nan > 500 is False: would slip through
        pytest.param(float("inf"), id="inf"),
        pytest.param(float("-inf"), id="minus-inf"),
        pytest.param([650], id="list"),
    ],
)
def test_anything_that_is_not_a_finite_number_is_denied(amount):
    assert check_refund(refund(amount)) == INVALID_AMOUNT_REASON


def test_missing_amount_is_denied():
    assert check_refund({"order_id": "45678", "customer_id": "C004"}) == INVALID_AMOUNT_REASON


# --- Matcher ------------------------------------------------------------------
# A matcher that names no real tool never fires and raises no error. These
# tests pin it to the exact tool name the SDK uses.


def test_matcher_is_the_sdk_name_of_process_refund():
    assert REFUND_MATCHER == "mcp__support__process_refund"


def test_matcher_names_a_tool_the_agent_can_actually_call():
    assert REFUND_MATCHER in build_options(model="test-model").allowed_tools


def test_agent_registers_the_hook_for_pre_tool_use():
    hooks = build_options(model="test-model").hooks

    assert hooks is not None
    [matcher] = hooks["PreToolUse"]
    assert matcher.matcher == REFUND_MATCHER
    assert matcher.hooks == [enforce_refund_limit]


# --- enforce_refund_limit (callback) ------------------------------------------


@pytest.mark.anyio
async def test_callback_denies_with_the_reason_and_logs_it(caplog):
    with caplog.at_level(logging.WARNING, logger="ex1_support_agent.hooks"):
        output = await enforce_refund_limit(
            pre_tool_use(REFUND_MATCHER, refund(650)), "toolu_1", {"signal": None}
        )

    assert output == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": OVER_LIMIT_REASON,
        }
    }
    assert "Denied" in caplog.text
    assert "650" in caplog.text


@pytest.mark.anyio
async def test_callback_allows_a_refund_within_the_limit():
    output = await enforce_refund_limit(
        pre_tool_use(REFUND_MATCHER, refund(129.99)), "toolu_1", {"signal": None}
    )

    assert output == {}  # no opinion: the normal permission rules apply


@pytest.mark.anyio
async def test_callback_ignores_other_tools():
    output = await enforce_refund_limit(
        pre_tool_use(mcp_tool_name("lookup_order"), {"amount": 9999}), "toolu_1", {"signal": None}
    )

    assert output == {}


@pytest.mark.anyio
async def test_callback_ignores_other_events():
    event: HookInput = post_tool_use(REFUND_MATCHER, "{}")

    assert await enforce_refund_limit(event, "toolu_1", {"signal": None}) == {}


# --- normalize_timestamps -----------------------------------------------------


def test_to_iso_is_utc_with_a_z_suffix():
    assert to_iso(1_792_368_000) == "2026-10-19T00:00:00Z"


def test_order_12345_is_delivered_on_october_19_not_18():
    # Step 4 regression: the model converted this timestamp itself and said October 18.
    order = backend.lookup_order("12345")
    assert order is not None

    normalized = normalize_timestamps(order)

    assert isinstance(normalized, dict)
    assert normalized["delivered_at"] == "2026-10-19T00:00:00Z"


def test_known_keys_are_converted_at_any_depth():
    data = {
        "refunded_at": 0,
        "created_at": 86_400,
        "log": [{"event": "picked", "at": 1_792_368_000}],
    }

    assert normalize_timestamps(data) == {
        "refunded_at": "1970-01-01T00:00:00Z",
        "created_at": "1970-01-02T00:00:00Z",
        "log": [{"event": "picked", "at": "2026-10-19T00:00:00Z"}],
    }


def test_key_names_decide_not_the_size_of_the_number():
    data = {"order_total_cents": 1_792_368_000, "carrier_ref": 1_792_368_000}

    assert normalize_timestamps(data) == data


@pytest.mark.parametrize("value", [None, "2026-10-19", True, 1.5])
def test_non_int_values_under_timestamp_keys_are_left_alone(value):
    assert normalize_timestamps({"delivered_at": value}) == {"delivered_at": value}


def test_normalize_does_not_change_its_input():
    data = {"delivered_at": 0, "items": [{"at": 0}]}

    normalize_timestamps(data)

    assert data == {"delivered_at": 0, "items": [{"at": 0}]}


# --- trim_order ---------------------------------------------------------------


def test_trim_keeps_only_the_support_fields():
    order = json.loads(lookup_order_output("12345"))

    assert trim_order(order) == {
        "found": True,
        "order_id": "12345",
        "customer_id": "C001",
        "status": "delivered",
        "total": 129.99,
        "delivered_at": order["delivered_at"],
        "items": [
            {"name": "Wireless Headphones", "qty": 1, "unit_price": 99.99},
            {"name": "USB-C Cable", "qty": 2, "unit_price": 15.0},
        ],
    }


def test_trim_drops_fields_it_does_not_know():
    # Allow-list: a field added to the backend later never leaks by default.
    trimmed = trim_order({"order_id": "1", "new_internal_field": "secret", "items": []})

    assert trimmed == {"order_id": "1", "items": []}


def test_trim_keeps_a_not_found_answer_as_it_is():
    assert trim_order({"found": False, "order_id": "99999"}) == {
        "found": False,
        "order_id": "99999",
    }


# --- clean_tool_output --------------------------------------------------------


def test_lookup_order_output_is_trimmed_and_normalized():
    cleaned = clean_tool_output(LOOKUP_ORDER, lookup_order_output("12345"))

    assert cleaned is not None
    data = json.loads(cleaned)
    assert data["delivered_at"] == "2026-10-19T00:00:00Z"
    for noise in ("warehouse_code", "carrier_internal_ref", "sku_audit_log"):
        assert noise not in data
    assert all("sku" not in item for item in data["items"])


def test_refund_output_is_normalized_but_not_trimmed():
    response = json.dumps({"refund_id": "RF-0001", "status": "refunded", "refunded_at": 0})

    cleaned = clean_tool_output(REFUND_MATCHER, response)

    assert cleaned is not None
    assert json.loads(cleaned) == {
        "refund_id": "RF-0001",
        "status": "refunded",
        "refunded_at": "1970-01-01T00:00:00Z",
    }


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(
            json.dumps({"errorCategory": "transient", "isRetryable": True, "at": 0}),
            id="structured-error",
        ),
        pytest.param("Input validation error: 'customer_id' is required", id="plain-text"),
        pytest.param(json.dumps([1, 2]), id="json-list"),
        pytest.param(None, id="null"),
        pytest.param([{"type": "text", "text": "{}"}], id="content-blocks"),
    ],
)
def test_anything_but_a_successful_json_object_passes_through(response):
    assert clean_tool_output(LOOKUP_ORDER, response) is None


# --- PostToolUse wiring and callback ------------------------------------------


def test_post_tool_use_matcher_covers_exactly_lookup_order_and_process_refund():
    tools = CLEAN_OUTPUT_MATCHER.split("|")

    assert tools == ["mcp__support__lookup_order", "mcp__support__process_refund"]
    assert set(tools) <= set(build_options(model="test-model").allowed_tools)


def test_agent_registers_the_post_tool_use_hook():
    hooks = build_options(model="test-model").hooks

    assert hooks is not None
    cleaning = hooks["PostToolUse"][0]
    assert cleaning.matcher == CLEAN_OUTPUT_MATCHER
    assert [hook.__name__ for hook in cleaning.hooks] == ["replace_tool_output"]


@pytest.mark.anyio
async def test_callback_replaces_the_output_instead_of_adding_context():
    raw = lookup_order_output("12345")

    output = await replace_tool_output(
        post_tool_use(LOOKUP_ORDER, raw), "toolu_1", {"signal": None}
    )

    # Only updatedToolOutput: no additionalContext, so the raw data is gone.
    assert output == {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "updatedToolOutput": clean_tool_output(LOOKUP_ORDER, raw),
        }
    }


@pytest.mark.anyio
async def test_callback_leaves_an_error_result_untouched():
    error = json.dumps({"errorCategory": "transient", "isRetryable": True})

    output = await replace_tool_output(
        post_tool_use(LOOKUP_ORDER, error), "toolu_1", {"signal": None}
    )

    assert output == {}


@pytest.mark.anyio
async def test_post_callback_ignores_other_events():
    event: HookInput = pre_tool_use(LOOKUP_ORDER, {"order_id": "12345"})

    assert await replace_tool_output(event, "toolu_1", {"signal": None}) == {}


# --- Authorization: withhold_unowned_order ------------------------------------
# Identity is not authorization: a verified customer sees only their own orders.

ORDER_12345: dict[str, Any] = {
    "found": True,
    "order_id": "12345",
    "customer_id": "C001",
    "status": "delivered",
    "total": 129.99,
}


def withheld_12345(reason: str) -> dict[str, Any]:
    return {"found": True, "order_id": "12345", "details_withheld": reason}


def test_before_verification_the_model_is_told_to_verify_then_retry():
    assert withhold_unowned_order(ORDER_12345, None) == withheld_12345(NOT_VERIFIED_REASON)
    assert "get_customer" in NOT_VERIFIED_REASON
    assert "call lookup_order again" in NOT_VERIFIED_REASON


def test_the_verified_owner_gets_the_full_order():
    assert withhold_unowned_order(ORDER_12345, "C001") == ORDER_12345


def test_another_verified_customer_is_told_not_to_retry_or_escalate():
    # C002 is the other Maria Schmidt: verified, but not for order 12345. A single
    # "verify first" reason made the model retry and escalate a fake system bug.
    assert withhold_unowned_order(ORDER_12345, "C002") == withheld_12345(NOT_ON_ACCOUNT_REASON)
    assert "Do not retry and do not escalate" in NOT_ON_ACCOUNT_REASON
    assert "couldn't find it on their account" in NOT_ON_ACCOUNT_REASON


def test_the_two_reasons_are_different_instructions():
    assert NOT_VERIFIED_REASON != NOT_ON_ACCOUNT_REASON
    assert "lookup_order again" not in NOT_ON_ACCOUNT_REASON


def test_a_not_found_answer_passes_through():
    not_found = {"found": False, "order_id": "99999"}

    assert withhold_unowned_order(not_found, None) == not_found


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("verified", "sees_details"),
    [
        pytest.param(None, False, id="nobody-verified"),
        pytest.param("C001", True, id="owner-verified"),
        pytest.param("C002", False, id="other-customer-verified"),
    ],
)
async def test_the_model_sees_order_details_only_for_its_verified_owner(verified, sees_details):
    cleaner = make_output_cleaner(lambda: verified)

    output = await cleaner(
        post_tool_use(LOOKUP_ORDER, lookup_order_output("12345")), "toolu_1", {"signal": None}
    )

    seen = json.loads(output["hookSpecificOutput"]["updatedToolOutput"])  # type: ignore[typeddict-item]
    assert ("total" in seen) is sees_details
    assert ("details_withheld" in seen) is not sees_details
    assert seen["order_id"] == "12345"


@pytest.mark.anyio
async def test_the_cleaner_asks_for_the_verified_customer_on_every_call():
    verified: list[str | None] = [None]
    cleaner = make_output_cleaner(lambda: verified[0])
    event = post_tool_use(LOOKUP_ORDER, lookup_order_output("12345"))

    before = await cleaner(event, "toolu_1", {"signal": None})
    verified[0] = "C001"  # the customer verifies; the model calls lookup_order again
    after = await cleaner(event, "toolu_2", {"signal": None})

    assert "details_withheld" in str(before)
    assert "details_withheld" not in str(after)


@pytest.mark.anyio
async def test_refund_output_is_never_withheld():
    cleaner = make_output_cleaner(lambda: None)
    response = json.dumps({"refund_id": "RF-0001", "order_id": "12345", "amount": 10.0})

    output = await cleaner(post_tool_use(REFUND_MATCHER, response), "toolu_1", {"signal": None})

    assert "details_withheld" not in str(output)
