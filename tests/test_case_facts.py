import json
from typing import Any

import pytest
from claude_agent_sdk import HookInput, PostToolUseHookInput

from ex1_support_agent import backend
from ex1_support_agent.agent import ALL_TOOLS_MATCHER, build_options
from ex1_support_agent.case_facts import (
    CaseFacts,
    OrderFacts,
    RefundFacts,
    make_fact_recorder,
    record,
    render,
)
from ex1_support_agent.hooks import mcp_tool_name

C004 = {"customer_id": "C004", "name": "Aiko Tanaka", "email": "aiko.tanaka@example.com"}
ORDER_12345 = {
    "found": True,
    "order_id": "12345",
    "customer_id": "C001",
    "status": "delivered",
    "total": 129.99,
    "delivered_at": "2026-10-19T00:00:00Z",
    "items": [],
}


def post_tool_use(tool: str, tool_input: dict[str, Any], response: object) -> PostToolUseHookInput:
    return {
        "hook_event_name": "PostToolUse",
        "session_id": "session-test",
        "transcript_path": "/dev/null",
        "cwd": "/project",
        "tool_name": mcp_tool_name(tool),
        "tool_input": tool_input,
        "tool_response": response,
        "tool_use_id": "toolu_1",
    }


# --- record: verification -----------------------------------------------------


@pytest.mark.parametrize(
    "tool_input",
    [
        pytest.param({"email": "aiko.tanaka@example.com"}, id="by-email"),
        pytest.param({"customer_id": "C004"}, id="by-id"),
        pytest.param({"name": "Aiko Tanaka", "email": "aiko.tanaka@example.com"}, id="name+email"),
    ],
)
def test_one_match_by_email_or_id_verifies_the_customer(tool_input):
    facts = CaseFacts()

    record(facts, "get_customer", tool_input, {"matches": [C004], "count": 1})

    assert facts.verified_customer_id == "C004"


def test_a_name_alone_never_verifies_even_with_one_match():
    facts = CaseFacts()

    record(facts, "get_customer", {"name": "Aiko Tanaka"}, {"matches": [C004], "count": 1})

    assert facts.verified_customer_id is None


@pytest.mark.parametrize("count", [0, 2])
def test_zero_or_several_matches_do_not_verify(count):
    facts = CaseFacts()
    result = {"matches": [C004] * count, "count": count}

    record(facts, "get_customer", {"email": "maria@example.com"}, result)

    assert facts.verified_customer_id is None


# --- record: orders, refunds, tickets -----------------------------------------


def test_lookup_order_records_the_exact_order_facts():
    facts = CaseFacts(verified_customer_id="C001")

    record(facts, "lookup_order", {"order_id": "#12345"}, ORDER_12345)

    assert facts.orders == {
        "12345": OrderFacts(
            customer_id="C001",
            status="delivered",
            total=129.99,
            delivered_at="2026-10-19T00:00:00Z",
        )
    }


@pytest.mark.parametrize("verified", [None, "C002"])
def test_orders_are_recorded_whoever_is_verified(verified):
    # Recording is the full truth; render() is the gate for what the model sees.
    facts = CaseFacts(verified_customer_id=verified)

    record(facts, "lookup_order", {"order_id": "12345"}, ORDER_12345)

    assert facts.orders["12345"].customer_id == "C001"


def test_an_unknown_order_records_nothing():
    facts = CaseFacts()

    record(facts, "lookup_order", {"order_id": "99999"}, {"found": False, "order_id": "99999"})

    assert facts.orders == {}


def test_a_refund_is_recorded_and_updates_the_order_status():
    facts = CaseFacts(verified_customer_id="C001")
    record(facts, "lookup_order", {}, ORDER_12345)
    refund = {
        "refund_id": "RF-0001",
        "order_id": "12345",
        "customer_id": "C001",
        "amount": 49.99,
        "status": "refunded",
        "refunded_at": "2026-10-22T00:00:00Z",
    }

    record(facts, "process_refund", {}, refund)

    assert facts.refunds == [
        RefundFacts(customer_id="C001", refund_id="RF-0001", order_id="12345", amount=49.99)
    ]
    assert facts.orders["12345"].status == "refunded"


def test_a_refund_for_an_order_never_looked_up_is_still_recorded():
    facts = CaseFacts()
    refund = {
        "refund_id": "RF-0001",
        "order_id": "12345",
        "customer_id": "C001",
        "amount": 10.0,
        "status": "refunded",
    }

    record(facts, "process_refund", {}, refund)

    assert len(facts.refunds) == 1
    assert facts.orders == {}


def test_escalation_records_the_ticket():
    facts = CaseFacts()

    record(facts, "escalate_to_human", {}, {"ticket_id": "ESC-0001"})

    assert facts.escalation_ticket == "ESC-0001"


def test_an_unknown_tool_changes_nothing():
    facts = CaseFacts()

    record(facts, "something_else", {}, {"ticket_id": "ESC-9"})

    assert facts == CaseFacts()


# --- render -------------------------------------------------------------------


def test_render_an_empty_case():
    assert render(CaseFacts()).splitlines() == [
        "<case_facts>",
        "Recorded by the system from tool results. These values are exact: use them "
        "instead of your memory of the conversation.",
        "verified_customer_id: none (not verified)",
        "orders:",
        "  (none)",
        "refunds:",
        "  (none)",
        "escalation_ticket: none",
        "</case_facts>",
    ]


def test_render_keeps_exact_values_in_a_stable_order():
    facts = CaseFacts(
        verified_customer_id="C002",
        orders={
            "56789": OrderFacts(
                customer_id="C002", status="refunded", total=15.0, delivered_at=None
            ),
            "23456": OrderFacts(
                customer_id="C002", status="shipped", total=249.0, delivered_at=None
            ),
            "12345": OrderFacts(
                customer_id="C002",
                status="delivered",
                total=129.99,
                delivered_at="2026-10-19T00:00:00Z",
            ),
        },
        refunds=[
            RefundFacts(customer_id="C002", refund_id="RF-0001", order_id="12345", amount=129.99)
        ],
        escalation_ticket="ESC-0001",
    )

    assert render(facts).splitlines()[2:] == [
        "verified_customer_id: C002",
        "orders:",
        "  - 12345: status=delivered total=129.99 delivered_at=2026-10-19T00:00:00Z",
        "  - 23456: status=shipped total=249.00 delivered_at=none",
        "  - 56789: status=refunded total=15.00 delivered_at=none",
        "refunds:",
        "  - RF-0001: order=12345 amount=129.99",
        "escalation_ticket: ESC-0001",
        "</case_facts>",
    ]


def test_render_is_deterministic():
    facts = CaseFacts(
        verified_customer_id="C001",
        orders={"12345": OrderFacts(customer_id="C001", status="x", total=1.5, delivered_at=None)},
    )

    assert render(facts) == render(facts.model_copy(deep=True))


# --- render: redaction --------------------------------------------------------
# <case_facts> goes into the context every turn, so it is a path for data just
# like tool output. It must withhold what the output cleaner withholds.

FACTS_12345 = OrderFacts(
    customer_id="C001", status="delivered", total=129.99, delivered_at="2026-10-19T00:00:00Z"
)
FULL_LINE = "  - 12345: status=delivered total=129.99 delivered_at=2026-10-19T00:00:00Z"
NOT_VERIFIED_LINE = "  - 12345: details withheld (customer not verified)"
NOT_ON_ACCOUNT_LINE = "  - 12345: details withheld (not on the verified customer's account)"


@pytest.mark.parametrize(
    ("verified", "line"),
    [
        pytest.param(None, NOT_VERIFIED_LINE, id="unverified"),
        pytest.param("C001", FULL_LINE, id="owner-C001"),
        pytest.param("C002", NOT_ON_ACCOUNT_LINE, id="other-customer-C002"),
    ],
)
def test_render_shows_order_details_only_to_the_verified_owner(verified, line):
    facts = CaseFacts(verified_customer_id=verified, orders={"12345": FACTS_12345})

    rendered = render(facts)

    assert line in rendered.splitlines()
    if line != FULL_LINE:
        assert "129.99" not in rendered
        assert "2026-10-19" not in rendered


def test_render_withholds_again_when_the_verified_customer_changes():
    # Verified as C001, order recorded, then re-verified as C002: the old facts stay
    # in the object, but render() decides by the verification as it is NOW.
    facts = CaseFacts(verified_customer_id="C001", orders={"12345": FACTS_12345})
    assert FULL_LINE in render(facts)

    facts.verified_customer_id = "C002"

    assert NOT_ON_ACCOUNT_LINE in render(facts)


def test_render_withholds_refunds_of_another_customer():
    refund = RefundFacts(customer_id="C001", refund_id="RF-0001", order_id="12345", amount=129.99)
    facts = CaseFacts(verified_customer_id="C002", refunds=[refund])

    rendered = render(facts)

    assert "  - order 12345: details withheld (not on the verified customer's account)" in rendered
    assert "RF-0001" not in rendered
    assert "129.99" not in rendered


# --- The recorder hook --------------------------------------------------------


def lookup_order_response(order_id: str) -> str:
    order = backend.lookup_order(order_id)
    assert order is not None
    return json.dumps({"found": True, **order})


@pytest.mark.anyio
async def test_recorder_reads_the_cleaned_output_not_the_raw_one():
    facts = CaseFacts(verified_customer_id="C001")
    recorder = make_fact_recorder(facts)

    output = await recorder(
        post_tool_use("lookup_order", {"order_id": "12345"}, lookup_order_response("12345")),
        "toolu_1",
        {"signal": None},
    )

    assert output == {}  # it only observes; the output cleaner changes the output
    assert facts.orders["12345"].delivered_at == "2026-10-19T00:00:00Z"  # ISO, not 1792368000


@pytest.mark.anyio
async def test_recorder_ignores_error_results():
    facts = CaseFacts()
    error = json.dumps({"errorCategory": "transient", "isRetryable": True})

    await make_fact_recorder(facts)(
        post_tool_use("lookup_order", {"order_id": "12345"}, error), "toolu_1", {"signal": None}
    )

    assert facts == CaseFacts()


@pytest.mark.anyio
async def test_recorder_ignores_other_events():
    facts = CaseFacts()
    event: HookInput = {
        "hook_event_name": "PreToolUse",
        "session_id": "session-test",
        "transcript_path": "/dev/null",
        "cwd": "/project",
        "tool_name": mcp_tool_name("escalate_to_human"),
        "tool_input": {},
        "tool_use_id": "toolu_1",
    }

    assert await make_fact_recorder(facts)(event, "toolu_1", {"signal": None}) == {}
    assert facts == CaseFacts()


@pytest.mark.anyio
async def test_each_recorder_writes_only_to_its_own_facts():
    first, second = CaseFacts(), CaseFacts()
    response = json.dumps({"ticket_id": "ESC-0001"})

    await make_fact_recorder(first)(
        post_tool_use("escalate_to_human", {}, response), "toolu_1", {"signal": None}
    )

    assert first.escalation_ticket == "ESC-0001"
    assert second == CaseFacts()


# --- Wiring -------------------------------------------------------------------


def test_options_register_the_cleaner_then_the_recorder():
    hooks = build_options(model="test-model", facts=CaseFacts()).hooks

    assert hooks is not None
    cleaning, recording = hooks["PostToolUse"]
    assert [hook.__name__ for hook in cleaning.hooks] == ["replace_tool_output"]
    assert [hook.__name__ for hook in recording.hooks] == ["record_case_facts"]
    assert recording.matcher == ALL_TOOLS_MATCHER
    assert sorted(ALL_TOOLS_MATCHER.split("|")) == sorted(
        build_options(model="test-model").allowed_tools
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("email", "sees_details"),
    [
        pytest.param("maria.schmidt@example.com", True, id="C001-owner"),
        pytest.param("m.schmidt@example.org", False, id="C002-other-maria"),
    ],
)
async def test_verification_recorded_by_one_hook_unlocks_details_in_the_other(email, sees_details):
    # Both hooks share this session's CaseFacts through build_options.
    facts = CaseFacts()
    hooks = build_options(model="test-model", facts=facts).hooks
    assert hooks is not None
    cleaner = hooks["PostToolUse"][0].hooks[0]
    recorder = hooks["PostToolUse"][1].hooks[0]
    customers = backend.get_customer(email=email)
    response = json.dumps({"matches": customers, "count": len(customers)})

    await recorder(
        post_tool_use("get_customer", {"email": email}, response), "t1", {"signal": None}
    )
    output = await cleaner(
        post_tool_use("lookup_order", {"order_id": "12345"}, lookup_order_response("12345")),
        "t2",
        {"signal": None},
    )

    assert ("details_withheld" not in str(output)) is sees_details
