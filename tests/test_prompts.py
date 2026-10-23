import re

import pytest

from ex1_support_agent.agent import build_options
from ex1_support_agent.backend import SEED_ORDERS
from ex1_support_agent.prompts import (
    DO_NOT_ESCALATE,
    ESCALATE_WHEN,
    EXAMPLES,
    ROLE,
    SECTIONS,
    SYSTEM_PROMPT,
    VERIFICATION,
)

# The prompt is guidance, not code, but its key rules are tested like code:
# deleting one by accident fails the build.


def test_prompt_is_the_five_sections_in_order():
    assert SECTIONS == (ROLE, VERIFICATION, ESCALATE_WHEN, DO_NOT_ESCALATE, EXAMPLES)
    assert "\n\n".join(SECTIONS) == SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("section", "tag"),
    [
        (ROLE, "role"),
        (VERIFICATION, "verification_and_disclosure"),
        (ESCALATE_WHEN, "escalate_when"),
        (DO_NOT_ESCALATE, "do_not_escalate"),
        (EXAMPLES, "examples"),
    ],
)
def test_each_section_is_wrapped_in_its_own_tag(section, tag):
    assert section.startswith(f"<{tag}>")
    assert section.endswith(f"</{tag}>")


def test_role_is_short():
    body = ROLE.removeprefix("<role>").removesuffix("</role>")
    assert body.count(". ") + 1 <= 2


@pytest.mark.parametrize(
    "phrase",
    [
        "exactly one match",
        "A name alone is never enough",
        "share no order details",
        "only confirm that an order number exists",
        "Never show internal IDs",
        "ticket_id from escalate_to_human is fine",
        "never in your reply",  # no internal notes to the customer
        "Do not promise what they will decide",
        "Never promise a refund before process_refund has succeeded",
        "Refund the amount the customer asks for",
        "ask what is wrong with the order",
        "Never mention or describe it to the customer",
        "ask before calling process_refund",
        "never offer a follow-up refund on the same order",
    ],
)
def test_verification_and_disclosure_rules(phrase):
    assert phrase in VERIFICATION


@pytest.mark.parametrize(
    "phrase",
    [
        "asks for a human",
        "even before verification",
        "price match",
        "refund is blocked by the system",
        "disputes a business rule",
        "transient",
        "internal error",
    ],
)
def test_escalation_criteria(phrase):
    assert phrase in ESCALATE_WHEN


@pytest.mark.parametrize(
    "phrase",
    ["Frustration or anger alone", "Several customers matching", "not explained"],
)
def test_do_not_escalate_criteria(phrase):
    assert phrase in DO_NOT_ESCALATE


def test_two_to_four_examples_each_ending_with_a_decision_and_reason():
    examples = re.findall(r"<example>\n(.*?)\n</example>", EXAMPLES, flags=re.DOTALL)

    assert 2 <= len(examples) <= 4
    for example in examples:
        last_line = example.splitlines()[-1]
        assert last_line.startswith("Decision:"), example
        assert ". " in last_line, f"decision without a reason: {last_line}"


@pytest.mark.parametrize(
    "case",
    ["Third time", "get me a person", "52 days ago", "cheaper at another store"],
)
def test_examples_cover_the_ambiguous_cases(case):
    assert case in EXAMPLES


def test_prompt_never_states_the_refund_limit():
    # The PreToolUse hook owns the limit; a second copy here could drift.
    assert "500" not in SYSTEM_PROMPT


def test_sdk_agent_uses_this_prompt():
    assert build_options(model="test-model").system_prompt == SYSTEM_PROMPT


def test_examples_share_no_product_or_amount_with_the_seed_data():
    # Few-shot leakage: the model once refunded $99.99 because an example used the
    # seed's headphones at that price. Examples must generalize, not be copied.
    seed_values = set()
    for order in SEED_ORDERS.values():
        seed_values.add(f"{order['total']:.2f}")
        for item in order["items"]:
            seed_values.add(f"{item['unit_price']:.2f}")  # 99.99 was a unit price
            seed_values.add(item["name"].lower())
            seed_values.update(word for word in item["name"].lower().split() if len(word) > 4)
    seed_values.update({"45 days", "monitor", "headphones", "cookbook"})

    examples = EXAMPLES.lower()
    leaked = sorted(value for value in seed_values if value in examples)
    assert leaked == []


def test_role_limits_the_agent_to_support_topics():
    assert "help only with orders, refunds and escalations" in ROLE
    assert "politely decline anything else" in ROLE
