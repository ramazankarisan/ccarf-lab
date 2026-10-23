import pytest

from ex1_support_agent import backend
from ex1_support_agent.backend import (
    DAY,
    SEED_CUSTOMERS,
    SEED_ORDERS,
    SEED_TIME,
    BackendError,
    BackendTimeout,
    InvalidInput,
    NotAuthorized,
    PolicyViolation,
    create_escalation,
    fail_next,
    get_customer,
    lookup_order,
    process_refund,
    reset,
)


@pytest.fixture(autouse=True)
def clean_backend():
    """Every test starts from the seed data and leaves nothing behind."""
    backend.reset()
    yield
    backend.reset()


# --- Exceptions ---------------------------------------------------------------


@pytest.mark.parametrize(
    "error_class", [BackendTimeout, InvalidInput, PolicyViolation, NotAuthorized]
)
def test_every_error_is_a_backend_error_with_a_description(error_class):
    error = error_class("something went wrong")

    assert isinstance(error, BackendError)
    assert error.description == "something went wrong"
    assert str(error) == "something went wrong"


def test_backend_timeout_does_not_shadow_builtin_timeout_error():
    assert not issubclass(BackendTimeout, TimeoutError)


# --- Seed data ----------------------------------------------------------------


def test_two_customers_named_maria_schmidt_with_different_ids_and_emails():
    marias = {
        customer_id: customer
        for customer_id, customer in SEED_CUSTOMERS.items()
        if customer["name"] == "Maria Schmidt"
    }

    assert len(marias) == 2
    emails = {customer["email"] for customer in marias.values()}
    assert len(emails) == 2


def test_happy_path_order_12345_is_recent_delivered_and_under_500():
    order = SEED_ORDERS["12345"]

    assert order["status"] == "delivered"
    assert order["total"] < 500
    assert SEED_TIME - order["delivered_at"] < 30 * DAY


def test_one_order_is_shipped_but_not_delivered():
    shipped = [o for o in SEED_ORDERS.values() if o["status"] == "shipped"]

    assert len(shipped) >= 1
    assert all(o["delivered_at"] is None for o in shipped)


def test_one_order_was_delivered_more_than_30_days_ago():
    old = [
        o
        for o in SEED_ORDERS.values()
        if o["status"] == "delivered" and SEED_TIME - o["delivered_at"] > 30 * DAY
    ]

    assert len(old) >= 1


def test_one_recent_delivered_order_totals_650_for_the_hook_test():
    big = [
        o
        for o in SEED_ORDERS.values()
        if o["total"] == 650
        and o["status"] == "delivered"
        and SEED_TIME - o["delivered_at"] < 30 * DAY
    ]

    assert len(big) == 1


@pytest.mark.parametrize("order_id", list(SEED_ORDERS))
def test_order_ids_are_five_digits_and_reference_known_customers(order_id):
    assert len(order_id) == 5
    assert order_id.isdigit()
    assert SEED_ORDERS[order_id]["customer_id"] in SEED_CUSTOMERS


@pytest.mark.parametrize("order_id", list(SEED_ORDERS))
def test_every_order_carries_noisy_fields_for_the_hook_to_trim(order_id):
    order = SEED_ORDERS[order_id]

    assert {"warehouse_code", "carrier_internal_ref", "sku_audit_log"} <= order.keys()


# --- get_customer -------------------------------------------------------------


def test_get_customer_by_name_returns_both_maria_schmidts():
    results = get_customer(name="Maria Schmidt")

    assert {c["customer_id"] for c in results} == {"C001", "C002"}


def test_get_customer_by_email_is_case_insensitive():
    results = get_customer(email="MARIA.SCHMIDT@example.com")

    assert [c["customer_id"] for c in results] == ["C001"]


def test_get_customer_combines_filters():
    results = get_customer(name="Maria Schmidt", email="m.schmidt@example.org")

    assert [c["customer_id"] for c in results] == ["C002"]


def test_get_customer_no_match_returns_empty_list():
    assert get_customer(name="Nobody Here") == []


def test_get_customer_without_arguments_raises_invalid_input():
    with pytest.raises(InvalidInput):
        get_customer()


# --- lookup_order -------------------------------------------------------------


@pytest.mark.parametrize("order_id", ["12345", "#12345"])
def test_lookup_order_accepts_plain_and_hash_prefixed_ids(order_id):
    order = lookup_order(order_id)

    assert order["order_id"] == "12345"
    assert order["customer_id"] == "C001"


@pytest.mark.parametrize("order_id", ["1234", "123456", "abcde", "##12345", " 12345", "", None])
def test_lookup_order_rejects_malformed_ids(order_id):
    with pytest.raises(InvalidInput):
        lookup_order(order_id)


def test_lookup_order_unknown_well_formed_id_returns_none():
    assert lookup_order("99999") is None


def test_lookup_order_returns_a_copy_not_live_state():
    order = lookup_order("12345")
    order["status"] = "tampered"

    assert lookup_order("12345")["status"] == "delivered"


# --- process_refund -----------------------------------------------------------


def test_refund_happy_path_marks_order_refunded():
    refund = process_refund("12345", "C001", 129.99)

    assert refund["refund_id"] == "RF-0001"
    assert refund["status"] == "refunded"
    assert lookup_order("12345")["status"] == "refunded"


def test_refund_of_650_is_allowed_by_the_backend():
    # The $500 limit is deliberately NOT here: Step 5's hook enforces it.
    refund = process_refund("45678", "C004", 650)

    assert refund["status"] == "refunded"


@pytest.mark.parametrize("amount", [0, -10, "50", True, float("nan"), float("inf"), float("-inf")])
def test_refund_rejects_invalid_amount(amount):
    with pytest.raises(InvalidInput):
        process_refund("12345", "C001", amount)


def test_refund_rejects_malformed_order_id():
    with pytest.raises(InvalidInput):
        process_refund("12-345", "C001", 10)


def test_refund_for_someone_elses_order_raises_not_authorized():
    # C002 is the *other* Maria Schmidt: same name, different customer.
    with pytest.raises(NotAuthorized):
        process_refund("12345", "C002", 10)


def test_refund_for_unknown_order_raises_not_authorized():
    with pytest.raises(NotAuthorized):
        process_refund("99999", "C001", 10)


@pytest.mark.parametrize(
    ("order_id", "customer_id", "amount", "reason"),
    [
        ("23456", "C002", 10, "not delivered"),
        ("56789", "C002", 10, "already been refunded"),
        ("34567", "C003", 10, "more than 30 days"),
        ("12345", "C001", 500, "exceeds the order total"),
    ],
)
def test_refund_policy_violations(order_id, customer_id, amount, reason):
    with pytest.raises(PolicyViolation, match=reason):
        process_refund(order_id, customer_id, amount)


def test_refund_twice_is_blocked_the_second_time():
    process_refund("12345", "C001", 10)

    with pytest.raises(PolicyViolation, match="already been refunded"):
        process_refund("12345", "C001", 10)


def test_input_errors_are_checked_before_ownership():
    # Wrong customer AND bad amount: the format error wins.
    with pytest.raises(InvalidInput):
        process_refund("12345", "C002", -1)


def test_ownership_is_checked_before_policy():
    # Wrong customer AND shipped order: the permission error wins.
    with pytest.raises(NotAuthorized):
        process_refund("23456", "C001", 10)


def test_refund_window_uses_the_overridable_clock(monkeypatch):
    # Move the clock 40 days ahead: order 12345 (delivered 3 days before seed)
    # falls outside the 30-day window.
    monkeypatch.setattr(backend, "now", lambda: SEED_TIME + 40 * DAY)

    with pytest.raises(PolicyViolation, match="more than 30 days"):
        process_refund("12345", "C001", 10)


# --- create_escalation --------------------------------------------------------


def test_create_escalation_returns_incrementing_ticket_ids():
    first = create_escalation("C001", {"reason": "refund over limit"})
    second = create_escalation("C004", {"reason": "angry customer"})

    assert first["ticket_id"] == "ESC-0001"
    assert second["ticket_id"] == "ESC-0002"
    assert len(backend._escalations) == 2


def test_create_escalation_works_before_the_customer_is_identified():
    ticket = create_escalation(None, {"reason": "customer asked for a human"})

    assert ticket["ticket_id"] == "ESC-0001"
    assert ticket["customer_id"] is None


@pytest.mark.parametrize("summary", [{}, "not a dict", None])
def test_create_escalation_rejects_empty_or_non_dict_summary(summary):
    with pytest.raises(InvalidInput):
        create_escalation("C001", summary)


# --- Test controls: fail_next and reset ---------------------------------------


def test_fail_next_raises_n_times_then_recovers():
    fail_next(2)

    with pytest.raises(BackendTimeout):
        lookup_order("12345")
    with pytest.raises(BackendTimeout):
        lookup_order("12345")
    assert lookup_order("12345")["order_id"] == "12345"


@pytest.mark.parametrize(
    "call",
    [
        lambda: get_customer(name="John Doe"),
        lambda: lookup_order("12345"),
        lambda: process_refund("12345", "C001", 10),
        lambda: create_escalation("C001", {"reason": "test"}),
    ],
    ids=["get_customer", "lookup_order", "process_refund", "create_escalation"],
)
def test_fail_next_applies_to_every_function(call):
    fail_next(1)

    with pytest.raises(BackendTimeout):
        call()


def test_timeout_happens_before_any_state_change():
    fail_next(1)

    with pytest.raises(BackendTimeout):
        process_refund("12345", "C001", 10)
    assert lookup_order("12345")["status"] == "delivered"


def test_fail_next_rejects_negative_counts():
    with pytest.raises(ValueError, match="n must be >= 0"):
        fail_next(-1)


def test_reset_restores_seed_data_ids_and_pending_failures():
    process_refund("12345", "C001", 10)
    create_escalation("C001", {"reason": "test"})
    fail_next(3)

    reset()

    assert lookup_order("12345")["status"] == "delivered"
    assert process_refund("12345", "C001", 10)["refund_id"] == "RF-0001"
    assert create_escalation("C001", {"reason": "x"})["ticket_id"] == "ESC-0001"
