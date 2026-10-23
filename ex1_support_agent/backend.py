"""Fake support backend: plain Python functions over in-memory data.

No MCP and no LLM here. Later steps wrap these functions as tools, so every
edge case the agent must handle (errors, policy rules, timeouts) lives here.
"""

import copy
import math
import re
from typing import Any

# A JSON-like record: customers, orders, refunds and tickets.
type Record = dict[str, Any]

# --- Exceptions ---------------------------------------------------------------


class BackendError(Exception):
    """Base class for every error the backend raises."""

    def __init__(self, description: str) -> None:
        super().__init__(description)
        self.description = description


class BackendTimeout(BackendError):
    """Simulated outage. Transient: retrying may succeed."""


class InvalidInput(BackendError):
    """Malformed input, e.g. a bad order ID format or an amount <= 0."""


class PolicyViolation(BackendError):
    """Well-formed input blocked by a business rule. Retrying will not help."""


class NotAuthorized(BackendError):
    """The order does not belong to the given customer."""


# --- Seed data ----------------------------------------------------------------
# All emails use example.com / example.org (reserved for docs), so no real PII.
# Dates are fixed relative to SEED_TIME so the scenarios never drift with the
# real calendar.

DAY = 24 * 60 * 60
SEED_TIME = 1_792_627_200  # 2026-10-22 00:00:00 UTC


def _days_before_seed(days: int) -> int:
    return SEED_TIME - days * DAY


SEED_CUSTOMERS: dict[str, Record] = {
    # Two customers share a name on purpose: a name search returns 2 results.
    "C001": {
        "name": "Maria Schmidt",
        "email": "maria.schmidt@example.com",
        "address": "Hauptstrasse 1, 10115 Berlin, DE",
    },
    "C002": {
        "name": "Maria Schmidt",
        "email": "m.schmidt@example.org",
        "address": "Marienplatz 8, 80331 Munich, DE",
    },
    "C003": {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "address": "221B Baker Street, London NW1 6XE, UK",
    },
    "C004": {
        "name": "Aiko Tanaka",
        "email": "aiko.tanaka@example.com",
        "address": "1-1 Chiyoda, Tokyo 100-0001, JP",
    },
}

SEED_ORDERS: dict[str, Record] = {
    # Happy path: delivered recently, total under $500.
    "12345": {
        "customer_id": "C001",
        "total": 129.99,
        "status": "delivered",
        "delivered_at": _days_before_seed(3),
        "items": [
            {"sku": "HP-100", "name": "Wireless Headphones", "qty": 1, "unit_price": 99.99},
            {"sku": "CB-200", "name": "USB-C Cable", "qty": 2, "unit_price": 15.00},
        ],
        "warehouse_code": "BER-03",
        "carrier_internal_ref": "DHL-INT-7781-0042-A",
        "sku_audit_log": [
            {"sku": "HP-100", "event": "picked", "at": _days_before_seed(6)},
            {"sku": "CB-200", "event": "picked", "at": _days_before_seed(6)},
            {"sku": "HP-100", "event": "packed", "at": _days_before_seed(5)},
        ],
    },
    # Shipped but not delivered yet: refund blocked by policy.
    "23456": {
        "customer_id": "C002",
        "total": 249.00,
        "status": "shipped",
        "delivered_at": None,
        "items": [
            {"sku": "KB-310", "name": "Mechanical Keyboard", "qty": 1, "unit_price": 249.00},
        ],
        "warehouse_code": "MUC-01",
        "carrier_internal_ref": "UPS-INT-3310-8812-C",
        "sku_audit_log": [
            {"sku": "KB-310", "event": "picked", "at": _days_before_seed(2)},
        ],
    },
    # Delivered more than 30 days ago: outside the refund window.
    "34567": {
        "customer_id": "C003",
        "total": 89.50,
        "status": "delivered",
        "delivered_at": _days_before_seed(45),
        "items": [
            {"sku": "BK-042", "name": "Python Cookbook", "qty": 1, "unit_price": 89.50},
        ],
        "warehouse_code": "LON-02",
        "carrier_internal_ref": "RM-INT-0045-1190-B",
        "sku_audit_log": [
            {"sku": "BK-042", "event": "picked", "at": _days_before_seed(48)},
        ],
    },
    # Delivered recently, $650: the backend allows it; Step 5's hook must block it.
    "45678": {
        "customer_id": "C004",
        "total": 650.00,
        "status": "delivered",
        "delivered_at": _days_before_seed(5),
        "items": [
            {"sku": "MN-270", "name": "27-inch Monitor", "qty": 1, "unit_price": 650.00},
        ],
        "warehouse_code": "TYO-05",
        "carrier_internal_ref": "JP-INT-5521-0071-D",
        "sku_audit_log": [
            {"sku": "MN-270", "event": "picked", "at": _days_before_seed(9)},
            {"sku": "MN-270", "event": "qc_passed", "at": _days_before_seed(9)},
        ],
    },
    # Already refunded: a second refund is blocked by policy.
    "56789": {
        "customer_id": "C002",
        "total": 59.90,
        "status": "refunded",
        "delivered_at": _days_before_seed(10),
        "items": [
            {"sku": "MS-115", "name": "Wireless Mouse", "qty": 1, "unit_price": 59.90},
        ],
        "warehouse_code": "MUC-01",
        "carrier_internal_ref": "UPS-INT-9902-4417-E",
        "sku_audit_log": [
            {"sku": "MS-115", "event": "picked", "at": _days_before_seed(13)},
        ],
    },
}

# Live state the functions read and change. Filled from the seed by reset().
_customers: dict[str, Record] = {}
_orders: dict[str, Record] = {}
_refunds: list[Record] = []
_escalations: list[Record] = []
_pending_failures = 0


# --- Test controls --------------------------------------------------------------


def reset() -> None:
    """Restore the seed data and clear refunds, tickets and pending failures."""
    global _pending_failures
    # Mutate in place (not reassign) so any existing reference stays valid.
    _customers.clear()
    _customers.update(copy.deepcopy(SEED_CUSTOMERS))
    _orders.clear()
    _orders.update(copy.deepcopy(SEED_ORDERS))
    _refunds.clear()
    _escalations.clear()
    _pending_failures = 0


def fail_next(n: int = 1) -> None:
    """Make the next n backend calls raise BackendTimeout; later calls succeed.

    A counter instead of an on/off flag lets a test prove that a retry recovers.
    """
    global _pending_failures
    if n < 0:
        raise ValueError(f"n must be >= 0; got {n}.")
    _pending_failures = n


def _simulate_outage() -> None:
    """Raise BackendTimeout while failures are pending. Called first by every function."""
    global _pending_failures
    if _pending_failures > 0:
        _pending_failures -= 1
        raise BackendTimeout("The backend did not respond in time. Please try again.")


reset()


# --- Clock --------------------------------------------------------------------


def now() -> int:
    """Current time as a Unix int.

    Frozen at SEED_TIME so the seed scenarios ("delivered 3 days ago") stay
    valid on any real date. Tests override this function instead of the code
    calling time.time() inline.
    """
    return SEED_TIME


# --- Functions ----------------------------------------------------------------

REFUND_WINDOW = 30 * DAY
_ORDER_ID_PATTERN = re.compile(r"#?(\d{5})")


def _normalize_order_id(order_id: str) -> str:
    """Turn "12345" or "#12345" into "12345"; reject anything else."""
    match = _ORDER_ID_PATTERN.fullmatch(order_id) if isinstance(order_id, str) else None
    if match is None:
        raise InvalidInput(
            f"Order ID must be 5 digits, optionally prefixed with '#'; got {order_id!r}."
        )
    return match.group(1)


def _is_positive_amount(amount: object) -> bool:
    """True only for a real, finite number above 0.

    bool is a subclass of int in Python (True == 1), and NaN slips past every
    comparison (nan <= 0 and nan > total are both False), so both need an
    explicit check. LLM-generated tool inputs do produce values like these.
    """
    return (
        isinstance(amount, (int, float))
        and not isinstance(amount, bool)
        and math.isfinite(amount)
        and amount > 0
    )


def get_customer(
    *,
    customer_id: str | None = None,
    email: str | None = None,
    name: str | None = None,
) -> list[Record]:
    """Find customers matching every given field. No match returns []."""
    _simulate_outage()
    if customer_id is None and email is None and name is None:
        raise InvalidInput("Provide at least one of customer_id, email or name.")

    matches = []
    for cid, customer in _customers.items():
        if customer_id is not None and cid != customer_id:
            continue
        if email is not None and customer["email"].lower() != email.lower():
            continue
        if name is not None and customer["name"].lower() != name.lower():
            continue
        matches.append({"customer_id": cid, **customer})
    return matches


def lookup_order(order_id: str) -> Record | None:
    """Return the order, or None when a well-formed ID is unknown."""
    _simulate_outage()
    normalized_id = _normalize_order_id(order_id)
    order = _orders.get(normalized_id)
    if order is None:
        return None
    return {"order_id": normalized_id, **copy.deepcopy(order)}


def process_refund(order_id: str, customer_id: str, amount: float) -> Record:
    """Refund a delivered order. Checks run in a fixed order:
    input format -> ownership -> business policy.
    """
    _simulate_outage()
    normalized_id = _normalize_order_id(order_id)
    if not _is_positive_amount(amount):
        raise InvalidInput(f"Refund amount must be a number greater than 0; got {amount!r}.")

    order = _orders.get(normalized_id)
    # Unknown orders get the same answer as someone else's order, so the
    # error never reveals which order IDs exist.
    if order is None or order["customer_id"] != customer_id:
        raise NotAuthorized(f"Order {normalized_id} does not belong to customer {customer_id}.")

    if order["status"] == "refunded":
        raise PolicyViolation(f"Order {normalized_id} has already been refunded.")
    if order["status"] != "delivered":
        raise PolicyViolation(
            f"Order {normalized_id} is {order['status']}, not delivered yet; "
            "it can only be refunded after delivery."
        )
    if now() - order["delivered_at"] > REFUND_WINDOW:
        raise PolicyViolation(
            f"Order {normalized_id} was delivered more than 30 days ago; "
            "the refund window has closed."
        )
    if amount > order["total"]:
        raise PolicyViolation(
            f"Refund amount {amount:.2f} exceeds the order total {order['total']:.2f}."
        )

    order["status"] = "refunded"
    refund = {
        "refund_id": f"RF-{len(_refunds) + 1:04d}",
        "order_id": normalized_id,
        "customer_id": customer_id,
        "amount": amount,
        "status": "refunded",
        "refunded_at": now(),
    }
    _refunds.append(refund)
    return dict(refund)


def create_escalation(customer_id: str | None, summary: Record) -> Record:
    """Open a ticket for a human agent and return it with its ticket_id.

    customer_id may be None: a customer who asks for a human must be escalated
    right away, even before they have been identified.
    """
    _simulate_outage()
    if not isinstance(summary, dict) or not summary:
        raise InvalidInput("Escalation summary must be a non-empty dict.")

    ticket = {
        "ticket_id": f"ESC-{len(_escalations) + 1:04d}",
        "customer_id": customer_id,
        "summary": copy.deepcopy(summary),
        "created_at": now(),
    }
    _escalations.append(ticket)
    return copy.deepcopy(ticket)
