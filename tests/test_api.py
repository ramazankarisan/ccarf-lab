from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

import pytest
from claude_agent_sdk import Message
from fastapi.testclient import TestClient

from ex1_support_agent import backend
from ex1_support_agent.api import create_app
from ex1_support_agent.case_facts import CaseFacts
from ex1_support_agent.loop import UnexpectedStopReasonError
from ex1_support_agent.prompts import SYSTEM_PROMPT
from ex1_support_agent.session import SupportSession
from tests.test_loop import FakeModel, reply, text, tool_use
from tests.test_session import FakeClient, result

# Offline: a scripted model for the Client SDK mode, a scripted session for the
# Agent SDK mode. The MCP tools and the backend are real.


class FakeSessions:
    """Opens scripted Agent SDK sessions and records which ones were closed."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.opened = 0
        self.closed = 0

    @asynccontextmanager
    async def open(self) -> AsyncIterator[SupportSession]:
        self.opened += 1
        turns: list[list[Message]] = [[result(text)] for text in self.replies]
        try:
            yield SupportSession(FakeClient(*turns), CaseFacts())
        finally:
            self.closed += 1


def make_client(model: FakeModel | None = None, sessions: FakeSessions | None = None) -> TestClient:
    app = create_app(
        model="test-model",
        model_client=model or FakeModel(),
        open_agent_session=(sessions or FakeSessions()).open,
    )
    return TestClient(app)


@pytest.fixture
def api() -> Iterator[TestClient]:
    with make_client() as client:
        yield client


# --- Data ---------------------------------------------------------------------


def test_data_returns_the_live_customers_and_orders(api):
    data = api.get("/api/data").json()

    assert set(data) == {"customers", "orders", "refunds", "escalations"}
    assert data["orders"]["12345"]["total"] == 129.99
    assert data["customers"]["C004"]["name"] == "Aiko Tanaka"


def test_reset_restores_the_seed_data(api):
    backend.process_refund("12345", "C001", 10)

    data = api.post("/api/reset").json()

    assert data["refunds"] == []
    assert data["orders"]["12345"]["status"] == "delivered"


# --- Client SDK mode ----------------------------------------------------------


def test_client_mode_runs_the_loop_with_real_tools():
    model = FakeModel(
        reply(tool_use("t1", "lookup_order", order_id="12345"), stop_reason="tool_use"),
        reply(text("It was delivered."), stop_reason="end_turn"),
    )
    with make_client(model=model) as client:
        body = client.post("/api/chat", json={"mode": "client", "message": "order 12345?"}).json()

    assert body["reply"] == "It was delivered."
    assert body["tool_calls"] == [
        {"name": "lookup_order", "input": {"order_id": "12345"}, "is_error": False}
    ]
    assert body["status"] == "completed"
    assert body["usage"].startswith("tokens in")


def test_client_mode_remembers_earlier_messages():
    model = FakeModel(
        reply(text("Which email?"), stop_reason="end_turn"),
        reply(text("Thanks."), stop_reason="end_turn"),
    )
    with make_client(model=model) as client:
        first = client.post("/api/chat", json={"mode": "client", "message": "hi"}).json()
        client.post(
            "/api/chat",
            json={
                "mode": "client",
                "message": "a@example.com",
                "conversation_id": first["conversation_id"],
            },
        )

    second_request = model.messages.requests[1]["messages"]
    assert [m["role"] for m in second_request] == ["user", "assistant", "user"]


def test_an_agent_failure_becomes_a_502_without_details():
    class FailingMessages:
        async def create(self, **kwargs):
            raise UnexpectedStopReasonError("secret internal detail")

    class FailingModel:
        messages = FailingMessages()

    app = create_app(
        model="test-model", model_client=FailingModel(), open_agent_session=FakeSessions().open
    )
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"mode": "client", "message": "hi"})

    assert response.status_code == 502
    assert response.json() == {"detail": "The agent failed (UnexpectedStopReasonError)."}


# --- Agent SDK mode -----------------------------------------------------------


def test_agent_mode_keeps_one_session_per_conversation():
    sessions = FakeSessions("Which email?", "Thanks.")
    with make_client(sessions=sessions) as client:
        first = client.post("/api/chat", json={"mode": "agent", "message": "hi"}).json()
        second = client.post(
            "/api/chat",
            json={
                "mode": "agent",
                "message": "a@example.com",
                "conversation_id": first["conversation_id"],
            },
        ).json()

    assert (first["reply"], second["reply"]) == ("Which email?", "Thanks.")
    assert second["conversation_id"] == first["conversation_id"]
    assert sessions.opened == 1
    assert first["usage"] == "cost n/a"


def test_reset_closes_open_agent_sessions():
    sessions = FakeSessions("Hi")
    with make_client(sessions=sessions) as client:
        client.post("/api/chat", json={"mode": "agent", "message": "hi"})
        client.post("/api/reset")

        assert sessions.closed == 1


def test_sessions_are_closed_when_the_server_stops():
    sessions = FakeSessions("Hi")
    with make_client(sessions=sessions) as client:
        client.post("/api/chat", json={"mode": "agent", "message": "hi"})

    assert sessions.closed == 1


# --- Validation ---------------------------------------------------------------


@pytest.mark.parametrize("mode", ["client", "agent"])
def test_an_unknown_conversation_is_a_404(api, mode):
    response = api.post(
        "/api/chat", json={"mode": mode, "message": "hi", "conversation_id": "nope"}
    )

    assert response.status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"mode": "other", "message": "hi"}, id="unknown-mode"),
        pytest.param({"mode": "client", "message": ""}, id="empty-message"),
        pytest.param({"mode": "client", "message": "x" * 2001}, id="too-long"),
    ],
)
def test_invalid_requests_are_rejected(api, body):
    assert api.post("/api/chat", json=body).status_code == 422


def test_client_mode_uses_the_same_full_prompt_as_agent_mode():
    model = FakeModel(reply(text("Hi"), stop_reason="end_turn"))
    with make_client(model=model) as client:
        client.post("/api/chat", json={"mode": "client", "message": "1+1"})

    assert model.messages.requests[0]["system"] == SYSTEM_PROMPT
