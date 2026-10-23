from collections.abc import AsyncIterator

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeSDKClient,
    Message,
    ResultMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from ex1_support_agent.case_facts import CaseFacts, OrderFacts, render
from ex1_support_agent.session import ChatClient, SupportSession

# Offline: a scripted fake stands in for ClaudeSDKClient.


def result(text: str) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="session-test",
        result=text,
    )


class FakeClient:
    """Records every prompt; replays one scripted list of messages per turn."""

    def __init__(self, *turns: list[Message]) -> None:
        self.turns = list(turns)
        self.prompts: list[str] = []

    async def query(self, prompt: str) -> None:
        self.prompts.append(prompt)

    async def receive_response(self) -> AsyncIterator[Message]:
        for message in self.turns.pop(0):
            yield message


def test_the_real_client_fits_the_chat_client_protocol():
    # mypy checks this assignment; constructing the client starts nothing.
    client: ChatClient = ClaudeSDKClient()

    assert client is not None


@pytest.mark.anyio
async def test_first_turn_is_sent_as_typed():
    client = FakeClient([result("Hi")])
    session = SupportSession(client, CaseFacts())

    await session.send("check my order #12345")

    assert client.prompts == ["check my order #12345"]


@pytest.mark.anyio
async def test_later_turns_get_the_current_case_facts_in_front():
    facts = CaseFacts()
    client = FakeClient([result("Which email?")], [result("Thanks")])
    session = SupportSession(client, facts)

    await session.send("check my order #12345")
    # A hook would record this during turn 1; here we set it directly.
    facts.verified_customer_id = "C001"
    facts.orders["12345"] = OrderFacts(
        customer_id="C001", status="delivered", total=129.99, delivered_at="2026-10-19T00:00:00Z"
    )
    await session.send("it's maria@example.com")

    second = client.prompts[1]
    assert second == f"{render(facts)}\n\nit's maria@example.com"
    assert "total=129.99" in second
    assert second.endswith("it's maria@example.com")


@pytest.mark.anyio
async def test_facts_are_rendered_fresh_on_every_turn():
    facts = CaseFacts()
    client = FakeClient([result("1")], [result("2")], [result("3")])
    session = SupportSession(client, facts)

    await session.send("one")
    await session.send("two")
    facts.escalation_ticket = "ESC-0001"
    await session.send("three")

    assert "escalation_ticket: none" in client.prompts[1]
    assert "escalation_ticket: ESC-0001" in client.prompts[2]
    assert session.turns == 3


@pytest.mark.anyio
async def test_send_returns_only_this_turns_tool_calls_and_result():
    first_turn: list[Message] = [
        AssistantMessage(
            content=[ToolUseBlock(id="t1", name="mcp__support__lookup_order", input={})],
            model="test-model",
        ),
        UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="{}")]),
        result("Found it."),
    ]
    session = SupportSession(FakeClient(first_turn, [result("Bye")]), CaseFacts())

    first = await session.send("check my order")
    second = await session.send("thanks")

    assert [call.name for call in first.tool_calls] == ["mcp__support__lookup_order"]
    assert second.tool_calls == []
    assert second.result is not None
    assert second.result.result == "Bye"
