"""Fixtures and helpers for the live scenario tests.

These tests call the real Claude API. The agent is not deterministic, so the
helpers read what it did (tool calls, arguments, results, case facts), never
how it phrased the reply.
"""

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import pytest
from dotenv import load_dotenv

from ex1_support_agent.agent import SDKRun
from ex1_support_agent.hooks import mcp_tool_name
from ex1_support_agent.session import SupportSession, open_session

TOOL_PREFIX = mcp_tool_name("")  # "mcp__support__"


@pytest.fixture
def model() -> str:
    """The model from .env. Loaded here, not at import, so offline runs never see the key."""
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("MODEL"):
        pytest.skip("live tests need ANTHROPIC_API_KEY and MODEL in .env")
    return os.environ["MODEL"]


@pytest.fixture
async def session(model: str) -> AsyncIterator[SupportSession]:
    """A fresh conversation. Its client starts its own stdio server, so the backend
    starts from the seed data every time."""
    async with open_session(model) as fresh:
        yield fresh


# --- Helpers ----------------------------------------------------------------------


def tools(run: SDKRun) -> list[str]:
    """Short tool names in call order, e.g. ["lookup_order", "get_customer"]."""
    return [call.name.removeprefix(TOOL_PREFIX) for call in run.tool_calls]


def calls(run: SDKRun, name: str) -> list[dict[str, Any]]:
    """The inputs of every call to one tool, in order."""
    return [
        call.input
        for call in run.tool_calls
        if call.name == mcp_tool_name(name) and isinstance(call.input, dict)
    ]


def results(run: SDKRun, name: str) -> list[str]:
    """The tool result text the model received for one tool, after the hooks ran."""
    return [
        output
        for call, output in zip(run.tool_calls, run.outputs, strict=True)
        if call.name == mcp_tool_name(name)
    ]


def errors(run: SDKRun, name: str) -> list[bool]:
    """is_error for every call to one tool, in order."""
    return [call.is_error for call in run.tool_calls if call.name == mcp_tool_name(name)]


def reply(run: SDKRun) -> str:
    """The final text. Only for negative checks: what must never appear."""
    return (run.result.result or "") if run.result else ""


def summary(escalation: dict[str, Any]) -> dict[str, Any]:
    """The handoff summary of an escalate_to_human call, parsed if sent as a JSON string."""
    value = escalation["summary"]
    parsed: dict[str, Any] = json.loads(value) if isinstance(value, str) else value
    return parsed
