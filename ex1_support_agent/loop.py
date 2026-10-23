"""Raw agentic loop on the plain anthropic SDK.

The model asks for tools, the loop runs them through the MCP server and feeds
the results back, and stop_reason alone decides when to stop.

Run against the real API:
    uv run python -m ex1_support_agent.loop "check my order #12345"
With a simulated backend outage on the first call:
    uv run python -m ex1_support_agent.loop --fail-next 1 "check my order #12345"
"""

import argparse
import json
import logging
import os
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import anyio
from anthropic import APIError, AsyncAnthropic
from anthropic.types import (
    ContentBlock,
    Message,
    MessageParam,
    TextBlock,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
    Usage,
)
from dotenv import load_dotenv
from mcp import Client
from mcp.types import TextContent

from ex1_support_agent import backend
from ex1_support_agent.server import mcp

logger = logging.getLogger(__name__)

# Minimal on purpose: this loop shows the mechanics. prompts.py has the full prompt.
SYSTEM_PROMPT = "You are a customer support agent. Use the tools."
MAX_TOKENS = 16_000

Status = Literal["completed", "truncated", "refused", "iteration_cap"]


class UnexpectedStopReasonError(RuntimeError):
    """The model stopped for a reason this loop does not handle. Never guess."""


# --- Model client -------------------------------------------------------------
# The loop only needs `model_client.messages.create(...)`. Typing it as a
# Protocol (structural typing) lets the real AsyncAnthropic
# and a scripted test fake both fit, without either one inheriting anything.


class MessagesAPI(Protocol):
    @property
    def create(self) -> Callable[..., Awaitable[Message]]: ...


class ModelClient(Protocol):
    @property
    def messages(self) -> MessagesAPI: ...


# --- Tools --------------------------------------------------------------------


async def list_anthropic_tools(tools_client: Client) -> list[ToolParam]:
    """The MCP server's tools in Anthropic format.

    Taken from the server rather than the backend, so the model sees exactly the
    descriptions and schemas tested in test_server.py.
    """
    result = await tools_client.list_tools()
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.input_schema,
        }
        for tool in result.tools
    ]


def _loop_error(tool_use_id: str, category: str, description: str) -> ToolResultBlockParam:
    """A tool_result for failures the loop itself detects, in the server's error shape."""
    payload = {
        "errorCategory": category,
        "isRetryable": False,
        "description": description,
        "customerMessage": None,
    }
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(payload),
        "is_error": True,
    }


async def run_tool_call(
    call: ToolUseBlock, tools_client: Client, tool_names: set[str]
) -> ToolResultBlockParam:
    """Run one tool_use block through MCP. Never raises: failures become is_error results."""
    if call.name not in tool_names:
        return _loop_error(
            call.id, "validation", f"Unknown tool {call.name!r}. Use only the tools provided."
        )
    if not isinstance(call.input, dict):
        return _loop_error(call.id, "validation", "Tool input must be a JSON object.")

    try:
        result = await tools_client.call_tool(call.name, call.input)
    except Exception:
        logger.exception("MCP call to %s failed", call.name)
        return _loop_error(
            call.id, "internal", "The tool could not be reached. Do not retry; escalate."
        )

    # Pass the MCP text through unchanged: structured errors stay JSON,
    # and the SDK's plain-text schema errors stay plain text.
    text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
    return {
        "type": "tool_result",
        "tool_use_id": call.id,
        "content": text,
        "is_error": bool(result.is_error),
    }


async def run_tool_calls(
    content: Sequence[ContentBlock], tools_client: Client, tool_names: set[str]
) -> list[ToolResultBlockParam]:
    """Run every tool_use block in one response, in order.

    Sequential on purpose: tools like process_refund change state, so a fixed
    order keeps runs reproducible. All results go back in ONE user message.
    """
    return [
        await run_tool_call(block, tools_client, tool_names)
        for block in content
        if isinstance(block, ToolUseBlock)
    ]


# --- Result -------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    name: str
    input: object  # as the model sent it; normally a dict
    is_error: bool


@dataclass
class TokenUsage:
    """Usage summed over every API call in the run: that is what gets billed."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def add(self, usage: Usage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        # The cache fields can be None when caching is not involved.
        self.cache_creation_input_tokens += usage.cache_creation_input_tokens or 0
        self.cache_read_input_tokens += usage.cache_read_input_tokens or 0


@dataclass
class AgentRun:
    status: Status
    final_text: str
    messages: list[MessageParam]
    tool_calls: list[ToolCall]
    iterations: int
    usage: TokenUsage


def text_of(content: Sequence[ContentBlock]) -> str:
    """Join the text blocks only; thinking and tool_use blocks are skipped."""
    return "".join(block.text for block in content if isinstance(block, TextBlock))


# --- The loop -----------------------------------------------------------------


async def run_agent(
    user_message: str,
    *,
    model_client: ModelClient,
    tools_client: Client,
    model: str,
    max_iterations: int = 10,
    history: Sequence[MessageParam] = (),
) -> AgentRun:
    """Run the agent until stop_reason says it is done, or the iteration cap is hit.

    history is the messages of earlier turns (a previous AgentRun.messages), so
    the API, which is stateless, sees the whole conversation again.
    No retries here: on a tool error the model reads isRetryable and decides.
    """
    tools = await list_anthropic_tools(tools_client)
    tool_names = {tool["name"] for tool in tools}
    messages: list[MessageParam] = [*history, {"role": "user", "content": user_message}]
    tool_calls: list[ToolCall] = []
    usage = TokenUsage()
    last_content: Sequence[ContentBlock] = []

    def finish(status: Status, iterations: int) -> AgentRun:
        return AgentRun(status, text_of(last_content), messages, tool_calls, iterations, usage)

    for iteration in range(1, max_iterations + 1):
        response = await model_client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )
        usage.add(response.usage)
        last_content = response.content
        # The FULL content: text, tool_use and thinking blocks must all go back.
        messages.append({"role": "assistant", "content": response.content})

        match response.stop_reason:
            case "tool_use":
                results = await run_tool_calls(response.content, tools_client, tool_names)
                calls = [block for block in response.content if isinstance(block, ToolUseBlock)]
                tool_calls.extend(
                    ToolCall(call.name, call.input, bool(result.get("is_error")))
                    for call, result in zip(calls, results, strict=True)
                )
                messages.append({"role": "user", "content": results})
            case "end_turn":
                return finish("completed", iteration)
            case "max_tokens":
                return finish("truncated", iteration)
            case "refusal":
                return finish("refused", iteration)
            case other:
                raise UnexpectedStopReasonError(f"Unhandled stop_reason: {other!r}")

    logger.warning("Agent hit the iteration cap of %d without finishing", max_iterations)
    return finish("iteration_cap", max_iterations)


# --- CLI ----------------------------------------------------------------------


def format_run(agent_run: AgentRun) -> str:
    """Human-readable summary: tool calls, final text, status and cost."""
    lines = ["Tool calls:"]
    for number, call in enumerate(agent_run.tool_calls, start=1):
        outcome = "ERROR" if call.is_error else "ok"
        lines.append(f"  {number}. {call.name} {json.dumps(call.input)}  [{outcome}]")
    if not agent_run.tool_calls:
        lines.append("  (none)")
    lines += ["", "Final text:", f"  {agent_run.final_text or '(none)'}", ""]
    usage = agent_run.usage
    lines.append(
        f"Status: {agent_run.status} | iterations: {agent_run.iterations} | "
        f"tokens in: {usage.input_tokens:,} out: {usage.output_tokens:,}"
    )
    return "\n".join(lines)


def leaf_errors(error: BaseException) -> list[BaseException]:
    """The real errors inside a (possibly nested) ExceptionGroup."""
    if isinstance(error, BaseExceptionGroup):
        return [leaf for inner in error.exceptions for leaf in leaf_errors(inner)]
    return [error]


async def main(user_message: str) -> AgentRun:  # pragma: no cover - live API call
    load_dotenv()
    async with Client(mcp) as tools_client:
        return await run_agent(
            user_message,
            model_client=AsyncAnthropic(),  # reads ANTHROPIC_API_KEY from the environment
            tools_client=tools_client,
            model=os.environ["MODEL"],
        )


def cli() -> None:  # pragma: no cover - live API call
    parser = argparse.ArgumentParser(description="Run the support agent once.")
    parser.add_argument("message", nargs="+", help="the customer's message")
    parser.add_argument(
        "--fail-next",
        type=int,
        default=0,
        metavar="N",
        help="make the next N backend calls time out, to watch the agent handle errors",
    )
    args = parser.parse_args()
    if args.fail_next < 0:
        parser.error("--fail-next must be 0 or more")
    # Works because the MCP server, and so the backend, runs in this process.
    backend.fail_next(args.fail_next)
    # force: the MCP server already configured logging at INFO when it was imported.
    logging.basicConfig(level=logging.WARNING, force=True)

    try:
        agent_run = anyio.run(main, " ".join(args.message))
    # Errors raised inside `async with Client(...)` arrive wrapped in an
    # ExceptionGroup; except* picks the matching ones out of the group.
    except* (APIError, UnexpectedStopReasonError) as group:
        for error in leaf_errors(group):
            print(f"Agent stopped: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
    print(format_run(agent_run))


if __name__ == "__main__":
    cli()
