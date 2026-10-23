"""The support agent on the Claude Agent SDK, using the MCP server over stdio.

The SDK runs the agentic loop that loop.py wrote by hand; hooks (hooks.py)
make specific behavior deterministic.

Run against the real API:
    uv run python -m ex1_support_agent.agent "check my order #12345"
"""

import argparse
import json
import logging
import os
import sys
from collections.abc import AsyncIterable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKError,
    HookMatcher,
    McpServerConfig,
    Message,
    ResultMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)
from dotenv import load_dotenv

from ex1_support_agent.case_facts import CaseFacts, make_fact_recorder
from ex1_support_agent.hooks import (
    CLEAN_OUTPUT_MATCHER,
    REFUND_MATCHER,
    SERVER_NAME,
    enforce_refund_limit,
    make_escalation_guard,
    make_output_cleaner,
    mcp_tool_name,
)
from ex1_support_agent.loop import ToolCall
from ex1_support_agent.prompts import SYSTEM_PROMPT
from ex1_support_agent.server import mcp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOL_NAMES = ("get_customer", "lookup_order", "process_refund", "escalate_to_human")
MAX_TURNS = 10  # same guard as loop.py's max_iterations


ALL_TOOLS_MATCHER = "|".join(mcp_tool_name(tool) for tool in TOOL_NAMES)


STDIO_SERVER: McpServerConfig = {
    "type": "stdio",
    "command": "uv",
    "args": ["run", "python", "-m", "ex1_support_agent.server"],
}


def in_process_server() -> McpServerConfig:
    """The same MCP server, served inside this process.

    The web API uses it so both agents share one backend, and the data panel
    sees every refund and ticket. A stdio server is a separate process with
    its own copy of the data. mcp 2.x has no public accessor for the
    low-level Server the SDK needs, hence the private attribute.
    """
    return {"type": "sdk", "name": SERVER_NAME, "instance": mcp._lowlevel_server}


def build_options(
    model: str, facts: CaseFacts | None = None, *, in_process: bool = False
) -> ClaudeAgentOptions:
    """Agent options that expose our 4 MCP tools and nothing else.

    facts is this conversation's CaseFacts: a recorder hook writes into it, and
    the output cleaner reads its verified customer to decide which order
    details the model may see. Without one, a fresh CaseFacts is used.
    in_process serves the MCP server in this process instead of over stdio.
    """
    case: CaseFacts = facts if facts is not None else CaseFacts()
    post_tool_use = [
        HookMatcher(
            matcher=CLEAN_OUTPUT_MATCHER,
            hooks=[make_output_cleaner(lambda: case.verified_customer_id)],
        ),
        HookMatcher(matcher=ALL_TOOLS_MATCHER, hooks=[make_fact_recorder(case)]),
    ]
    return ClaudeAgentOptions(
        model=model,
        system_prompt=SYSTEM_PROMPT,  # a plain string replaces Claude Code's own prompt
        max_turns=MAX_TURNS,
        cwd=PROJECT_ROOT,  # so `uv run` finds this project when it starts the server
        mcp_servers={SERVER_NAME: in_process_server() if in_process else STDIO_SERVER},
        tools=[],  # no built-in tools at all: no Bash, Read, Write, WebFetch ...
        allowed_tools=[mcp_tool_name(tool) for tool in TOOL_NAMES],  # exact names, no wildcard
        # Isolation: ignore the developer's own ~/.claude settings, hooks,
        # CLAUDE.md files and other MCP servers, so every machine runs the same agent.
        setting_sources=[],
        strict_mcp_config=True,
        hooks={
            "PreToolUse": [
                HookMatcher(
                    matcher=REFUND_MATCHER,
                    hooks=[
                        enforce_refund_limit,
                        make_escalation_guard(lambda: case.escalated_order_ids),
                    ],
                )
            ],
            "PostToolUse": post_tool_use,
        },
    )


# --- Result -------------------------------------------------------------------


@dataclass
class SDKRun:
    tool_calls: list[ToolCall] = field(default_factory=list)
    result: ResultMessage | None = None


def collect(messages: Iterable[Message]) -> SDKRun:
    """Pair each tool_use with its tool_result and keep the final ResultMessage."""
    sdk_run = SDKRun()
    requested: dict[str, ToolUseBlock] = {}
    for message in messages:
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    requested[block.id] = block
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ToolResultBlock) and block.tool_use_id in requested:
                    call = requested[block.tool_use_id]
                    sdk_run.tool_calls.append(ToolCall(call.name, call.input, bool(block.is_error)))
        elif isinstance(message, ResultMessage):
            sdk_run.result = message
    return sdk_run


def format_sdk_run(sdk_run: SDKRun) -> str:
    """Same layout as loop.format_run, with the SDK's result and cost."""
    lines = ["Tool calls:"]
    for number, call in enumerate(sdk_run.tool_calls, start=1):
        outcome = "ERROR" if call.is_error else "ok"
        lines.append(f"  {number}. {call.name} {json.dumps(call.input)}  [{outcome}]")
    if not sdk_run.tool_calls:
        lines.append("  (none)")
    result = sdk_run.result
    final_text = result.result if result else None
    lines += ["", "Final text:", f"  {final_text or '(none)'}", ""]
    if result is None:
        lines.append("Result: (no result message)")
    else:
        cost = f"${result.total_cost_usd:.4f}" if result.total_cost_usd is not None else "n/a"
        lines.append(f"Result: {result.subtype} | turns: {result.num_turns} | cost: {cost}")
    return "\n".join(lines)


# --- CLI ----------------------------------------------------------------------


async def gather(messages: AsyncIterable[Message]) -> list[Message]:
    return [message async for message in messages]


async def main(user_message: str) -> SDKRun:  # pragma: no cover - live API call
    load_dotenv()
    options = build_options(model=os.environ["MODEL"])
    return collect(await gather(query(prompt=user_message, options=options)))


def cli() -> None:  # pragma: no cover - live API call
    parser = argparse.ArgumentParser(description="Run the SDK support agent once.")
    parser.add_argument("message", nargs="+", help="the customer's message")
    args = parser.parse_args()
    # force: importing loop.py loads the MCP server, which set logging to INFO.
    logging.basicConfig(level=logging.WARNING, force=True)

    try:
        sdk_run = anyio.run(main, " ".join(args.message))
    except ClaudeSDKError as error:
        print(f"Agent stopped: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
    print(format_sdk_run(sdk_run))
    if sdk_run.result is None or sdk_run.result.is_error:
        sys.exit(1)


if __name__ == "__main__":
    cli()
