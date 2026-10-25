from collections.abc import AsyncIterator

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    Message,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from mcp import Client

from ex1_support_agent.agent import (
    MAX_TURNS,
    PROJECT_ROOT,
    TOOL_NAMES,
    build_options,
    collect,
    format_sdk_run,
    gather,
    result_text,
    stdio_server,
)
from ex1_support_agent.loop import ToolCall
from ex1_support_agent.prompts import SYSTEM_PROMPT
from ex1_support_agent.server import mcp

# Offline only: these build options and parse SDK messages, never start an agent.


def result_message(
    *, total_cost_usd: float | None = 0.0123, result: str | None = "Your order was delivered."
) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1200,
        duration_api_ms=1000,
        is_error=False,
        num_turns=2,
        session_id="session-test",
        total_cost_usd=total_cost_usd,
        result=result,
    )


# --- Options ------------------------------------------------------------------


@pytest.mark.anyio
async def test_tool_names_match_the_mcp_server():
    # The allow-list is written out by hand; this catches a tool added to the
    # server but never allowed (or a typo, which would silently allow nothing).
    async with Client(mcp) as client:
        server_tools = {t.name for t in (await client.list_tools()).tools}

    assert set(TOOL_NAMES) == server_tools


def test_options_allow_exactly_the_four_mcp_tools_and_no_built_ins():
    options = build_options(model="test-model")

    assert options.tools == []
    assert sorted(options.allowed_tools) == [
        "mcp__support__escalate_to_human",
        "mcp__support__get_customer",
        "mcp__support__lookup_order",
        "mcp__support__process_refund",
    ]


def test_options_run_our_server_over_stdio_with_model_prompt_and_guard():
    options = build_options(model="test-model")

    assert options.mcp_servers == {
        "support": {
            "type": "stdio",
            "command": "uv",
            "args": ["run", "python", "-m", "ex1_support_agent.server"],
        }
    }
    assert options.model == "test-model"
    assert options.system_prompt == SYSTEM_PROMPT
    assert options.max_turns == MAX_TURNS
    assert options.cwd == PROJECT_ROOT


def test_in_process_option_serves_the_same_server_in_this_process():
    options = build_options(model="test-model", in_process=True)

    assert options.mcp_servers == {
        "support": {"type": "sdk", "name": "support", "instance": mcp._lowlevel_server}
    }


def test_server_env_is_passed_to_the_stdio_server_process():
    options = build_options(model="test-model", server_env={"SUPPORT_FAIL_NEXT": "1"})

    assert options.mcp_servers == {"support": stdio_server({"SUPPORT_FAIL_NEXT": "1"})}
    assert stdio_server({"SUPPORT_FAIL_NEXT": "1"})["env"] == {"SUPPORT_FAIL_NEXT": "1"}


def test_server_env_cannot_reach_an_in_process_server():
    # In process, the backend is this process's own: call backend.fail_next() instead.
    with pytest.raises(ValueError, match="stdio"):
        build_options(model="test-model", in_process=True, server_env={"SUPPORT_FAIL_NEXT": "1"})


def test_options_ignore_the_developers_local_claude_config():
    options = build_options(model="test-model")

    assert options.setting_sources == []
    assert options.strict_mcp_config is True


# --- Collecting results -------------------------------------------------------


def test_collect_pairs_tool_calls_with_their_results():
    messages: list[Message] = [
        AssistantMessage(
            content=[
                TextBlock(text="Let me check."),
                ToolUseBlock(id="t1", name="mcp__support__lookup_order", input={"order_id": "1"}),
                ToolUseBlock(id="t2", name="mcp__support__get_customer", input={"name": "X"}),
            ],
            model="test-model",
        ),
        UserMessage(
            content=[
                ToolResultBlock(tool_use_id="t1", content="{}", is_error=False),
                ToolResultBlock(tool_use_id="t2", content="boom", is_error=True),
            ]
        ),
        UserMessage(content="a plain text user message is skipped"),
        result_message(),
    ]

    sdk_run = collect(messages)

    assert sdk_run.tool_calls == [
        ToolCall("mcp__support__lookup_order", {"order_id": "1"}, is_error=False),
        ToolCall("mcp__support__get_customer", {"name": "X"}, is_error=True),
    ]
    assert sdk_run.outputs == ["{}", "boom"]
    assert sdk_run.result is not None
    assert sdk_run.result.result == "Your order was delivered."


@pytest.mark.parametrize(
    ("content", "text"),
    [
        (None, ""),
        ("plain", "plain"),
        (
            [
                {"type": "text", "text": "a"},
                {"type": "image", "data": "x"},
                {"type": "text", "text": "b"},
            ],
            "a\nb",
        ),
    ],
)
def test_result_text_reads_every_content_shape(content, text):
    assert result_text(content) == text


def test_collect_ignores_a_result_without_a_matching_tool_use():
    messages = [UserMessage(content=[ToolResultBlock(tool_use_id="unknown", content="x")])]

    assert collect(messages).tool_calls == []


@pytest.mark.anyio
async def test_gather_reads_the_whole_stream():
    async def stream() -> AsyncIterator[Message]:
        yield result_message()

    assert len(await gather(stream())) == 1


# --- Formatting ---------------------------------------------------------------


def test_format_shows_tool_calls_final_text_and_cost():
    sdk_run = collect(
        [
            AssistantMessage(
                content=[ToolUseBlock(id="t1", name="lookup", input={"order_id": "1"})],
                model="test-model",
            ),
            UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="x", is_error=True)]),
            result_message(),
        ]
    )

    assert format_sdk_run(sdk_run) == (
        "Tool calls:\n"
        '  1. lookup {"order_id": "1"}  [ERROR]\n'
        "\n"
        "Final text:\n"
        "  Your order was delivered.\n"
        "\n"
        "Result: success | turns: 2 | cost: $0.0123"
    )


def test_format_without_tools_result_or_cost():
    assert format_sdk_run(collect([])).splitlines() == [
        "Tool calls:",
        "  (none)",
        "",
        "Final text:",
        "  (none)",
        "",
        "Result: (no result message)",
    ]
    no_cost = collect([result_message(total_cost_usd=None, result=None)])
    assert format_sdk_run(no_cost).endswith("cost: n/a")
