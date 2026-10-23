import copy
import json
import logging
from typing import Any

import pytest
from anthropic import AsyncAnthropic
from anthropic.types import ContentBlock, Message, TextBlock, ThinkingBlock, ToolUseBlock, Usage
from mcp import Client

from ex1_support_agent import backend
from ex1_support_agent.loop import (
    SYSTEM_PROMPT,
    AgentRun,
    ModelClient,
    TokenUsage,
    ToolCall,
    UnexpectedStopReasonError,
    format_run,
    leaf_errors,
    list_anthropic_tools,
    run_agent,
)
from ex1_support_agent.server import mcp

# --- Scripted fake model -------------------------------------------------------
# Returns prepared replies in order and records every request. If the loop asks
# for more replies than scripted, pop() raises: that proves no extra calls.


class FakeMessages:
    def __init__(self, replies: list[Message]) -> None:
        self.replies = replies
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Message:
        self.requests.append(copy.deepcopy(kwargs))  # snapshot: the loop mutates messages
        return self.replies.pop(0)


class FakeModel:
    def __init__(self, *replies: Message) -> None:
        self.messages = FakeMessages(list(replies))


def reply(*blocks: ContentBlock, stop_reason: str, usage: Usage | None = None) -> Message:
    return Message.model_construct(
        id="msg_test",
        type="message",
        role="assistant",
        model="fake-model",
        content=list(blocks),
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=usage or Usage(input_tokens=100, output_tokens=20),
    )


def text(value: str) -> TextBlock:
    return TextBlock(type="text", text=value, citations=None)


def tool_use(tool_id: str, tool_name: str, /, **tool_input: Any) -> ToolUseBlock:
    # Positional-only (the "/") so a tool input called "name" can't collide.
    return ToolUseBlock(type="tool_use", id=tool_id, name=tool_name, input=tool_input)


async def run(model: FakeModel, message: str = "check my order #12345", **kwargs: Any) -> AgentRun:
    async with Client(mcp) as tools_client:
        return await run_agent(
            message, model_client=model, tools_client=tools_client, model="fake-model", **kwargs
        )


def tool_results(agent_run: AgentRun, index: int) -> list[dict[str, Any]]:
    """The tool_result blocks in messages[index], which must be a user message."""
    message = agent_run.messages[index]
    assert message["role"] == "user"
    content = message["content"]
    assert isinstance(content, list)
    return [dict(block) for block in content]


# --- Wiring -------------------------------------------------------------------


@pytest.mark.anyio
async def test_tools_come_from_the_mcp_server_in_anthropic_format():
    async with Client(mcp) as client:
        server_tools = {t.name: t for t in (await client.list_tools()).tools}
        tools = await list_anthropic_tools(client)

    assert {t["name"] for t in tools} == set(server_tools)
    for tool in tools:
        assert set(tool) == {"name", "description", "input_schema"}
        server_tool = server_tools[tool["name"]]
        assert tool["description"] == server_tool.description
        assert tool["input_schema"] == server_tool.input_schema


def test_real_anthropic_client_fits_the_model_client_protocol():
    # mypy checks this assignment; constructing the client makes no API call.
    client: ModelClient = AsyncAnthropic(api_key="test-key-not-used")

    assert client.messages.create is not None


@pytest.mark.anyio
async def test_each_request_sends_model_system_prompt_tools_and_history():
    model = FakeModel(reply(text("Hi!"), stop_reason="end_turn"))

    await run(model)

    request = model.messages.requests[0]
    assert request["model"] == "fake-model"
    assert request["system"] == SYSTEM_PROMPT
    assert {t["name"] for t in request["tools"]} >= {"lookup_order", "process_refund"}
    assert request["messages"] == [{"role": "user", "content": "check my order #12345"}]


# --- The loop -----------------------------------------------------------------


@pytest.mark.anyio
async def test_end_turn_completes_after_one_call():
    agent_run = await run(FakeModel(reply(text("Hello!"), stop_reason="end_turn")))

    assert agent_run.status == "completed"
    assert agent_run.iterations == 1
    assert [m["role"] for m in agent_run.messages] == ["user", "assistant"]


@pytest.mark.anyio
async def test_tool_use_runs_the_tool_and_feeds_the_result_back():
    model = FakeModel(
        reply(
            text("Let me check."),
            tool_use("toolu_1", "lookup_order", order_id="#12345"),
            stop_reason="tool_use",
        ),
        reply(text("Your order was delivered."), stop_reason="end_turn"),
    )

    agent_run = await run(model)

    assert agent_run.status == "completed"
    assert agent_run.iterations == 2
    # The FULL assistant content was kept: text AND tool_use.
    assistant_content = agent_run.messages[1]["content"]
    assert [block.type for block in assistant_content] == ["text", "tool_use"]  # type: ignore[union-attr]
    # One user message answers the tool_use_id, with the MCP text unchanged.
    [result] = tool_results(agent_run, 2)
    assert result["tool_use_id"] == "toolu_1"
    assert result["is_error"] is False
    assert json.loads(result["content"])["customer_id"] == "C001"
    # The second request carried the whole history.
    assert len(model.messages.requests[1]["messages"]) == 3


@pytest.mark.anyio
async def test_parallel_tool_calls_return_all_results_in_one_user_message():
    model = FakeModel(
        reply(
            tool_use("toolu_a", "lookup_order", order_id="12345"),
            tool_use("toolu_b", "get_customer", name="Maria Schmidt"),
            stop_reason="tool_use",
        ),
        reply(text("Done."), stop_reason="end_turn"),
    )

    agent_run = await run(model)

    results = tool_results(agent_run, 2)
    assert [r["tool_use_id"] for r in results] == ["toolu_a", "toolu_b"]
    assert all(r["is_error"] is False for r in results)
    assert [m["role"] for m in agent_run.messages] == ["user", "assistant", "user", "assistant"]


# Misleading text on purpose: only stop_reason decides, never what the text says.
@pytest.mark.anyio
@pytest.mark.parametrize(
    ("replies", "tools_run", "iterations"),
    [
        pytest.param(
            [
                reply(
                    text("Done! All finished."),
                    tool_use("toolu_1", "lookup_order", order_id="12345"),
                    stop_reason="tool_use",
                ),
                reply(text("Your order was delivered."), stop_reason="end_turn"),
            ],
            ["lookup_order"],
            2,
            id="text-says-done-but-stop_reason-is-tool_use",
        ),
        pytest.param(
            [reply(text("Next I'll call lookup_order"), stop_reason="end_turn")],
            [],
            1,
            id="text-promises-a-tool-but-stop_reason-is-end_turn",
        ),
    ],
)
async def test_termination_follows_stop_reason_not_the_text(replies, tools_run, iterations):
    agent_run = await run(FakeModel(*replies))

    assert agent_run.status == "completed"
    assert [call.name for call in agent_run.tool_calls] == tools_run
    assert agent_run.iterations == iterations


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("stop_reason", "status"),
    [("max_tokens", "truncated"), ("refusal", "refused")],
)
async def test_stop_reasons_that_are_not_success(stop_reason, status):
    agent_run = await run(FakeModel(reply(text("Partial"), stop_reason=stop_reason)))

    assert agent_run.status == status
    assert agent_run.iterations == 1


@pytest.mark.anyio
@pytest.mark.parametrize("stop_reason", ["pause_turn", "stop_sequence", None])
async def test_unhandled_stop_reason_raises_instead_of_guessing(stop_reason):
    # Raised inside `async with Client(...)`, whose task group wraps it in an
    # ExceptionGroup; callers must look inside the group.
    with pytest.raises(ExceptionGroup) as excinfo:
        await run(FakeModel(reply(text("?"), stop_reason=stop_reason)))

    assert excinfo.group_contains(UnexpectedStopReasonError, match="Unhandled stop_reason")


@pytest.mark.anyio
async def test_iteration_cap_stops_the_loop_and_logs_a_warning(caplog):
    endless = [
        reply(tool_use(f"toolu_{i}", "lookup_order", order_id="12345"), stop_reason="tool_use")
        for i in range(3)
    ]

    with caplog.at_level(logging.WARNING, logger="ex1_support_agent.loop"):
        agent_run = await run(FakeModel(*endless), max_iterations=3)

    assert agent_run.status == "iteration_cap"
    assert agent_run.iterations == 3
    assert "iteration cap" in caplog.text


# --- Tool failures never crash the loop ----------------------------------------


async def first_tool_result_after_failure(
    model_first_reply: Message, **kwargs: Any
) -> dict[str, Any]:
    model = FakeModel(model_first_reply, reply(text("Sorry."), stop_reason="end_turn"))
    agent_run = await run(model, **kwargs)
    assert agent_run.status == "completed"
    [result] = tool_results(agent_run, 2)
    assert result["is_error"] is True
    return result


@pytest.mark.anyio
async def test_backend_error_arrives_as_the_servers_structured_json():
    result = await first_tool_result_after_failure(
        reply(tool_use("toolu_1", "lookup_order", order_id="12-345"), stop_reason="tool_use")
    )

    assert json.loads(result["content"])["errorCategory"] == "validation"


@pytest.mark.anyio
async def test_non_object_tool_input_becomes_an_error_result():
    bad_call = ToolUseBlock.model_construct(
        type="tool_use", id="toolu_1", name="lookup_order", input="12345"
    )

    result = await first_tool_result_after_failure(reply(bad_call, stop_reason="tool_use"))

    assert "must be a JSON object" in result["content"]


@pytest.mark.anyio
async def test_unknown_tool_becomes_an_error_result():
    result = await first_tool_result_after_failure(
        reply(tool_use("toolu_1", "delete_database"), stop_reason="tool_use")
    )

    assert "Unknown tool 'delete_database'" in result["content"]


@pytest.mark.anyio
async def test_schema_validation_error_passes_through_as_plain_text():
    result = await first_tool_result_after_failure(
        reply(
            tool_use("toolu_1", "process_refund", order_id="12345", amount=10),
            stop_reason="tool_use",
        )
    )

    assert "customer_id" in result["content"]
    assert "errorCategory" not in result["content"]


@pytest.mark.anyio
async def test_mcp_client_exception_becomes_an_error_result(monkeypatch):
    async def broken_call_tool(self: Client, *args: Any, **kwargs: Any) -> None:
        raise ConnectionError("pipe closed")

    monkeypatch.setattr(Client, "call_tool", broken_call_tool)

    result = await first_tool_result_after_failure(
        reply(tool_use("toolu_1", "lookup_order", order_id="12345"), stop_reason="tool_use")
    )

    assert json.loads(result["content"])["errorCategory"] == "internal"
    assert "pipe closed" not in result["content"]


@pytest.mark.anyio
async def test_loop_does_not_retry_a_retryable_error():
    # One pending timeout. The model gets the transient error once; the loop
    # does not call the tool again on its own (the model would have to).
    backend.fail_next(1)

    result = await first_tool_result_after_failure(
        reply(tool_use("toolu_1", "lookup_order", order_id="12345"), stop_reason="tool_use")
    )

    error = json.loads(result["content"])
    assert error["errorCategory"] == "transient"
    assert error["isRetryable"] is True


# --- Return value -------------------------------------------------------------


@pytest.mark.anyio
async def test_final_text_comes_from_the_last_responses_text_blocks_only():
    model = FakeModel(
        reply(
            text("Let me check."),
            tool_use("toolu_1", "lookup_order", order_id="12345"),
            stop_reason="tool_use",
        ),
        reply(
            ThinkingBlock(type="thinking", thinking="", signature="sig"),
            text("Your order "),
            text("was delivered."),
            stop_reason="end_turn",
        ),
    )

    agent_run = await run(model)

    assert agent_run.final_text == "Your order was delivered."


@pytest.mark.anyio
async def test_tool_calls_record_name_input_and_error_flag():
    model = FakeModel(
        reply(
            tool_use("toolu_a", "lookup_order", order_id="12345"),
            tool_use("toolu_b", "lookup_order", order_id="12-345"),
            stop_reason="tool_use",
        ),
        reply(tool_use("toolu_c", "delete_database"), stop_reason="tool_use"),
        reply(text("Done."), stop_reason="end_turn"),
    )

    agent_run = await run(model)

    assert agent_run.tool_calls == [
        ToolCall("lookup_order", {"order_id": "12345"}, is_error=False),
        ToolCall("lookup_order", {"order_id": "12-345"}, is_error=True),
        ToolCall("delete_database", {}, is_error=True),
    ]
    assert agent_run.iterations == 3


@pytest.mark.anyio
async def test_usage_is_summed_over_every_api_call():
    model = FakeModel(
        reply(
            tool_use("toolu_1", "lookup_order", order_id="12345"),
            stop_reason="tool_use",
            usage=Usage(
                input_tokens=2000,
                output_tokens=50,
                cache_creation_input_tokens=1800,
                cache_read_input_tokens=0,
            ),
        ),
        reply(
            text("Done."),
            stop_reason="end_turn",
            usage=Usage(
                input_tokens=300,
                output_tokens=30,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=1800,
            ),
        ),
    )

    usage = (await run(model)).usage

    assert usage.input_tokens == 2300
    assert usage.output_tokens == 80
    assert usage.cache_creation_input_tokens == 1800
    assert usage.cache_read_input_tokens == 1800


@pytest.mark.anyio
async def test_truncated_run_keeps_the_partial_text():
    agent_run = await run(FakeModel(reply(text("Your order was deliv"), stop_reason="max_tokens")))

    assert agent_run.status == "truncated"
    assert agent_run.final_text == "Your order was deliv"


@pytest.mark.anyio
async def test_iteration_cap_run_has_no_final_text_when_the_last_reply_was_only_tools():
    model = FakeModel(
        reply(tool_use("toolu_1", "lookup_order", order_id="12345"), stop_reason="tool_use")
    )

    agent_run = await run(model, max_iterations=1)

    assert agent_run.status == "iteration_cap"
    assert agent_run.final_text == ""
    assert len(agent_run.tool_calls) == 1


# --- CLI output ---------------------------------------------------------------


def test_format_run_shows_tool_calls_final_text_and_status():
    agent_run = AgentRun(
        status="completed",
        final_text="Your order was delivered.",
        messages=[],
        tool_calls=[
            ToolCall("lookup_order", {"order_id": "#12345"}, is_error=False),
            ToolCall("process_refund", {"order_id": "23456"}, is_error=True),
        ],
        iterations=3,
        usage=TokenUsage(input_tokens=6900, output_tokens=210),
    )

    output = format_run(agent_run)

    assert '1. lookup_order {"order_id": "#12345"}  [ok]' in output
    assert '2. process_refund {"order_id": "23456"}  [ERROR]' in output
    assert "Your order was delivered." in output
    assert "Status: completed | iterations: 3 | tokens in: 6,900 out: 210" in output


def test_format_run_without_tools_or_text():
    agent_run = AgentRun("refused", "", [], [], 1, TokenUsage())

    output = format_run(agent_run)

    assert output.count("(none)") == 2
    assert "Status: refused" in output


def test_leaf_errors_unwraps_nested_exception_groups():
    timeout, refusal = TimeoutError("slow"), ValueError("bad")
    nested = ExceptionGroup("outer", [ExceptionGroup("inner", [timeout]), refusal])

    assert leaf_errors(nested) == [timeout, refusal]
    assert leaf_errors(timeout) == [timeout]
