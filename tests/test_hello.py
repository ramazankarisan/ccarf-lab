from anthropic.types import Message, TextBlock, ThinkingBlock, ToolUseBlock

from ex1_support_agent.hello import extract_text


def test_extract_text_joins_text_blocks_and_skips_the_rest():
    # A reply can start with thinking or contain tool calls, so content[0]
    # is not guaranteed to be text.
    message = Message.model_construct(
        content=[
            ThinkingBlock(type="thinking", thinking="", signature="sig"),
            TextBlock(type="text", text="Your order ", citations=None),
            ToolUseBlock(type="tool_use", id="toolu_1", name="lookup_order", input={}),
            TextBlock(type="text", text="has shipped.", citations=None),
        ]
    )

    assert extract_text(message) == "Your order has shipped."


def test_extract_text_returns_empty_string_without_text_blocks():
    message = Message.model_construct(content=[])

    assert extract_text(message) == ""
