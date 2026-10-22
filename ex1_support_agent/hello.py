"""Smoke test: one Messages API call, end to end.

Run with:  uv run python -m ex1_support_agent.hello
"""

import os

from anthropic import Anthropic
from anthropic.types import Message
from dotenv import load_dotenv

SYSTEM_PROMPT = "You are a friendly customer support assistant. Answer in one short sentence."
USER_MESSAGE = "Hi! Can you confirm you are up and running?"
MAX_TOKENS = 1024


def extract_text(message: Message) -> str:
    """Join all text blocks. A response may also hold thinking or tool_use blocks,
    so never assume content[0] is the text."""
    return "".join(block.text for block in message.content if block.type == "text")


def main() -> None:
    load_dotenv()
    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    model = os.environ["MODEL"]

    message = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_MESSAGE}],
    )

    print(extract_text(message))
    print(f"stop_reason: {message.stop_reason} | usage: {message.usage}")


if __name__ == "__main__":
    main()
