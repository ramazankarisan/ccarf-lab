"""A multi-turn support conversation on ClaudeSDKClient, with case facts.

query() in agent.py runs one message and ends. ClaudeSDKClient keeps one
conversation open across many messages, like a chat window.

From the second turn on, every user message is sent with the current
<case_facts> block in front of it, so the model always has the exact IDs,
amounts and dates however long the history gets.

Chat interactively:
    uv run python -m ex1_support_agent.session
"""

import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

import anyio
from claude_agent_sdk import ClaudeSDKClient, ClaudeSDKError, Message
from dotenv import load_dotenv

from ex1_support_agent.agent import SDKRun, build_options, collect, format_sdk_run
from ex1_support_agent.case_facts import CaseFacts, render


class ChatClient(Protocol):
    """The part of ClaudeSDKClient a session uses, so tests can pass a fake."""

    async def query(self, prompt: str) -> None: ...
    def receive_response(self) -> AsyncIterator[Message]: ...


class SupportSession:
    """One customer conversation: one client, one CaseFacts, many turns."""

    def __init__(self, client: ChatClient, facts: CaseFacts) -> None:
        self.client = client
        self.facts = facts
        self.turns = 0

    def build_prompt(self, user_text: str) -> str:
        """The first turn goes as typed; later turns get the case facts in front."""
        if self.turns == 0:
            return user_text
        return f"{render(self.facts)}\n\n{user_text}"

    async def send(self, user_text: str) -> SDKRun:
        """Send one customer message and return this turn's tool calls and result."""
        prompt = self.build_prompt(user_text)
        self.turns += 1
        await self.client.query(prompt)
        # receive_response() stops after this turn's ResultMessage.
        return collect([message async for message in self.client.receive_response()])


@asynccontextmanager
async def open_session(model: str) -> AsyncIterator[SupportSession]:  # pragma: no cover - live
    """Connect a client whose recorder hook writes into this session's own facts."""
    facts = CaseFacts()
    async with ClaudeSDKClient(options=build_options(model, facts=facts)) as client:
        yield SupportSession(client, facts)


# --- CLI ----------------------------------------------------------------------

QUIT_WORDS = {"quit", "exit"}
HELP = "Type a message as the customer. 'facts' shows the case facts, 'quit' ends the chat."


async def chat(model: str) -> None:  # pragma: no cover - interactive, live API
    print(HELP)
    async with open_session(model) as session:
        while True:
            try:
                # input() blocks; running it in a thread keeps the SDK's tasks alive.
                text = (await anyio.to_thread.run_sync(input, "\ncustomer> ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in QUIT_WORDS:
                break
            if text.lower() == "facts":
                print(render(session.facts))
                continue
            if text:
                print(format_sdk_run(await session.send(text)))


def cli() -> None:  # pragma: no cover - interactive, live API
    load_dotenv()
    # force: importing the agent loads the MCP server, which set logging to INFO.
    logging.basicConfig(level=logging.WARNING, force=True)
    try:
        anyio.run(chat, os.environ["MODEL"])
    except ClaudeSDKError as error:
        print(f"Session stopped: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
