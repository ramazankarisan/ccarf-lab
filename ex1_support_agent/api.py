"""HTTP API for the web app: both agents behind one chat endpoint.

    uv run uvicorn ex1_support_agent.api:create_app_from_env --factory --port 8000

"client" runs loop.py, the by-hand loop on the Client SDK.
"agent" runs session.py on the Agent SDK, with hooks and case facts.
Both get the same system prompt, MCP tools and backend data, so the only
difference is what the code guarantees on top of the prompt.
"""

import logging
import os
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import Any, Literal

from anthropic import APIError, AsyncAnthropic
from anthropic.types import MessageParam
from claude_agent_sdk import ClaudeSDKError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from mcp import Client
from pydantic import BaseModel, Field

from ex1_support_agent import backend
from ex1_support_agent.loop import ModelClient, ToolCall, UnexpectedStopReasonError, run_agent
from ex1_support_agent.prompts import SYSTEM_PROMPT
from ex1_support_agent.server import mcp
from ex1_support_agent.session import SupportSession, open_session

logger = logging.getLogger(__name__)

Mode = Literal["client", "agent"]
OpenSession = Callable[[], AbstractAsyncContextManager[SupportSession]]


class ChatRequest(BaseModel):
    mode: Mode
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None  # None starts a new conversation


class ToolCallOut(BaseModel):
    name: str
    input: Any
    is_error: bool


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    tool_calls: list[ToolCallOut]
    status: str
    usage: str


def _tool_calls(calls: list[ToolCall]) -> list[ToolCallOut]:
    return [ToolCallOut(name=c.name, input=c.input, is_error=c.is_error) for c in calls]


class ChatService:
    """Holds every open conversation: message histories and Agent SDK sessions."""

    def __init__(
        self,
        *,
        model: str,
        model_client: ModelClient,
        tools_client: Client,
        open_agent_session: OpenSession,
    ) -> None:
        self.model = model
        self.model_client = model_client
        self.tools_client = tools_client
        self.open_agent_session = open_agent_session
        self.histories: dict[str, list[MessageParam]] = {}
        self.sessions: dict[str, SupportSession] = {}
        self.stack = AsyncExitStack()  # owns the open Agent SDK sessions

    async def chat(self, request: ChatRequest) -> ChatResponse:
        if request.mode == "client":
            return await self._chat_client(request)
        return await self._chat_agent(request)

    async def _chat_client(self, request: ChatRequest) -> ChatResponse:
        conversation_id = _known_or_new(request.conversation_id, self.histories)
        agent_run = await run_agent(
            request.message,
            model_client=self.model_client,
            tools_client=self.tools_client,
            model=self.model,
            history=self.histories.get(conversation_id, []),
            system_prompt=SYSTEM_PROMPT,
        )
        self.histories[conversation_id] = agent_run.messages
        usage = agent_run.usage
        return ChatResponse(
            conversation_id=conversation_id,
            reply=agent_run.final_text,
            tool_calls=_tool_calls(agent_run.tool_calls),
            status=agent_run.status,
            usage=f"tokens in {usage.input_tokens:,} / out {usage.output_tokens:,}",
        )

    async def _chat_agent(self, request: ChatRequest) -> ChatResponse:
        conversation_id = _known_or_new(request.conversation_id, self.sessions)
        if conversation_id not in self.sessions:
            session = await self.stack.enter_async_context(self.open_agent_session())
            self.sessions[conversation_id] = session
        sdk_run = await self.sessions[conversation_id].send(request.message)
        result = sdk_run.result
        cost = result.total_cost_usd if result else None
        return ChatResponse(
            conversation_id=conversation_id,
            reply=(result.result if result else None) or "",
            tool_calls=_tool_calls(sdk_run.tool_calls),
            status=result.subtype if result else "no result",
            usage=f"${cost:.4f}" if cost is not None else "cost n/a",
        )

    async def reset(self) -> None:
        """Close every conversation and restore the seed data."""
        await self.stack.aclose()
        self.stack = AsyncExitStack()
        self.histories.clear()
        self.sessions.clear()
        backend.reset()


def _known_or_new(conversation_id: str | None, known: dict[str, Any]) -> str:
    if conversation_id is None:
        return uuid.uuid4().hex
    if conversation_id not in known:
        raise HTTPException(404, "Unknown conversation. Start a new one.")
    return conversation_id


def create_app(
    *, model: str, model_client: ModelClient, open_agent_session: OpenSession
) -> FastAPI:
    """The API with its clients passed in, so tests can use fakes."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with Client(mcp) as tools_client:
            app.state.chat = ChatService(
                model=model,
                model_client=model_client,
                tools_client=tools_client,
                open_agent_session=open_agent_session,
            )
            yield
            await app.state.chat.stack.aclose()

    app = FastAPI(title="ccarf-lab support agent", lifespan=lifespan)

    @app.get("/api/data")
    async def data() -> dict[str, Any]:
        return backend.snapshot()

    @app.post("/api/chat")
    async def chat(request: ChatRequest) -> ChatResponse:
        try:
            response: ChatResponse = await app.state.chat.chat(request)
        except (APIError, ClaudeSDKError, UnexpectedStopReasonError) as error:
            # Details go to the server log; the browser gets no internals.
            logger.exception("Agent call failed")
            raise HTTPException(502, f"The agent failed ({type(error).__name__}).") from error
        return response

    @app.post("/api/reset")
    async def reset() -> dict[str, Any]:
        await app.state.chat.reset()
        return backend.snapshot()

    return app


def create_app_from_env() -> FastAPI:  # pragma: no cover - real clients, used by uvicorn
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, force=True)
    model = os.environ["MODEL"]
    return create_app(
        model=model,
        model_client=AsyncAnthropic(),  # reads ANTHROPIC_API_KEY from the environment
        # In-process MCP server: both modes share one backend, so the data
        # panel sees every refund and ticket.
        open_agent_session=lambda: open_session(model, in_process=True),
    )
