"""MCP server exposing the fake research corpus as two tools.

Run over stdio:  uv run python -m ex4_research.server [TOOL ...]

With tool names, the server registers only those tools, so each subagent can
get a server that has its own tool and nothing else. Without, it has both.

Same pattern as ex1_support_agent.server: mcp 2.x MCPServer, docstrings as tool
descriptions, structured isError results the agent can act on.
"""

import functools
import json
import os
import re
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent
from pydantic import Field

from ex1_support_agent.server import tool_description
from ex4_research.faults import FailurePlan, SearchTimeout, plan_from_env

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
WEB_DIR = CORPUS_DIR / "web"
DOCS_DIR = CORPUS_DIR / "docs"

# Lowercase letters, digits and hyphens: no dots, slashes or drive letters, so a
# doc_id can never name a path. The containment check in load_document is a
# second line of defense, not the only one.
DOC_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,99}$")
MAX_RESULTS = 5
STOPWORD_TEXT = (
    "a an and are as at be by for from how in is it of on or the to was what when which who "
    "why will with"
)
STOPWORDS = frozenset(STOPWORD_TEXT.split())
Category = Literal["current", "news archive"]

# Injected failures for this process (see faults.py). Tests replace it.
FAILURES: FailurePlan = FailurePlan()

TOOLS: dict[str, Callable[..., dict[str, Any] | CallToolResult]] = {}


class UnknownDocument(Exception):
    """load_document got a doc_id that is not in the report library."""


def error_result(category: str, description: str, *, retryable: bool = False) -> CallToolResult:
    """An isError tool result: what went wrong, whether a retry can help, what to do next."""
    payload = {"errorCategory": category, "isRetryable": retryable, "description": description}
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))], is_error=True
    )


def research_tool[**P](
    fn: Callable[P, dict[str, Any]],
) -> Callable[P, dict[str, Any] | CallToolResult]:
    """Collect fn as a research tool; build_server registers it. UnknownDocument becomes isError."""

    @functools.wraps(fn)  # keeps name, docstring and signature for the input schema
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> dict[str, Any] | CallToolResult:
        try:
            return fn(*args, **kwargs)
        except UnknownDocument as exc:
            return error_result("validation", str(exc))
        except SearchTimeout as exc:
            return error_result(
                "transient",
                f"web_search timed out: {exc}. Retrying the same call may succeed.",
                retryable=True,
            )

    TOOLS[fn.__name__] = wrapper
    return wrapper


def read_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def web_pages() -> list[dict[str, Any]]:
    return [read_json(path) for path in sorted(WEB_DIR.glob("*.json"))]


def doc_ids() -> list[str]:
    return sorted(path.stem for path in DOCS_DIR.glob("*.json"))


def terms(text: str) -> set[str]:
    """Lowercase words of text, without stopwords: 'EV battery' -> {'ev', 'battery'}."""
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS}


def search(query: str, category: str = "current") -> list[dict[str, Any]]:
    """Pages of category that share at least one term with query, best match first."""
    wanted = terms(query)
    scored = []
    for page in web_pages():
        if page["category"] != category:
            continue
        score = len(wanted & terms(f"{page['title']} {page['text']}"))
        if score:
            scored.append((score, page))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["url"]))  # ties in a stable order
    return [page for _, page in scored[:MAX_RESULTS]]


def resolve_doc(doc_id: str) -> Path | None:
    """The file for doc_id inside DOCS_DIR, or None if the id is malformed or unknown."""
    if not DOC_ID_PATTERN.fullmatch(doc_id):
        return None
    path = (DOCS_DIR / f"{doc_id}.json").resolve()
    if path.parent != DOCS_DIR or not path.is_file():
        return None
    return path


@research_tool
def web_search(
    query: Annotated[
        str,
        Field(
            description="Keywords to search for, e.g. 'EU battery recycling market 2025'.",
            min_length=1,
        ),
    ],
    category: Annotated[
        Category,
        Field(description="'current' for recent pages, 'news archive' for older news coverage."),
    ] = "current",
) -> dict[str, Any]:
    """Search the web for pages about a topic and return the best matches.

    Returns {"results": [...], "count": n} with up to 5 pages, best match
    first. Each page has url, title, published (ISO date), category and its
    full text, so you do not need to open it separately. Matching is by
    keywords, so use a few specific words rather than a sentence. category
    "current" (the default) searches recent pages; "news archive" searches
    older news coverage, for history and earlier forecasts. count 0 means
    nothing matched: that is a normal answer, not an error, so try other
    keywords. A timeout returns an error with isRetryable true: retrying the
    same call may succeed. Pages are of mixed quality: check who published each
    one and when before relying on it. Some pages mention a longer report by
    doc_id; this tool cannot open those.
    """
    if FAILURES.before_search(query, category):
        return {"results": [], "count": 0}
    results = search(query, category)
    return {"results": results, "count": len(results)}


@research_tool
def load_document(
    doc_id: Annotated[
        str,
        Field(description="Report ID, e.g. 'acme-market-report-2024'. Not a URL or file path."),
    ],
) -> dict[str, Any]:
    """Load the full text of one research report from the report library.

    Pass a doc_id, for example "acme-market-report-2024": lowercase letters,
    digits and hyphens only. Returns doc_id, title, published (ISO date) and
    text. This tool reads only the report library: it cannot open URLs, web
    pages or files. An unknown or malformed doc_id returns an error that lists
    the available doc_ids; pick one from that list instead of guessing again.
    """
    path = resolve_doc(doc_id)
    if path is None:
        raise UnknownDocument(
            f"No report with doc_id {doc_id!r}. Available doc_ids: {', '.join(doc_ids())}."
        )
    return read_json(path)


def build_server(tool_names: Iterable[str] = ()) -> MCPServer:
    """A server with the named tools, or with every tool if no names are given."""
    names = list(tool_names) or list(TOOLS)
    unknown = [name for name in names if name not in TOOLS]
    if unknown:
        raise ValueError(f"Unknown tools: {', '.join(unknown)}. Known: {', '.join(TOOLS)}.")
    server = MCPServer("research-corpus")
    for name in names:
        server.tool(description=tool_description(TOOLS[name]))(TOOLS[name])
    return server


mcp = build_server()


if __name__ == "__main__":
    FAILURES = plan_from_env(os.environ)
    build_server(sys.argv[1:]).run()  # stdio transport by default
