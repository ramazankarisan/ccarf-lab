import json
from typing import Any

import pytest
from mcp import Client
from mcp.types import TextContent

from ex4_research.server import (
    DOCS_DIR,
    WEB_DIR,
    build_server,
    doc_ids,
    mcp,
    resolve_doc,
    search,
    terms,
)

# Client(server) connects in-process: no subprocess, no network.


async def call(tool: str, arguments: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Call a tool and return (is_error, its JSON result)."""
    async with Client(mcp) as client:
        result = await client.call_tool(tool, arguments)
    block = result.content[0]
    assert isinstance(block, TextContent)
    data: dict[str, Any] = json.loads(block.text)
    return bool(result.is_error), data


# --- Corpus -------------------------------------------------------------------


def test_corpus_has_8_web_pages_and_3_reports():
    pages = [json.loads(p.read_text()) for p in WEB_DIR.glob("*.json")]
    docs = [json.loads(p.read_text()) for p in DOCS_DIR.glob("*.json")]
    assert len(pages) == 8
    assert len(docs) == 3
    keys = {"url", "title", "published", "category", "text"}
    assert all(set(page) == keys for page in pages)
    assert {page["category"] for page in pages} == {"current", "news archive"}
    assert all(set(doc) == {"doc_id", "title", "published", "text"} for doc in docs)


def test_each_report_file_is_named_after_its_doc_id():
    for path in DOCS_DIR.glob("*.json"):
        assert json.loads(path.read_text())["doc_id"] == path.stem


def test_corpus_has_two_different_2025_market_sizes():
    """Step 4 needs two credible sources that disagree, each with its scope stated."""
    nordwell = json.loads((DOCS_DIR / "nordwell-eu-outlook-2026.json").read_text())["text"]
    ebca = json.loads((DOCS_DIR / "ebca-circularity-report-2025.json").read_text())["text"]
    assert "EUR 1.9 billion" in nordwell
    assert "EU-27" in nordwell
    assert "EUR 3.4 billion" in ebca
    assert "United Kingdom" in ebca


# --- web_search ---------------------------------------------------------------


def test_terms_drops_stopwords_and_case():
    assert terms("How big is the EU Market?") == {"big", "eu", "market"}


def test_search_ranks_the_best_match_first():
    results = search("Nordwell EV battery recycling revenue 2025")
    assert results[0]["url"].startswith("https://nordwell-research.example.com/")
    assert len(results) <= 5


@pytest.mark.anyio
async def test_web_search_returns_pages_with_dates():
    is_error, data = await call("web_search", {"query": "plant capacity utilisation"})
    assert not is_error
    assert data["count"] == len(data["results"]) >= 1
    assert set(data["results"][0]) == {"url", "title", "published", "category", "text"}


@pytest.mark.anyio
async def test_web_search_keeps_the_news_archive_separate():
    _, current = await call("web_search", {"query": "black mass payables 2022 forecast"})
    _, archive = await call(
        "web_search", {"query": "black mass payables 2022 forecast", "category": "news archive"}
    )
    assert {page["category"] for page in current["results"]} == {"current"}
    assert {page["url"].split("/")[2] for page in archive["results"]} == {
        "metals-brief.example.com",
        "autotrend-daily.example.com",
    }


@pytest.mark.anyio
async def test_web_search_rejects_an_unknown_category():
    async with Client(mcp) as client:
        result = await client.call_tool("web_search", {"query": "x", "category": "blogs"})
    assert result.is_error


@pytest.mark.anyio
async def test_web_search_with_no_match_is_an_empty_list_not_an_error():
    is_error, data = await call("web_search", {"query": "zeppelin"})
    assert not is_error
    assert data == {"results": [], "count": 0}


@pytest.mark.anyio
async def test_web_search_does_not_return_reports():
    _, data = await call("web_search", {"query": "Scope and method"})
    assert all("doc_id" not in page for page in data["results"])


# --- load_document ------------------------------------------------------------


@pytest.mark.anyio
async def test_load_document_returns_a_report():
    is_error, data = await call("load_document", {"doc_id": "icm-feedstock-study-2026"})
    assert not is_error
    assert data["doc_id"] == "icm-feedstock-study-2026"
    assert data["published"] == "2026-06-30"


@pytest.mark.parametrize(
    "doc_id",
    [
        "../web/ev-recycling-boom-blog",  # a web page, by relative path
        "..",
        "../../server",
        "/etc/passwd",
        "nordwell-eu-outlook-2026.json",  # a file name, not an id
        "https://ebca.example.com/news/battery-circularity-figures-2025",  # a URL
        "corpus/docs/nordwell-eu-outlook-2026",
        "NORDWELL-EU-OUTLOOK-2026",
        "nordwell-eu-outlook-2026\n",
        "",
        "no-such-report",
    ],
)
@pytest.mark.anyio
async def test_load_document_rejects_anything_outside_the_report_library(doc_id):
    assert resolve_doc(doc_id) is None
    is_error, data = await call("load_document", {"doc_id": doc_id})
    assert is_error
    assert data["errorCategory"] == "validation"
    assert "Available doc_ids: " + ", ".join(doc_ids()) in data["description"]


def test_resolve_doc_rejects_a_symlink_that_leaves_the_library(tmp_path, monkeypatch):
    """Even a well-formed id must resolve to a file directly inside DOCS_DIR."""
    from ex4_research import server

    library = tmp_path / "docs"
    library.mkdir()
    (tmp_path / "secret.json").write_text("{}")
    (library / "innocent.json").symlink_to(tmp_path / "secret.json")
    monkeypatch.setattr(server, "DOCS_DIR", library.resolve())
    assert server.resolve_doc("innocent") is None


@pytest.mark.anyio
async def test_tool_descriptions_are_cleaned_docstrings():
    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert set(tools) == {"web_search", "load_document"}
    for tool in tools.values():
        assert tool.description
        assert "\n " not in tool.description


@pytest.mark.anyio
@pytest.mark.parametrize("tool", ["web_search", "load_document"])
async def test_a_server_built_for_one_tool_has_only_that_tool(tool):
    async with Client(build_server([tool])) as client:
        names = [t.name for t in (await client.list_tools()).tools]
    assert names == [tool]


def test_build_server_rejects_an_unknown_tool():
    with pytest.raises(ValueError, match="Unknown tools: fetch_url"):
        build_server(["fetch_url"])


# --- injected failures (EX4_FAIL, see faults.py) ------------------------------


@pytest.fixture
def failures(monkeypatch):
    """Install a failure plan for this test only."""
    from ex4_research import server
    from ex4_research.faults import parse_plan

    def install(value: str) -> None:
        monkeypatch.setattr(server, "FAILURES", parse_plan(value))

    return install


@pytest.mark.anyio
async def test_timeout_once_fails_the_first_call_with_a_retryable_error(failures):
    failures("timeout_once")
    is_error, error = await call("web_search", {"query": "EU recycling market"})
    assert is_error
    assert error["errorCategory"] == "transient"
    assert error["isRetryable"] is True
    assert "Retrying the same call may succeed" in error["description"]
    is_error, data = await call("web_search", {"query": "EU recycling market"})
    assert not is_error
    assert data["count"] >= 1


@pytest.mark.anyio
async def test_category_timeout_fails_every_search_of_that_category(failures):
    failures("category_timeout:news archive")
    for _ in range(3):
        is_error, error = await call("web_search", {"query": "x", "category": "news archive"})
        assert is_error
        assert error["isRetryable"] is True
        assert "'news archive' search backend" in error["description"]
    is_error, _ = await call("web_search", {"query": "EU recycling market"})
    assert not is_error


@pytest.mark.anyio
async def test_empty_once_returns_a_valid_empty_result_first(failures):
    failures("empty_once")
    is_error, data = await call("web_search", {"query": "EU recycling market"})
    assert not is_error
    assert data == {"results": [], "count": 0}
    _, data = await call("web_search", {"query": "EU recycling market"})
    assert data["count"] >= 1
