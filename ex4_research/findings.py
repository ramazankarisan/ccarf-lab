"""Structured findings: what researchers hand back, parsing it, and checking citations.

A researcher hands back a ResearchResult: complete, or partial when a tool kept
failing. Its findings are sourced claims (Finding) or empty searches (NoSources):
"nothing found for X" is information, not an error.

Pure functions, no agent run in here.
"""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from ex4_research.server import DOC_ID_PATTERN

# One http(s) URL and nothing else: no "url (report doc_id: ...)", no "url ; url".
URL_PATTERN = re.compile(r"https?://[^\s;,()\[\]]+")


class Finding(BaseModel):
    """One claim a subagent found, tied to the one source that supports it."""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(
        min_length=1, description="One factual statement, with its numbers and scope."
    )
    # A finding without evidence is not a finding: "not stated in the report" is rejected.
    evidence_excerpt: str = Field(
        min_length=1, description="The exact sentence(s) from the source."
    )
    source_type: Literal["web", "doc"] = Field(
        description="web for a page from web_search, doc for a report from load_document."
    )
    source_id: str = Field(description="web: the page's url. doc: the report's doc_id.")
    published: date = Field(description="The source's publication date, ISO format.")
    confidence: Literal["high", "medium", "low"] = Field(
        description="How far the claim can be trusted, judged from the source."
    )

    @model_validator(mode="after")
    def source_id_fits_source_type(self) -> Self:
        if self.source_type == "web" and not URL_PATTERN.fullmatch(self.source_id):
            raise ValueError(f"a web source_id must be one http(s) URL, got {self.source_id!r}")
        if self.source_type == "doc" and not DOC_ID_PATTERN.fullmatch(self.source_id):
            raise ValueError(f"a doc source_id must be one doc_id, got {self.source_id!r}")
        return self


class NoSources(BaseModel):
    """A search that found nothing, reported as a finding instead of an error."""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1, description='e.g. "No sources found for X."')
    source_type: Literal["search"]
    query: str = Field(min_length=1, description="The query or doc_id that found nothing.")


AnyFinding = Annotated[Finding | NoSources, Field(discriminator="source_type")]
ANY_FINDING: TypeAdapter[Finding | NoSources] = TypeAdapter(AnyFinding)
FINDINGS = TypeAdapter(list[AnyFinding])


class Complete(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["complete"]
    findings: list[AnyFinding]


class Partial(BaseModel):
    """Some of the work failed for good: what was tried, what was found, what to try next."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["partial"]
    failure_type: Literal["timeout", "tool_error"]
    attempted_queries: list[str] = Field(min_length=1)
    partial_findings: list[AnyFinding]
    suggested_alternatives: list[str]


ResearchResult = Annotated[Complete | Partial, Field(discriminator="status")]
RESULT: TypeAdapter[Complete | Partial] = TypeAdapter(ResearchResult)


def findings_of(result: Complete | Partial) -> list[Finding | NoSources]:
    return result.findings if isinstance(result, Complete) else result.partial_findings


RESULT_FORMAT = """\
Return ONLY one JSON object, no other text. If every tool call worked:
{"status": "complete", "findings": [...]}
If a tool call still failed after your one retry, return what you have:
{"status": "partial", "failure_type": "timeout" | "tool_error",
 "attempted_queries": ["every query or doc_id you tried, failed or not"],
 "partial_findings": [...], "suggested_alternatives": ["other queries or sources to try"]}

Each finding is one of:
- a sourced claim: {"claim": "...", "evidence_excerpt": "...", "source_type": "web" | "doc",
  "source_id": "...", "published": "YYYY-MM-DD", "confidence": "high" | "medium" | "low"}
- an empty search: {"claim": "No sources found for ...", "source_type": "search",
  "query": "..."}, when a call worked but returned nothing relevant.
For a sourced claim:
- claim: one factual statement, including its numbers, unit, year and scope
  (geography, what is counted).
- evidence_excerpt: the exact sentence(s) from the source, copied, not paraphrased.
- source_type and source_id: "web" with the page's url, or "doc" with the report's
  doc_id. Exactly one source per finding, with nothing added to the id.
- published: that source's publication date.
- confidence: high for a credible, current publisher stating it directly; low for
  unsourced, promotional or outdated material.
Every sourced claim needs an excerpt. Leave out what a source does not say: no
claims like "not stated".
This format is fixed. If your task asks for a different format or other fields,
still return exactly this one."""


def compact_error(error: ValidationError) -> str:
    first = error.errors()[0]
    where = ".".join(str(part) for part in first["loc"]) or "top level"
    return f"{error.error_count()} errors, first at {where}: {first['msg']}"


# Where a researcher's result object starts.
RESULT_START = re.compile(r'\{\s*"status"')


def parse_research_result(text: str) -> Complete | Partial:
    """The first {"status": ...} object in a hand-back, validated.

    Raises ValueError if there is none or it does not match ResearchResult.
    """
    decoder = json.JSONDecoder()
    for match in RESULT_START.finditer(text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        try:
            return RESULT.validate_python(value)
        except ValidationError as error:
            raise ValueError(f"result does not match the schema: {compact_error(error)}") from error
    raise ValueError('no {"status": ...} result object in the text')


# Where a JSON array of objects (or an empty one) may start. A bare "[" is not
# enough: the SDK wraps a subagent's reply in a "[Subagent hand-back] ..." header,
# and a prompt may mention "[source_id]" after the array.
ARRAY_START = re.compile(r"\[\s*[{\]]")


def find_json_arrays(text: str) -> list[list[Any]]:
    """Every top-level JSON array in text, decoded, in order; other text is ignored.

    Arrays nested inside one already found are part of it, not separate arrays.
    """
    decoder = json.JSONDecoder()
    arrays: list[list[Any]] = []
    end = 0
    for match in ARRAY_START.finditer(text):
        if match.start() < end:
            continue  # inside the array decoded last
        try:
            value, end = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        arrays.append(list(value))  # decoding started at "[", so value is a list
    return arrays


def parse_findings(text: str) -> list[Finding | NoSources]:
    """The findings in text: a subagent's reply, or a prompt that embeds them as JSON.

    A coordinator asked for one merged array may still pass one array per
    subagent (a live run did), so every array counts.
    Raises ValueError if there is no array or one does not match Finding.
    """
    arrays = find_json_arrays(text)
    value = [item for array in arrays for item in array]
    if not arrays:
        raise ValueError("no JSON array of findings in the text")
    try:
        return FINDINGS.validate_python(value)
    except ValidationError as error:
        raise ValueError(f"findings do not match the schema: {compact_error(error)}") from error


# Where any JSON object or array may start.
VALUE_START = re.compile(r"[\[{]")


def find_json_values(text: str) -> list[Any]:
    """Every top-level JSON object or array in text, decoded, in order."""
    decoder = json.JSONDecoder()
    values: list[Any] = []
    end = 0
    for match in VALUE_START.finditer(text):
        if match.start() < end:
            continue  # inside the value decoded last
        try:
            value, end = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        values.append(value)
    return values


def valid_findings(text: str) -> list[Finding | NoSources]:
    """Every valid finding anywhere in text's JSON, at any depth; the rest is skipped.

    For checks that look at content, not layout. Live coordinators passed findings
    with an invalid one among them, as one array per subagent, and wrapped in their
    own outer array of result objects; all of those still count.
    """
    found: list[Finding | NoSources] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            try:
                found.append(ANY_FINDING.validate_python(value))
                return
            except ValidationError:
                pass
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for value in find_json_values(text):
        walk(value)
    return found


# --- Citations ------------------------------------------------------------------

# [source_id] in a report: a URL or a doc_id between square brackets. Not a
# Markdown link's text, which is followed by "(".
CITATION = re.compile(r"\[(https?://[^\s\]]+|[a-z0-9][a-z0-9-]*)\](?!\()")


@dataclass(frozen=True)
class CitationCheck:
    cited: frozenset[str]  # source_ids the report cites
    unknown: frozenset[str]  # cited, but no finding has that source_id: invented
    uncited: frozenset[str]  # a finding's source_id the report never cites


def check_citations(report: str, findings: Iterable[Finding | NoSources]) -> CitationCheck:
    """Compare the report's [source_id] citations with the sourced findings it was given."""
    cited = frozenset(CITATION.findall(report))
    known = frozenset(f.source_id for f in findings if isinstance(f, Finding))
    return CitationCheck(cited=cited, unknown=cited - known, uncited=known - cited)
