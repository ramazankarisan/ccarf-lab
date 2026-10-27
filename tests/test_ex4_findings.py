import json
from typing import Any

import pytest
from pydantic import ValidationError

from ex4_research.findings import (
    RESULT_FORMAT,
    Complete,
    Finding,
    NoSources,
    Partial,
    check_citations,
    find_json_arrays,
    findings_of,
    parse_findings,
    parse_research_result,
    valid_findings,
)

DOC_FINDING: dict[str, Any] = {
    "claim": "EU-27 EV recycling revenue was EUR 1.9 billion in 2025.",
    "evidence_excerpt": "generated EUR 1.9 billion of revenue in 2025",
    "source_type": "doc",
    "source_id": "nordwell-eu-outlook-2026",
    "published": "2026-02-11",
    "confidence": "high",
}
WEB_FINDING: dict[str, Any] = {
    **DOC_FINDING,
    "source_type": "web",
    "source_id": "https://ebca.example.com/news/battery-circularity-figures-2025",
    "confidence": "medium",
}
FINDING_JSON = json.dumps([DOC_FINDING])

# How the SDK hands a subagent's reply back to the coordinator (shortened from a live run).
HAND_BACK = (
    "[Subagent hand-back] The text below is the final report of a subagent. "
    "The report follows:\n  {body}\nagentId: a4f17a98b11639ead\n"
    "<usage>subagent_tokens: 13396\ntool_uses: 7</usage>"
)


# --- Schema -------------------------------------------------------------------


@pytest.mark.parametrize("data", [DOC_FINDING, WEB_FINDING])
def test_a_finding_has_one_typed_source(data):
    finding = Finding.model_validate(data)
    assert finding.source_type == data["source_type"]
    assert finding.source_id == data["source_id"]


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"source_type": "blog"}, "source_type"),
        ({"source_type": "web"}, "a web source_id must be one http"),
        ({"source_type": "doc", "source_id": WEB_FINDING["source_id"]}, "a doc source_id"),
        # Both shapes came back from live runs before source was split:
        (
            {"source_type": "web", "source_id": WEB_FINDING["source_id"] + " (report doc_id: x)"},
            "must be one http",
        ),
        (
            {
                "source_type": "web",
                "source_id": "https://a.example.com/x ; https://b.example.com/y",
            },
            "must be one http",
        ),
        ({"source_id": "../web/blog"}, "a doc source_id"),
        ({"source": "nordwell-eu-outlook-2026"}, "Extra inputs are not permitted"),
        ({"evidence_excerpt": ""}, "at least 1 character"),
        ({"claim": ""}, "at least 1 character"),
        ({"confidence": "certain"}, "confidence"),
        ({"published": "last year"}, "published"),
    ],
)
def test_finding_rejects(changes, error):
    with pytest.raises(ValidationError, match=error):
        Finding.model_validate({**DOC_FINDING, **changes})


def test_every_field_is_required():
    for field in DOC_FINDING:
        data = {key: value for key, value in DOC_FINDING.items() if key != field}
        with pytest.raises(ValidationError, match=field):
            Finding.model_validate(data)


# --- Parsing ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        FINDING_JSON,
        f"```json\n{FINDING_JSON}\n```",
        f"Here you go:\n```\n{FINDING_JSON}\n```",
        HAND_BACK.format(body=FINDING_JSON),
        HAND_BACK.format(body=f"```json\n{FINDING_JSON}\n```"),
        HAND_BACK.format(body=json.dumps([DOC_FINDING], indent=2)),
        # A synthesizer prompt: question, the findings, then instructions with brackets.
        f"Question: how big?\nFindings:\n{FINDING_JSON}\nCite as [source_id].",
    ],
)
def test_parse_findings_finds_the_array_wherever_it_is(text):
    [finding] = parse_findings(text)
    assert isinstance(finding, Finding)
    assert finding.source_id == "nordwell-eu-outlook-2026"
    assert finding.published.isoformat() == "2026-02-11"


@pytest.mark.parametrize("text", ["[]", "[ ]", HAND_BACK.format(body="[]")])
def test_parse_findings_accepts_an_empty_list(text):
    assert parse_findings(text) == []


def test_find_json_arrays_skips_brackets_that_are_not_json():
    assert find_json_arrays('[{not json}] then [ {"a": 1} ]') == [[{"a": 1}]]


def test_find_json_arrays_returns_top_level_arrays_only():
    assert find_json_arrays('[{"a": [{"b": 1}]}] and [] and [{"c": 2}]') == [
        [{"a": [{"b": 1}]}],
        [],
        [{"c": 2}],
    ]


def test_parse_findings_reads_one_array_per_subagent():
    """A live coordinator passed the synthesizer one array per subagent, not one merged array."""
    text = f"Web findings:\n{json.dumps([WEB_FINDING])}\nReport findings:\n{FINDING_JSON}"
    assert [f.source_type for f in parse_findings(text)] == ["web", "doc"]


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("I found nothing.", "no JSON array"),
        ("[Subagent hand-back] nothing found", "no JSON array"),
        ('{"claim": "an object, not a list"}', "no JSON array"),
        (FINDING_JSON.replace('"doc"', '"web"'), "1 errors, first at 0.web: Value error"),
        (json.dumps([{**DOC_FINDING, "value": 1.9}]), "first at 0.doc.value: Extra inputs"),
    ],
)
def test_parse_findings_rejects_what_does_not_match(text, error):
    with pytest.raises(ValueError, match=error):
        parse_findings(text)


# --- Citations ----------------------------------------------------------------


def findings() -> list[Finding]:
    return [Finding.model_validate(DOC_FINDING), Finding.model_validate(WEB_FINDING)]


def test_check_citations_counts_known_sources():
    report = (
        "EUR 1.9 billion [nordwell-eu-outlook-2026]. EUR 3.4 billion "
        "[https://ebca.example.com/news/battery-circularity-figures-2025]."
    )
    check = check_citations(report, findings())
    assert check.cited == {"nordwell-eu-outlook-2026", WEB_FINDING["source_id"]}
    assert check.unknown == frozenset()
    assert check.uncited == frozenset()


def test_check_citations_flags_invented_and_unused_sources():
    check = check_citations("EUR 2 billion [made-up-report-2025].", findings())
    assert check.unknown == {"made-up-report-2025"}
    assert check.uncited == {"nordwell-eu-outlook-2026", WEB_FINDING["source_id"]}


def test_markdown_link_text_is_not_a_citation():
    check = check_citations("See [nordwell](https://nordwell.example.com).", findings())
    assert check.cited == frozenset()


# --- research results (Step 3) ------------------------------------------------

NO_SOURCES: dict[str, Any] = {
    "claim": "No sources found for EU tonnage recycled in 2023.",
    "source_type": "search",
    "query": "EU battery tonnage recycled 2023",
}
PARTIAL: dict[str, Any] = {
    "status": "partial",
    "failure_type": "timeout",
    "attempted_queries": ["battery recycling forecast 2022", "battery recycling forecast 2022"],
    "partial_findings": [WEB_FINDING, NO_SOURCES],
    "suggested_alternatives": ["search 'current' for 2025 growth figures"],
}


def complete(*findings: dict[str, Any]) -> str:
    return json.dumps({"status": "complete", "findings": list(findings)})


def test_a_complete_result_holds_findings_of_both_kinds():
    result = parse_research_result(HAND_BACK.format(body=complete(DOC_FINDING, NO_SOURCES)))
    assert isinstance(result, Complete)
    doc, empty = findings_of(result)
    assert isinstance(doc, Finding)
    assert isinstance(empty, NoSources)
    assert empty.query == "EU battery tonnage recycled 2023"


def test_a_partial_result_says_what_failed_was_tried_and_could_be_tried():
    result = parse_research_result(f"```json\n{json.dumps(PARTIAL)}\n```")
    assert isinstance(result, Partial)
    assert result.failure_type == "timeout"
    assert len(result.attempted_queries) == 2
    assert len(findings_of(result)) == 2
    assert result.suggested_alternatives


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"failure_type": "network"}, "failure_type"),
        ({"attempted_queries": []}, "attempted_queries"),
        ({"suggested_alternatives": None}, "suggested_alternatives"),
        ({"findings": []}, "Extra inputs"),
    ],
)
def test_partial_result_rejects(changes, error):
    with pytest.raises(ValueError, match=error):
        parse_research_result(json.dumps({**PARTIAL, **changes}))


@pytest.mark.parametrize(
    ("text", "error"),
    [
        (FINDING_JSON, 'no {"status": ...} result object'),
        ('{"status": "done", "findings": []}', "does not match the schema"),
        ('{"status": broken} {"status": "complete", "findings": []}', ""),
    ],
)
def test_parse_research_result_needs_a_valid_status_object(text, error):
    if error:
        with pytest.raises(ValueError, match=error):
            parse_research_result(text)
    else:
        assert parse_research_result(text) == Complete(status="complete", findings=[])


@pytest.mark.parametrize(
    "data",
    [{**NO_SOURCES, "claim": ""}, {**NO_SOURCES, "query": ""}, {**NO_SOURCES, "source_id": "x"}],
)
def test_no_sources_rejects(data):
    with pytest.raises(ValidationError):
        NoSources.model_validate(data)


def test_valid_findings_reads_both_kinds_inside_result_objects():
    found = valid_findings(json.dumps(PARTIAL) + complete(DOC_FINDING))
    assert [type(f) for f in found] == [Finding, NoSources, Finding]


def test_citations_ignore_no_sources_findings():
    check = check_citations("x", [NoSources.model_validate(NO_SOURCES)])
    assert check.uncited == frozenset()


def test_result_format_describes_both_statuses_and_both_kinds():
    for text in ('"status": "complete"', '"status": "partial"', '"source_type": "search"'):
        assert text in RESULT_FORMAT


def test_find_json_values_skips_what_does_not_parse():
    from ex4_research.findings import find_json_values

    assert find_json_values('{not json} then {"a": [1, {"b": 2}]} and [3]') == [
        {"a": [1, {"b": 2}]},
        [3],
    ]
