"""Step 4 variants: A is Step 3 unchanged, B adds field descriptions and nothing else."""

import json
import re
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from ex2_extraction.ground_truth import load_ground_truth
from ex2_extraction.schema import Category, Invoice, LineItem, json_schema, strip_descriptions
from ex2_extraction.variants import EXAMPLES, VARIANTS, A, B, C, Variant

STEP3_SCHEMA = json.loads(Path("tests/fixtures/ex2_step3_schema.json").read_text(encoding="utf-8"))


def schema_of(variant: Variant) -> dict[str, Any]:
    return cast(dict[str, Any], variant.tool["input_schema"])


def descriptions(schema: Any) -> list[str]:
    """Every description annotation in a schema (not the LineItem field named description)."""
    found: list[str] = []
    if isinstance(schema, list):
        for item in schema:
            found += descriptions(item)
    elif isinstance(schema, dict):
        for key, value in schema.items():
            if key == "properties":
                for prop in value.values():
                    found += descriptions(prop)
            elif key == "description" and isinstance(value, str):
                found.append(value)
            else:
                found += descriptions(value)
    return found


def test_variant_a_is_exactly_the_step_3_schema_and_prompt():
    from ex2_extraction.extractor import RECORD_INVOICE, SYSTEM_PROMPT

    assert schema_of(A) == STEP3_SCHEMA
    assert A.tool == RECORD_INVOICE
    assert A.system_prompt == SYSTEM_PROMPT
    assert descriptions(schema_of(A)) == []


def test_variant_b_changes_only_the_descriptions():
    """One change at a time: same prompt, and the schema minus descriptions is A's."""
    assert B.system_prompt == A.system_prompt
    assert strip_descriptions(schema_of(B)) == schema_of(A)
    assert B.tool["name"] == A.tool["name"]
    assert B.tool["description"] == A.tool["description"]


@pytest.mark.parametrize(
    ("model", "where"),
    [(Invoice, ("properties",)), (LineItem, ("$defs", "LineItem", "properties"))],
)
def test_every_field_has_a_description_in_variant_b(model: type[BaseModel], where: tuple[str, ...]):
    properties: dict[str, Any] = schema_of(B)
    for key in where:
        properties = properties[key]

    for field in model.model_fields:
        assert properties[field].get("description"), field


def test_stripping_keeps_the_line_item_field_called_description():
    item_properties = json_schema(with_descriptions=False)["$defs"]["LineItem"]["properties"]

    assert "description" in item_properties
    assert item_properties["description"] == {"title": "Description", "type": "string"}


def test_the_category_description_defines_all_five_values():
    text = schema_of(B)["properties"]["category"]["description"]

    for value in Category:
        assert f"{value.value}:" in text


@pytest.mark.parametrize(
    ("field", "rule"),
    [
        ("line_items", "use []"),
        ("vendor_tax_id", "without its label (EIN, VAT No, USt-IdNr., Steuernummer)"),
        ("vendor_tax_id", "Never the buyer's"),
        ("stated_subtotal", "A document with a single total has a null subtotal"),
        ("invoice_date", "YYYY-MM-DD"),
        ("invoice_date", "decide from the document's cues"),
        ("stated_total", "never add up the items"),
    ],
)
def test_the_key_descriptions_keep_their_rules(field: str, rule: str):
    assert rule in schema_of(B)["properties"][field]["description"]


def test_nullable_fields_stay_required_with_descriptions():
    assert set(schema_of(B)["required"]) == set(Invoice.model_fields)


# --- Leakage: nothing from the corpus in what the model reads before the document ------


def corpus_terms() -> set[str]:
    """Vendors, numbers, products and tax IDs from the ground truth, plus "catering":
    whether the model generalizes to it is what variant C measures."""
    terms = {"catering"}
    for entry in load_ground_truth().values():
        invoice = entry.invoice
        terms |= {invoice.vendor_name, invoice.invoice_number}
        terms |= {invoice.vendor_tax_id} if invoice.vendor_tax_id else set()
        amounts = [invoice.stated_subtotal, invoice.stated_tax, invoice.stated_total]
        for item in invoice.line_items:
            terms.add(item.description)
            amounts += [item.unit_price, item.amount]
        terms |= {str(amount) for amount in amounts if amount is not None and amount >= 10}
    return terms


def appears(term: str, text: str) -> bool:
    """A whole-token match, ignoring case: 50.00 is not found inside 950.00."""
    return re.search(rf"(?<![\w.]){re.escape(term)}(?![\w]|\.\d)", text, re.IGNORECASE) is not None


def model_facing_text(variant_name: str) -> str:
    variant = VARIANTS[variant_name]
    return "\n".join([variant.system_prompt, *descriptions(schema_of(variant))])


@pytest.mark.parametrize("variant_name", sorted(VARIANTS))
def test_no_corpus_term_leaks_into_prompts_or_descriptions(variant_name: str):
    text = model_facing_text(variant_name)

    leaked = [term for term in corpus_terms() if appears(term, text)]
    assert leaked == []


def test_the_leak_check_would_catch_a_leak():
    assert {"Café Lindgren Catering", "1203.00", "catering"} <= corpus_terms()
    assert appears("catering", "food and Catering for events")
    assert appears("50.00", "a fee of 50.00 EUR")
    assert not appears("50.00", "GBP 950.00")
    assert not appears("50.00", "50.001")


# --- Variant C: B plus three few-shot examples ----------------------------------------


def test_variant_c_changes_only_the_prompt():
    """One change from B: the same schema with descriptions, the rules unchanged, the
    examples appended."""
    assert C.tool == B.tool
    assert C.system_prompt.startswith(B.system_prompt)
    assert C.system_prompt == f"{B.system_prompt}\n\n{EXAMPLES}"


@pytest.mark.parametrize(("name", "count"), [("A", 0), ("B", 0), ("C", 3)])
def test_only_variant_c_has_examples_and_exactly_three(name: str, count: int):
    prompt = VARIANTS[name].system_prompt

    assert prompt.count("<example>") == count
    assert prompt.count("</example>") == count


@pytest.mark.parametrize(
    "decisive",
    [
        "line_items: []\nstated_subtotal: null",  # a receipt with only a total
        "category: other\ncategory_detail: venue hire",  # outside the enum
        "vendor_tax_id: GB000000555",  # the seller's ID, without its label
    ],
)
def test_each_example_targets_one_ambiguous_case(decisive: str):
    assert decisive in EXAMPLES


def test_every_example_has_a_snippet_fields_and_a_reason():
    blocks = EXAMPLES.split("<example>")[1:]

    for block in blocks:
        for tag in ("snippet", "decisive_fields", "reason"):
            assert f"<{tag}>" in block
            assert f"</{tag}>" in block


def test_the_seller_example_does_not_show_the_label_in_the_answer():
    fields = EXAMPLES.split("<decisive_fields>")[3].split("</decisive_fields>")[0]

    assert "VAT" not in fields
    assert "GB000000777" not in fields  # the buyer's number
