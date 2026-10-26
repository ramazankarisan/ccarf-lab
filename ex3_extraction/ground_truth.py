"""The corpus and its ground truth: what each document really says.

Ground truth records the document, not the arithmetic. For a document whose line
items don't add up, stated_total is the printed total; spotting the gap is code's
job in Step 3. Steps 2 to 6 score every extraction against these entries.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ex3_extraction.schema import Invoice

CORPUS_DIR = Path(__file__).parent / "corpus"
GROUND_TRUTH_FILE = CORPUS_DIR / "ground_truth.json"

EDGE_CASES = {
    1: "clean table, everything present",
    2: "narrative email",
    3: "terse receipt",
    4: "no vendor tax ID",
    5: "no due date",
    6: "category outside the enum",
    7: "line items don't sum to the stated total",
    8: "German number and date format",
    9: "subtotal, VAT and total all printed",
    10: "no total printed",
    11: "ambiguous date, settled by a hint",
    12: "buyer's VAT number but no vendor tax ID, many reference numbers",
}


class GroundTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_cases: list[int] = Field(min_length=1)
    invoice: Invoice


def load_ground_truth() -> dict[str, GroundTruth]:
    """Every corpus file name mapped to its expected extraction."""
    raw = json.loads(GROUND_TRUTH_FILE.read_text(encoding="utf-8"))
    return {name: GroundTruth.model_validate(entry) for name, entry in raw.items()}


def load_document(name: str) -> str:
    """The text of one corpus document, e.g. load_document("inv_01.txt")."""
    return (CORPUS_DIR / name).read_text(encoding="utf-8")
