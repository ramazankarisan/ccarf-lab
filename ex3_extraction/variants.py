"""Step 4's variants: which intervention fixes what, measured one change at a time.

A: the Step 3 pipeline as it is, no field descriptions, rules only.
B: the same, plus a description on every field of the schema.
C: B plus three few-shot examples in the system prompt.

Each variant runs through the same Step 3 pipeline and is scored the same way, so a
difference between two neighbours can be attributed to the one thing that changed.
"""

from dataclasses import dataclass

from anthropic.types import ToolParam

from ex3_extraction.extractor import SYSTEM_PROMPT, record_invoice_tool
from ex3_extraction.schema import json_schema


@dataclass(frozen=True)
class Variant:
    name: str
    tool: ToolParam
    system_prompt: str


# Three examples, one per ambiguous case that failed or nearly failed, each a short
# snippet, the decisive fields and a one-line reason. Nothing from the corpus: no
# vendor, number, product or tax ID, and not "catering". Whether the model gets from
# venue hire to catering is part of what variant C measures.
EXAMPLES = """\
<examples>
<example>
<snippet>
SHORT-STAY CAR PARK WEST
Ticket 5527-K
2026-09-14 08:12
Paid by card  EUR 17.80
</snippet>
<decisive_fields>
line_items: []
stated_subtotal: null
stated_tax: null
stated_total: 17.80
</decisive_fields>
<reason>A receipt that prints only one amount has no item rows and no subtotal.</reason>
</example>
<example>
<snippet>
Riverside Hall Ltd
Hire of the main hall with seating for 80, Saturday all day ...... GBP 950.00
</snippet>
<decisive_fields>
category: other
category_detail: venue hire
</decisive_fields>
<reason>Renting a room is not hardware, software, services or travel, so it is other, \
named in category_detail.</reason>
</example>
<example>
<snippet>
Northgate Print Studio, VAT No: GB000000555
Bill to: Moorside Partners, VAT Reg. GB000000777
</snippet>
<decisive_fields>
vendor_tax_id: GB000000555
</decisive_fields>
<reason>The seller's tax ID, without its label. The bill-to number belongs to the buyer.\
</reason>
</example>
</examples>"""

A = Variant("A", record_invoice_tool(json_schema(with_descriptions=False)), SYSTEM_PROMPT)
B = Variant("B", record_invoice_tool(json_schema(with_descriptions=True)), SYSTEM_PROMPT)
C = Variant("C", B.tool, f"{SYSTEM_PROMPT}\n\n{EXAMPLES}")

VARIANTS = {variant.name: variant for variant in (A, B, C)}
