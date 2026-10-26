# Exercise 3: structured extraction

An invoice extraction pipeline. Step 1 defines what a correct extraction is before
any model call: a schema, a corpus of hand-written invoices and their ground truth.

## The steps

| Step | Adds | Main files |
|---|---|---|
| 1 | The invoice schema, 12 invoices covering 12 edge cases, hand-written ground truth (no API) | `schema.py`, `corpus/`, `ground_truth.py` |

## Design decisions

| Decision | Why |
|---|---|
| Absent fields are required but nullable | The model must address every field, and "not in the document" has an honest answer (null) instead of a made-up tax ID. |
| Money is `Decimal`, never `float` | `129.99` as a float is `129.98999…`; totals must compare exactly. |
| Subtotal, tax and total are separate fields, each as printed | With only a total, every VAT invoice looks like its items don't add up. Code compares the printed values; the model never computes or "fixes" them. |
| Currency is an ISO 4217 pattern (`^[A-Z]{3}$`), not an enum | Codes are a closed format, not a business taxonomy. A CHF invoice stays CHF instead of being forced into EUR. |
| Unknown keys are rejected | A typo like `vendor_taxid` would otherwise look like a missing tax ID. |
| Ground truth records what the document says | For an invoice whose items don't add up, the expected total is the printed one. Spotting the gap is code's job. |
| Corpus files have neutral names (`inv_01.txt`) | A name like `no_tax_id.txt` would leak the answer if it ever reached a prompt. The edge cases live in `ground_truth.json`. |

## The corpus

| Doc | Edge cases | The trap |
|---|---|---|
| 01 | clean table | none: the baseline |
| 02 | narrative email | quantity, rate, due date and tax ID are spread through the prose |
| 03 | terse receipt, no tax ID, no due date | three nulls and no line items |
| 04 | no tax ID | nothing to copy |
| 05 | no due date, net + VAT + total | "settled on 14 October" is not a due date |
| 06 | category outside the enum | catering → `other` + detail |
| 07 | items don't sum to the total | $1,163 in items, $1,203 printed |
| 08 | German formats, net + VAT + total | `19.10.2026`, `1.234,72 €` |
| 09 | net + VAT + total | "TOTAL (excl. VAT)" comes before the real total |
| 10 | no total printed | null, not a sum |
| 11 | ambiguous date | `03/04/2026` with a Texas address and Net 30 → March 4 |
| 12 | buyer's VAT number only | the printed VAT number is the guest company's, so the vendor's is null |

All data is fictional: placeholder tax IDs, `example.com` addresses, 555-01xx phone numbers.
