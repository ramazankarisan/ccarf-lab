# Exercise 2: rules for this package

- Fields that a document may lack are required but nullable (`str | None` with no
  default). Never give them a default: the model must address every field.
- Money is `Decimal`, never `float`. Schemas reject unknown keys (`extra="forbid"`).
- Subtotal, tax and total are extracted separately, as printed. Code compares them;
  the model never computes or corrects totals. Currency is an ISO 4217 code pattern.
- Ground truth records what the document says (the printed total, not the sum of the
  line items). Checking the numbers is code's job.
- Model-facing text (prompts, descriptions, examples) shares nothing with the corpus:
  no vendors, numbers, products or tax IDs from it.
