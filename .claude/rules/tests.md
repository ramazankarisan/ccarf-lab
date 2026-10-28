---
paths:
  - "**/tests/**/*.py"
---

# Test conventions

- Unit tests run offline: no live API calls, no network. Fake the model with scripted
  responses or message streams, and the MCP server in-process with `Client(mcp)`.
- Live tests live in `tests/live/`, are marked `live`, and run only on request.
- Reuse the fixtures in `tests/fixtures/`. A fixture captured from a live run says so
  and keeps its fictional data.
- One test module per source module: `tests/test_<module>.py` for Exercise 1,
  `tests/test_ex2_*.py` and `tests/test_ex3_*.py` for the others.
- Async tests use `@pytest.mark.anyio`.
- Assert on structure (tool calls, arguments, schema fields, report sections), never
  on model wording.
