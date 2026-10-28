# Live runs

These call the real Claude API and cost money. Run them only when asked, never from a
test or a hook. They read `ANTHROPIC_API_KEY` and `MODEL` from `.env`.

| Command | What it runs |
|---|---|
| `uv run python -m ex1_support_agent.hello` | smoke test of the API key |
| `uv run python -m ex1_support_agent.loop "…"` / `… .agent "…"` / `… .session` | Exercise 1 agents |
| `uv run python -m ex2_extraction.pipeline --variant B` | Exercise 2 extraction, 3 runs |
| `uv run python -m ex3_research.agent "…"` | Exercise 3 research pipeline |
| `uv run pytest -m live` | live scenarios for Exercise 1 |

Live scenarios assert on tool calls, arguments and case facts, never on wording, and
each must pass 3 runs out of 3.
