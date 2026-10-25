# ccarf-lab

Hands-on exercises for the Claude Certified Architect exam. Exercise 1 builds a
customer-support agent step by step: fake backend → MCP tools → agent loop →
Agent SDK hooks → escalation, case facts and multi-turn sessions. See README.md.

## Commands

- Install: `uv sync` and `npm --prefix web install` (installs the lefthook git hooks)
- All checks: `npx --prefix web lefthook run pre-commit --all-files`
- Web: `npm --prefix web run lint|typecheck|test`; app via `uv run uvicorn
  ex1_support_agent.api:create_app_from_env --factory --port 8000` + `npm --prefix web run dev`
- Tests: `uv run pytest --cov` (must stay at or above 90% coverage)
- Live scenarios (cost money, only on request): `uv run pytest -m live`. Assert on tool
  calls, arguments and case facts, never on wording; each scenario must pass 3 of 3.
- Types: `uv run mypy ex1_support_agent tests`
- Smoke test against the real API: `uv run python -m ex1_support_agent.hello`
- Live runs (cost money, never in tests): `uv run python -m ex1_support_agent.loop "…"`,
  `… .agent "…"`, and the interactive `uv run python -m ex1_support_agent.session`

Run all checks before saying a change is done.

## Rules

- Never read, print or commit `.env`. Secrets come from environment variables only;
  document new variables in `.env.example` with placeholder values.
- Tests run offline. Never call the real Claude API from a test or a hook.
- Use `uv add` / `uv add --dev` for dependencies, never `pip install`.
- Commit messages follow Conventional Commits (`feat(ex1): …`, `chore: …`).

## Design decisions to keep

- `backend.py` has no AI in it. Its error classes map to agent behavior:
  `BackendTimeout` transient (retry), `InvalidInput` validation (fix input),
  `PolicyViolation` business (don't retry), `NotAuthorized` permission (don't retry).
- `process_refund` checks in a fixed order: format → ownership → policy.
- The $500 refund limit lives only in the PreToolUse hook (`hooks.py`). Do not add it
  to `backend.py` or to the system prompt.
- Order details reach the model only for the verified owner. Any new path into the
  model's context (tool output, `<case_facts>`, summaries) must apply the same rule.
- Prompt examples must not reuse products, amounts or day counts from the seed data.
- Never call `time.time()` in backend logic; use `now()` so tests control the clock.
