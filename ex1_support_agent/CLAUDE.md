# Exercise 1: rules for this package

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
- Run the app: `uv run uvicorn ex1_support_agent.api:create_app_from_env --factory
  --port 8000` and `npm --prefix web run dev`.
