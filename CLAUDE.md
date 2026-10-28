# ccarf-lab

Exercises for the Claude Certified Architect exam, one package each, each with its own
README and its own CLAUDE.md for exercise-specific rules: `ex1_support_agent/`,
`ex2_extraction/`, `ex3_research/`. `web/` is the React app for Exercise 1.

## Stack and commands

- Python 3.12 with uv; Node 20+ for `web/`.
- Install: `uv sync` and `npm --prefix web install` (also installs the git hooks).
- Tests: `uv run pytest --cov` (offline; coverage must stay at or above 90%).
- Types: `uv run mypy ex1_support_agent ex2_extraction ex3_research tests`
- Web: `npm --prefix web run lint|typecheck|test`
- All checks: `npx --prefix web lefthook run pre-commit --all-files`. Run them before
  saying a change is done.

## Rules

- Never read, print or commit `.env`. Secrets come from environment variables only;
  document a new variable in `.env.example` with a placeholder value.
- Use `uv add` / `uv add --dev` for dependencies, never `pip install`.
- Commit messages follow Conventional Commits (`feat(ex1): …`, `chore: …`).
- Data in corpora, fixtures and prompts is fictional: made-up names, example.com.

Commands that call the real API cost money:

@docs/live-runs.md
