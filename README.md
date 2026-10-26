# ccarf-lab

Hands-on exercises I built while preparing for the Claude Certified Architect exam.
Each exercise adds one idea at a time, and each idea is pinned down by tests.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![Type-checked with mypy --strict](https://img.shields.io/badge/mypy-strict-informational)
![Linted with Ruff](https://img.shields.io/badge/lint-ruff-informational)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## Exercises

| Exercise | What it builds | Status |
|---|---|---|
| [1. Customer-support agent](ex1_support_agent/README.md) | An agent on the Claude API and the Agent SDK: MCP tools, structured errors, hooks that enforce business rules, escalation, case facts, a web app | done |
| [3. Structured extraction](ex3_extraction/README.md) | An invoice extraction pipeline: a Pydantic schema, tool_choice, validation retries, few-shot, the Batches API, confidence-based review routing | in progress |

## Quickstart

You need [uv](https://docs.astral.sh/uv/), Node.js 20+, an
[Anthropic API key](https://console.anthropic.com/settings/keys) and, for the git
hooks, [gitleaks](https://github.com/gitleaks/gitleaks).

```bash
git clone https://github.com/ramazankarisan/ccarf-lab.git
cd ccarf-lab
uv sync
npm --prefix web install          # also installs the git hooks (lefthook)
cp .env.example .env              # then put your key in .env
```

Then follow the README of the exercise you want to run.

## Quality checks

```bash
uv run pytest --cov                 # offline, 100% line and branch coverage
uv run mypy ex1_support_agent ex3_extraction tests  # strict type checking
npm --prefix web test               # React tests (Vitest + Testing Library)
uv run pytest -m live               # live scenarios against the real API (cost money)
npx --prefix web lefthook run pre-commit --all-files
```

[Lefthook](https://lefthook.dev) runs the git hooks for both stacks (`lefthook.yml`):

| Hook | Checks |
|---|---|
| pre-commit | gitleaks secret scan · Ruff lint and format (with bandit security rules) · mypy strict · `uv.lock` in sync · oxlint and Prettier · TypeScript |
| commit-msg | Conventional Commits |
| pre-push | pytest with a 90% coverage gate · Vitest |

## Project layout

```text
ex1_support_agent/   Exercise 1: support agent (own README)
ex3_extraction/      Exercise 3: invoice extraction (own README)
tests/               one test module per source module (test_ex3_* for Exercise 3)
tests/live/          live scenario tests (marker: live, skipped by default)
web/                 React + TypeScript app for Exercise 1
docs/                screenshots
lefthook.yml         git hooks for both stacks
CLAUDE.md            working rules for AI coding agents in this repo
```

## License

[MIT](LICENSE)
