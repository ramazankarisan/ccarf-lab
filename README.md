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
| [2. Structured extraction](ex2_extraction/README.md) | An invoice extraction pipeline: a Pydantic schema as a forced tool, semantic checks with targeted retries, an evaluation against ground truth, prompt variants measured one change at a time, the Batches API | done |

## Highlights

- **Rules in code, not just in the prompt.** Exercise 1 enforces a $500 refund limit
  and per-order authorization in Agent SDK hooks, so the model can't be talked out of
  them. Exercise 2 checks invoice arithmetic in code and sends documents that don't add
  up to a human instead of letting the model "fix" them.
- **Measured, not assumed.** Exercise 2 scores every field against hand-written ground
  truth over 3 runs per change: 421/432 for the zero-shot baseline, 432/432 with
  checks, retries and field descriptions. Few-shot examples were tried and dropped:
  no gain, about 15% more input tokens per call.
- **Agents tested like software.** 100% line and branch coverage offline, with
  scripted fake models; live scenarios assert on tool calls and must pass 3 runs out of 3.
- **Production habits.** Strict typing, secret scanning and Conventional Commits in
  git hooks; fictional data only; API keys stay on the server.

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
uv run mypy ex1_support_agent ex2_extraction tests  # strict type checking
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
ex2_extraction/      Exercise 2: invoice extraction (own README)
tests/               one test module per source module (test_ex2_* for Exercise 2)
tests/live/          live scenario tests (marker: live, skipped by default)
web/                 React + TypeScript app for Exercise 1
docs/                screenshots
lefthook.yml         git hooks for both stacks
CLAUDE.md            working rules for AI coding agents in this repo
```

## License

[MIT](LICENSE)
