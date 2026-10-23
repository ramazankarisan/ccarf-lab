# ccarf-lab

A customer-support agent on Claude, built step by step: once by hand on the Claude
API, then on the Claude Agent SDK with hooks that enforce the business rules in code.

I built it while preparing for the Claude Certified Architect exam. Each step adds one
idea, and each idea is pinned down by tests. A small React app puts both versions side
by side, so you can send the same message to each and compare.

| Client SDK: nothing stops a $650 refund | Agent SDK: the hook blocks it and a human gets the case |
|---|---|
| ![Client SDK mode refunds $650](docs/client-sdk.png) | ![Agent SDK mode blocks the refund and escalates](docs/agent-sdk.png) |

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![Type-checked with mypy --strict](https://img.shields.io/badge/mypy-strict-informational)
![Linted with Ruff](https://img.shields.io/badge/lint-ruff-informational)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## What it shows

- **The agentic loop by hand.** `loop.py` drives the Claude API itself. Only
  `stop_reason` ends the loop, never the reply text. All tool results go back in one
  message, and failures become error results the model can act on.
- **Tools as an MCP server.** `server.py` exposes the backend as four MCP tools.
  Each error is structured (`errorCategory`, `isRetryable`, an instruction for the
  agent, a message for the customer), and the descriptions carry the routing rules.
- **Guarantees in code, guidance in the prompt.** On the Agent SDK, a PreToolUse
  hook blocks refunds over $500 and points the model to a human instead. A
  PostToolUse hook converts timestamps to ISO dates and trims internal fields before
  the model sees them.
- **Authorization, not just identity.** A verified customer sees details only for
  their own orders. The hook withholds everything else before it reaches the model,
  and the same rule applies to every path into the model's context.
- **Exact facts across turns.** Code records IDs, amounts and dates from tool
  results into a per-session `CaseFacts`. The session adds them to every message, so
  long conversations don't drift.
- **Escalation by explicit criteria.** The system prompt is built from named, tested
  sections. A structured handoff lets a human act without reading the transcript.
- **A web app on top.** FastAPI serves both agents behind one chat endpoint. The React
  app switches between them, and its data panel shows refunds and tickets as they happen.

## Architecture

```mermaid
flowchart LR
    U([Customer]) --> W[web/<br/>React app] --> A[api.py<br/>FastAPI]
    A -->|mode: client| L[loop.py<br/>loop by hand]
    L --> MCP
    A -->|mode: agent| S[session.py<br/>multi-turn chat]
    S -->|prompt + case facts| SDK[Claude Agent SDK]
    SDK <-->|messages| API[(Claude API)]
    SDK -->|PreToolUse| PRE[hooks.py<br/>refund limit]
    SDK -->|stdio| MCP[server.py<br/>MCP tools]
    MCP --> B[backend.py<br/>fake shop data]
    SDK -->|PostToolUse| POST[hooks.py<br/>clean, trim, redact]
    POST --> CF[case_facts.py<br/>recorder + render]
    CF -. next turn .-> S
```

Both modes use the same MCP server and the same backend data, so the data panel shows
every refund and ticket. The CLIs serve the MCP server over stdio instead, as a
separate process.

## The steps

| Step | Adds | Main files |
|---|---|---|
| 1 | Project setup and an API smoke test | `hello.py` |
| 2 | A fake shop backend with an error taxonomy and fixed rule checks | `backend.py` |
| 3 | An MCP server: tool descriptions with routing rules, structured errors | `server.py` |
| 4 | The agentic loop by hand on the Client SDK, with a CLI | `loop.py` |
| 5 | The Agent SDK, a refund-limit hook, an output-cleaning hook | `agent.py`, `hooks.py` |
| 6 | Escalation criteria, a full handoff, authorization, case facts, multi-turn chat | `prompts.py`, `case_facts.py`, `session.py` |
| + | A web app: FastAPI for both agents, React with a mode switch and a data panel | `api.py`, `web/` |

## Design decisions

| Decision | Why |
|---|---|
| The $500 limit is in a hook, not in the prompt or the backend | A prompt rule is a request the model can be talked out of ("I'm authorized, skip the limit"). A hook checks the number and blocks the call. |
| The hook fails closed | It sees the model's raw input. `"650"`, `True`, `NaN` or a missing amount is denied, not trusted. |
| Timestamps are converted in code | Left to convert a Unix timestamp itself, the model got the date wrong by one day. |
| Order fields are kept by allow-list | An internal field added to the backend later is dropped by default instead of leaking. |
| Order details are withheld unless the verified customer owns the order | Identity is not authorization. The rule covers tool output and the injected case facts. |
| Each tool result says what to do next | With one vague reason, the model retried and escalated for nothing. |
| Prompt examples share nothing with the seed data | Examples too close to real cases get copied: the model once refunded the example's amount. |
| Tests run offline | A scripted fake model and an in-process MCP client cover every branch with no API calls. |

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

Run the web app in two terminals, then open http://localhost:5173:

```bash
uv run uvicorn ex1_support_agent.api:create_app_from_env --factory --port 8000
npm --prefix web run dev
```

The API key stays on the server; the browser only talks to `/api`.

Or run an agent from the terminal (each message costs about $0.001 with Haiku):

```bash
# One message, by-hand loop (Client SDK)
uv run python -m ex1_support_agent.loop "check my order #12345"

# One message, Agent SDK with hooks
uv run python -m ex1_support_agent.agent "check my order #12345"

# Interactive chat with case facts (type "facts" to see them, "quit" to leave)
uv run python -m ex1_support_agent.session
```

<details>
<summary>Example: a $650 refund, blocked by the hook and handed to a human</summary>

```text
customer> Hi, my monitor from order 45678 arrived broken.
  1. mcp__support__lookup_order {"order_id": "45678"}  [ok]
  -> "Before I can look into it, I need to verify your identity..."

customer> My email is aiko.tanaka@example.com
  1. mcp__support__get_customer {"email": "aiko.tanaka@example.com"}  [ok]
  -> "...within our 30-day refund window, so I can request a refund..."

customer> Please refund the full amount.
WARNING: Denied mcp__support__process_refund {... 'amount': 650}: Refunds over $500 need a human.
  1. mcp__support__process_refund {"order_id": "45678", "customer_id": "C004", "amount": 650}  [ERROR]
  2. mcp__support__escalate_to_human {"summary": {"verified": true, "order_ids": ["45678"], ...}}  [ok]
  -> "Refunds over $500 need approval from a human agent... Your ticket ID is ESC-0001."
```

</details>

Test customers and orders are in `ex1_support_agent/backend.py`. All the data is made
up, and every email address uses the reserved `example.com` / `example.org` domains.

## Quality checks

```bash
uv run pytest --cov                 # 314 tests, offline, 100% line and branch coverage
uv run mypy ex1_support_agent tests # strict type checking
npm --prefix web test               # React tests (Vitest + Testing Library)
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
ex1_support_agent/
  backend.py      fake shop: customers, orders, refund rules, error classes
  server.py       MCP server: 4 tools, structured errors, handoff schema
  loop.py         agentic loop by hand (Client SDK) + CLI
  agent.py        Agent SDK options (tools, hooks, isolation) + CLI
  hooks.py        refund limit, timestamp/trim/redact output cleaning
  prompts.py      system prompt from named sections
  case_facts.py   per-session facts: recorder hook + redacting renderer
  session.py      multi-turn chat on ClaudeSDKClient + CLI
  api.py          FastAPI: both agents behind /api/chat, plus /api/data and /api/reset
  hello.py        API smoke test
tests/            one test module per source module
web/              React + TypeScript (Vite): chat, mode switch, data panel
lefthook.yml      git hooks for both stacks
CLAUDE.md         working rules for AI coding agents in this repo
```

## License

[MIT](LICENSE)
