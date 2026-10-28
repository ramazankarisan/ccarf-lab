# Exercise 3: rules for this package

- Every researcher hand-back is one `ResearchResult` object (`findings.py`): `complete`
  with findings, or `partial` with `failure_type`, `attempted_queries`,
  `partial_findings` and `suggested_alternatives`. Each finding is a `Finding` (one
  typed source) or a `NoSources` (an empty search). Don't add a second format.
- Output formats live in the subagent definitions, never in the coordinator's prompt.
- Findings reach the synthesizer through code (`completeness.py` attaches them). Never
  ask a model to copy findings between agents.
- The coordinator's only tool is the spawn tool (`Agent`; `Task` on the exam). Each
  researcher gets its own MCP server process with only its tool, plus the owner check
  in the PreToolUse hook: `AgentDefinition.tools` alone leaked a tool.
- A hand-back is validated on SubagentStop and sent back at most once; the second one
  always goes through, marked `passed` or `format_failed`.
- Transient tool errors are structured (`isRetryable: true`) and retried once by the
  subagent. An empty search is a finding, not an error.
- The report has Established, Contested and Coverage sections; contested figures are
  never averaged or dropped. Keep the corpus's two 2025 figures in conflict by scope.
