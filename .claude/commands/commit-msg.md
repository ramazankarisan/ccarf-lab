---
description: Draft a commit message for the staged changes
allowed-tools: Bash(git diff:*), Bash(git log:*)
---

Draft a commit message for the staged changes below. Do not commit, and do not change
any files.

Staged files:
!`git diff --cached --stat`

Staged diff:
!`git diff --cached`

Recent commit messages, for the style:
!`git log --oneline -8`

Write it as Conventional Commits: `type(scope): subject`, with a type from feat, fix,
docs, style, refactor, perf, test, build, ci, chore or revert, and the exercise as the
scope when one exercise changes (`ex1`, `ex2`, `ex3`). Keep the subject under 72
characters, in the imperative. Add a body only when the reason for the change isn't
obvious from the subject: say why, not what. If nothing is staged, say so instead.
