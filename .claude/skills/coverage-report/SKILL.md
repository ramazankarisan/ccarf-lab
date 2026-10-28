---
description: Reports test coverage for one exercise package and returns only the files below 90% with their uncovered line ranges. Use it to check coverage before a commit, or when asked how well an exercise is tested.
argument-hint: "<exercise folder, e.g. ex3_research>"
context: fork
allowed-tools: Bash(uv run pytest:*), Read, Grep
---

Report test coverage for the package `$ARGUMENTS`.

1. `$ARGUMENTS` must be one of the exercise package folders at the repository root
   (`ex1_support_agent`, `ex2_extraction`, `ex3_research`). If it is not, stop and say
   which folders exist.
2. Run the offline test suite with coverage for that package only:
   `uv run pytest -q --cov=$ARGUMENTS --cov-report=term-missing:skip-covered`
   Live tests are excluded by the project's pytest settings; do not add `-m live`.
3. Return only a short summary, nothing else:
   - the test result in one line (passed, failed, total coverage of the package);
   - every file below 90% as `path: NN% (uncovered: 12-18, 40)`, using the ranges from
     the Missing column;
   - if no file is below 90%, the line "All files are at 90% or above."
   - if tests failed, the names of the failing tests.

Do not paste the full pytest output and do not suggest fixes.
