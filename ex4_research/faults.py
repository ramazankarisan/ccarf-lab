"""Failure injection for web_search, driven by the EX4_FAIL environment variable.

The agent starts the MCP server as a subprocess, so a test or the CLI sets
EX4_FAIL in that process's environment (as Exercise 1 does with SUPPORT_FAIL_NEXT).

    EX4_FAIL=timeout_once                  the first call times out; the same query again works
    EX4_FAIL=category_timeout:news archive every search of that category times out
    EX4_FAIL=empty_once                    the first call returns a valid empty result

State is per server process. Each subagent gets its own process, so "first call"
means the first call of each subagent.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

FAIL_ENV = "EX4_FAIL"
CASES = ("timeout_once", "category_timeout", "empty_once")


class SearchTimeout(Exception):
    """A web_search call that timed out: transient, so a retry may succeed."""


@dataclass
class FailurePlan:
    case: str | None = None
    category: str | None = None  # for category_timeout
    calls: int = field(default=0, init=False)
    timed_out: str | None = field(default=None, init=False)  # query of the injected timeout

    def before_search(self, query: str, category: str) -> bool:
        """Raise SearchTimeout if this call must time out; True if it must return nothing."""
        self.calls += 1
        if self.case == "category_timeout" and category == self.category:
            raise SearchTimeout(f"the {category!r} search backend did not respond")
        if self.case == "timeout_once" and self.calls == 1:
            self.timed_out = query
            raise SearchTimeout("the search backend did not respond")
        return self.case == "empty_once" and self.calls == 1


def parse_plan(value: str | None) -> FailurePlan:
    """FailurePlan from EX4_FAIL; empty or unset means no failures.

    Raises ValueError for an unknown case, so a typo cannot silently test nothing.
    """
    if not value:
        return FailurePlan()
    case, _, argument = value.partition(":")
    if case not in CASES:
        raise ValueError(f"Unknown {FAIL_ENV} case {case!r}. Known: {', '.join(CASES)}.")
    if case == "category_timeout":
        if not argument:
            raise ValueError(f"{FAIL_ENV}=category_timeout needs a category, e.g. ':news archive'.")
        return FailurePlan(case, argument)
    return FailurePlan(case)


def plan_from_env(environ: Mapping[str, str]) -> FailurePlan:
    return parse_plan(environ.get(FAIL_ENV))
