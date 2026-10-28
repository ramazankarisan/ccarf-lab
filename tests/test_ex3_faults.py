import pytest

from ex3_research.faults import FailurePlan, SearchTimeout, parse_plan, plan_from_env


@pytest.mark.parametrize("value", [None, ""])
def test_no_value_means_no_failures(value):
    plan = parse_plan(value)
    assert plan.case is None
    assert plan.before_search("q", "current") is False


def test_timeout_once_times_out_only_the_first_call():
    plan = parse_plan("timeout_once")
    with pytest.raises(SearchTimeout):
        plan.before_search("EU market", "current")
    assert plan.timed_out == "EU market"
    assert plan.before_search("EU market", "current") is False


def test_category_timeout_times_out_only_that_category_every_time():
    plan = parse_plan("category_timeout:news archive")
    assert plan.category == "news archive"
    for _ in range(3):
        with pytest.raises(SearchTimeout, match="news archive"):
            plan.before_search("q", "news archive")
    assert plan.before_search("q", "current") is False


def test_empty_once_empties_only_the_first_call():
    plan = FailurePlan("empty_once")
    assert plan.before_search("q", "current") is True
    assert plan.before_search("q", "current") is False


@pytest.mark.parametrize(
    ("value", "error"),
    [("timeout_twice", "Unknown EX3_FAIL case"), ("category_timeout", "needs a category")],
)
def test_parse_plan_rejects_typos(value, error):
    with pytest.raises(ValueError, match=error):
        parse_plan(value)


def test_plan_from_env_reads_ex3_fail():
    assert plan_from_env({"EX3_FAIL": "empty_once"}).case == "empty_once"
    assert plan_from_env({}).case is None
