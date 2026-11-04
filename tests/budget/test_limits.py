from decimal import Decimal

from footnote.budget.limits import LIMITS, MAX_CALLS, PRICES, call_cost, call_max_cost


def test_call_max_cost_exact() -> None:
    assert call_max_cost(LIMITS["agent"], PRICES["gpt-5-mini"]) == Decimal("0.012")


def test_grader_and_long_context_maxima() -> None:
    assert call_max_cost(LIMITS["grader"], PRICES["gpt-4.1-mini"]) == Decimal("0.0024")
    assert call_max_cost(LIMITS["long_context"], PRICES["gpt-5-mini"]) == Decimal("0.056")


def test_limit_values_and_call_counts() -> None:
    assert {k: (v.max_input_tokens, v.max_completion_tokens) for k, v in LIMITS.items()} == {
        "agent": (24_000, 3_000),
        "plain_retrieval": (12_000, 3_000),
        "long_context": (200_000, 3_000),
        "no_docs": (2_000, 3_000),
        "grader": (4_000, 500),
    }
    assert MAX_CALLS == {"agent": 8, "baseline": 2, "grader": 2}


def test_call_cost_is_exact_decimal() -> None:
    assert call_cost(1_000, 100, PRICES["gpt-5-mini"]) == Decimal("0.00045")
