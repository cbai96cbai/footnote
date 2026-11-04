"""Per-call limits, prices and the maximum cost of one call."""

from dataclasses import dataclass
from decimal import Decimal

MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class Price:
    """Dollars per 1M tokens."""

    input: Decimal
    output: Decimal


PRICES: dict[str, Price] = {
    "gpt-5-mini": Price(Decimal("0.25"), Decimal("2.00")),
    "gpt-4.1-mini": Price(Decimal("0.40"), Decimal("1.60")),
}


@dataclass(frozen=True)
class CallLimits:
    max_input_tokens: int
    max_completion_tokens: int
    reasoning_effort: str | None = None


LIMITS: dict[str, CallLimits] = {
    "agent": CallLimits(16_000, 3_000, "low"),
    "plain_retrieval": CallLimits(12_000, 3_000, "low"),
    "long_context": CallLimits(200_000, 3_000, "low"),
    "no_docs": CallLimits(1_000, 3_000, "low"),
    "grader": CallLimits(4_000, 500, None),
}

MAX_CALLS: dict[str, int] = {"agent": 8, "baseline": 2, "grader": 2}


def call_cost(prompt_tokens: int, completion_tokens: int, price: Price) -> Decimal:
    """Cost of a call; completion tokens already include reasoning tokens."""
    return Decimal(prompt_tokens) * price.input / MILLION + (
        Decimal(completion_tokens) * price.output / MILLION
    )


def call_max_cost(limits: CallLimits, price: Price) -> Decimal:
    return call_cost(limits.max_input_tokens, limits.max_completion_tokens, price)
