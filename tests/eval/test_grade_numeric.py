from decimal import Decimal

from footnote.answers import AgentAnswer, FinalValue
from footnote.benchmark import AgentQuestion
from footnote.eval.gold import Gold
from footnote.eval.grade import grade


class NoClient:
    def chat(self, *a: object, **k: object) -> object:
        raise AssertionError("numeric grading must not call the model")


def q(kind: str = "metrics-generated") -> AgentQuestion:
    return AgentQuestion("q1", "ACME", "ACME_2020_10K", kind, "What was revenue in USD millions?")


def gold(text: str) -> Gold:
    return Gold("q1", text, "", [])


def ans(value: str, unit: str, text: str = "free text 999") -> AgentAnswer:
    return AgentAnswer(
        answer=text,
        final_value=FinalValue(value=Decimal(value), unit=unit),  # type: ignore[arg-type]
        declined=False,
        decline_reason=None,
        citations=[],
    )


def run(g: str, value: str, unit: str) -> str:
    v = grade(q(), gold(g), ans(value, unit), "ok", client=NoClient())  # type: ignore[arg-type]
    assert v.method == "numeric"
    assert v.usage.cost == 0
    return v.label


def test_exact_and_within_half_percent() -> None:
    assert run("$1577.00", "1577", "usd") == "correct"
    assert run("$1577.00", "1580", "usd") == "correct"


def test_magnitude_error_incorrect() -> None:
    assert run("$1577.00", "1.577", "usd") == "incorrect"


def test_gold_ratio_answer_percent() -> None:
    assert run("0.66", "66", "percent") == "correct"


def test_percent_wrong_value() -> None:
    assert run("1.9%", "2.9", "percent") == "incorrect"


def test_gold_percent_answer_ratio() -> None:
    assert run("65.4%", "0.654", "ratio") == "correct"


def test_free_text_ignored() -> None:
    v = grade(q(), gold("$1577.00"), ans("1577", "usd", "about 1.2"), "ok", client=NoClient())  # type: ignore[arg-type]
    assert v.label == "correct"


def test_gold_scale_word_ignored() -> None:
    assert run("$8.74 billion", "8.74", "usd") == "correct"
