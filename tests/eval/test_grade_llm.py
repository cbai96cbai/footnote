from decimal import Decimal
from typing import Any

from footnote.answers import AgentAnswer, FinalValue
from footnote.benchmark import AgentQuestion
from footnote.budget.limits import LIMITS, MAX_CALLS
from footnote.eval.gold import Gold
from footnote.eval.grade import grade
from footnote.eval.grade_llm import GraderOutput
from footnote.llm.client import CallFailed, ChatResult, Usage

U = Usage(100, 20, Decimal("0.001"))


def good(label: str = "correct") -> ChatResult:
    return ChatResult({}, U, "stop", 0.1, GraderOutput(label=label, rationale="r"))  # type: ignore[arg-type]


def bad(finish: str = "stop", err: str | None = "ValidationError: 1 errors") -> ChatResult:
    return ChatResult({}, U, finish, 0.1, None, err)


class FakeClient:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: Any, **kw: Any) -> ChatResult:
        self.calls.append({"messages": messages, **kw})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[no-any-return]


QUESTION = AgentQuestion("q1", "ACME", "ACME_2020_10K", "domain-relevant", "Who audits ACME?")
GOLD = Gold("q1", "Ernst & Young", "", [])


def answer(**kw: Any) -> AgentAnswer:
    base: dict[str, Any] = dict(
        answer="The auditor is EY.",
        final_value=FinalValue(value=Decimal("3"), unit="count"),
        declined=False,
        decline_reason=None,
        citations=[],
    )
    base.update(kw)
    return AgentAnswer(**base)


def test_grader_prompt_blind_to_system() -> None:
    from footnote.answers import Citation

    c = FakeClient([good()])
    a = answer(citations=[Citation(page_index=7, quote="SECRETQUOTE")])
    v = grade(QUESTION, GOLD, a, "ok", client=c)  # type: ignore[arg-type]
    assert v.label == "correct" and v.method == "llm" and v.usage == U
    text = " ".join(str(m["content"]) for m in c.calls[0]["messages"])
    assert "Who audits ACME?" in text and "Ernst & Young" in text and "The auditor is EY." in text
    assert "SECRETQUOTE" not in text
    for banned in ("footnote-agent", "baseline", "trace", "version", "citation"):
        assert banned not in text.lower()
    assert c.calls[0]["limits"] == LIMITS["grader"]
    assert c.calls[0]["response_schema"] is GraderOutput


def test_invalid_output_retried_once_then_error() -> None:
    c = FakeClient([bad(), bad("length", None), good()])
    v = grade(QUESTION, GOLD, answer(), "ok", client=c)  # type: ignore[arg-type]
    assert len(c.calls) == MAX_CALLS["grader"] == 2
    assert v.label == "error" and v.method == "llm" and v.usage.cost == 2 * U.cost


def test_retry_succeeds_and_sums_usage() -> None:
    c = FakeClient([bad(), good("incorrect")])
    v = grade(QUESTION, GOLD, answer(), "ok", client=c)  # type: ignore[arg-type]
    assert v.label == "incorrect" and v.usage.cost == 2 * U.cost
    assert v.usage.input_tokens == 200


def test_call_failed_charged_and_retried() -> None:
    mx = Usage(0, 0, Decimal("0.5"), charged_max=True)
    c = FakeClient([CallFailed("x", mx), CallFailed("y", mx)])
    v = grade(QUESTION, GOLD, answer(), "ok", client=c)  # type: ignore[arg-type]
    assert v.label == "error" and v.usage.cost == Decimal("1.0") and v.usage.charged_max


def test_declined_and_error_not_sent_to_grader() -> None:
    c = FakeClient([])
    d = grade(QUESTION, GOLD, answer(declined=True), "declined", client=c)  # type: ignore[arg-type]
    e = grade(QUESTION, GOLD, None, "error", client=c)  # type: ignore[arg-type]
    assert (d.label, d.method, e.label, e.method) == ("declined", "none", "error", "none")
    assert not c.calls and d.usage.cost == 0 and e.usage.cost == 0


def test_numeric_question_without_final_value_goes_to_llm() -> None:
    c = FakeClient([good()])
    q = AgentQuestion("q1", "ACME", "d", "metrics-generated", "Revenue?")
    v = grade(q, Gold("q1", "$1577.00", "", []), answer(final_value=None), "ok", client=c)  # type: ignore[arg-type]
    assert v.method == "llm" and len(c.calls) == 1
