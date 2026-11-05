from decimal import Decimal

from footnote.answers import AgentAnswer, Citation, FinalValue
from footnote.benchmark import AgentQuestion
from footnote.eval.grade import Verdict
from footnote.eval.summary import build_summary
from footnote.llm.client import Usage
from footnote.prediction import Prediction

U = Usage(100, 20, Decimal("0.001"))
IDENTITY = {
    "family": "fake",
    "version": "1",
    "config_hash": "h",
    "split": "dev",
    "rubric_hash": "r",
}
QS = [AgentQuestion(f"q{i}", "Acme", "ACME_2022_10K", "x", "?") for i in range(4)]


def pred(q: AgentQuestion, value: int | None, status: str = "answered") -> Prediction:
    final = None if value is None else FinalValue(value=Decimal(value), unit="usd")
    ans = AgentAnswer(
        answer="a",
        final_value=final,
        declined=status == "declined",
        decline_reason=None,
        citations=[Citation(page_index=0, quote="page")],
    )
    return Prediction(q.id, "fake", "1", ans, status, None, [], [], U, 0.5)  # type: ignore[arg-type]


def test_the_summary_counts_answered_predictions_with_no_final_value() -> None:
    preds = {
        "q0": pred(QS[0], None),  # answered, no number
        "q1": pred(QS[1], 5),  # answered with a number
        "q2": pred(QS[2], None, "declined"),  # declined: not counted
        "q3": pred(QS[3], None),  # answered, no number
    }
    verdicts = {q.id: Verdict(q.id, "correct", "numeric", "r", U) for q in QS}  # type: ignore[arg-type]
    summary = build_summary(IDENTITY, QS, preds, verdicts)
    assert summary["answered_without_final_value"] == 2
    assert summary["coverage"] == 0.75
