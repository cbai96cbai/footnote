import pytest

from footnote.eval.grade import Verdict
from footnote.eval.grader_check import grader_agreement


def v(qid: str, label: str) -> Verdict:
    return Verdict(qid, label, "llm", "r")  # type: ignore[arg-type]


def lab(qid: str, label: str, labeler: str) -> dict[str, str]:
    return {"question_id": qid, "label": label, "labeler": labeler}


def test_agreement_reference_ai_unless_all_human() -> None:
    verdicts = {
        q: v(q, g)
        for q, g in (("a", "correct"), ("b", "correct"), ("c", "incorrect"), ("d", "incorrect"))
    }
    companies = {"a": "X", "b": "X", "c": "Y", "d": "Z"}
    ai = [
        lab("a", "correct", "claude-opus"),
        lab("b", "incorrect", "gpt-5"),
        lab("c", "correct", "claude-opus"),
        lab("d", "incorrect", "Camellia"),
    ]
    r = grader_agreement(verdicts, ai, companies=companies)
    assert r["reference"] == "ai"
    assert (r["n"], r["agree"], r["disagree"]) == (4, 2, 2)
    assert r["agreement_rate"] == pytest.approx(0.5)
    assert r["false_positive_rate"] == pytest.approx(
        0.5
    )  # b of b, d: grader correct, ref incorrect
    assert r["false_negative_rate"] == pytest.approx(
        0.5
    )  # c of a, c: grader incorrect, ref correct
    lo, hi = r["agreement_ci95"]
    assert lo <= 0.5 <= hi
    human = [lab(q, v_.label, "Camellia") for q, v_ in verdicts.items()]
    h = grader_agreement(verdicts, human, companies=companies)
    assert h["reference"] == "human" and h["agreement_rate"] == 1.0
    assert h["false_positive_rate"] == 0.0 and h["false_negative_rate"] == 0.0


def test_agreement_edge_cases() -> None:
    verdicts = {
        "a": v("a", "correct"),
        "e": v("e", "error"),
        "n": Verdict("n", "declined", "none", ""),
    }
    labels = [
        lab("a", "correct", "me"),
        lab("e", "correct", "me"),
        lab("n", "incorrect", "me"),
        lab("zzz", "correct", "me"),
    ]
    r = grader_agreement(verdicts, labels, companies={"a": "X", "e": "X", "n": "Y", "zzz": "Z"})
    assert r["n"] == 1 and r["agreement_ci95"] == (1.0, 1.0)
    assert r["false_positive_rate"] is None  # no reference-incorrect items
    assert grader_agreement({}, [], companies={})["reference"] == "ai"
    assert grader_agreement({}, [], companies={})["agreement_rate"] is None


def test_missing_company_raises() -> None:
    with pytest.raises(ValueError, match="b"):
        grader_agreement(
            {"a": v("a", "correct")},
            [lab("a", "correct", "me"), lab("b", "correct", "me")],
            companies={"a": "X"},
        )


def test_agreement_rejects_labels_other_than_correct_or_incorrect() -> None:
    verdicts = {"a": v("a", "correct")}
    for bad in ("Correct", "partial", ""):
        with pytest.raises(ValueError, match="'correct' or 'incorrect'"):
            grader_agreement(verdicts, [lab("a", bad, "camellia")], companies={"a": "X"})
