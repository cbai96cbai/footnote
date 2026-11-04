import json
from pathlib import Path

from typer.testing import CliRunner

from footnote.cli import app
from footnote.eval.failures import failure_type, select_for_labeling
from footnote.eval.gold import Evidence, Gold
from footnote.eval.grade import Verdict
from footnote.eval.records import append_prediction, append_verdict
from footnote.prediction import NO_USAGE, Prediction


def pred(
    status: str = "answered",
    verify_ok: bool | None = True,
    pages: list[int] | None = None,
    system: str = "agent",
) -> Prediction:
    return Prediction(
        "q",
        system,
        "1",
        None,
        status,  # type: ignore[arg-type]
        verify_ok,
        [],
        [],
        NO_USAGE,
        0.0,
        [3] if pages is None else pages,
    )


def verdict(label: str, method: str = "llm", qid: str = "q") -> Verdict:
    return Verdict(qid, label, method, "r")  # type: ignore[arg-type]


GOLD = Gold("q", "1", "j", [Evidence("d", 3, "t", "f")])


def test_failure_type_rules() -> None:
    assert failure_type(pred("error"), verdict("error", "none"), GOLD) == "error"
    assert failure_type(pred("declined"), verdict("declined", "none"), GOLD) == "declined"
    assert failure_type(pred(), verdict("declined", "none"), GOLD) == "declined"
    assert failure_type(pred(verify_ok=False), verdict("incorrect"), GOLD) == "unsupported"
    assert failure_type(pred(pages=[1, 2]), verdict("incorrect"), GOLD) == "retrieval_miss"
    assert failure_type(pred(), verdict("incorrect", "numeric"), GOLD) == "wrong_number"
    assert failure_type(pred(), verdict("incorrect", "llm"), GOLD) == "reasoning"
    # precedence: unsupported beats retrieval_miss; retrieval_miss beats wrong_number
    assert (
        failure_type(pred(verify_ok=False, pages=[]), verdict("incorrect"), GOLD) == "unsupported"
    )
    assert failure_type(pred(pages=[]), verdict("incorrect", "numeric"), GOLD) == "retrieval_miss"
    # correct answers and grader errors are not failures
    assert failure_type(pred(verify_ok=False), verdict("correct"), GOLD) is None
    assert failure_type(pred(), verdict("error", "llm"), GOLD) is None


def test_unsupported_only_for_systems_that_verify_themselves() -> None:
    for system in ("long-context", "plain-retrieval", "no-docs"):
        # the eval-time check is informational: the other rules decide the type
        failed = pred(verify_ok=False, system=system)
        assert failure_type(failed, verdict("incorrect", "numeric"), GOLD) == "wrong_number"
        assert failure_type(failed, verdict("incorrect"), GOLD) == "reasoning"
        assert (
            failure_type(pred(verify_ok=False, pages=[], system=system), verdict("incorrect"), GOLD)
            == "retrieval_miss"
        )
        assert failure_type(pred(verify_ok=None, system=system), verdict("incorrect"), GOLD) == (
            "reasoning"
        )


def test_labeling_sample_stratified() -> None:
    verdicts: dict[str, Verdict] = {}
    for i in range(40):
        qid = f"c{i}"
        verdicts[qid] = verdict("correct", qid=qid)
    for kind, n in (("reasoning", 20), ("unsupported", 20), ("retrieval_miss", 3)):
        for i in range(n):
            qid = f"{kind}{i}"
            verdicts[qid] = verdict("incorrect", qid=qid)
    verdicts["num"] = verdict("correct", "numeric", "num")
    verdicts["gerr"] = verdict("error", "llm", "gerr")
    verdicts["dec"] = verdict("declined", "none", "dec")
    types = {
        q: (None if v.label == "correct" else q.rstrip("0123456789")) for q, v in verdicts.items()
    }
    strata = {q: t or "ok" for q, t in types.items()}
    first = select_for_labeling(verdicts, types, n=12, seed=7)
    assert first == select_for_labeling(verdicts, types, n=12, seed=7)
    assert len(first) == len(set(first)) == 12
    assert not {"num", "gerr", "dec"} & set(first)
    counts = {s: sum(strata[q] == s for q in first) for s in set(strata.values())}
    assert (
        counts["ok"]
        == counts["reasoning"]
        == counts["unsupported"]
        == counts["retrieval_miss"]
        == 3
    )
    # a thin stratum gives all it has and the rest is spread over the others
    small = select_for_labeling(verdicts, types, n=30, seed=7)
    assert sum(strata[q] == "retrieval_miss" for q in small) == 3 and len(small) == 30
    assert select_for_labeling(verdicts, types, n=30, seed=8) != small
    assert len(select_for_labeling(verdicts, types, n=1000, seed=7)) == 83


def test_failures_command_writes_file_without_text(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from footnote import cli

    run = tmp_path / "r1"
    for qid, p, v in (
        ("q1", pred("declined"), verdict("declined", "none", "q1")),
        ("q2", pred(pages=[]), verdict("incorrect", "llm", "q2")),
        ("q3", pred(), verdict("correct", "llm", "q3")),
    ):
        append_prediction(
            run,
            Prediction(
                qid,
                *[
                    getattr(p, f)
                    for f in (
                        "system",
                        "version",
                        "answer",
                        "status",
                        "verify_ok",
                        "verify_problems",
                        "trace",
                        "usage",
                        "latency_s",
                        "pages_read",
                    )
                ],
            ),
        )
        append_verdict(run, v)
    monkeypatch.setattr(
        cli,
        "load_gold",
        lambda: {
            q: Gold(q, "SECRETANS", "SECRETJUST", [Evidence("d", 3, "t", "f")])
            for q in ("q1", "q2", "q3")
        },
    )
    out = CliRunner().invoke(app, ["failures", "r1", "--results-dir", str(tmp_path)])
    assert out.exit_code == 0 and "declined" in out.output and "retrieval_miss" in out.output
    text = (run / "failures.md").read_text()
    assert "q1" in text and "q2" in text and "q3" not in text
    assert "SECRET" not in text and json.dumps(text)
