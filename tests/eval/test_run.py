import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from footnote.answers import AgentAnswer
from footnote.benchmark import AgentQuestion
from footnote.budget.limits import LIMITS, PRICES, call_max_cost
from footnote.budget.tally import Tally
from footnote.config import Settings
from footnote.eval.gold import Gold
from footnote.eval.grade_llm import GraderOutput
from footnote.eval.identity import RunIdentityMismatch
from footnote.eval.registry import TestRunRefused
from footnote.eval.run import run_eval
from footnote.llm.client import CallFailed, ChatResult, Usage
from footnote.prediction import Prediction
from footnote.systems import register_system

U = Usage(100, 20, Decimal("0.001"))
MAX_Q = Decimal("0.05")
GRADER_MAX = call_max_cost(LIMITS["grader"], PRICES["gpt-4.1-mini"])
QS = [
    AgentQuestion(f"q{i}", f"Co{i // 2}", f"Co{i // 2}_2020_10K", "domain-relevant", f"Q{i}?")
    for i in range(4)
]
GOLD = {q.id: Gold(q.id, "gold", "", []) for q in QS}


class FakeClient:
    settings = Settings()

    def __init__(self) -> None:
        self.graded: list[str] = []

    def price_for(self, deployment: str) -> Any:
        return PRICES["gpt-4.1-mini"]

    def chat(self, messages: Any, **kw: Any) -> ChatResult:
        assert kw["deployment"] == "grader-x"
        self.graded.append(str(messages[-1]["content"]))
        out = GraderOutput(label="correct", rationale="r")
        return ChatResult({}, U, "stop", 0.1, out)


def prediction(q: AgentQuestion, status: str = "answered") -> Prediction:
    ans = (
        None
        if status == "declined"
        else AgentAnswer(
            answer="a", final_value=None, declined=False, decline_reason=None, citations=[]
        )
    )
    return Prediction(q.id, "fake", "1", ans, status, None, [], [], U, 0.5)  # type: ignore[arg-type]


def setup(tmp_path: Path, runner: Any, family: str = "fake") -> dict[str, Any]:
    register_system(family, runner, MAX_Q, {"version": "1"})
    return dict(
        client=FakeClient(),
        tally=Tally(tmp_path / "spend.jsonl"),
        results_root=tmp_path / "results",
        settings=Settings(grader_deployment="grader-x"),
        questions=QS,
        gold=GOLD,
        pages_loader=lambda doc: {0: "page"},
        split_ids={"dev": [q.id for q in QS], "test": [q.id for q in QS], "gate": []},
    )


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_run_admits_max_before_first_call(tmp_path: Path) -> None:
    seen: list[str] = []

    def runner(q: AgentQuestion, client: Any) -> Prediction:
        seen.append((tmp_path / "spend.jsonl").read_text())
        return prediction(q, "declined")

    kw = setup(tmp_path, runner)
    summary = run_eval("fake", "dev", "r1", **kw)
    admit = json.loads(seen[0].splitlines()[0])
    assert admit["event"] == "admit" and len(seen[0].splitlines()) == 1
    assert Decimal(admit["amount"]) == 4 * (MAX_Q + 2 * GRADER_MAX)
    settle = json.loads((tmp_path / "spend.jsonl").read_text().splitlines()[-1])
    assert settle["event"] == "settle" and Decimal(settle["amount"]) == 4 * U.cost
    assert summary["n"] == 4 and summary["declined"] == 4 and summary["rubric_hash"]
    assert summary["tally_admitted"] == str(Decimal(admit["amount"]))
    stored = json.loads((tmp_path / "results" / "r1" / "summary.json").read_text())
    assert stored["system"] == "fake" and stored["config_hash"] == summary["config_hash"]


def test_resume_skips_done_questions(tmp_path: Path) -> None:
    calls: list[str] = []
    crash_on = {"q2"}

    def runner(q: AgentQuestion, client: Any) -> Prediction:
        if q.id in crash_on:
            raise KeyboardInterrupt  # not an Exception: a real crash
        calls.append(q.id)
        return prediction(q)

    kw = setup(tmp_path, runner)
    with pytest.raises(KeyboardInterrupt):
        run_eval("fake", "dev", "r1", **kw)
    assert calls == ["q0", "q1"]
    crash_on.clear()
    run_eval("fake", "dev", "r1", **kw)
    assert calls == ["q0", "q1", "q2", "q3"]
    d = tmp_path / "results" / "r1"
    assert [p["question_id"] for p in lines(d / "predictions.jsonl")] == ["q0", "q1", "q2", "q3"]
    assert len(lines(d / "verdicts.jsonl")) == 4
    spend = lines(tmp_path / "spend.jsonl")
    assert [s["event"] for s in spend] == ["admit", "admit", "settle"]
    assert Decimal(spend[1]["amount"]) == 2 * (MAX_Q + 2 * GRADER_MAX)  # only the remaining two


def test_prediction_without_verdict_gets_graded(tmp_path: Path) -> None:
    kw = setup(tmp_path, lambda q, c: prediction(q))
    run_eval("fake", "dev", "r1", **kw)
    d = tmp_path / "results" / "r1"
    done = lines(d / "verdicts.jsonl")[:3]
    (d / "verdicts.jsonl").write_text("".join(json.dumps(v) + "\n" for v in done))
    kw["client"].graded.clear()
    run_eval("fake", "dev", "r1", **kw)
    assert len(kw["client"].graded) == 1 and len(lines(d / "verdicts.jsonl")) == 4


def test_question_error_recorded_and_run_continues(tmp_path: Path) -> None:
    def runner(q: AgentQuestion, client: Any) -> Prediction:
        if q.id == "q1":
            raise CallFailed("timeout", Usage(16000, 3000, Decimal("0.01"), True))
        if q.id == "q2":
            raise ValueError("boom")
        return prediction(q)

    kw = setup(tmp_path, runner)
    summary = run_eval("fake", "dev", "r1", **kw)
    preds = {p["question_id"]: p for p in lines(tmp_path / "results" / "r1" / "predictions.jsonl")}
    assert len(preds) == 4 and summary["errors"] == 2 and summary["correct"] == 2
    assert (
        preds["q1"]["status"] == "error"
        and "CallFailed: timeout" in preds["q1"]["trace"][0]["summary"]
    )
    assert preds["q1"]["usage"]["charged_max"] is True
    assert Decimal(preds["q1"]["usage"]["cost"]) == MAX_Q
    assert "ValueError: boom" in preds["q2"]["trace"][0]["summary"]
    assert Decimal(preds["q2"]["usage"]["cost"]) == MAX_Q
    assert summary["cost_usd"] == pytest.approx(float(2 * MAX_Q) + 0.004)


def test_test_split_registry_guard(tmp_path: Path) -> None:
    kw = setup(tmp_path, lambda q, c: prediction(q))
    run_eval("fake", "test", "t1", **kw)
    with pytest.raises(TestRunRefused):
        run_eval("fake", "test", "t2", **kw)
    s = run_eval("fake", "test", "t2", rerun_test=True, reason="grader fixed", **kw)
    assert s["rerun_reason"] == "grader fixed"
    run_eval("fake", "dev", "d1", **kw)  # dev is never limited


def test_torn_last_line_resumes(tmp_path: Path) -> None:
    kw = setup(tmp_path, lambda q, c: prediction(q))
    run_eval("fake", "dev", "r1", **kw)
    d = tmp_path / "results" / "r1"
    for name in ("predictions.jsonl", "verdicts.jsonl"):
        text = (d / name).read_text()
        (d / name).write_text(
            text[: text.rstrip("\n").rfind("\n") + 1] + text.splitlines()[-1][:20]
        )
    calls: list[str] = []
    register_system(
        "fake", lambda q, c: (calls.append(q.id), prediction(q))[1], MAX_Q, {"version": "1"}
    )
    run_eval("fake", "dev", "r1", **kw)
    assert calls == ["q3"]
    for name in ("predictions.jsonl", "verdicts.jsonl"):
        assert [r["question_id"] for r in lines(d / name)] == ["q0", "q1", "q2", "q3"]


def test_raising_runner_charged_system_maximum(tmp_path: Path) -> None:
    def runner(q: AgentQuestion, client: Any) -> Prediction:
        raise ValueError("boom")

    kw = setup(tmp_path, runner)
    s = run_eval("fake", "dev", "r1", **kw)
    p = lines(tmp_path / "results" / "r1" / "predictions.jsonl")[0]
    assert p["usage"]["charged_max"] is True and Decimal(p["usage"]["cost"]) == MAX_Q
    assert s["errors"] == 4 and s["cost_usd"] == pytest.approx(4 * float(MAX_Q))


class FlakyClient(FakeClient):
    broken = True

    def chat(self, messages: Any, **kw: Any) -> ChatResult:
        if self.broken and "Q1?" in str(messages[-1]["content"]):
            raise CallFailed("grader timeout", Usage(4000, 500, Decimal("0.003"), True))
        return super().chat(messages, **kw)


def test_grader_error_regraded_on_resume_and_excluded_from_accuracy(tmp_path: Path) -> None:
    calls: list[str] = []

    def runner(q: AgentQuestion, client: Any) -> Prediction:
        calls.append(q.id)
        return prediction(q)

    kw = setup(tmp_path, runner)
    kw["client"] = FlakyClient()
    s = run_eval("fake", "dev", "r1", **kw)
    assert s["grader_errors"] == 1 and s["complete"] is False
    assert s["n"] == 4 and s["n_graded"] == 3 and s["accuracy"] == 1.0 and s["correct"] == 3
    kw["client"].broken = False
    s = run_eval("fake", "dev", "r1", **kw)
    assert calls == ["q0", "q1", "q2", "q3"]  # the system was not re-run
    assert s["grader_errors"] == 0 and s["complete"] is True and s["n_graded"] == 4
    assert s["cost_usd"] == pytest.approx(
        4 * 0.001 + 4 * 0.001 + 2 * 0.003
    )  # the grader retried once


def test_incomplete_test_run_stays_started_and_config_change_refused(tmp_path: Path) -> None:
    kw = setup(tmp_path, lambda q, c: prediction(q))
    kw["client"] = FlakyClient()
    run_eval("fake", "test", "t1", **kw)
    reg = json.loads((tmp_path / "results" / "test_runs.json").read_text())
    assert reg["fake"]["state"] == "started"
    register_system("fake", lambda q, c: prediction(q), MAX_Q, {"version": "1", "prompt": "new"})
    with pytest.raises(RunIdentityMismatch, match="config_hash"):
        run_eval("fake", "test", "t1", **kw)


def test_unverified_answers_are_verified_agent_ones_left_alone(tmp_path: Path) -> None:
    def runner(q: AgentQuestion, client: Any) -> Prediction:
        p = prediction(q)
        if q.id == "q1":
            return replace(p, verify_ok=True, verify_problems=["agent's own"])
        if q.id == "q2":
            return prediction(q, "declined")
        assert p.answer is not None
        bad = p.answer.model_copy(update={"answer": "It was 777.", "citations": []})
        return replace(p, answer=bad) if q.id == "q3" else p

    kw = setup(tmp_path, runner)
    run_eval("fake", "dev", "r1", **kw)
    by_id = {r["question_id"]: r for r in lines(tmp_path / "results" / "r1" / "predictions.jsonl")}
    assert by_id["q0"]["verify_ok"] is True and by_id["q0"]["verify_problems"] == []
    assert by_id["q1"]["verify_ok"] is True and by_id["q1"]["verify_problems"] == ["agent's own"]
    assert by_id["q2"]["verify_ok"] is None
    assert by_id["q3"]["verify_ok"] is False and "777" in by_id["q3"]["verify_problems"][0]


def test_missing_filing_pages_recorded_as_failed_verification(tmp_path: Path) -> None:
    def broken(doc: str) -> dict[int, str]:
        raise FileNotFoundError(doc)

    kw = setup(tmp_path, lambda q, c: prediction(q))
    kw["pages_loader"] = broken
    run_eval("fake", "dev", "r1", **kw)
    rows = lines(tmp_path / "results" / "r1" / "predictions.jsonl")
    assert all(r["verify_ok"] is False and "unavailable" in r["verify_problems"][0] for r in rows)
