import json
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from typer.testing import CliRunner

from footnote.benchmark import AgentQuestion
from footnote.budget.tally import Tally
from footnote.cli import app
from footnote.eval.records import append_prediction
from footnote.prediction import NO_USAGE, Prediction


def test_compare_prints_paired_difference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from footnote import cli

    qs = [AgentQuestion(f"q{i}", f"Co{i}", "D", "domain-relevant", "?") for i in range(6)]
    monkeypatch.setattr(cli, "load_questions", lambda: qs)
    ids = [q.id for q in qs]
    for name, labels in (("a", ["correct"] * 6), ("b", ["incorrect"] * 6)):
        d = tmp_path / name
        d.mkdir()
        u = {"input_tokens": 0, "output_tokens": 0, "cost": "0", "charged_max": False}
        for i in ids:
            append_prediction(
                d, Prediction(i, "fake", "1", None, "answered", None, [], [], NO_USAGE, 0.0)
            )
        (d / "verdicts.jsonl").write_text(
            "".join(
                json.dumps(
                    {"question_id": i, "label": lb, "method": "llm", "rationale": "", "usage": u}
                )
                + "\n"
                for i, lb in zip(ids, labels, strict=True)
            )
        )
    out = CliRunner().invoke(app, ["compare", "a", "b", "--results-dir", str(tmp_path)])
    assert out.exit_code == 0 and "a minus b: +1.000" in out.output


# ---- `footnote eval` end to end with a fake SDK: no network, no Azure -----------------------

QS = [
    AgentQuestion(f"q{i}", f"Co{i}", f"Co{i}_2020_10K", "domain-relevant", f"Question {i}?")
    for i in range(3)
]
ANSWER = {
    "answer": "about right",
    "final_value": None,
    "declined": False,
    "decline_reason": None,
    "citations": [],
}
GRADE = {"label": "correct", "rationale": "matches"}


def _completion(payload: dict[str, Any]) -> Any:
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(payload)},
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }
    )


class FakeSDK:
    """Answers the agent deployment with ANSWER and the grader with GRADE, or refuses with 429."""

    def __init__(self, rate_limited: bool = False) -> None:
        self.rate_limited = rate_limited
        self.calls: list[str] = []
        self.chat = self
        self.completions = self

    def create(self, **kw: Any) -> Any:
        self.calls.append(kw["model"])
        if self.rate_limited and kw["model"] == "footnote-agent":
            req = httpx.Request("POST", "https://x.openai.azure.com/chat")
            raise openai.RateLimitError(
                "slow down", response=httpx.Response(429, request=req), body=None
            )
        return _completion(ANSWER if kw["model"] == "footnote-agent" else GRADE)


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from footnote import cli
    from footnote.eval import run as run_mod
    from footnote.eval.gold import Gold
    from footnote.llm.client import LLMClient

    monkeypatch.setattr(run_mod, "load_questions", lambda: QS)
    monkeypatch.setattr(
        run_mod, "load_split", lambda: {"gate": [q.id for q in QS], "dev": [], "test": []}
    )
    monkeypatch.setattr(
        run_mod, "load_gold", lambda: {q.id: Gold(q.id, "gold", "", []) for q in QS}
    )
    monkeypatch.setattr("footnote.eval.verify_pass.load_pages", lambda doc: [])
    monkeypatch.setattr(cli, "Tally", lambda: Tally(tmp_path / "spend.jsonl"))
    monkeypatch.setattr(cli, "register_known_systems", run_mod.register_known_systems)
    sdk = FakeSDK()
    monkeypatch.setattr(
        cli, "LLMClient", lambda settings: LLMClient(settings, sdk=sdk, sleep=lambda _: None)
    )
    return tmp_path


def _eval(tmp_path: Path, *extra: str, system: str = "no-docs", run_id: str = "g1") -> Any:
    return CliRunner().invoke(
        app,
        ["eval", "--system", system, "--split", "gate", "--run-id", run_id]
        + ["--results-dir", str(tmp_path / "results"), *extra],
    )


def test_eval_registers_admits_runs_summarizes_and_exits_zero(cli_env: Path) -> None:
    out = _eval(cli_env)
    assert out.exit_code == 0, out.output
    assert "3/3 correct" in out.output and "0 errors" in out.output
    summary = json.loads((cli_env / "results" / "g1" / "summary.json").read_text())
    assert summary["system"] == "no-docs" and summary["complete"] is True
    assert summary["n"] == 3 and summary["retriable_errors"] == 0
    events = [json.loads(x)["event"] for x in (cli_env / "spend.jsonl").read_text().splitlines()]
    assert events == ["admit", "settle"]
    assert len((cli_env / "results" / "g1" / "predictions.jsonl").read_text().splitlines()) == 3


def test_eval_exits_nonzero_while_rate_limited_and_a_resume_finishes(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from footnote import cli
    from footnote.llm.client import LLMClient

    limited = FakeSDK(rate_limited=True)
    monkeypatch.setattr(cli, "LLMClient", lambda s: LLMClient(s, sdk=limited, sleep=lambda _: None))
    out = _eval(cli_env)
    assert out.exit_code == 1 and "incomplete: 3 questions rate limited" in out.output
    summary = json.loads((cli_env / "results" / "g1" / "summary.json").read_text())
    assert summary["complete"] is False and summary["retriable_errors"] == 3
    assert limited.calls.count("footnote-agent") == 3 * 6  # six tries each, charged nothing
    assert summary["cost_usd"] == 0
    healthy = FakeSDK()
    monkeypatch.setattr(cli, "LLMClient", lambda s: LLMClient(s, sdk=healthy, sleep=lambda _: None))
    out = _eval(cli_env)
    assert out.exit_code == 0, out.output
    assert json.loads((cli_env / "results" / "g1" / "summary.json").read_text())["complete"]


def test_eval_prints_one_line_for_a_locked_run_and_an_identity_mismatch(cli_env: Path) -> None:
    from footnote.eval.identity import run_lock

    with run_lock(cli_env / "results" / "g1"):
        out = _eval(cli_env)
    assert out.exit_code == 1 and "already running" in out.output
    assert out.exception is None or isinstance(out.exception, SystemExit)
    assert "Traceback" not in out.output

    assert _eval(cli_env).exit_code == 0
    out = _eval(cli_env, system="plain-retrieval")
    assert out.exit_code == 1 and "was started differently" in out.output
    assert isinstance(out.exception, SystemExit) and "Traceback" not in out.output


def test_unknown_system_is_a_one_line_error(cli_env: Path) -> None:
    out = _eval(cli_env, system="nope")
    assert out.exit_code == 1 and "unknown system 'nope'" in out.output
    assert isinstance(out.exception, SystemExit) and "Traceback" not in out.output
    assert len(out.output.strip().splitlines()) == 1
