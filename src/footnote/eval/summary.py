"""summary.json: one run's headline numbers."""

import hashlib
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any

from footnote.benchmark import AgentQuestion
from footnote.eval.grade import Verdict
from footnote.eval.stats import cluster_bootstrap
from footnote.prediction import Prediction, is_retriable_error

REPO_ROOT = Path(__file__).resolve().parents[3]
RUBRIC = Path(__file__).with_name("rubric.md")


def rubric_hash() -> str:
    return hashlib.sha256(RUBRIC.read_bytes()).hexdigest()


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def is_grader_error(v: Verdict) -> bool:
    """Grading itself failed; the system's answer was never judged."""
    return v.method == "llm" and v.label == "error"


def is_unscored(pred: Prediction, verdict: Verdict) -> bool:
    """Never a result: grading failed, or the service refused the request and it is to be rerun."""
    return is_grader_error(verdict) or is_retriable_error(pred)


def outcomes_by_company(
    questions: list[AgentQuestion], verdicts: dict[str, Verdict], preds: dict[str, Prediction]
) -> dict[str, list[bool]]:
    """Scored questions only, by company."""
    out: dict[str, list[bool]] = {}
    for q in questions:
        if is_unscored(preds[q.id], verdicts[q.id]):
            continue
        out.setdefault(q.company, []).append(verdicts[q.id].label == "correct")
    return out


def build_summary(
    identity: dict[str, str],
    questions: list[AgentQuestion],
    preds: dict[str, Prediction],
    verdicts: dict[str, Verdict],
) -> dict[str, Any]:
    n = len(questions)
    ps = [preds[q.id] for q in questions]
    correct = sum(verdicts[q.id].label == "correct" for q in questions)
    cost = sum((p.usage.cost for p in ps), Decimal(0)) + sum(
        (verdicts[q.id].usage.cost for q in questions), Decimal(0)
    )
    answered = [p for p in ps if p.status == "answered"]
    verified = [p for p in answered if p.verify_ok is not None]
    grader_errors = sum(is_grader_error(verdicts[q.id]) for q in questions)
    retriable_errors = sum(is_retriable_error(p) for p in ps)
    n_graded = sum(not is_unscored(preds[q.id], verdicts[q.id]) for q in questions)
    ci = None
    if n_graded:
        _, lo, hi = cluster_bootstrap(outcomes_by_company(questions, verdicts, preds))
        ci = [lo, hi]
    return {
        "system": identity["family"],
        "version": identity["version"],
        "config_hash": identity["config_hash"],
        "split": identity["split"],
        "n": n,
        "correct": correct,
        "n_graded": n_graded,
        "grader_errors": grader_errors,
        "retriable_errors": retriable_errors,
        "complete": grader_errors == 0 and retriable_errors == 0,
        "accuracy": correct / n_graded if n_graded else None,
        "ci95": ci,
        "coverage": len(answered) / n,
        "declined": sum(p.status == "declined" for p in ps),
        "errors": sum(p.status == "error" for p in ps),
        "cost_usd": float(cost),
        "cost_per_question": float(cost / n),
        "mean_latency_s": sum(p.latency_s for p in ps) / n,
        "verified_rate": (sum(bool(p.verify_ok) for p in verified) / len(verified))
        if verified
        else None,
        "rubric_hash": identity["rubric_hash"],
        "git_sha": git_sha(),
    }
