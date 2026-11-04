"""Run one system over one split: predict, grade, summarize. Resumable by run id."""

import importlib
import importlib.util
import json
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from footnote.benchmark import AgentQuestion, load_questions
from footnote.budget.limits import LIMITS, call_max_cost
from footnote.budget.tally import Tally
from footnote.config import Settings
from footnote.eval.gold import Gold, load_gold
from footnote.eval.grade import NO_USAGE, Verdict, grade
from footnote.eval.records import (
    append_prediction,
    append_verdict,
    read_predictions,
    read_verdicts,
)
from footnote.eval.registry import complete_test_run, start_test_run
from footnote.eval.split import SplitName, load_split
from footnote.eval.summary import build_summary
from footnote.llm.client import LLMClient, Usage
from footnote.prediction import Prediction, Step
from footnote.systems import SystemSpec, get_system

DEFAULT_RESULTS = Path("results")


def register_known_systems() -> None:
    """Import the packages that register the agent and the baselines, when they exist."""
    for name in ("footnote.baselines", "footnote.agent"):
        if importlib.util.find_spec(name) is not None:
            importlib.import_module(name)


def _error_prediction(
    spec: SystemSpec, q: AgentQuestion, exc: Exception, started: float
) -> Prediction:
    usage = getattr(exc, "usage", None)
    usage = usage if isinstance(usage, Usage) else NO_USAGE
    step = Step("error", {}, f"{type(exc).__name__}: {exc}", usage)
    return Prediction(
        q.id,
        spec.family,
        spec.version,
        None,
        "error",
        None,
        [],
        [step],
        usage,
        time.monotonic() - started,
    )


def _predict(spec: SystemSpec, q: AgentQuestion, client: LLMClient) -> Prediction:
    started = time.monotonic()
    try:
        return spec.runner(q, client)
    except Exception as exc:  # noqa: BLE001 - any failure of one question is that question's result
        return _error_prediction(spec, q, exc, started)


def _grade(
    q: AgentQuestion, gold: Gold, p: Prediction, client: LLMClient, deployment: str
) -> Verdict:
    try:
        return grade(q, gold, p.answer, p.status, client=client, deployment=deployment)
    except Exception as exc:  # noqa: BLE001
        usage = getattr(exc, "usage", None)
        usage = usage if isinstance(usage, Usage) else NO_USAGE
        return Verdict(q.id, "error", "llm", f"{type(exc).__name__}: {exc}", usage)


def _selected(
    questions: list[AgentQuestion] | None,
    split_ids: dict[SplitName, list[str]] | None,
    split: SplitName,
) -> list[AgentQuestion]:
    ids = set((split_ids or load_split())[split])
    return [q for q in (questions or load_questions()) if q.id in ids]


def run_eval(
    family: str,
    split: SplitName,
    run_id: str,
    *,
    client: LLMClient,
    tally: Tally,
    settings: Settings,
    results_root: Path = DEFAULT_RESULTS,
    rerun_test: bool = False,
    reason: str | None = None,
    questions: list[AgentQuestion] | None = None,
    gold: dict[str, Gold] | None = None,
    split_ids: dict[SplitName, list[str]] | None = None,
) -> dict[str, Any]:
    spec = get_system(family)
    selected = _selected(questions, split_ids, split)
    rerun_reason = reason if rerun_test else None
    if split == "test":
        rerun_reason = start_test_run(
            results_root, family, spec.config_hash, run_id, rerun=rerun_test, reason=reason
        )
    gold = gold if gold is not None else load_gold()
    run_dir = results_root / run_id
    preds, verdicts = read_predictions(run_dir), read_verdicts(run_dir)
    remaining = [q for q in selected if q.id not in verdicts]
    grader_price = client.price_for(settings.grader_deployment)
    per_question = spec.max_cost_per_question + 2 * call_max_cost(LIMITS["grader"], grader_price)
    admitted = per_question * len(remaining)
    if remaining:
        tally.admit(run_id, admitted)
    spent = Decimal(0)
    for q in remaining:
        if q.id not in preds:
            preds[q.id] = _predict(spec, q, client)
            append_prediction(run_dir, preds[q.id])
            spent += preds[q.id].usage.cost
        verdicts[q.id] = _grade(q, gold[q.id], preds[q.id], client, settings.grader_deployment)
        append_verdict(run_dir, verdicts[q.id])
        spent += verdicts[q.id].usage.cost
    if remaining:
        tally.settle(run_id, spent)  # an earlier crashed admission of this run stays at its maximum
    summary = build_summary(spec, split, selected, preds, verdicts)
    summary["tally_admitted"] = str(admitted)
    summary["tally_settled"] = str(spent)
    if rerun_reason:
        summary["rerun_reason"] = rerun_reason
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    if split == "test":
        complete_test_run(results_root, family, run_id)
    return summary
