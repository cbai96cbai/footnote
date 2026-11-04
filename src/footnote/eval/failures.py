"""Why a prediction failed, and which graded answers a person should label."""

import random
from typing import Literal, get_args

from footnote.eval.gold import Gold
from footnote.eval.grade import Verdict
from footnote.eval.summary import is_grader_error
from footnote.eval.verify_pass import SELF_VERIFYING_SYSTEMS
from footnote.prediction import Prediction

FailureType = Literal[
    "declined", "unsupported", "retrieval_miss", "wrong_number", "reasoning", "error"
]
FAILURE_TYPES: tuple[str, ...] = get_args(FailureType)

# The no-documents system reads no pages, so "none of the gold pages was read" says nothing about it.
SYSTEMS_WITHOUT_PAGES = frozenset({"no-docs"})


def failure_type(pred: Prediction, verdict: Verdict, gold: Gold) -> FailureType | None:
    """The first matching cause; None for a correct answer or a grader error (not failures).

    "unsupported" applies only to a system that checks its own answers (an answer, or a decline
    after the check failed); a baseline's eval-time check is informational and never the cause.
    "retrieval_miss" applies only to systems that read pages."""
    if verdict.label == "correct" or is_grader_error(verdict):
        return None
    if pred.status == "error":
        return "error"
    if pred.verify_ok is False and pred.system in SELF_VERIFYING_SYSTEMS:
        return "unsupported"
    if pred.status == "declined" or verdict.label == "declined":
        return "declined"
    if pred.system not in SYSTEMS_WITHOUT_PAGES and not any(
        e.page_index in pred.pages_read for e in gold.evidence
    ):
        return "retrieval_miss"
    if verdict.method == "numeric":
        return "wrong_number"
    return "reasoning"


def select_for_labeling(
    verdicts: dict[str, Verdict],
    failure_types: dict[str, str | None],
    *,
    n: int = 30,
    seed: int = 7,
) -> list[str]:
    """Question ids for a person to label: LLM-graded correct/incorrect only, spread as evenly
    as the data allows over label x failure type (a thin stratum gives all it has)."""
    strata: dict[tuple[str, str | None], list[str]] = {}
    for qid in sorted(verdicts):
        v = verdicts[qid]
        if v.method == "llm" and v.label in ("correct", "incorrect"):
            strata.setdefault((v.label, failure_types.get(qid)), []).append(qid)
    rng = random.Random(seed)
    queues = [rng.sample(strata[k], len(strata[k])) for k in sorted(strata, key=str)]
    chosen: list[str] = []
    while len(chosen) < n and any(queues):
        for q in queues:
            if q and len(chosen) < n:
                chosen.append(q.pop())
    return chosen


REASONS: dict[str, str] = {
    "declined": "the system declined to answer",
    "unsupported": "the system's own citation check rejected the answer",
    "retrieval_miss": "none of the gold evidence pages was read",
    "wrong_number": "the final number does not match the gold number",
    "reasoning": "the right pages were read but the answer was graded incorrect",
    "error": "the run ended in an error",
}


def classify_run(
    preds: dict[str, Prediction], verdicts: dict[str, Verdict], gold: dict[str, Gold]
) -> dict[str, FailureType]:
    """Failure type of every failed question of a run (question id -> type)."""
    out: dict[str, FailureType] = {}
    for qid in sorted(verdicts):
        if qid in preds and qid in gold:
            t = failure_type(preds[qid], verdicts[qid], gold[qid])
            if t is not None:
                out[qid] = t
    return out


def failures_markdown(run_id: str, types: dict[str, FailureType]) -> str:
    """Question id, type and a fixed one-line reason; no document or gold text."""
    lines = [f"# Failures in {run_id}", "", "| question | type | reason |", "| --- | --- | --- |"]
    lines += [f"| {q} | {t} | {REASONS[t]} |" for q, t in sorted(types.items())]
    return "\n".join(lines) + "\n"
