"""How far the LLM grader agrees with a reference labeling of the same answers."""

from collections.abc import Mapping
from typing import Any

from footnote.eval.grade import Verdict
from footnote.eval.stats import cluster_bootstrap
from footnote.eval.summary import is_grader_error

AI_LABELER_PREFIXES = ("claude-", "gpt-")


def is_person(labeler: str) -> bool:
    """A labeler is a person unless its name starts with 'claude-' or 'gpt-'."""
    return not labeler.startswith(AI_LABELER_PREFIXES)


def _rate(flags: dict[str, list[bool]]) -> tuple[float | None, tuple[float, float] | None]:
    """Share of True flags with a company-clustered 95% interval (None when no items)."""
    values = [f for fs in flags.values() for f in fs]
    if not values:
        return None, None
    rate = sum(values) / len(values)
    _, lo, hi = cluster_bootstrap(flags)
    return rate, (lo, hi)


def grader_agreement(
    verdicts: dict[str, Verdict],
    labels: list[dict[str, str]],
    *,
    companies: Mapping[str, str],
) -> dict[str, Any]:
    """Agreement of the grader with labels {question_id, label, labeler}.

    Only questions with a correct/incorrect grader verdict and a label are compared.
    `companies` (question id -> company) is required for the company-clustered intervals; a
    labeled question id missing from it raises ValueError.
    False positive = grader says correct where the reference says incorrect (rate over
    reference-incorrect items); false negative = the reverse (over reference-correct items).
    `reference` is "human" only if every label's labeler is a person, otherwise "ai".
    """
    missing = sorted({lab["question_id"] for lab in labels} - companies.keys())
    if missing:
        raise ValueError(f"no company for labeled question ids: {missing}")
    agree: dict[str, list[bool]] = {}
    fp: dict[str, list[bool]] = {}
    fn: dict[str, list[bool]] = {}
    pairs = 0
    for lab in labels:
        qid, ref = lab["question_id"], lab["label"]
        v = verdicts.get(qid)
        if v is None or is_grader_error(v) or v.label not in ("correct", "incorrect"):
            continue
        pairs += 1
        co = companies[qid]
        agree.setdefault(co, []).append(v.label == ref)
        target = fp if ref == "incorrect" else fn
        target.setdefault(co, []).append(v.label != ref)
    n_agree = sum(sum(x) for x in agree.values())
    rate, ci = _rate(agree)
    fpr, fp_ci = _rate(fp)
    fnr, fn_ci = _rate(fn)
    human = bool(labels) and all(is_person(lab["labeler"]) for lab in labels)
    return {
        "reference": "human" if human else "ai",
        "n": pairs,
        "agree": n_agree,
        "disagree": pairs - n_agree,
        "agreement_rate": rate,
        "agreement_ci95": ci,
        "false_positive_rate": fpr,
        "false_positive_ci95": fp_ci,
        "false_negative_rate": fnr,
        "false_negative_ci95": fn_ci,
    }
