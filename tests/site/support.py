import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from footnote.answers import AgentAnswer, Citation
from footnote.benchmark import AgentQuestion, DocInfo
from footnote.eval.gold import Gold
from footnote.eval.grade import Verdict
from footnote.eval.records import append_prediction, append_verdict
from footnote.llm.client import Usage
from footnote.prediction import Prediction, Step

U = Usage(100, 20, Decimal("0.001"))
LONG = "word " * 300  # 1500 characters
PAGE_TEXT = "RAW PAGE TEXT SHOULD NEVER APPEAR " * 20

QUESTIONS = [
    AgentQuestion("q1", "Acme", "ACME_2022_10K", "domain-relevant", "What was Acme revenue?"),
    AgentQuestion("q2", "Acme", "ACME_2022_10K", "novel-generated", "What was Acme income?"),
    AgentQuestion("q3", "Pfizer", "PFE_2022_10K", "domain-relevant", "What was Pfizer revenue?"),
]
DOCS = {
    "ACME_2022_10K": DocInfo(
        "ACME_2022_10K", "Acme", "Industrials", "10K", 2022, "https://a.test/x"
    ),
    "PFE_2022_10K": DocInfo(
        "PFE_2022_10K", "Pfizer", "Health Care", "10K", 2022, "https://p.test/x"
    ),
}
GOLD = {q.id: Gold(q.id, LONG if q.id == "q1" else "gold", "", []) for q in QUESTIONS}
LINKS: dict[str, Any] = {
    "ACME_2022_10K": {"ok": False, "status": 404},
    "PFE_2022_10K": {"ok": True, "status": 200},
}


def prediction(
    q: AgentQuestion,
    system: str,
    quote: str = LONG,
    page_index: int = 4,
    verify_ok: bool | None = True,
    problems: list[str] | None = None,
) -> Prediction:
    ans = AgentAnswer(
        answer=f"{system} answer",
        final_value=None,
        declined=False,
        decline_reason=None,
        citations=[Citation(page_index=page_index, quote=quote)],
    )
    step = Step("read_page", {"page_index": 4}, LONG, U)
    step2 = Step("search", {"query": "revenue"}, "found 3 pages", U)
    return Prediction(
        q.id, system, "v1", ans, "answered", verify_ok, problems or [], [step, step2], U, 2.5, [4]
    )


def write_run(
    root: Path,
    run_id: str,
    system: str,
    split: str,
    ids: list[Any],
    quote: str = LONG,
    page_index: int = 4,
    verify_ok: bool | None = True,
    problems: list[str] | None = None,
    version: str = "v1",
    labels: list[str] | None = None,
) -> None:
    run = root / run_id
    qs = [q for q in ids if not isinstance(q, str)] or [q for q in QUESTIONS if q.id in ids]
    for i, q in enumerate(qs):
        append_prediction(run, prediction(q, system, quote, page_index, verify_ok, problems))
        label = labels[i] if labels else ("correct" if i % 2 == 0 else "incorrect")
        append_verdict(run, Verdict(q.id, label, "numeric", "r", U))  # type: ignore[arg-type]
    n = len(qs)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "system": system,
                "version": version,
                "split": split,
                "n": n,
                "correct": (n + 1) // 2,
                "n_graded": n,
                "complete": True,
                "accuracy": ((n + 1) // 2) / n,
                "ci95": [0.1, 0.9],
                "coverage": 1.0,
                "cost_per_question": 0.01,
                "mean_latency_s": 2.5,
            }
        )
    )


def publish(root: Path, run_ids: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "published.json").write_text(json.dumps(run_ids), encoding="utf-8")
