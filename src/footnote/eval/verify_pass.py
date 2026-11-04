"""The evaluation-time grounding check for predictions that arrive unverified (the baselines)."""

from collections.abc import Callable
from dataclasses import replace

from footnote.benchmark import AgentQuestion
from footnote.corpus.parse import load_pages
from footnote.prediction import Prediction
from footnote.verify import verify_answer

# Verification is a per-system metric. The agent checks its own answers before it submits them;
# the baselines are checked here for information only (they have no calculator, so a computed
# value cannot pass); the no-documents system has no pages to cite and is not checked at all.
SELF_VERIFYING_SYSTEMS = frozenset({"agent"})
UNVERIFIED_SYSTEMS = frozenset({"no-docs"})

PagesLoader = Callable[[str], dict[int, str]]


def load_page_texts(doc_name: str) -> dict[int, str]:
    return {p.page_index: p.text for p in load_pages(doc_name)}


def with_verification(p: Prediction, q: AgentQuestion, load: PagesLoader) -> Prediction:
    """Verify an answered prediction whose verify_ok is None; leave every other one as it is.

    The no-documents system is never verified: its verify_ok stays None.
    """
    if (
        p.system in UNVERIFIED_SYSTEMS
        or p.status != "answered"
        or p.answer is None
        or p.verify_ok is not None
    ):
        return p
    try:
        pages = load(q.doc_name)
    except Exception as exc:  # noqa: BLE001 - recorded on the prediction, never silently skipped
        return replace(p, verify_ok=False, verify_problems=[f"filing pages unavailable: {exc}"])
    ok, problems = verify_answer(p.answer, pages, q.question, calcs=[])
    return replace(p, verify_ok=ok, verify_problems=problems)
