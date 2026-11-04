"""Baseline: the top passages of the told filing for the question text, in one call."""

from typing import Any, Protocol

from footnote.baselines.common import (
    INSTRUCTIONS,
    family_config,
    filing_line,
    max_cost_per_question,
    run_calls,
)
from footnote.benchmark import AgentQuestion
from footnote.budget.limits import LIMITS
from footnote.llm.client import LLMClient
from footnote.prediction import NO_USAGE, Prediction, Step
from footnote.retrieval.index import DocIndex, ScoredPassage
from footnote.systems import register_system

FAMILY = "plain-retrieval"
K = 8
USER_TEMPLATE = "{filing}\n\nPassages from the filing:\n\n{passages}\n\nQuestion: {question}"
PASSAGE_TEMPLATE = "[page {index}]\n{text}"
JOINER = "\n\n"
PACKING = (
    f"query = the question text; the top {K} passages from DocIndex.search in rank order, each as "
    "its [page N] block (N = the passage's page_index); pages_read = the pages of those passages, "
    "deduplicated, in rank order"
)


class Searchable(Protocol):
    def search(self, query: str, k: int = ...) -> list[ScoredPassage]: ...


def open_index(doc_name: str) -> Searchable:
    return DocIndex.for_doc(doc_name)


def _messages(q: AgentQuestion, hits: list[ScoredPassage]) -> list[dict[str, Any]]:
    body = JOINER.join(
        PASSAGE_TEMPLATE.format(index=h.passage.page_index, text=h.passage.text) for h in hits
    )
    user = USER_TEMPLATE.format(filing=filing_line(q), passages=body, question=q.question)
    return [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": user}]


def run(q: AgentQuestion, client: LLMClient) -> Prediction:
    hits = open_index(q.doc_name).search(q.question, k=K)
    pages = list(dict.fromkeys(h.passage.page_index for h in hits))
    search_step = Step(
        "search",
        {"query": q.question, "k": K, "passage_ids": [h.passage.id for h in hits]},
        f"{len(hits)} passages from {len(pages)} pages",
        NO_USAGE,
    )
    return run_calls(
        q,
        client,
        FAMILY,
        LIMITS["plain_retrieval"],
        _messages(q, hits),
        pre_steps=[search_step],
        pages_read=pages,
    )


SPEC = register_system(
    FAMILY,
    run,
    max_cost_per_question(LIMITS["plain_retrieval"]),
    family_config(FAMILY, "plain_retrieval", USER_TEMPLATE, PACKING)
    | {"retrieval": {"k": K, "query": "the question text"}},
)
