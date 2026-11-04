"""Baseline: the told filing's pages, in order, until the input limit is reached."""

from typing import Any

from footnote.answers import AgentAnswer
from footnote.baselines.common import (
    INSTRUCTIONS,
    family_config,
    filing_line,
    max_cost_per_question,
    run_calls,
)
from footnote.benchmark import AgentQuestion
from footnote.budget.limits import LIMITS
from footnote.corpus.normalize import display_text
from footnote.corpus.parse import PARSER_VERSION, Page, load_pages
from footnote.llm.client import LLMClient
from footnote.llm.tokens import count_input_tokens, count_text_tokens
from footnote.prediction import NO_USAGE, Prediction, Step
from footnote.systems import register_system

FAMILY = "long-context"
USER_TEMPLATE = "{filing}\n\nPages of the filing:\n\n{pages}\n\nQuestion: {question}"
PAGE_TEMPLATE = "[page {index}]\n{text}"
JOINER = "\n\n"
PACKING = (
    "pages in order, each counted once as its [page N] block plus the joiner; added while the "
    "total fits the input limit minus the rest of the prompt (system text, question, response "
    "schema, framing per count_input_tokens); then the whole request is counted again and "
    "trailing pages are dropped while it is still over the limit"
)


def _block(p: Page) -> str:
    return PAGE_TEMPLATE.format(index=p.page_index, text=display_text(p.text))


def _messages(q: AgentQuestion, pages: list[Page]) -> list[dict[str, Any]]:
    body = JOINER.join(_block(p) for p in pages)
    user = USER_TEMPLATE.format(filing=filing_line(q), pages=body, question=q.question)
    return [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": user}]


def pack(q: AgentQuestion, pages: list[Page], limit: int) -> list[Page]:
    rest = count_input_tokens(_messages(q, []), None, AgentAnswer)
    budget = limit - rest
    used = 0
    chosen: list[Page] = []
    for p in pages:
        cost = count_text_tokens(_block(p)) + count_text_tokens(JOINER)
        if used + cost > budget:
            break
        used += cost
        chosen.append(p)
    while chosen and count_input_tokens(_messages(q, chosen), None, AgentAnswer) > limit:
        chosen.pop()
    return chosen


def run(q: AgentQuestion, client: LLMClient) -> Prediction:
    limits = LIMITS["long_context"]
    pages = load_pages(q.doc_name)
    chosen = pack(q, pages, limits.max_input_tokens)
    pack_step = Step(
        "pack",
        {
            "pages_in_filing": len(pages),
            "pages_included": len(chosen),
            "last_page_included": chosen[-1].page_index if chosen else None,
            "truncated": len(chosen) < len(pages),
        },
        f"included {len(chosen)} of {len(pages)} pages",
        NO_USAGE,
    )
    return run_calls(
        q,
        client,
        FAMILY,
        limits,
        _messages(q, chosen),
        pre_steps=[pack_step],
        pages_read=[p.page_index for p in chosen],
    )


SPEC = register_system(
    FAMILY,
    run,
    max_cost_per_question(LIMITS["long_context"]),
    family_config(FAMILY, "long_context", USER_TEMPLATE, PACKING)
    | {"parser_version": PARSER_VERSION},
)
