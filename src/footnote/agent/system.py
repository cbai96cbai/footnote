"""The agent as a registered system: loads the filing, runs the loop, publishes its config."""

import time
from dataclasses import asdict
from typing import Any

from footnote.agent.loop import FAMILY, VERSION, run_agent
from footnote.agent.prompts import SYSTEM_PROMPT, TOOLS, USER_TEMPLATE
from footnote.agent.tools import cap
from footnote.benchmark import AgentQuestion
from footnote.budget.limits import LIMITS, MAX_CALLS, PRICES, call_max_cost
from footnote.corpus.parse import load_pages
from footnote.llm.client import LLMClient
from footnote.prediction import NO_USAGE, Prediction, Step
from footnote.retrieval.index import DocIndex
from footnote.systems import register_system


def run(q: AgentQuestion, client: LLMClient) -> Prediction:
    started = time.monotonic()
    try:
        pages = {p.page_index: p.text for p in load_pages(q.doc_name)}
        index = DocIndex.for_doc(q.doc_name)
    except Exception as exc:  # the filing could not be opened; nothing was charged
        return Prediction(
            q.id,
            FAMILY,
            VERSION,
            None,
            "error",
            None,
            [],
            [Step("error", {}, cap(f"filing not loaded: {type(exc).__name__}: {exc}"), NO_USAGE)],
            NO_USAGE,
            time.monotonic() - started,
        )
    return run_agent(q, client, pages, index)


CONFIG: dict[str, Any] = {
    "family": FAMILY,
    "version": VERSION,
    "limits": asdict(LIMITS["agent"]),
    "max_calls": MAX_CALLS["agent"],
    "prompt": {"system": SYSTEM_PROMPT, "user": USER_TEMPLATE},
    "tools": TOOLS,
}

SPEC = register_system(
    FAMILY,
    run,
    MAX_CALLS["agent"] * call_max_cost(LIMITS["agent"], PRICES["gpt-5-mini"]),
    CONFIG,
)
