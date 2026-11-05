"""The agent as a registered system: loads the filing, runs the loop, publishes its config."""

import time
from dataclasses import asdict
from typing import Any

from footnote.agent import tools as tool_limits
from footnote.agent.loop import FAMILY, VERSION, run_agent
from footnote.agent.prompts import (
    LAST_REPLY,
    NUDGE_LENGTH,
    NUDGE_NO_TOOL,
    REPLIES_LEFT,
    RETRY_TEMPLATE,
    SUBMIT_NEXT,
    SYSTEM_PROMPT,
    TOOLS,
    USER_TEMPLATE,
)
from footnote.agent.submit import (
    FORCE,
    FORCE_WHEN_LEFT,
    IGNORED_AFTER,
    IGNORED_EXTRA,
    MAX_PROBLEM_TOKENS,
    RESEARCH_INPUT_TARGET,
)
from footnote.agent.tools import cap
from footnote.benchmark import AgentQuestion
from footnote.budget.limits import LIMITS, MAX_CALLS, PRICES, call_max_cost
from footnote.corpus.normalize import DISPLAY_RULE
from footnote.corpus.parse import load_pages
from footnote.llm.client import LLMClient
from footnote.prediction import NO_USAGE, Prediction, Step
from footnote.retrieval.index import DocIndex, retrieval_settings
from footnote.systems import register_system
from footnote.verify import VERIFY_VERSION


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
    "prompt": {
        "system": SYSTEM_PROMPT,
        "user": USER_TEMPLATE,
        "nudge_length": NUDGE_LENGTH,
        "nudge_no_tool": NUDGE_NO_TOOL,
        "verification_retry": RETRY_TEMPLATE,
        "replies_left": REPLIES_LEFT,
        "last_reply": LAST_REPLY,
        "submit_next": SUBMIT_NEXT,
        "ignored_extra_submission": IGNORED_EXTRA,
        "ignored_after_submission": IGNORED_AFTER,
    },
    "submit_forced_when_left": FORCE_WHEN_LEFT,
    "research_input_target_tokens": RESEARCH_INPUT_TARGET,
    "max_problem_tokens": MAX_PROBLEM_TOKENS,
    "submit_tool_choice": FORCE,
    "tool_limits": {
        "read_page_max_tokens": tool_limits.READ_PAGE_MAX_TOKENS,
        "search_k": tool_limits.SEARCH_K,
        "summary_chars": tool_limits.SUMMARY_CHARS,
        "trimming": "oldest tool results are replaced by their summaries until the call fits",
    },
    "tools": TOOLS,
    "verify_version": VERIFY_VERSION,
    "display": DISPLAY_RULE,
    "retrieval": retrieval_settings(),
}

SPEC = register_system(
    FAMILY,
    run,
    MAX_CALLS["agent"] * call_max_cost(LIMITS["agent"], PRICES["gpt-5-mini"]),
    CONFIG,
)
