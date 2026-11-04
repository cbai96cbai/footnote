"""Baseline: the model answers from memory, told the filing but given no pages."""

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
from footnote.prediction import Prediction
from footnote.systems import register_system

FAMILY = "no-docs"
USER_TEMPLATE = (
    "{filing}\nNo pages of the filing are provided. This overrides the rule about declining: "
    "answer from your own knowledge of the company's public filings and give your best "
    "final_value; set declined to true only if you have no basis at all for an answer.\n\n"
    "Question: {question}"
)
PACKING = "none: no pages are placed in the prompt"


def run(q: AgentQuestion, client: LLMClient) -> Prediction:
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(filing=filing_line(q), question=q.question),
        },
    ]
    return run_calls(q, client, FAMILY, LIMITS["no_docs"], messages)


SPEC = register_system(
    FAMILY,
    run,
    max_cost_per_question(LIMITS["no_docs"]),
    family_config(FAMILY, "no_docs", USER_TEMPLATE, PACKING),
)
