"""Every system's fixed prompt plus any of the 150 real questions fits that system's input limit."""

import pytest

from footnote.agent.loop import _Agent
from footnote.agent.prompts import TOOLS
from footnote.answers import AgentAnswer
from footnote.baselines import plain_retrieval
from footnote.baselines.common import INSTRUCTIONS, filing_line
from footnote.baselines.no_docs import USER_TEMPLATE
from footnote.benchmark import load_questions
from footnote.budget.limits import LIMITS
from footnote.llm.tokens import count_input_tokens, count_text_tokens
from footnote.retrieval.chunk import MAX_TOKENS, Passage
from footnote.retrieval.index import ScoredPassage

pytestmark = pytest.mark.corpus


def test_no_docs_prompt_fits_for_all_questions() -> None:
    qs = load_questions()
    assert len(qs) == 150
    limit = LIMITS["no_docs"].max_input_tokens
    sizes = []
    for q in qs:
        user = USER_TEMPLATE.format(filing=filing_line(q), question=q.question)
        msgs = [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": user}]
        sizes.append(count_input_tokens(msgs, None, AgentAnswer))
    assert max(sizes) <= limit - 300, f"largest no-docs request {max(sizes)} of {limit}"


def test_plain_retrieval_prompt_fits_with_largest_passages_for_all_questions() -> None:
    word = "revenue "
    text = word * (MAX_TOKENS // count_text_tokens(word))
    assert count_text_tokens(text) <= MAX_TOKENS
    hits = [ScoredPassage(Passage(f"D:p{i}:0", "D", i, text), 1.0) for i in range(8)]
    limit = LIMITS["plain_retrieval"].max_input_tokens
    for q in load_questions():
        msgs = plain_retrieval._messages(q, hits)
        assert count_input_tokens(msgs, None, AgentAnswer) <= limit, q.id


def test_agent_first_turn_fits_for_all_questions() -> None:
    limit = LIMITS["agent"].max_input_tokens
    for q in load_questions():
        agent = _Agent(q, None, {}, None)  # type: ignore[arg-type]
        assert count_input_tokens(agent.messages, TOOLS, None) <= limit, q.id
