"""Submitting the answer: the forced-submission rule and what the check says about one attempt."""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from footnote.agent.prompts import RETRY_TEMPLATE
from footnote.agent.tools import cap, cut_to_tokens
from footnote.answers import AgentAnswer
from footnote.verify import Calc, verify_answer

SUBMIT = "submit_answer"
FORCE_WHEN_LEFT = 2  # the last two replies belong to submission
# research calls stay at or under this many input tokens (older tool results are trimmed to get
# there); only the submission phase may go up to the per-call limit, which leaves headroom for the
# last research reply, the submission and its feedback
RESEARCH_INPUT_TARGET = 13_000
MAX_PROBLEM_TOKENS = 1_000  # the problem list in the retry feedback is cut to this size
FORCE: dict[str, Any] = {"type": "function", "function": {"name": SUBMIT}}
IGNORED_EXTRA = "ignored: one submission per reply"
IGNORED_AFTER = "ignored: the answer was already submitted"


def declined(reason: str) -> AgentAnswer:
    return AgentAnswer(
        answer="", final_value=None, declined=True, decline_reason=reason, citations=[]
    )


def tool_choice_for(replies_left: int) -> dict[str, Any] | None:
    return FORCE if replies_left <= FORCE_WHEN_LEFT else None


def _capped(problems: str) -> str:
    """The problem list cut to MAX_PROBLEM_TOKENS (including the ellipsis that marks the cut)."""
    text, cut = cut_to_tokens(problems, MAX_PROBLEM_TOKENS - 1)
    return text + "…" if cut else text


@dataclass(frozen=True)
class Attempt:
    """What one submit_answer call came to. `feedback` is the tool message the model sees."""

    kind: Literal["invalid", "declined", "accepted", "failed"]
    feedback: str
    summary: str
    answer: AgentAnswer | None = None
    problems: tuple[str, ...] = ()


def judge(
    args: dict[str, Any], pages: dict[int, str], question: str, calcs: list[Calc], *, retried: bool
) -> Attempt:
    try:
        answer = AgentAnswer.model_validate(args)
    except ValidationError as exc:
        text = f"error: answer does not match the schema ({exc.error_count()} errors)"
        return Attempt("invalid", text, cap(text))
    if answer.declined:
        return Attempt("declined", "declined", "declined", answer)
    ok, problems = verify_answer(answer, pages, question, calcs)
    if ok:
        return Attempt("accepted", "accepted", "verified", answer)
    joined = "; ".join(problems)
    if retried:
        feedback = f"problems: {joined}"
    else:
        feedback = RETRY_TEMPLATE.format(problems=_capped(joined))
    return Attempt(
        "failed", feedback, cap("verification failed: " + joined), answer, tuple(problems)
    )
