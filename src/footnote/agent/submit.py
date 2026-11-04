"""Submitting the answer: the forced-submission rule and what the check says about one attempt."""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from footnote.agent.prompts import RETRY_TEMPLATE
from footnote.agent.tools import cap
from footnote.answers import AgentAnswer
from footnote.verify import Calc, verify_answer

SUBMIT = "submit_answer"
FORCE_WHEN_LEFT = 2  # the last two replies belong to submission
SUBMISSION_RESERVE = 3_000  # input tokens research must leave free for the submission phase
FORCE: dict[str, Any] = {"type": "function", "function": {"name": SUBMIT}}
IGNORED_EXTRA = "ignored: one submission per reply"
IGNORED_AFTER = "ignored: the answer was already submitted"


def declined(reason: str) -> AgentAnswer:
    return AgentAnswer(
        answer="", final_value=None, declined=True, decline_reason=reason, citations=[]
    )


def tool_choice_for(replies_left: int) -> dict[str, Any] | None:
    return FORCE if replies_left <= FORCE_WHEN_LEFT else None


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
    feedback = f"problems: {joined}" if retried else RETRY_TEMPLATE.format(problems=joined)
    return Attempt(
        "failed", feedback, cap("verification failed: " + joined), answer, tuple(problems)
    )
