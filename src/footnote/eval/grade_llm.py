"""LLM grading for answers the numeric rule does not cover. Sees only question, gold, answer."""

from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from footnote.answers import AgentAnswer
from footnote.benchmark import AgentQuestion
from footnote.budget.limits import LIMITS, MAX_CALLS
from footnote.eval.gold import Gold
from footnote.llm.client import CallFailed, LLMClient, Usage

RUBRIC = (Path(__file__).parent / "rubric.md").read_text(encoding="utf-8")
_BAD_FINISH = {"length", "content_filter", "no_choices"}


class GraderOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Literal["correct", "incorrect"]
    rationale: str


def _add(a: Usage, b: Usage) -> Usage:
    return Usage(
        a.input_tokens + b.input_tokens,
        a.output_tokens + b.output_tokens,
        a.cost + b.cost,
        a.charged_max or b.charged_max,
    )


def build_messages(
    question: AgentQuestion, gold: Gold, answer: AgentAnswer
) -> list[dict[str, str]]:
    fv = answer.final_value
    shown = f"{fv.value} ({fv.unit})" if fv else "none"
    body = (
        f"Question: {question.question}\n\nReference answer: {gold.answer}\n\n"
        f"System answer: {answer.answer}\nSystem final value: {shown}"
    )
    return [{"role": "system", "content": RUBRIC}, {"role": "user", "content": body}]


def grade_with_llm(
    question: AgentQuestion, gold: Gold, answer: AgentAnswer, *, client: LLMClient, deployment: str
) -> tuple[GraderOutput | None, str, Usage]:
    """(output, reason when None, total charged usage). At most MAX_CALLS['grader'] calls."""
    messages = build_messages(question, gold, answer)
    total = Usage(0, 0, Decimal(0))
    reason = ""
    for _ in range(MAX_CALLS["grader"]):
        try:
            res = client.chat(
                messages,
                deployment=deployment,
                limits=LIMITS["grader"],
                response_schema=GraderOutput,
            )
        except CallFailed as exc:
            total = _add(total, exc.usage)
            reason = f"grader call failed: {exc}"
            continue
        total = _add(total, res.usage)
        if res.parsed is not None and res.finish_reason not in _BAD_FINISH:
            return res.parsed, "", total
        reason = f"grader output unusable ({res.finish_reason}; {res.parse_error})"
    return None, reason, total
