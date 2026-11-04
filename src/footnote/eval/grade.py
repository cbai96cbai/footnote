"""Grade one answer against its gold: numeric when it can be, else the LLM grader."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from footnote.answers import AgentAnswer
from footnote.benchmark import AgentQuestion
from footnote.eval.gold import Gold
from footnote.eval.grade_llm import grade_with_llm
from footnote.eval.grade_numeric import numeric_match
from footnote.llm.client import LLMClient, Usage
from footnote.numbers import parse_gold_number

NO_USAGE = Usage(0, 0, Decimal(0))


@dataclass(frozen=True)
class Verdict:
    question_id: str
    label: Literal["correct", "incorrect", "declined", "error"]
    method: Literal["numeric", "llm", "none"]
    rationale: str
    usage: Usage = field(default=NO_USAGE)  # total charged grader usage across attempts


def grade(
    question: AgentQuestion,
    gold: Gold,
    answer: AgentAnswer | None,
    status: str,
    *,
    client: LLMClient,
    deployment: str = "footnote-grader",
) -> Verdict:
    qid = question.id
    if status == "declined" or (answer is not None and answer.declined):
        return Verdict(qid, "declined", "none", "the system declined to answer")
    if status == "error":
        return Verdict(qid, "error", "none", "the system run ended in an error")
    if answer is None:
        return Verdict(qid, "error", "none", f"no answer with status {status!r}")
    parsed = parse_gold_number(gold.answer)
    if question.question_type == "metrics-generated" and parsed is not None and answer.final_value:
        ok = numeric_match(parsed, answer.final_value)
        fv = answer.final_value
        return Verdict(
            qid,
            "correct" if ok else "incorrect",
            "numeric",
            f"gold {gold.answer!r} vs final value {fv.value} {fv.unit}",
        )
    out, reason, usage = grade_with_llm(
        question, gold, answer, client=client, deployment=deployment
    )
    if out is None:
        return Verdict(qid, "error", "llm", reason, usage)
    return Verdict(qid, out.label, "llm", out.rationale, usage)
