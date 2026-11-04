"""What a system returns for one question, and how it is stored."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from footnote.answers import AgentAnswer
from footnote.llm.client import Usage

NO_USAGE = Usage(0, 0, Decimal(0))


# A question the service refused before processing (HTTP 429 on every attempt) was never
# answered by the system: its error step carries this reason and a resume runs it again.
RATE_LIMITED = "rate_limited"


def add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        a.input_tokens + b.input_tokens,
        a.output_tokens + b.output_tokens,
        a.cost + b.cost,
        a.charged_max or b.charged_max,
    )


@dataclass(frozen=True)
class Step:
    tool: str
    arguments: dict[str, Any]
    summary: str
    usage: Usage


@dataclass(frozen=True)
class Prediction:
    question_id: str
    system: str
    version: str
    answer: AgentAnswer | None
    status: Literal["answered", "declined", "error"]
    verify_ok: bool | None
    verify_problems: list[str]
    trace: list[Step]
    usage: Usage
    latency_s: float
    pages_read: list[int] = field(default_factory=list)


def is_retriable_error(p: Prediction) -> bool:
    """An error caused only by rate limiting: not a result, to be run again."""
    return p.status == "error" and any(
        s.tool == "error" and s.arguments.get("reason") == RATE_LIMITED for s in p.trace
    )


def usage_to_dict(u: Usage) -> dict[str, Any]:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cost": str(u.cost),
        "charged_max": u.charged_max,
    }


def usage_from_dict(d: dict[str, Any]) -> Usage:
    return Usage(d["input_tokens"], d["output_tokens"], Decimal(d["cost"]), d["charged_max"])


def prediction_to_dict(p: Prediction) -> dict[str, Any]:
    return {
        "question_id": p.question_id,
        "system": p.system,
        "version": p.version,
        "answer": p.answer.model_dump(mode="json") if p.answer else None,
        "status": p.status,
        "verify_ok": p.verify_ok,
        "verify_problems": p.verify_problems,
        "trace": [
            {
                "tool": s.tool,
                "arguments": s.arguments,
                "summary": s.summary,
                "usage": usage_to_dict(s.usage),
            }
            for s in p.trace
        ],
        "usage": usage_to_dict(p.usage),
        "latency_s": p.latency_s,
        "pages_read": p.pages_read,
    }


def prediction_from_dict(d: dict[str, Any]) -> Prediction:
    return Prediction(
        question_id=d["question_id"],
        system=d["system"],
        version=d["version"],
        answer=AgentAnswer.model_validate(d["answer"]) if d["answer"] else None,
        status=d["status"],
        verify_ok=d["verify_ok"],
        verify_problems=d["verify_problems"],
        trace=[
            Step(s["tool"], s["arguments"], s["summary"], usage_from_dict(s["usage"]))
            for s in d["trace"]
        ],
        usage=usage_from_dict(d["usage"]),
        latency_s=d["latency_s"],
        pages_read=d["pages_read"],
    )
