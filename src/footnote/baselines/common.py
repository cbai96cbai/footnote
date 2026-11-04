"""Shared pieces of the two baselines: the instruction text, the call loop and the settings."""

import time
from dataclasses import asdict
from decimal import Decimal
from typing import Any

from footnote.answers import AgentAnswer
from footnote.benchmark import AgentQuestion
from footnote.budget.limits import LIMITS, MAX_CALLS, PRICES, CallLimits, call_max_cost
from footnote.corpus.normalize import DISPLAY_RULE
from footnote.llm.client import CallFailed, LLMClient, Usage
from footnote.prediction import NO_USAGE, RATE_LIMITED, Prediction, Step, add_usage

VERSION = "1"

INSTRUCTIONS = """\
You answer one question about one SEC filing. Reply only with a JSON object with these fields.
- answer: the answer in plain prose.
- final_value: the single final number the question asks for, as {"value": <number>, "unit": ...}.
  Give the number in the unit the question asks for (if it asks "in USD millions", the value is
  the number of millions). Use unit "usd" for money, "percent" for percentages (65.4% is 65.4),
  "ratio" for ratios, "count" for counts, "other" otherwise. Use null when the question has no
  single numeric answer.
- declined: true only when the material provided to you cannot support an answer; otherwise false.
- decline_reason: why you declined, or null.
- citations: exact quotes copied from the provided pages that support the answer, each with its
  page number as written in the [page N] marker. Use an empty list when no pages are provided."""

FILING_LINE = "The question is about this filing: {doc_name} ({company})."


def filing_line(q: AgentQuestion) -> str:
    return FILING_LINE.format(doc_name=q.doc_name, company=q.company)


def max_cost_per_question(limits: CallLimits, model: str = "gpt-5-mini") -> Decimal:
    return MAX_CALLS["baseline"] * call_max_cost(limits, PRICES[model])


def family_config(family: str, limits_key: str, user_template: str, packing: str) -> dict[str, Any]:
    return {
        "family": family,
        "version": VERSION,
        "limits": asdict(LIMITS[limits_key]),
        "max_calls": MAX_CALLS["baseline"],
        "prompt": {"system": INSTRUCTIONS, "user": user_template},
        "packing": packing,
        "display": DISPLAY_RULE,
    }


def _wrongly_ended(finish: str) -> bool:
    return finish in ("content_filter", "no_choices")


def run_calls(
    q: AgentQuestion,
    client: LLMClient,
    family: str,
    limits: CallLimits,
    messages: list[dict[str, Any]],
    *,
    pre_steps: list[Step] | None = None,
    pages_read: list[int] | None = None,
) -> Prediction:
    """At most MAX_CALLS["baseline"] calls; retry only on invalid output or a cut-off."""
    started = time.monotonic()
    trace: list[Step] = list(pre_steps or [])
    usage = NO_USAGE
    answer: AgentAnswer | None = None
    reason = ""
    rate_limited = False

    def step(summary: str, u: Usage, **args: Any) -> None:
        trace.append(Step("chat", args, summary, u))

    for attempt in range(1, MAX_CALLS["baseline"] + 1):
        try:
            res = client.chat(
                messages,
                deployment=client.settings.agent_deployment,
                limits=limits,
                response_schema=AgentAnswer,
            )
        except CallFailed as exc:
            usage = add_usage(usage, exc.usage)
            step(f"call failed: {exc}", exc.usage, attempt=attempt)
            reason = f"call failed: {exc}"
            rate_limited = exc.rate_limited
            break
        except Exception as exc:  # nothing was sent or charged; recorded, never raised
            step(f"call not made: {type(exc).__name__}: {exc}", NO_USAGE, attempt=attempt)
            reason = f"call not made: {type(exc).__name__}: {exc}"
            break
        usage = add_usage(usage, res.usage)
        step(
            f"finish_reason={res.finish_reason} parse_error={res.parse_error}",
            res.usage,
            attempt=attempt,
            finish_reason=res.finish_reason,
        )
        if res.parsed is not None:
            answer = res.parsed
            break
        if _wrongly_ended(res.finish_reason):
            reason = f"finish_reason {res.finish_reason}"
            break
        reason = f"finish_reason {res.finish_reason}, parse_error {res.parse_error}"
    status: Any = "error"
    if answer is not None:
        status = "declined" if answer.declined else "answered"
    elif reason:
        args = {"reason": RATE_LIMITED} if rate_limited else {}
        trace.append(Step("error", args, reason, NO_USAGE))
    return Prediction(
        question_id=q.id,
        system=family,
        version=VERSION,
        answer=answer,
        status=status,
        verify_ok=None,
        verify_problems=[],
        trace=trace,
        usage=usage,
        latency_s=time.monotonic() - started,
        pages_read=pages_read or [],
    )
