"""Live compatibility check against the Azure deployments. Run once, by hand:
uv run pytest -m live tests/live. Admitted through the Tally with its own maximum first."""

import json
from decimal import Decimal
from typing import Any

import pytest
from pydantic import BaseModel

from footnote.budget.limits import LIMITS, PRICES, CallLimits, call_max_cost
from footnote.budget.tally import Tally, settle_if_complete
from footnote.config import Settings
from footnote.llm.client import CallFailed, ChatResult, LLMClient
from footnote.llm.tokens import count_input_tokens

pytestmark = pytest.mark.live

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }
]


class Answer(BaseModel):
    value: int


class Verdict(BaseModel):
    correct: bool


def test_live_compat() -> None:
    s = Settings.from_env()
    short = CallLimits(LIMITS["agent"].max_input_tokens, 16, "low")
    plan = [  # (limits, deployment) for the four calls below
        (LIMITS["agent"], s.agent_deployment),
        (LIMITS["agent"], s.agent_deployment),
        (LIMITS["grader"], s.grader_deployment),
        (short, s.agent_deployment),
    ]
    maximum = sum(
        (
            call_max_cost(
                lim, PRICES["gpt-5-mini" if dep == s.agent_deployment else "gpt-4.1-mini"]
            )
            for lim, dep in plan
        ),
        Decimal(0),
    )
    tally = Tally()
    tally.admit("live-compat", maximum)
    spent = Decimal(0)
    completed = False
    try:
        client = LLMClient(s)
        _chat = client.chat

        def charged_chat(*a: Any, **k: Any) -> ChatResult:
            nonlocal spent
            try:
                r = _chat(*a, **k)
            except CallFailed as exc:
                spent += exc.usage.cost
                raise
            spent += r.usage.cost
            return r

        client.chat = charged_chat  # type: ignore[method-assign]  # every call is accumulated
        agent, grader = s.agent_deployment, s.grader_deployment

        # one tool-call round trip
        msgs: list[dict[str, object]] = [
            {"role": "user", "content": "Use the add tool to add 2 and 3."}
        ]
        first = client.chat(msgs, deployment=agent, limits=LIMITS["agent"], tools=TOOLS)
        assert first.message["tool_calls"]
        call = first.message["tool_calls"][0]
        args = json.loads(call["function"]["arguments"])
        msgs += [
            first.message,
            {"role": "tool", "tool_call_id": call["id"], "content": str(args["a"] + args["b"])},
        ]
        second = client.chat(msgs, deployment=agent, limits=LIMITS["agent"], tools=TOOLS)
        assert "5" in (second.message.get("content") or "")
        assert count_input_tokens(msgs, TOOLS, None) >= second.usage.input_tokens

        # structured verdict on the grader
        gmsgs = [{"role": "user", "content": "Is 2 + 2 = 4? Answer as JSON."}]
        graded = client.chat(
            gmsgs, deployment=grader, limits=LIMITS["grader"], response_schema=Verdict
        )
        assert isinstance(graded.parsed, Verdict) and graded.parsed.correct
        assert count_input_tokens(gmsgs, None, Verdict) >= graded.usage.input_tokens

        # a call cut off at 16 completion tokens surfaces finish_reason == "length"
        cut = client.chat(
            [{"role": "user", "content": "Count from 1 to 200, one per line."}],
            deployment=agent,
            limits=short,
        )
        assert cut.finish_reason == "length"
        completed = True
    finally:
        # an interrupt or escaping error may hide a sent request's cost: stay at the maximum
        settle_if_complete(tally, "live-compat", spent, completed)
