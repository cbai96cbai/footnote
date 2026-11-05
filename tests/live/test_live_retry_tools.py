"""Live check that tool_choice "required" with the tools restricted to calculate and submit_answer
is accepted by the configured deployment and api-version, and returns a tool call. Run once, by
hand: uv run pytest -m live tests/live. Admitted through the Tally with its own maximum first."""

from decimal import Decimal

import pytest

from footnote.agent.prompts import RETRY_TOOL_CHOICE, RETRY_TOOLS, SYSTEM_PROMPT
from footnote.budget.limits import LIMITS, PRICES, call_max_cost
from footnote.budget.tally import Tally, settle_if_complete
from footnote.config import Settings
from footnote.llm.client import CallFailed, LLMClient

pytestmark = pytest.mark.live


def test_live_required_with_the_retry_tools_returns_a_tool_call() -> None:
    s = Settings.from_env()
    maximum = call_max_cost(LIMITS["agent"], PRICES["gpt-5-mini"])
    tally = Tally()
    tally.admit("live-retry-tools", maximum)
    spent = Decimal(0)
    completed = False
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Filing: none. Question: what is 2 + 3? Decline."},
        ]
        try:
            res = LLMClient(s).chat(
                messages,
                deployment=s.agent_deployment,
                limits=LIMITS["agent"],
                tools=RETRY_TOOLS,
                tool_choice=RETRY_TOOL_CHOICE,
            )
        except CallFailed as exc:
            spent += exc.usage.cost
            raise
        spent += res.usage.cost
        calls = res.message.get("tool_calls") or []
        assert calls and {c["function"]["name"] for c in calls} <= {"calculate", "submit_answer"}
        completed = True
    finally:
        settle_if_complete(tally, "live-retry-tools", spent, completed)
