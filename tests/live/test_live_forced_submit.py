"""Live check that tool_choice forcing submit_answer is accepted by the configured deployment and
api-version. Run once, by hand: uv run pytest -m live tests/live. Admitted through the Tally with
its own maximum first."""

from decimal import Decimal

import pytest

from footnote.agent.prompts import SYSTEM_PROMPT, TOOLS
from footnote.budget.limits import LIMITS, PRICES, call_max_cost
from footnote.budget.tally import Tally, settle_if_complete
from footnote.config import Settings
from footnote.llm.client import CallFailed, LLMClient

pytestmark = pytest.mark.live

FORCE = {"type": "function", "function": {"name": "submit_answer"}}


def test_live_forced_submit_answer() -> None:
    s = Settings.from_env()
    maximum = call_max_cost(LIMITS["agent"], PRICES["gpt-5-mini"])
    tally = Tally()
    tally.admit("live-forced-submit", maximum)
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
                tools=TOOLS,
                tool_choice=FORCE,
            )
        except CallFailed as exc:
            spent += exc.usage.cost
            raise
        spent += res.usage.cost
        calls = res.message.get("tool_calls") or []
        assert [c["function"]["name"] for c in calls] == ["submit_answer"]
        completed = True
    finally:
        settle_if_complete(tally, "live-forced-submit", spent, completed)
