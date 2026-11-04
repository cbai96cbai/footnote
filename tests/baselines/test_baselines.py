import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from footnote.answers import AgentAnswer
from footnote.baselines import baseline_settings, long_context, no_docs
from footnote.benchmark import AgentQuestion
from footnote.budget.limits import LIMITS, MAX_CALLS
from footnote.config import Settings
from footnote.corpus.parse import Page
from footnote.llm.client import CallFailed, ChatResult, Usage
from footnote.llm.tokens import count_input_tokens

Q = AgentQuestion("fb_1", "Acme", "ACME_2020_10K", "metrics-generated", "What was revenue?")
GOOD = AgentAnswer(
    answer="12", final_value=None, declined=False, decline_reason=None, citations=[]
)
U = Usage(100, 50, Decimal("0.001"))


class FakeClient:
    settings = Settings()

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages: Any, **kw: Any) -> ChatResult:
        self.calls.append(list(messages))
        assert kw["deployment"] == self.settings.agent_deployment
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


def res(parsed: Any = GOOD, finish: str = "stop", err: str | None = None) -> ChatResult:
    return ChatResult({}, U, finish, 0.1, parsed, err)


def test_long_context_truncates_and_records_last_page(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [Page("ACME_2020_10K", i, "word " * 60_000) for i in range(4)]
    monkeypatch.setattr(long_context, "load_pages", lambda doc: pages)
    client = FakeClient([res()])
    pred = long_context.run(Q, client)  # type: ignore[arg-type]
    sent = client.calls[0]
    assert count_input_tokens(sent, None, AgentAnswer) <= LIMITS["long_context"].max_input_tokens
    assert pred.pages_read == [0, 1, 2]
    pack = pred.trace[0]
    assert pack.arguments["last_page_included"] == 2
    assert pack.arguments["truncated"] is True
    assert "[page 2]" in sent[1]["content"] and "[page 3]" not in sent[1]["content"]
    assert pred.verify_ok is None and pred.verify_problems == []
    assert pred.status == "answered"


def test_long_context_short_filing_not_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [Page("ACME_2020_10K", i, f"text {i}") for i in range(3)]
    monkeypatch.setattr(long_context, "load_pages", lambda doc: pages)
    pred = long_context.run(Q, FakeClient([res()]))  # type: ignore[arg-type]
    assert pred.pages_read == [0, 1, 2]
    assert pred.trace[0].arguments["truncated"] is False


def test_baseline_prompts_contain_no_gold_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(long_context, "load_pages", lambda doc: [Page("ACME_2020_10K", 0, "x")])
    for run in (no_docs.run, long_context.run):
        client = FakeClient([res()])
        run(Q, client)  # type: ignore[arg-type]
        text = json.dumps(client.calls[0])
        assert Q.question in text and Q.doc_name in text
        for banned in ("justification", "evidence", "gold", "financebench_id"):
            assert banned not in text.lower()
    # the no-documents prompt must fit its own 1,000-token limit
    msgs = [{"role": "system", "content": no_docs.INSTRUCTIONS}, {"role": "user", "content": "q"}]
    assert count_input_tokens(msgs, None, AgentAnswer) < LIMITS["no_docs"].max_input_tokens - 100


def test_baseline_retry_only_on_invalid_or_length() -> None:
    assert MAX_CALLS["baseline"] == 2
    # invalid output then a good one: retried once
    c = FakeClient([res(None, "stop", "ValidationError: 1 errors"), res()])
    p = no_docs.run(Q, c)  # type: ignore[arg-type]
    assert len(c.calls) == 2 and p.status == "answered" and p.usage.input_tokens == 200
    # length then good
    c = FakeClient([res(None, "length"), res()])
    assert no_docs.run(Q, c).status == "answered"  # type: ignore[arg-type]
    # content filter and no choices: no retry
    for finish in ("content_filter", "no_choices"):
        c = FakeClient([res(None, finish), res()])
        p = no_docs.run(Q, c)  # type: ignore[arg-type]
        assert len(c.calls) == 1 and p.status == "error"
        assert finish in p.trace[-1].summary
    # two failures: error, never a third call
    c = FakeClient([res(None, "length"), res(None, "stop", "bad"), res()])
    p = no_docs.run(Q, c)  # type: ignore[arg-type]
    assert len(c.calls) == 2 and p.status == "error" and p.usage.input_tokens == 200
    # a failed send keeps its charge and is not retried
    c = FakeClient([CallFailed("timeout", Usage(0, 0, Decimal("0.5"), True)), res()])
    p = no_docs.run(Q, c)  # type: ignore[arg-type]
    assert len(c.calls) == 1 and p.status == "error" and p.usage.cost == Decimal("0.5")
    # declined
    d = AgentAnswer(answer="", final_value=None, declined=True, decline_reason="x", citations=[])
    assert no_docs.run(Q, FakeClient([res(d)])).status == "declined"  # type: ignore[arg-type]


def test_settings_file_matches_constants() -> None:
    path = Path(__file__).resolve().parents[2] / "data" / "baseline_settings.json"
    assert json.loads(path.read_text()) == json.loads(json.dumps(baseline_settings()))
    for fam, spec in (("no-docs", no_docs.SPEC), ("long-context", long_context.SPEC)):
        assert spec.config["limits"]["max_input_tokens"] == LIMITS[
            "no_docs" if fam == "no-docs" else "long_context"
        ].max_input_tokens
        assert spec.max_cost_per_question > 0
