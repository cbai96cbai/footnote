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
from footnote.prediction import is_retriable_error

Q = AgentQuestion("fb_1", "Acme", "ACME_2020_10K", "metrics-generated", "What was revenue?")
GOOD = AgentAnswer(answer="12", final_value=None, declined=False, decline_reason=None, citations=[])
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
        assert (
            spec.config["limits"]["max_input_tokens"]
            == LIMITS["no_docs" if fam == "no-docs" else "long_context"].max_input_tokens
        )
        assert spec.max_cost_per_question > 0


def test_no_docs_prompt_answers_from_knowledge_long_context_does_not() -> None:
    assert "from your own knowledge" in no_docs.USER_TEMPLATE
    assert "from your own knowledge" not in long_context.USER_TEMPLATE
    assert "from your own knowledge" not in long_context.INSTRUCTIONS


def test_rate_limited_call_is_a_retriable_error_charged_nothing() -> None:
    limited = CallFailed(
        "rate limited on every attempt", Usage(0, 0, Decimal(0)), rate_limited=True
    )
    c = FakeClient([limited, res()])
    p = no_docs.run(Q, c)  # type: ignore[arg-type]
    assert len(c.calls) == 1 and p.status == "error" and p.usage.cost == 0
    assert is_retriable_error(p)
    c = FakeClient([CallFailed("timeout", Usage(0, 0, Decimal("0.5"), True))])
    assert not is_retriable_error(no_docs.run(Q, c))  # type: ignore[arg-type]


def test_config_hash_changes_with_every_retrieval_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    from footnote.agent import system as agent_system
    from footnote.baselines import plain_retrieval
    from footnote.corpus import parse
    from footnote.retrieval import index as index_mod
    from footnote.retrieval.embed import FastEmbedder
    from footnote.systems import hash_config

    def hashes() -> tuple[str, str]:
        s = index_mod.retrieval_settings()
        return (
            hash_config({**agent_system.CONFIG, "retrieval": s}),
            hash_config({**plain_retrieval.SPEC.config, "retrieval": {"k": 8, **s}}),
        )

    base = hashes()
    changes = [
        (index_mod, "CHUNKER_VERSION", "99"),
        (index_mod, "MAX_TOKENS", 351),
        (index_mod, "OVERLAP_TOKENS", 51),
        (index_mod, "RRF_K", 61),
        (index_mod, "PARSER_VERSION", "pdfium-99"),
        (FastEmbedder, "model_name", "other/model"),
        (FastEmbedder, "library_version", "0.0.1"),
    ]
    for obj, name, value in changes:
        with monkeypatch.context() as m:
            m.setattr(obj, name, value)
            assert all(a != b for a, b in zip(hashes(), base, strict=True)), name
    # both configs publish what they read
    assert agent_system.CONFIG["retrieval"] == index_mod.retrieval_settings()
    assert plain_retrieval.SPEC.config["retrieval"]["chunker_version"] == "1"
    assert long_context.SPEC.config["parser_version"] == parse.PARSER_VERSION


def test_long_context_shows_hyphen_marks_as_hyphens_and_counts_tokens_on_that_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = [Page("ACME_2020_10K", 0, "Regulation S\x02T and forward\x02looking")]
    monkeypatch.setattr(long_context, "load_pages", lambda doc: pages)
    client = FakeClient([res()])
    long_context.run(Q, client)  # type: ignore[arg-type]
    user = client.calls[0][1]["content"]
    assert "Regulation S-T and forward-looking" in user and "\x02" not in user
    assert long_context._block(pages[0]) == "[page 0]\nRegulation S-T and forward-looking"


def test_every_config_states_the_display_rule() -> None:
    from footnote.agent.system import CONFIG
    from footnote.corpus.normalize import DISPLAY_RULE

    assert CONFIG["display"] == DISPLAY_RULE
    assert all(c["display"] == DISPLAY_RULE for c in baseline_settings().values())


def test_baselines_share_the_quote_rules_and_a_new_version() -> None:
    from footnote.baselines import common, plain_retrieval
    from footnote.systems import hash_config

    text = " ".join(common.INSTRUCTIONS.split())
    assert "one unbroken piece of text from one page, at most 400 characters" in text
    assert "a sentence, or one table row with its label" in text
    assert "Never join separate lines or rows with '...'" in text
    assert "Never put a calculation in a quote" in text
    assert "calculate" not in text.lower() and "earlier" not in text.lower()
    assert common.VERSION == "2"
    for spec in (no_docs.SPEC, plain_retrieval.SPEC, long_context.SPEC):
        assert spec.version == "2" and spec.config["version"] == "2"
        assert spec.config["prompt"]["system"] == common.INSTRUCTIONS
        old = {**spec.config, "prompt": {**spec.config["prompt"], "system": "old"}}
        assert hash_config(old) != hash_config(spec.config)


def test_the_check_version_is_in_each_baseline_config_and_changes_its_hash() -> None:
    from footnote.baselines import plain_retrieval
    from footnote.systems import hash_config
    from footnote.verify import VERIFY_VERSION

    for spec in (no_docs.SPEC, plain_retrieval.SPEC, long_context.SPEC):
        assert spec.config["verify_version"] == VERIFY_VERSION
        assert hash_config({**spec.config, "verify_version": "4"}) != hash_config(spec.config)
