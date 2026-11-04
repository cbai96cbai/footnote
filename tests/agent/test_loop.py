import json
from decimal import Decimal
from typing import Any

import pytest
from fakes import FakeIndex, ScriptedClient, failed, reply, text_reply, tool_call

from footnote.agent.loop import run_agent
from footnote.agent.system import CONFIG, SPEC
from footnote.agent.tools import Toolbox
from footnote.answers import AgentAnswer
from footnote.benchmark import AgentQuestion
from footnote.budget.limits import LIMITS, MAX_CALLS
from footnote.llm.client import CallFailed, ChatResult, Usage
from footnote.llm.tokens import count_input_tokens
from footnote.prediction import Prediction, is_retriable_error

Q = AgentQuestion(
    "fb_1", "Acme", "ACME_2020_10K", "metrics-generated", "What was revenue in USD millions?"
)
PAGES = {0: "Intro page.", 1: "Total revenue was $1,234 million in fiscal 2020.", 2: "x"}
INDEX = FakeIndex([(1, PAGES[1])])
GOOD = {
    "answer": "Revenue was 1,234 USD millions.",
    "final_value": {"value": 1234, "unit": "usd"},
    "declined": False,
    "decline_reason": None,
    "citations": [{"page_index": 1, "quote": "Total revenue was $1,234 million"}],
}
BAD = {**GOOD, "citations": [{"page_index": 1, "quote": "Total revenue was $9,999 million"}]}


def run(outcomes: list[ChatResult | Exception]) -> tuple[Prediction, ScriptedClient]:
    client = ScriptedClient(outcomes)
    return run_agent(Q, client, PAGES, INDEX), client  # type: ignore[arg-type]


def submit(args: Any, cid: str = "s") -> ChatResult:
    return reply(tool_call("submit_answer", args, cid))


def search(cid: str = "a") -> ChatResult:
    return reply(tool_call("search_filing", {"query": "revenue"}, cid))


def test_happy_path_verified() -> None:
    p, client = run(
        [search(), reply(tool_call("calculate", {"expression": "1234 * 1"}, "b")), submit(GOOD)]
    )
    assert p.status == "answered" and p.verify_ok is True and p.verify_problems == []
    assert p.answer is not None and p.answer.final_value is not None
    assert p.answer.final_value.value == Decimal(1234)
    assert p.pages_read == [1] and len(client.calls) == 3
    assert [s.tool for s in p.trace] == ["search_filing", "calculate", "submit_answer"]
    assert p.usage.input_tokens == 300 and sum(s.usage.input_tokens for s in p.trace) == 300
    assert all(len(s.summary) <= 200 for s in p.trace)
    assert "Total revenue was" not in p.trace[0].summary


def test_verify_failure_then_fixed() -> None:
    p, client = run([submit(BAD), submit(GOOD, "s2")])
    assert p.status == "answered" and p.verify_ok is True and len(client.calls) == 2
    sent = json.dumps(client.calls[1])
    assert "Verification found problems" in sent and "quote not found" in sent


def test_double_verify_failure_declines_with_reason() -> None:
    p, client = run([submit(BAD), submit(BAD, "s2"), submit(GOOD, "s3")])
    assert len(client.calls) == 2 and p.status == "declined" and p.verify_ok is False
    assert p.answer is not None and p.answer.decline_reason is not None
    assert "quote not found" in p.answer.decline_reason and p.verify_problems


def test_eighth_call_is_last() -> None:
    assert MAX_CALLS["agent"] == 8
    p, client = run([search(f"a{i}") for i in range(12)])
    assert len(client.calls) == 8 and p.status == "declined"
    assert p.answer is not None and p.answer.decline_reason == "call limit"
    assert p.usage.input_tokens == 800


def test_verify_retry_cannot_exceed_call_limit() -> None:
    p, client = run([search(f"a{i}") for i in range(7)] + [submit(BAD), submit(GOOD)])
    assert len(client.calls) == 8 and p.answer is not None
    assert p.answer.decline_reason == "call limit" and p.verify_ok is False


def test_calculate_rejects_non_arithmetic() -> None:
    p, client = run(
        [reply(tool_call("calculate", {"expression": "__import__('os')"})), submit(GOOD)]
    )
    tool_msg = client.calls[1][-1]
    assert tool_msg["role"] == "tool" and tool_msg["content"].startswith("error:")
    assert p.status == "answered"


def test_bad_tool_arguments_returned_as_error_and_call_counted() -> None:
    p, client = run(
        [
            reply(tool_call("search_filing", "{not json")),
            reply(tool_call("no_such_tool", {})),
            reply(tool_call("read_page", {"page_index": 99})),
            submit({"answer": "x"}),
            submit(GOOD, "s2"),
        ]
    )
    assert len(client.calls) == 5 and p.status == "answered"
    for i in (1, 2, 3, 4):
        assert client.calls[i][-1]["content"].startswith("error:")
    assert p.usage.input_tokens == 500 and p.pages_read == []


def test_length_finish_counts_and_continues_or_declines() -> None:
    p, client = run([text_reply("", "length"), submit(GOOD)])
    assert len(client.calls) == 2 and p.status == "answered" and p.usage.input_tokens == 200
    p, client = run([text_reply("", "length")] * 8)
    assert len(client.calls) == 8 and p.answer is not None
    assert p.answer.decline_reason == "call limit"


def test_plain_text_reply_is_nudged_and_filter_is_error() -> None:
    p, client = run([text_reply("I think 5"), submit(GOOD)])
    assert p.status == "answered" and client.calls[1][-1]["role"] == "user"
    p, _ = run([text_reply("", "content_filter")])
    assert p.status == "error" and p.usage.input_tokens == 100


def test_failed_call_keeps_charge_and_returns_error_prediction() -> None:
    p, client = run([search(), failed("0.5")])
    assert p.status == "error" and p.usage.cost == Decimal("0.001") + Decimal("0.5")
    assert p.usage.charged_max and len(client.calls) == 2


def test_read_page_tracks_pages_in_first_seen_order() -> None:
    p, _ = run(
        [
            reply(tool_call("read_page", {"page_index": 2})),
            search(),
            reply(tool_call("read_page", {"page_index": 2}, "z")),
            submit(GOOD),
        ]
    )
    assert p.pages_read == [2, 1]


def test_old_tool_results_are_trimmed_then_context_limit() -> None:
    big = {0: "word " * 2900, 1: PAGES[1], 2: "y " * 2900}
    outs: list[ChatResult | Exception] = [
        reply(tool_call("read_page", {"page_index": 0 if i % 2 == 0 else 2}, f"r{i}"))
        for i in range(7)
    ]
    client = ScriptedClient(outs + [submit(GOOD)])
    p = run_agent(Q, client, big, INDEX)  # type: ignore[arg-type]
    for sent, tools in zip(client.calls, client.tool_sets, strict=True):
        assert count_input_tokens(sent, tools, None) <= LIMITS["agent"].max_input_tokens
    assert any(
        m["content"].startswith("read page_index") for m in client.calls[-1] if m["role"] == "tool"
    )
    assert p.status == "answered"
    # a question that alone exceeds the limit is never sent
    huge = AgentQuestion("fb_2", "Acme", "ACME_2020_10K", "x", "word " * 20000)
    c2 = ScriptedClient([])
    p2 = run_agent(huge, c2, PAGES, INDEX)  # type: ignore[arg-type]
    assert c2.calls == [] and p2.answer is not None and p2.answer.decline_reason == "context limit"


def test_prompt_contains_no_gold_fields() -> None:
    p, client = run([submit(GOOD)])
    text = (json.dumps(client.calls[0]) + json.dumps(client.tool_sets[0])).lower()
    assert Q.question.lower() in text and Q.doc_name.lower() in text
    for banned in ("justification", "evidence", "gold", "financebench_id"):
        assert banned not in text


def test_declined_by_model_and_registration() -> None:
    d = AgentAnswer(
        answer="", final_value=None, declined=True, decline_reason="no data", citations=[]
    )
    p, _ = run([submit(d.model_dump(mode="json"))])
    assert p.status == "declined" and p.verify_ok is None
    assert SPEC.family == "agent" and SPEC.version == "1" and SPEC.config == CONFIG
    assert SPEC.max_cost_per_question > 0 and CONFIG["max_calls"] == 8


def test_two_tool_calls_in_one_reply_both_answered() -> None:
    p, client = run(
        [
            reply(
                tool_call("read_page", {"page_index": 0}, "a"),
                tool_call("read_page", {"page_index": 2}, "b"),
            ),
            submit(GOOD),
        ]
    )
    roles = [m.get("tool_call_id") for m in client.calls[1] if m["role"] == "tool"]
    assert roles == ["a", "b"] and p.usage.input_tokens == 200
    assert sum(s.usage.input_tokens for s in p.trace) == 200


def test_config_hash_covers_all_model_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from footnote.agent import prompts
    from footnote.systems import hash_config

    base = hash_config(CONFIG)
    for name in ("RETRY_TEMPLATE", "NUDGE_LENGTH", "NUDGE_NO_TOOL", "REPLIES_LEFT"):
        assert getattr(prompts, name) in json.dumps(CONFIG)
    changed = {**CONFIG, "prompt": {**CONFIG["prompt"], "verification_retry": "other"}}
    assert hash_config(changed) != base


def test_failure_while_handling_a_reply_returns_error_with_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(self: Toolbox, name: str, args: dict[str, Any]) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(Toolbox, "run", boom)
    p, client = run([search("a"), search("b")])
    assert p.status == "error" and len(client.calls) == 1
    assert p.usage.input_tokens == 100 and "boom" in p.trace[-1].summary


def test_declined_answers_carry_no_citations() -> None:
    with_cites = {**GOOD, "declined": True, "decline_reason": "no"}
    p, _ = run([submit(with_cites)])
    assert p.status == "declined" and p.answer is not None and p.answer.citations == []
    assert p.answer.decline_reason == "no"
    p, _ = run([submit(BAD), submit(BAD, "s2")])
    assert p.answer is not None and p.answer.citations == []


def test_budget_stated_in_prompt_and_each_tool_result() -> None:
    p, client = run([search("a"), search("b"), submit(GOOD)])
    assert "at most 8 replies" in client.calls[0][0]["content"]
    assert client.calls[1][-1]["content"].endswith("[replies left: 7]")
    assert client.calls[2][-1]["content"].endswith("[replies left: 6]")
    p, client = run([search(f"a{i}") for i in range(7)] + [submit(GOOD)])
    assert "last reply" in client.calls[7][-1]["content"]


def test_rate_limited_call_is_a_retriable_error_charged_nothing() -> None:
    limited = CallFailed(
        "rate limited on every attempt", Usage(0, 0, Decimal(0)), rate_limited=True
    )
    p, _ = run([limited])
    assert p.status == "error" and p.usage.cost == 0 and is_retriable_error(p)
    p, _ = run([failed("0.5")])  # a timeout is an error that stays an error
    assert p.status == "error" and not is_retriable_error(p)


def test_long_quotes_in_submit_arguments_are_cut_in_the_trace() -> None:
    long_quote = "Total revenue was $1,234 million " + "x" * 600
    args = {**GOOD, "citations": [{"page_index": 1, "quote": long_quote}]}
    p, _ = run([search(), submit(args), submit(GOOD, "t")])
    stored = p.trace[1].arguments["citations"][0]["quote"]
    assert len(stored) == 400 and stored.endswith("…") and stored[:-1] == long_quote[:399]
    assert p.trace[2].arguments["citations"][0]["quote"] == GOOD["citations"][0]["quote"]
