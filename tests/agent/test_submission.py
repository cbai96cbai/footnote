import json
from typing import Any

from fakes import FakeIndex, ScriptedClient, reply, text_reply, tool_call
from test_loop import BAD, GOOD, PAGES, run, search, submit

from footnote.agent.loop import run_agent
from footnote.agent.prompts import RETRY_TEMPLATE, SYSTEM_PROMPT, TOOLS, USER_TEMPLATE
from footnote.agent.submit import MAX_PROBLEM_TOKENS, judge
from footnote.agent.submit import RESEARCH_INPUT_TARGET as TARGET
from footnote.benchmark import AgentQuestion
from footnote.budget.limits import LIMITS
from footnote.llm.client import ChatResult
from footnote.llm.tokens import count_input_tokens, count_text_tokens

LIMIT = LIMITS["agent"].max_input_tokens
FORCE = {"type": "function", "function": {"name": "submit_answer"}}


def researching(n: int) -> list[ChatResult | Exception]:
    return [search(f"a{i}") for i in range(n)]


def test_submission_is_forced_for_the_last_two_replies_only() -> None:
    p, client = run([*researching(6), submit(GOOD, "s")])
    assert client.tool_choices == [None] * 6 + [FORCE]
    assert p.status == "answered" and len(client.calls) == 7
    p, client = run(researching(12))
    assert client.tool_choices == [None] * 6 + [FORCE, FORCE] and len(client.calls) == 8
    assert p.answer is not None and p.answer.decline_reason == "call limit"


def test_first_forced_submission_fails_and_the_forced_retry_passes() -> None:
    p, client = run([*researching(6), submit(BAD, "s1"), submit(GOOD, "s2")])
    assert client.tool_choices == [None] * 6 + [FORCE, FORCE]
    assert p.status == "answered" and p.verify_ok is True and len(client.calls) == 8
    assert "Verification found problems" in json.dumps(client.calls[7])


def test_an_earlier_malformed_submission_does_not_change_when_forcing_starts() -> None:
    p, client = run([submit({"answer": "x"}, "s1"), *researching(5), submit(GOOD, "s2")])
    assert client.tool_choices == [None] * 6 + [FORCE]
    assert p.status == "answered" and p.verify_ok is True and len(client.calls) == 7


def test_a_cut_off_reply_in_the_submission_phase_uses_one_of_the_two() -> None:
    p, client = run([*researching(6), ChatResult({}, search().usage, "length", 0.1), submit(GOOD)])
    assert client.tool_choices == [None] * 6 + [FORCE, FORCE]
    assert p.status == "answered" and len(client.calls) == 8
    p, client = run([*researching(6), text_reply("", "length"), text_reply("", "length")])
    assert len(client.calls) == 8 and p.answer is not None
    assert p.answer.decline_reason == "call limit"


def test_a_schema_invalid_submission_in_the_submission_phase_uses_one_of_the_two() -> None:
    p, client = run([*researching(6), submit({"answer": "x"}, "s1"), submit(GOOD, "s2")])
    assert client.tool_choices == [None] * 6 + [FORCE, FORCE]
    assert p.status == "answered" and p.verify_ok is True
    assert client.calls[7][-1]["content"].startswith("error: answer does not match")


def test_the_budget_markers_say_what_the_next_reply_must_be() -> None:
    p, client = run([*researching(6), submit(GOOD, "s")])
    assert client.calls[5][-1]["content"].endswith("[replies left: 3]")
    assert "must be submit_answer" in client.calls[6][-1]["content"]
    prompt = " ".join(client.calls[0][0]["content"].split())
    assert "at most 8 replies" in prompt and "next reply must be submit_answer" in prompt


def sent_tool_ids(client: ScriptedClient, call: int) -> list[str]:
    return [m["tool_call_id"] for m in client.calls[call] if m["role"] == "tool"]


def test_only_the_first_submission_in_a_reply_is_evaluated() -> None:
    p, client = run(
        [
            reply(tool_call("submit_answer", BAD, "s1"), tool_call("submit_answer", BAD, "s2")),
            submit(GOOD, "s3"),
        ]
    )
    assert p.status == "answered" and p.verify_ok is True and len(client.calls) == 2
    msgs = {m["tool_call_id"]: m["content"] for m in client.calls[1] if m["role"] == "tool"}
    assert "Verification found problems" in msgs["s1"]
    assert msgs["s2"].startswith("ignored: one submission per reply")
    p, client = run(
        [reply(tool_call("submit_answer", GOOD, "s1"), tool_call("submit_answer", BAD, "s2"))]
    )
    assert p.status == "answered" and len(client.calls) == 1


def test_calculate_then_submit_in_one_reply_sees_the_calculation() -> None:
    chained = {**GOOD, "final_value": {"value": 617, "unit": "usd"}}
    p, _ = run(
        [
            reply(
                tool_call("calculate", {"expression": "1234/2"}, "c"),
                tool_call("submit_answer", chained, "s"),
            )
        ]
    )
    assert p.status == "answered" and p.verify_ok is True
    assert [s.tool for s in p.trace] == ["calculate", "submit_answer"]


def test_submit_then_calculate_in_one_reply_runs_both() -> None:
    p, client = run(
        [
            reply(
                tool_call("submit_answer", BAD, "s"),
                tool_call("calculate", {"expression": "1+1"}, "c"),
            ),
            submit(GOOD, "s2"),
        ]
    )
    assert p.status == "answered" and len(client.calls) == 2
    msgs = {m["tool_call_id"]: m["content"] for m in client.calls[1] if m["role"] == "tool"}
    assert "Verification found problems" in msgs["s"] and msgs["c"].startswith("1+1 = 2")


def question_of(opening_tokens: int) -> AgentQuestion:
    """A question sized so that the opening call is `opening_tokens` input tokens."""
    empty = AgentQuestion("fb_9", "Acme", "ACME_2020_10K", "x", "")
    base = count_input_tokens(first_messages(empty), TOOLS, None)
    return AgentQuestion("fb_9", "Acme", "ACME_2020_10K", "x", "word " * (opening_tokens - base))


def first_messages(q: AgentQuestion) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                doc_name=q.doc_name, company=q.company, question=q.question
            ),
        },
    ]


def sent_size(client: ScriptedClient, i: int) -> int:
    return count_input_tokens(client.calls[i], client.tool_sets[i], None)


def test_trimming_keeps_the_retry_instruction_intact() -> None:
    many_bad = {**BAD, "citations": BAD["citations"] * 25}
    q = question_of(TARGET - 3500)
    read = reply(tool_call("read_page", {"page_index": 0}, "r"))
    client = ScriptedClient([read, submit(many_bad), submit(GOOD, "s2")])
    pages = {**PAGES, 0: "word " * 2900}
    p = run_agent(q, client, pages, FakeIndex([(1, PAGES[1])]))  # type: ignore[arg-type]
    assert p.status == "answered" and len(client.calls) == 3
    original = "[page_index 0]\n" + "word " * 2900
    old = next(m for m in client.calls[-1] if m.get("tool_call_id") == "r")
    assert old["content"].startswith("read page_index 0") and len(old["content"]) < len(original)
    assert old["content"] == next(s.summary for s in p.trace if s.tool == "read_page")
    retry = next(m for m in client.calls[-1] if m.get("tool_call_id") == "s")
    assert retry["content"].startswith("Verification found problems")
    assert retry["content"].count("quote not found") == 25


def test_a_retry_research_cannot_make_room_for_ends_in_context_limit_not_a_cut_one() -> None:
    # 25 citations add ~550 tokens to the call and ~550 more as feedback
    many_bad = {**BAD, "citations": BAD["citations"] * 25}
    client = ScriptedClient([submit(many_bad), submit(GOOD, "s2")])
    q = question_of(TARGET - 900)
    p = run_agent(q, client, PAGES, FakeIndex([(1, PAGES[1])]))  # type: ignore[arg-type]
    assert len(client.calls) == 1  # the retry cannot fit without cutting its own instruction
    assert p.answer is not None and p.answer.decline_reason == "context limit"
    assert p.verify_ok is False and p.verify_problems


def test_research_that_cannot_get_under_the_target_ends_in_context_limit() -> None:
    client = ScriptedClient([submit(GOOD)])
    q = question_of(TARGET + 500)
    assert sent_opening(q) <= LIMIT  # it would fit the per-call limit; research is held lower
    p = run_agent(q, client, PAGES, FakeIndex([(1, PAGES[1])]))  # type: ignore[arg-type]
    assert client.calls == [] and p.answer is not None
    assert p.answer.decline_reason == "context limit"


def sent_opening(q: AgentQuestion) -> int:
    return count_input_tokens(first_messages(q), TOOLS, None)


def test_a_long_last_research_reply_still_leaves_the_retry_a_call() -> None:
    big = {0: "word " * 2900, 1: PAGES[1], 2: "y " * 2900}
    reads = [reply(tool_call("read_page", {"page_index": i % 2 * 2}, f"r{i}")) for i in range(5)]
    long_reply = reply(tool_call("search_filing", {"query": "revenue " * 2900}, "long"))
    client = ScriptedClient([*reads, long_reply, submit(BAD, "s1"), submit(GOOD, "s2")])
    p = run_agent(question_of(12_700), client, big, FakeIndex([(1, PAGES[1])]))  # type: ignore[arg-type]
    assert p.status == "answered" and p.verify_ok is True and len(client.calls) == 8
    assert client.tool_choices == [None] * 6 + [FORCE, FORCE]
    assert all(sent_size(client, i) <= TARGET for i in range(6))
    assert sent_size(client, 7) > 15_000  # far beyond the research target; all tools are sent
    assert all(sent_size(client, i) <= LIMIT for i in (6, 7))
    retry = next(m for m in client.calls[7] if m.get("tool_call_id") == "s1")
    assert retry["content"].startswith("Verification found problems")


def retry_parts(feedback: str) -> tuple[str, str]:
    head, tail = RETRY_TEMPLATE.split("{problems}")
    assert feedback.startswith(head) and feedback.endswith(tail)
    return head, feedback[len(head) : len(feedback) - len(tail)]


def test_the_problem_list_in_the_feedback_is_capped_and_the_instruction_stays_whole() -> None:
    texts = {
        "ordinary": "Total revenue was $9,999 million",
        "non-latin": "総収益は九千九百九十九百万ドルでした" * 3,
    }
    for name, quote in texts.items():
        args = {
            **BAD,
            "citations": [{"page_index": 1, "quote": quote + str(i)} for i in range(300)],
        }
        attempt = judge(args, PAGES, "?", [], retried=False)
        assert attempt.kind == "failed" and len(attempt.problems) >= 300, name
        _, problems = retry_parts(attempt.feedback)
        assert count_text_tokens(problems) <= MAX_PROBLEM_TOKENS, name
        assert problems.endswith("…") and problems.startswith("quote not found"), name
    short = judge(BAD, PAGES, "?", [], retried=False)
    assert retry_parts(short.feedback)[1] == "; ".join(short.problems)


def test_config_hash_covers_the_new_texts_the_search_width_and_the_forcing_rule() -> None:
    from footnote.agent import prompts
    from footnote.agent.system import CONFIG
    from footnote.systems import hash_config

    base = hash_config(CONFIG)
    texts = list(CONFIG["prompt"].values())
    for name in ("SYSTEM_PROMPT", "RETRY_TEMPLATE", "SUBMIT_NEXT", "LAST_REPLY", "REPLIES_LEFT"):
        assert getattr(prompts, name) in texts
    assert CONFIG["version"] == "4" and CONFIG["tool_limits"]["search_k"] == 8
    assert CONFIG["research_input_target_tokens"] == TARGET == 13_000
    assert CONFIG["max_problem_tokens"] == MAX_PROBLEM_TOKENS == 1_000
    assert CONFIG["limits"]["max_input_tokens"] == 24_000
    assert CONFIG["submit_forced_when_left"] == 2 and CONFIG["submit_tool_choice"] == FORCE
    assert CONFIG["tools"] == prompts.TOOLS
    prompt = {**CONFIG["prompt"], "system": CONFIG["prompt"]["system"] + " "}
    for changed in (
        {**CONFIG, "submit_forced_when_left": 1},
        {**CONFIG, "research_input_target_tokens": 12_000},
        {**CONFIG, "max_problem_tokens": 500},
        {**CONFIG, "submit_tool_choice": None},
        {**CONFIG, "prompt": prompt},
        {**CONFIG, "prompt": {**CONFIG["prompt"], "submit_next": "x"}},
        {**CONFIG, "tool_limits": {**CONFIG["tool_limits"], "search_k": 5}},
        {**CONFIG, "version": "3"},
    ):
        assert hash_config(changed) != base


def test_every_model_visible_constant_and_the_check_version_change_the_agent_hash() -> None:
    from footnote.agent import submit
    from footnote.agent.system import CONFIG
    from footnote.systems import hash_config
    from footnote.verify import VERIFY_VERSION

    base = hash_config(CONFIG)
    texts = list(CONFIG["prompt"].values())
    assert submit.IGNORED_EXTRA in texts and submit.IGNORED_AFTER in texts
    assert CONFIG["verify_version"] == VERIFY_VERSION == "4"
    for key in ("ignored_extra_submission", "ignored_after_submission"):
        assert hash_config({**CONFIG, "prompt": {**CONFIG["prompt"], key: "other"}}) != base
    assert hash_config({**CONFIG, "verify_version": "3"}) != base


def test_a_schema_error_names_the_wrong_field_and_the_expected_shape() -> None:
    attempt = judge({**GOOD, "final_value": 16.52}, PAGES, "?", [], retried=False)
    assert attempt.kind == "invalid"
    assert "final_value:" in attempt.feedback
    assert '{"value": <number>, "unit": "usd" | "percent" | "ratio" | "count" | "other"}' in (
        attempt.feedback
    )
    assert "null" in attempt.feedback
    many = judge({"citations": 3}, PAGES, "?", [], retried=False)
    assert many.feedback.count("Field required") == 3 and "and 2 more" in many.feedback
    assert (
        "final_value must be"
        not in judge({**GOOD, "citations": 3}, PAGES, "?", [], retried=False).feedback
    )
