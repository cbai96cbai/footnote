from fakes import reply, text_reply, tool_call
from test_loop import GOOD, run, submit

from footnote.agent.prompts import RETRY_TEMPLATE, SYSTEM_PROMPT, TOOLS
from footnote.agent.system import CONFIG

HALF = {**GOOD, "final_value": {"value": 617, "unit": "usd"}}  # 1234 / 2 was done in the head
RETRY_NAMES = ["calculate", "submit_answer"]


def names(tools: list[dict[str, object]] | None) -> list[str]:
    return [t["function"]["name"] for t in tools or []]  # type: ignore[index]


def test_the_retry_may_calculate_then_submit_in_one_reply() -> None:
    both = reply(
        tool_call("calculate", {"expression": "1234/2"}, "c"),
        tool_call("submit_answer", HALF, "s2"),
    )
    p, client = run([submit(HALF, "s1"), both])
    assert p.status == "answered" and p.verify_ok is True and len(client.calls) == 2
    assert [s.tool for s in p.trace] == ["submit_answer", "calculate", "submit_answer"]


def test_the_retry_call_offers_only_calculate_and_submit_and_requires_a_tool() -> None:
    p, client = run([submit(HALF, "s1"), submit(GOOD, "s2")])
    assert client.tool_choices == [None, "required"]
    assert names(client.tool_sets[0]) == names(TOOLS) and len(TOOLS) == 4
    assert names(client.tool_sets[1]) == RETRY_NAMES
    assert client.tool_sets[1] == [t for t in TOOLS if t["function"]["name"] in RETRY_NAMES]


def test_the_retry_after_a_late_failure_is_also_restricted() -> None:
    from test_submission import researching

    p, client = run([*researching(6), submit(HALF, "s1"), submit(GOOD, "s2")])
    assert client.tool_choices[7] == "required" and names(client.tool_sets[7]) == RETRY_NAMES
    assert client.tool_choices[6] == {"type": "function", "function": {"name": "submit_answer"}}


def test_a_retry_reply_with_only_a_calculation_ends_declined_with_the_problems() -> None:
    only_calc = reply(tool_call("calculate", {"expression": "1234/2"}, "c"))
    p, client = run([submit(HALF, "s1"), only_calc, submit(HALF, "s3")])
    assert len(client.calls) == 2 and p.status == "declined" and p.verify_ok is False
    assert p.verify_problems and p.verify_problems[0].startswith("final value 617")
    assert p.answer is not None and p.answer.decline_reason is not None
    assert p.answer.decline_reason.startswith("final value 617")


def test_a_retry_reply_without_any_tool_call_ends_declined_too() -> None:
    p, client = run([submit(HALF, "s1"), text_reply("617"), submit(GOOD, "s3")])
    assert len(client.calls) == 2 and p.status == "declined" and p.verify_ok is False
    assert p.verify_problems


def test_a_malformed_submission_on_the_retry_still_gets_another_reply() -> None:
    p, client = run([submit(HALF, "s1"), submit({"answer": "x"}, "s2"), submit(GOOD, "s3")])
    assert p.status == "answered" and len(client.calls) == 3
    assert client.tool_choices == [None, "required", "required"]


def test_the_texts_tell_the_model_it_may_calculate_in_the_retry_reply() -> None:
    sentence = "in the same reply, call calculate for any missing step and then submit_answer"
    assert sentence in " ".join(RETRY_TEMPLATE.split())
    assert sentence in " ".join(SYSTEM_PROMPT.split())
    assert "search or read again" not in RETRY_TEMPLATE


def test_the_retry_restriction_is_registered_in_the_config() -> None:
    from footnote.systems import hash_config

    assert CONFIG["retry_tool_choice"] == "required"
    assert CONFIG["retry_tools"] == RETRY_NAMES
    base = hash_config(CONFIG)
    assert hash_config({**CONFIG, "retry_tool_choice": None}) != base
    assert hash_config({**CONFIG, "retry_tools": ["submit_answer"]}) != base


def test_the_agent_hash_differs_from_the_version_2_shape() -> None:
    from footnote.systems import hash_config

    older = {k: v for k, v in CONFIG.items() if not k.startswith("retry_")}
    older = {**older, "version": "2", "verify_version": "2"}
    assert CONFIG["version"] == "3" and CONFIG["verify_version"] == "3"
    assert hash_config(older) != hash_config(CONFIG)
    for key in ("version", "verify_version"):
        assert hash_config({**CONFIG, key: "2"}) != hash_config(CONFIG)
