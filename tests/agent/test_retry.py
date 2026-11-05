from fakes import reply, tool_call
from test_loop import GOOD, run, submit

from footnote.agent import prompts
from footnote.agent.prompts import RETRY_TEMPLATE, SYSTEM_PROMPT, TOOLS
from footnote.agent.system import CONFIG

HALF = {**GOOD, "final_value": {"value": 617, "unit": "usd"}}  # 1234 / 2 was done in the head
FORCE = {"type": "function", "function": {"name": "submit_answer"}}


def names(tools: list[dict[str, object]] | None) -> list[str]:
    return [t["function"]["name"] for t in tools or []]  # type: ignore[index]


def test_an_early_failure_leaves_the_next_reply_free_to_calculate_then_resubmit() -> None:
    from test_submission import researching

    calc = reply(tool_call("calculate", {"expression": "1234/2"}, "c"))
    p, client = run([*researching(2), submit(HALF, "s1"), calc, submit(HALF, "s2")])
    assert p.status == "answered" and p.verify_ok is True and len(client.calls) == 5
    assert client.tool_choices == [None] * 5  # five replies were left after the failure
    assert all(names(t) == names(TOOLS) for t in client.tool_sets)
    assert "compute it with calculate" in client.calls[3][-1]["content"]


def test_a_late_failure_gets_a_forced_resubmission_with_all_tools() -> None:
    from test_submission import researching

    p, client = run([*researching(6), submit(HALF, "s1"), submit(GOOD, "s2")])
    assert p.status == "answered" and len(client.calls) == 8
    assert client.tool_choices[6:] == [FORCE, FORCE]
    assert names(client.tool_sets[7]) == names(TOOLS) and len(TOOLS) == 4
    feedback = client.calls[7][-1]["content"]
    assert "resubmit with a final value" in feedback and "compute it" not in feedback


def test_a_second_failure_ends_declined_with_the_problems() -> None:
    p, client = run([submit(HALF, "s1"), submit(HALF, "s2")])
    assert p.status == "declined" and p.verify_ok is False and len(client.calls) == 2
    assert p.verify_problems and p.verify_problems[0].startswith("final value 617")


def test_a_free_retry_reply_that_only_calculates_does_not_end_the_question() -> None:
    only_calc = reply(tool_call("calculate", {"expression": "1234/2"}, "c"))
    p, client = run([submit(HALF, "s1"), only_calc, submit(HALF, "s3")])
    assert p.status == "answered" and len(client.calls) == 3


def test_a_malformed_submission_on_the_retry_still_gets_another_reply() -> None:
    p, client = run([submit(HALF, "s1"), submit({"answer": "x"}, "s2"), submit(GOOD, "s3")])
    assert p.status == "answered" and len(client.calls) == 3
    assert client.tool_choices == [None, None, None]


def test_the_texts_say_calculation_belongs_before_the_first_submission() -> None:
    sentence = (
        "Run calculate for every step before you submit: if a check fails with two or fewer "
        "replies left, the next reply can only resubmit."
    )
    assert sentence in " ".join(SYSTEM_PROMPT.split())
    assert "calculate" not in RETRY_TEMPLATE and "offers only" not in SYSTEM_PROMPT
    assert "search or read again" not in RETRY_TEMPLATE


def test_the_retry_has_no_special_rule_in_the_config() -> None:
    assert not [k for k in CONFIG if k.startswith("retry_")]
    assert not hasattr(prompts, "RETRY_TOOLS") and not hasattr(prompts, "RETRY_TOOL_CHOICE")


def test_the_agent_hash_differs_from_the_version_3_shape() -> None:
    from footnote.systems import hash_config

    older = {
        **CONFIG,
        "version": "3",
        "verify_version": "3",
        "retry_tool_choice": "required",
        "retry_tools": ["calculate", "submit_answer"],
    }
    assert CONFIG["version"] == "4" and CONFIG["verify_version"] == "4"
    assert hash_config(older) != hash_config(CONFIG)
    for key in ("version", "verify_version"):
        assert hash_config({**CONFIG, key: "3"}) != hash_config(CONFIG)


def test_the_prompt_rules_for_words_dashes_and_quote_length_are_pinned() -> None:
    system = " ".join(SYSTEM_PROMPT.split())
    retry = " ".join(RETRY_TEMPLATE.split())
    assert (
        "If the filing states the answer in words or shows it as a dash (for example 'there were "
        "no such costs', or '—' in a table), set final_value to null and state the answer in "
        "words." in system
    )
    assert (
        "If the filing does not show the figure separately, say that, cite what it does show, "
        "and do not submit 0." in system
    )
    assert "at most 400 characters (about 60 words)" in system
    assert "at most 400 characters (about 60 words)" in retry


def test_the_prompt_keeps_field_names_out_of_the_answer_text() -> None:
    assert (
        "Write the answer text for a reader: state the figure in words and numbers, and never "
        "name fields such as final_value or citations." in " ".join(SYSTEM_PROMPT.split())
    )


def test_the_free_or_forced_boundary_after_a_failure_is_three_versus_two_replies_left() -> None:
    from test_submission import researching

    # failure at reply 5 leaves 3 replies: the next is free
    p, client = run([*researching(4), submit(HALF, "s1"), submit(GOOD, "s2")])
    assert client.tool_choices[5] is None
    # failure at reply 6 leaves 2 replies: the next is forced
    p, client = run([*researching(5), submit(HALF, "s1"), submit(GOOD, "s2")])
    assert client.tool_choices[6] == FORCE
    assert "resubmit with a final value" in client.calls[6][-1]["content"]
    assert client.tool_choices[5] is None
