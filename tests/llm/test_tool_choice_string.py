from test_client import MSGS, TOOLS, client, ok, status_error

from footnote.budget.limits import LIMITS


def test_a_string_tool_choice_is_passed_through_unchanged() -> None:
    c, sdk = client([ok(), status_error(429), ok()])
    c.chat(
        MSGS,
        deployment="footnote-agent",
        limits=LIMITS["agent"],
        tools=TOOLS,
        tool_choice="required",
    )
    c.chat(
        MSGS,
        deployment="footnote-agent",
        limits=LIMITS["agent"],
        tools=TOOLS,
        tool_choice="required",
    )
    assert [k["tool_choice"] for k in sdk.calls] == ["required"] * 3
