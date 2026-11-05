from decimal import Decimal

from footnote.answers import AgentAnswer, FinalValue
from footnote.benchmark import DocInfo
from footnote.prediction import NO_USAGE, Prediction, Step
from footnote.site.wording import (
    decline_headline,
    filing,
    final_value_line,
    problem_notes,
    shown_steps,
)


def test_final_value_line_has_thousands_separators_and_the_stored_unit() -> None:
    assert (
        final_value_line(FinalValue(value=Decimal(1577), unit="usd")) == "Final value: 1,577 (usd)"
    )
    assert (
        final_value_line(FinalValue(value=Decimal("8.74"), unit="ratio"))
        == "Final value: 8.74 (ratio)"
    )
    assert final_value_line(FinalValue(value=Decimal("-1234567.5"), unit="count")) == (
        "Final value: -1,234,567.5 (count)"
    )
    assert final_value_line(None) == "No final number"


def test_filing_spells_the_form_with_a_hyphen() -> None:
    doc = DocInfo("D", "3M", "Industrials", "10K", 2018, "u")
    assert filing(doc) == "3M 10-K 2018"
    assert filing(DocInfo("D", "Acme", "x", "10Q", 2019, "u")) == "Acme 10-Q 2019"
    assert filing(DocInfo("D", "Acme", "x", "EARNINGS", 2019, "u")) == "Acme EARNINGS 2019"


def test_page_problem_wording_claims_only_what_the_record_shows() -> None:
    assert problem_notes(
        ["quote not found on page 59: 'x'; this exact text is on page_index 45"]
    ) == ["Cited page does not match: this text is on page 46, not page 60"]
    assert problem_notes(["quote not found on page 3: 'x'"]) == [
        "Quote not confirmed on the cited page"
    ]
    assert problem_notes(
        ["quote not found on page 3: 'x'; this exact text is on page_index 4, 8"]
    ) == ["Cited page does not match: this text is on pages 5 and 9, not page 4"]
    assert problem_notes(
        ["quote not found on page 3: 'x'; this exact text is on page_index 1, 2, 4"]
    ) == ["Cited page does not match: this text is on pages 2, 3 and 5, not page 4"]
    assert problem_notes(["cited page 70 is not in the filing"]) == [
        "Cited page 71 is not in the filing"
    ]
    assert problem_notes(
        ["cited page 70 is not in the filing; this exact text is on page_index 5"]
    ) == ["Cited page 71 is not in the filing; this text is on page 6"]


def test_other_problems_get_plain_notes_and_repeats_are_dropped() -> None:
    notes = problem_notes(
        [
            "final value 5 is not a correct rounding of a number in a cited quote",
            "quote too long (500 > 400 characters, about 60 words): cut it",
            "quote not found on page 1: 'a'",
            "quote not found on page 2: 'b'",
            "final value not supported by a cited quote or calculation: 7",
        ]
    )
    assert notes[0] == "Final number not traced to the quotes"
    assert notes.count("Final number not traced to the quotes") == 1
    assert notes.count("Quote not confirmed on the cited page") == 1
    assert any(n.startswith("quote too long") for n in notes)


def _step(tool: str, summary: str = "") -> Step:
    return Step(tool, {}, summary, NO_USAGE)


def _pred(trace: list[Step], *, declined: bool = True, reason: str | None = "r") -> Prediction:
    ans = AgentAnswer(
        answer="", final_value=None, declined=declined, decline_reason=reason, citations=[]
    )
    return Prediction("q", "agent", "v", ans, "declined", None, [], trace, NO_USAGE, 1.0)


REJECT = "verification failed: quote not found on page 1: 'a'"


def test_a_decline_states_only_what_the_trace_shows() -> None:
    own = _pred([_step("search_filing"), _step("submit_answer", "declined")])
    assert decline_headline(own, "Footnote agent") == "Footnote agent did not give an answer"
    twice = _pred([_step("submit_answer", REJECT), _step("submit_answer", REJECT)])
    assert decline_headline(twice, "Footnote agent") == (
        "The answer failed the code check twice, so it was withheld"
    )
    stopped = _pred([_step("submit_answer", REJECT), _step("decline", "call limit")])
    assert decline_headline(stopped, "Footnote agent") == (
        "The answer failed the code check and the run stopped before a retry"
    )
    declined_after_one = _pred([_step("submit_answer", REJECT), _step("submit_answer", "declined")])
    assert decline_headline(declined_after_one, "Plain retrieval") == (
        "Plain retrieval did not give an answer"
    )
    no_trace = _pred([])
    assert decline_headline(no_trace, "Long context") == "Long context did not give an answer"


def test_only_tool_steps_are_shown() -> None:
    trace = [
        _step("chat"),
        _step("pack"),
        _step("search_filing"),
        _step("read_page"),
        _step("calculate"),
        _step("submit_answer"),
        _step("search"),
        _step("decline"),
        _step("error"),
    ]
    names = [name for name, _ in shown_steps(trace)]
    assert names == ["search", "read page", "calculate", "submit", "search"]
