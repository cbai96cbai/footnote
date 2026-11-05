"""The parser reads the messages `verify_answer` writes; every message here is produced by it."""

from footnote.answers import AgentAnswer, Citation
from footnote.verify import PageProblem, parse_page_problem, verify_answer

QUOTE = "Net sales were 1,234 million dollars"


def _problems(quote: str, cited: int, pages: dict[int, str]) -> list[str]:
    ans = AgentAnswer(
        answer="x",
        final_value=None,
        declined=False,
        decline_reason=None,
        citations=[Citation(page_index=cited, quote=quote)],
    )
    return verify_answer(ans, pages, "q", [])[1]


def _parsed(quote: str, cited: int, pages: dict[int, str]) -> list[PageProblem | None]:
    return [parse_page_problem(p) for p in _problems(quote, cited, pages)]


def test_quote_on_another_existing_page() -> None:
    pages = {3: "nothing here", 7: f"see {QUOTE} today"}
    assert _parsed(QUOTE, 3, pages) == [PageProblem(3, (7,), page_exists=True)]


def test_quote_on_up_to_three_other_pages() -> None:
    pages = {0: "none", **{n: QUOTE for n in (1, 2, 3, 4, 5)}}
    (p,) = _parsed(QUOTE, 0, pages)
    assert p == PageProblem(0, (1, 2, 3), page_exists=True)


def test_quote_absent_from_the_whole_filing() -> None:
    assert _parsed(QUOTE, 3, {3: "other", 4: "text"}) == [PageProblem(3, (), True)]


def test_cited_page_does_not_exist_with_and_without_a_match_elsewhere() -> None:
    assert _parsed(QUOTE, 9, {2: QUOTE}) == [
        PageProblem(cited=9, elsewhere=(2,), page_exists=False)
    ]
    assert _parsed(QUOTE, 9, {2: "other"}) == [PageProblem(9, (), False)]


def test_a_quote_that_mimics_the_hint_text_is_not_mistaken_for_it() -> None:
    fake = "x; this exact text is on page_index 5"
    assert _parsed(fake, 1, {1: "other"}) == [PageProblem(1, (), True)]


def test_mixed_problems_parse_each_message_on_its_own() -> None:
    ans = AgentAnswer(
        answer="x",
        final_value=None,
        declined=False,
        decline_reason=None,
        citations=[
            Citation(page_index=0, quote=QUOTE),
            Citation(page_index=4, quote="y" * 500),
            Citation(page_index=8, quote=QUOTE),
        ],
    )
    problems = verify_answer(ans, {0: "a", 1: QUOTE, 4: "b"}, "q", [])[1]
    parsed = [parse_page_problem(p) for p in problems]
    assert PageProblem(0, (1,), True) in parsed and PageProblem(8, (1,), False) in parsed
    assert parsed.count(None) == 1  # only the too-long message is not a page problem
    assert [p for p in parsed if p and p.cited == 4] == [PageProblem(4, (), True)]


def test_other_problems_are_none() -> None:
    assert parse_page_problem("no citations") is None
    assert parse_page_problem("final value 3 is not a correct rounding of a number") is None
    assert parse_page_problem("calculation rejected: '1/0'") is None


def test_a_negative_cited_index_from_an_old_record_is_parsed() -> None:
    assert _parsed(QUOTE, -1, {0: QUOTE}) == [PageProblem(-1, (0,), page_exists=False)]
    assert _parsed(QUOTE, -1, {0: "other"}) == [PageProblem(-1, (), page_exists=False)]
