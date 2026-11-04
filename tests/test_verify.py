from decimal import Decimal

from footnote.answers import AgentAnswer, Citation
from footnote.verify import Calc, quote_in_page, verify_answer


def ans(text: str, quotes: list[str], declined: bool = False) -> AgentAnswer:
    return AgentAnswer(
        answer=text,
        final_value=None,
        declined=declined,
        decline_reason=None,
        citations=[Citation(page_index=0, quote=q) for q in quotes],
    )


def check(
    text: str,
    quotes: list[str],
    page: str,
    question: str = "What?",
    calcs: list[Calc] | None = None,
) -> tuple[bool, list[str]]:
    return verify_answer(ans(text, quotes), {0: page}, question, calcs or [])


def test_quote_matches_across_line_break_and_hyphen() -> None:
    assert quote_in_page("net sales increased", "Total net sa-\nles  increased 3%")
    assert quote_in_page("long-term debt", "the long-\nterm debt")
    assert quote_in_page("Net Sales", "NET SALES")


def test_quote_not_on_page_fails() -> None:
    assert not quote_in_page("net sales decreased", "net sales increased")
    assert not quote_in_page("", "anything")
    ok, problems = check("Sales fell.", ["net sales decreased"], "net sales increased")
    assert not ok and problems


def test_cited_page_missing_fails() -> None:
    ok, problems = verify_answer(ans("x", ["a"]), {5: "a"}, "?", [])
    assert not ok and problems


def test_number_from_quote_ok() -> None:
    assert check("Net sales fell $1.6 billion.", ["Net sales 1,577"], "Net sales 1,577 here")[0]


def test_invented_number_fails() -> None:
    ok, problems = check("Revenue was $777 million.", ["Net sales 1,577"], "Net sales 1,577")
    assert not ok and any("777" in p for p in problems)


def test_unscaled_number_needs_exact_rounding() -> None:
    assert not check("It was 1.6.", ["Net sales 1,577"], "Net sales 1,577")[0]


def test_calc_with_uncited_operand_fails() -> None:
    calc = Calc("999", [Decimal(999)], Decimal(999))
    assert not check("It is 999.", ["Net sales 1,577"], "Net sales 1,577", calcs=[calc])[0]


def test_calc_with_cited_operands_ok() -> None:
    page = "Gross profit 1,018 and revenue 1,557"
    calc = Calc("1018 / 1557", [Decimal(1018), Decimal(1557)], Decimal(1018) / Decimal(1557))
    ok, problems = check(
        "Margin was 65.4%.", ["Gross profit 1,018", "revenue 1,557"], page, calcs=[calc]
    )
    assert ok, problems


def test_percent_needs_calc_or_quote() -> None:
    assert not check("Margin was 65.4%.", ["Net sales 1,577"], "Net sales 1,577")[0]
    assert check("Rate was 65.4%.", ["Rate 65.4%"], "Rate 65.4%")[0]
    assert check("Rate was 65.4%.", ["Rate 0.654"], "Rate 0.654")[0]
    assert not check("Rate was 6540%.", ["Rate 65.4%"], "Rate 65.4%")[0]


def test_question_numbers_exempt_years_need_citation() -> None:
    assert check("In 2019 it grew.", [], "x", question="What happened in 2019?")[0]
    assert not check("In 2019 it grew.", [], "x", question="What happened last year?")[0]
    assert check("In 2019 it grew.", ["fiscal 2019 results"], "fiscal 2019 results")[0]


def test_accounting_parentheses_with_scale_word_after() -> None:
    page = "Change (1,577) million"
    assert check("It was -$1.6 billion.", ["(1,577) million"], page)[0]
    assert check("It was $(1.2) billion.", ["$(1.2) billion"], "Total $(1.2) billion")[0]


def test_declined_answer_checks_citations_only() -> None:
    a = ans("Not found; 999 pages searched.", [], declined=True)
    assert verify_answer(a, {0: "x"}, "?", []) == (True, [])
