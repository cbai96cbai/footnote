from decimal import Decimal

import pytest

from footnote.answers import AgentAnswer, Citation, FinalValue
from footnote.verify import Calc, evaluate_expression, quote_in_page, verify_answer


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
    calc = evaluate_expression("999")
    assert not check("It is 999.", ["Net sales 1,577"], "Net sales 1,577", calcs=[calc])[0]


def test_calc_with_cited_operands_ok() -> None:
    page = "Gross profit 1,018 and revenue 1,557"
    calc = evaluate_expression("1018 / 1557")
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


def fv_answer(text: str, quotes: list[str], value: str, unit: str) -> AgentAnswer:
    a = ans(text, quotes)
    return a.model_copy(update={"final_value": FinalValue(value=Decimal(value), unit=unit)})  # type: ignore[arg-type]


def test_final_value_must_be_grounded() -> None:
    page = "Net sales 1,577"
    good = fv_answer("Net sales were $1,577 million.", ["Net sales 1,577"], "1577", "usd")
    assert verify_answer(good, {0: page}, "?", [])[0]
    in_billions = fv_answer("Net sales were $1,577 million.", ["Net sales 1,577"], "1.6", "usd")
    assert verify_answer(in_billions, {0: page}, "?", [])[0]
    bad = fv_answer("Net sales were $1,577 million.", ["Net sales 1,577"], "4242", "usd")
    ok, problems = verify_answer(bad, {0: page}, "?", [])
    assert not ok and any("final value" in p for p in problems)
    pct = fv_answer("Rate 65.4%", ["Rate 65.4%"], "65.4", "percent")
    assert verify_answer(pct, {0: "Rate 65.4%"}, "?", [])[0]


def test_quote_matches_only_at_token_boundaries() -> None:
    assert not quote_in_page("77", "Net sales 1,777 here")
    assert not quote_in_page("9202", "the years 2019-2020")
    assert not quote_in_page("sales", "wholesales rose")
    assert quote_in_page("1,777", "Net sales 1,777 here")
    assert quote_in_page("Net sales", "Total Net sales, up")
    assert not check("It was 77.", ["77"], "Net sales 1,777")[0]


def test_evaluate_expression_is_safe_and_exact() -> None:
    c = evaluate_expression("(1,018 - 500) × 2 ÷ 4")
    assert c.operands == [Decimal(1018), Decimal(500), Decimal(2), Decimal(4)]
    assert c.result == Decimal(259)
    assert evaluate_expression("-3 + 2 ** 2").result == Decimal(1)
    for bad in ("__import__('os')", "abs(3)", "1 +", "1/0", "2 ** 99", "x + 1", "'a'", ""):
        with pytest.raises(ValueError):
            evaluate_expression(bad)


def test_calc_must_match_its_expression() -> None:
    page = "Net sales 1,577"
    forged = Calc("1577*1", [Decimal(1577)], Decimal(999))
    assert not check("It is 999.", ["Net sales 1,577"], page, calcs=[forged])[0]
    empty = Calc("", [], Decimal(999))
    assert not check("It is 999.", ["Net sales 1,577"], page, calcs=[empty])[0]
    wrong_ops = Calc("1577*1", [Decimal(5)], Decimal(1577))
    assert not check("It is 1,577 times two.", [], page, calcs=[wrong_ops])[0]
    honest = evaluate_expression("1577 + 1577")
    assert check("It is 3,154.", ["Net sales 1,577"], page, calcs=[honest])[0]


def test_quote_longer_than_400_characters_fails() -> None:
    page = "Net sales 1,577 " + "x " * 300
    ok, problems = check("Sales 1,577.", [page], page)
    assert not ok and any("too long" in p for p in problems)
    ok, _ = check("Sales 1,577.", [page[:400]], page)
    assert ok


def test_oversized_or_deeply_nested_expressions_are_rejected_not_crashed() -> None:
    for bad in ("-" * 5000 + "1", "1" + "+1" * 20000, "(" * 150 + "1" + ")" * 150):
        with pytest.raises(ValueError):
            evaluate_expression(bad)
    huge = Calc("-" * 5000 + "1", [Decimal(1)], Decimal(1))
    ok, problems = check("It is 1.", [], "x", calcs=[huge])
    assert not ok and any("calculation rejected" in p for p in problems)


def test_unrecognized_number_format_fails_closed() -> None:
    ok, problems = check("The value is 999e9.", ["Revenue"], "Revenue")
    assert not ok and any("unrecognized number format: 999e9" in p for p in problems)
    assert not check("It is 1.2E3.", ["Revenue 1,200"], "Revenue 1,200")[0]
    assert check("In FY2018 and Q3 under the 10-K, the 3rd.", ["Revenue"], "Revenue")[0]


def test_leading_decimal_point_is_read_as_a_decimal() -> None:
    assert not check("It was .5.", ["Rate 5"], "Rate 5")[0]
    assert check("It was .5.", ["Rate 0.5"], "Rate 0.5")[0]


def test_parenthesised_percent_is_negative() -> None:
    assert not check("It fell 50%.", ["Change (0.5)%"], "Change (0.5)%")[0]
    assert check("It was -0.5%.", ["Change (0.5)%"], "Change (0.5)%")[0]
    assert check("It was (5.2)%.", ["Change (5.2)%"], "Change (5.2)%")[0]


def test_question_number_is_not_a_calculation_operand() -> None:
    calc = evaluate_expression("2020+2020")
    q = "What happened in 2020?"
    assert not check("It is 4,040.", [], "x", question=q, calcs=[calc])[0]


def test_calculation_with_cited_operands_and_unit_constant() -> None:
    page = "Gross profit 1,018 and revenue 1,557"
    calc = evaluate_expression("1018/1557*100")
    ok, problems = check(
        "Margin was 65.4%.", ["Gross profit 1,018", "revenue 1,557"], page, calcs=[calc]
    )
    assert ok, problems
    assert not check("Margin was 65.4%.", ["Gross profit 1,018"], page, calcs=[calc])[0]
