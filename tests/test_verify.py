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


def test_declined_answer_checks_citations_only() -> None:
    a = ans("Not found; 999 pages searched.", [], declined=True)
    assert verify_answer(a, {0: "x"}, "?", []) == (True, [])


def fv_answer(text: str, quotes: list[str], value: str, unit: str) -> AgentAnswer:
    a = ans(text, quotes)
    return a.model_copy(update={"final_value": FinalValue(value=Decimal(value), unit=unit)})  # type: ignore[arg-type]


def test_quote_matches_only_at_token_boundaries() -> None:
    assert not quote_in_page("77", "Net sales 1,777 here")
    assert not quote_in_page("9202", "the years 2019-2020")
    assert not quote_in_page("sales", "wholesales rose")
    assert quote_in_page("1,777", "Net sales 1,777 here")
    assert quote_in_page("Net sales", "Total Net sales, up")
    assert not check("x", ["77"], "Net sales 1,777")[0]


def test_evaluate_expression_is_safe_and_exact() -> None:
    c = evaluate_expression("(1,018 - 500) × 2 ÷ 4")
    assert c.operands == [Decimal(1018), Decimal(500), Decimal(2), Decimal(4)]
    assert c.result == Decimal(259)
    assert evaluate_expression("-3 + 2 ** 2").result == Decimal(1)
    for bad in ("__import__('os')", "abs(3)", "1 +", "1/0", "2 ** 99", "x + 1", "'a'", ""):
        with pytest.raises(ValueError):
            evaluate_expression(bad)


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


PAGE = "Net sales 1,577 and gross profit 1,018 and revenue 1,557"


def fv_check(
    value: str,
    unit: str,
    quotes: list[str],
    calcs: list[Calc] | None = None,
    page: str = PAGE,
    text: str = "See filing.",
    question: str = "What?",
) -> tuple[bool, list[str]]:
    a = fv_answer(text, quotes, value, unit)
    return verify_answer(a, {0: page}, question, calcs or [])


def test_final_value_from_a_cited_number() -> None:
    assert fv_check("1577", "usd", ["Net sales 1,577"])[0]
    assert fv_check("1.6", "usd", ["Net sales 1,577"])[0]
    ok, problems = fv_check("999", "usd", ["Net sales 1,577"])
    assert not ok and any("final value" in p for p in problems)


def test_final_value_percent_from_a_calculation() -> None:
    calc = evaluate_expression("1018/1557*100")
    quotes = ["gross profit 1,018", "revenue 1,557"]
    assert fv_check("65.4", "percent", quotes, [calc])[0]
    assert not fv_check("65.4", "percent", ["gross profit 1,018"], [calc])[0]


def test_percent_is_not_made_from_a_plain_quoted_number() -> None:
    page = "Rate (0.5) percent"
    assert not fv_check("50", "percent", ["Rate (0.5) percent"], page=page)[0]
    assert fv_check("65.4", "percent", ["Rate 65.4%"], page="Rate 65.4%")[0]
    assert not fv_check("65.4", "percent", ["Rate 0.654"], page="Rate 0.654")[0]


def test_calculation_needs_at_least_one_cited_operand() -> None:
    avg = evaluate_expression("(1577+1018)/2")
    assert fv_check("1298", "usd", ["Net sales 1,577", "gross profit 1,018"], [avg])[0]
    half = evaluate_expression("1577/2")
    assert fv_check("789", "usd", ["Net sales 1,577"], [half])[0]
    consts = evaluate_expression("12*365")
    assert not fv_check("4380", "usd", ["Net sales 1,577"], [consts])[0]


def test_question_numbers_are_never_calculation_operands() -> None:
    calc = evaluate_expression("2020+2020")
    q = "What happened in 2020?"
    assert not fv_check("4040", "usd", ["Net sales 1,577"], [calc], question=q)[0]


def test_calculation_must_match_its_expression() -> None:
    forged = Calc("1577*1", [Decimal(1577)], Decimal(999))
    assert not fv_check("999", "usd", ["Net sales 1,577"], [forged])[0]
    empty = Calc("", [], Decimal(999))
    assert not fv_check("999", "usd", ["Net sales 1,577"], [empty])[0]
    wrong_ops = Calc("1577*1", [Decimal(5)], Decimal(1577))
    assert not fv_check("1577", "usd", ["Net sales 1,577"], [wrong_ops])[0]
    honest = evaluate_expression("1577 + 1577")
    assert fv_check("3154", "usd", ["Net sales 1,577"], [honest])[0]


def test_answered_needs_a_citation_declined_does_not() -> None:
    ok, problems = fv_check("1577", "usd", [])
    assert not ok and "no citations" in problems
    assert verify_answer(ans("Not found.", [], declined=True), {0: "x"}, "?", []) == (True, [])


def test_free_text_numbers_do_not_affect_the_result() -> None:
    text = "In USD999, FY2099, 999th, 999e9 and (0.5) percent."
    assert fv_check("1577", "usd", ["Net sales 1,577"], text=text)[0]
    assert fv_check("1577", "usd", ["Net sales 1,577"], text="Sales were 12.")[0]


def test_units_limit_the_scale_a_quoted_number_may_take() -> None:
    assert not fv_check("654", "percent", ["Rate 0.654"], page="Rate 0.654")[0]
    assert not fv_check("1.6", "percent", ["Net sales 1,577"])[0]
    assert not fv_check("1.58", "ratio", ["Net sales 1,577"])[0]
    assert not fv_check("65.4", "usd", ["Rate 65.4%"], page="Rate 65.4%")[0]
    assert not fv_check("65.4", "count", ["Rate 65.4%"], page="Rate 65.4%")[0]
    assert fv_check("65.4", "percent", ["Rate 65.4"], page="Rate 65.4")[0]
    assert fv_check("65.4", "percent", ["Rate 65.4%"], page="Rate 65.4%")[0]
    assert fv_check("1.6", "usd", ["Net sales 1,577"])[0]
    assert fv_check("1577", "ratio", ["Net sales 1,577"])[0]


def test_percent_from_a_calculation_result_times_100() -> None:
    calc = evaluate_expression("1018/1557")
    quotes = ["gross profit 1,018", "revenue 1,557"]
    assert fv_check("65.4", "percent", quotes, [calc])[0]
    assert not fv_check("65.4", "ratio", quotes, [calc])[0]
    assert not fv_check("65.4", "usd", quotes, [calc])[0]


def test_zero_is_supported_only_by_zero() -> None:
    assert not fv_check("0", "usd", ["Net sales 1,577"])[0]
    assert not fv_check("0", "percent", ["Net sales 1,577"])[0]
    assert fv_check("0", "usd", ["Net sales 0"], page="Net sales 0")[0]


def test_question_year_is_not_an_operand_even_beside_a_cited_number() -> None:
    calc = evaluate_expression("1577+2020")
    ok, _ = fv_check("3597", "usd", ["Net sales 1,577"], [calc], question="In 2020?")
    assert not ok


def test_extreme_final_values_are_reported_not_raised() -> None:
    for v in ("1E-90", "1E+90", "-1E+90"):
        ok, problems = fv_check(v, "usd", ["Net sales 1,577"])
        assert not ok and "final value out of range" in problems


def test_exponent_shaped_tokens_hold_no_number() -> None:
    from footnote.numbers import parse_numbers

    for token in ("1e-3", "2.5E+6", "1e9", "1E−3"):
        assert parse_numbers(f"rate {token} here") == []
    assert [n.value for n in parse_numbers("Q3 2018 rose 5 percent")] == [2018, 5]
    ok, _ = fv_check("3", "ratio", ["Rate 1e-3"], page="Rate 1e-3")
    assert not ok


def test_percent_and_ratio_ignore_the_quoted_scale_word() -> None:
    page = "Revenue 1.2 million"
    q = ["Revenue 1.2 million"]
    assert not fv_check("1200000", "ratio", q, page=page)[0]
    assert not fv_check("1200000", "percent", q, page=page)[0]
    assert fv_check("1200000", "usd", q, page=page)[0]
    assert fv_check("1.2", "ratio", q, page=page)[0]


def test_quote_with_plain_hyphen_matches_extractor_hyphen_mark() -> None:
    page = "Regulation S\x02T and forward\x02looking statements, Non\u00adGAAP measures"
    assert quote_in_page("Regulation S-T and forward-looking statements", page)
    assert quote_in_page("Non-GAAP measures", page)
    assert quote_in_page("forward\x02looking statements", page)


def test_number_in_quote_with_hyphen_mark_is_parsed_like_a_plain_hyphen() -> None:
    page = "Fiscal 2019\x022020 net sales were 1,577"
    for quote in ("2019\x022020 net sales were 1,577", "2019-2020 net sales were 1,577"):
        assert fv_check("1577", "usd", [quote], page=page)[0]
    # the mark between digits must not glue them into one number
    assert not fv_check("20192020", "usd", ["2019\x022020 net sales"], page=page)[0]


# -- chained calculations ------------------------------------------------------------------------
CHAIN_PAGE = "Inventory 6,854 and 7,367 and cost of sales 24,576"
CHAIN_QUOTES = ["Inventory 6,854 and 7,367", "cost of sales 24,576"]


def chain(*expressions: str) -> list[Calc]:
    return [evaluate_expression(e) for e in expressions]


def chain_check(
    value: str, unit: str, calcs: list[Calc], quotes: list[str] | None = None
) -> tuple[bool, list[str]]:
    return fv_check(value, unit, quotes or CHAIN_QUOTES, calcs, page=CHAIN_PAGE)


def test_chained_calculation_is_accepted() -> None:
    calcs = chain("(6854+7367)/2", "24576/7110.5")
    assert calcs[0].result == Decimal("7110.5")
    assert chain_check("3.46", "ratio", calcs)[0]
    # the same final value is not supported by the second calculation alone
    assert not chain_check("3.46", "ratio", calcs[1:])[0]


def test_chain_with_an_uncited_first_operand_is_rejected() -> None:
    calcs = chain("(6854+7000)/2", "24576/6927")
    assert not chain_check("3.55", "ratio", calcs)[0]
    assert not chain_check("3.55", "ratio", calcs[1:])[0]


def test_operand_equal_to_a_later_result_is_rejected() -> None:
    calcs = chain("24576/7110.5", "(6854+7367)/2")
    assert not chain_check("3.46", "ratio", calcs)[0]


def test_rounded_copy_of_an_earlier_result_is_rejected() -> None:
    for rounded in ("7110", "7111"):
        calcs = chain("(6854+7367)/2", f"24576/{rounded}")
        assert not chain_check("3.46", "ratio", calcs)[0]
    calcs = chain("(6854+7367)/2", "24576/7110.50")  # same value written with a trailing zero
    assert chain_check("3.46", "ratio", calcs)[0]


def test_final_value_may_round_a_chained_result() -> None:
    calcs = chain("(6854+7367)/2", "24576/7110.5")
    assert chain_check("3.456", "ratio", calcs)[0]
    assert not chain_check("3.5", "ratio", chain("(6854+7367)/2"), CHAIN_QUOTES[:1])[0]


def test_percent_times_100_still_comes_only_from_calculations() -> None:
    calcs = chain("(6854+7367)/2", "24576/7110.5")
    assert not chain_check("345.63", "percent", calcs[:1])[0]
    assert chain_check("345.63", "percent", calcs)[0]
    assert not chain_check("345.63", "percent", [])[0]


def test_repeating_decimal_result_keeps_every_digit_in_the_next_expression() -> None:
    quotes = ["gross profit 1,018", "revenue 1,557"]
    first = evaluate_expression("1018/1557")
    exact = format(first.result, "f")
    assert exact == "0.6538214515093127809890815671"
    second = evaluate_expression(f"{exact}*100")
    assert second.operands[0] == first.result
    assert fv_check("65.38", "ratio", quotes, [first, second])[0]
    assert fv_check("65.4", "percent", quotes, [first, second])[0]
    truncated = evaluate_expression("0.6538214515093128*100")
    assert not fv_check("65.38", "ratio", quotes, [first, truncated])[0]


def test_negative_result_is_reused_with_the_sign_outside_the_literal() -> None:
    page = "Loss of 5 versus 7 last year"
    quotes = ["Loss of 5 versus 7"]
    calcs = chain("5-7", "-2*100")
    assert calcs[1].operands == [Decimal(2), Decimal(100)]
    assert fv_check("200", "usd", quotes, calcs, page=page)[0]
    assert fv_check("-200", "usd", quotes, calcs, page=page)[0]
    # without the first calculation the second has no cited operand
    assert not fv_check("200", "usd", quotes, calcs[1:], page=page)[0]


def test_a_calculation_cannot_feed_itself() -> None:
    selfish = Calc("7110.5*1", [Decimal("7110.5"), Decimal(1)], Decimal("7110.5"))
    assert not chain_check("7110.5", "usd", [selfish])[0]
    assert not chain_check("7110.5", "usd", [selfish], CHAIN_QUOTES)[0]


def test_three_hop_chain() -> None:
    calcs = chain("(6854+7367)/2", "24576/7110.5", "3.456297*1000")
    exact = format(calcs[1].result, "f")
    calcs = chain("(6854+7367)/2", "24576/7110.5", f"{exact}*1000")
    assert chain_check("3456.3", "ratio", calcs)[0]
    assert not chain_check("3456.3", "ratio", calcs[:2] + chain("3.456297*1000"))[0]
    broken = chain(
        "(6854+7000)/2", "24576/6927", f"{format(chain('24576/6927')[0].result, 'f')}*1000"
    )
    assert not chain_check("3547.8", "ratio", broken)[0]


def test_citation_that_fails_its_own_check_feeds_nothing() -> None:
    calcs = chain("(6854+7367)/2", "24576/7110.5")
    bad_quotes = {
        "wrong page": ([CHAIN_QUOTES[0], CHAIN_QUOTES[1]], {0: "elsewhere", 1: CHAIN_PAGE}),
        "fabricated": (["Inventory 6,854 and 7,367", "cost of sales 24,577"], {0: CHAIN_PAGE}),
        "too long": (
            [CHAIN_QUOTES[0], CHAIN_PAGE + " " + "x" * 400],
            {0: CHAIN_PAGE + " " + "x" * 400},
        ),
    }
    for _name, (quotes, pages) in bad_quotes.items():
        a = fv_answer("See filing.", quotes, "3.46", "ratio")
        ok, problems = verify_answer(a, pages, "What?", calcs)
        assert not ok
        assert any(p.startswith(("quote", "cited page")) for p in problems)
        assert any(p.startswith("final value not supported") for p in problems)
    # a valid chain never rescues the bad quote itself
    a = fv_answer("See filing.", [CHAIN_QUOTES[0], "cost of sales 24,577"], "3.46", "ratio")
    ok, problems = verify_answer(a, {0: CHAIN_PAGE}, "What?", calcs)
    assert not ok and any("quote not found" in p for p in problems)


def test_a_value_that_is_not_a_correct_rounding_stays_rejected_with_chaining() -> None:
    page = "Debt 1,000 and equity 82.4"
    quotes = [page]
    calcs = chain("(1000+82.4)/2", "1000/541.2")
    assert calcs[1].result == Decimal("1.847745750184774575018477458")
    # truncated at 15 places instead of rounded (the shape of the 12.138637733574441 answer)
    assert not fv_check("1.847745750184774", "ratio", quotes, calcs, page=page)[0]
    assert fv_check("1.847745750184775", "ratio", quotes, calcs, page=page)[0]
    assert fv_check("1.85", "ratio", quotes, calcs, page=page)[0]
    # the recorded answer: 12.138637733574441 against the calculator's 12.13863773357444243520192887
    exact = "12.13863773357444243520192887"
    page = f"Interest coverage {exact}"
    calcs = chain(f"{exact}*1")
    assert not fv_check("12.138637733574441", "ratio", [page], calcs, page=page)[0]
    assert fv_check("12.138637733574442", "ratio", [page], calcs, page=page)[0]
    assert fv_check("12.14", "ratio", [page], calcs, page=page)[0]


def test_a_second_calculation_with_only_an_earlier_result_and_constants() -> None:
    calcs = chain("(6854+7367)/2", "7110.5*12")
    assert chain_check("85326", "usd", calcs)[0]
    assert not chain_check("85326", "usd", calcs[1:])[0]


def test_wrong_page_message_names_the_page_that_holds_the_quote() -> None:
    a = ans("x", ["Net sales 1,577"])
    a = a.model_copy(update={"citations": [Citation(page_index=2, quote="Net sales 1,577")]})
    ok, problems = verify_answer(a, {0: "Net sales 1,577", 1: "Other", 2: "Other"}, "?", [])
    assert not ok
    assert problems[0].startswith("quote not found on page 2")
    assert "this exact text is on page_index 0" in problems[0]
    ok, problems = verify_answer(a, {1: "Other", 2: "Other"}, "?", [])
    assert "this exact text" not in problems[0]
