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


def test_too_long_message_gives_the_length_in_words_and_the_fix() -> None:
    page = "Net sales 1,577 " + "x " * 300
    _, problems = check("Sales 1,577.", [page], page)
    n = len(page)
    assert problems == [
        f"quote too long ({n} > 400 characters, about 60 words): "
        "cut it to the sentence or table row that holds the figure"
    ]


def test_a_quote_both_too_long_and_not_on_its_page_gets_both_problems() -> None:
    quote = "Net sales 1,577 " + "y " * 300
    _, problems = check("Sales 1,577.", [quote], "Net sales 1,577 " + "x " * 300)
    assert len(problems) == 2
    assert problems[0].startswith(f"quote too long ({len(quote)} > 400 characters")
    assert problems[1].startswith("quote not found on page 0: 'Net sales 1,577 y y")
    _, problems = verify_answer(ans("x", [quote]), {1: quote}, "?", [])
    assert problems[0].startswith("quote too long")
    assert problems[1] == "cited page 0 is not in the filing; this exact text is on page_index 1"


def test_untraced_final_value_says_what_would_fix_it() -> None:
    ok, problems = fv_check("7.9", "percent", ["Net sales 1,577"], [], page="Net sales 1,577")
    assert not ok and problems == [
        "final value 7.9 is not a correct rounding of a number in a cited quote or of a "
        "calculate result: compute it with calculate from cited figures, then submit that "
        "result rounded"
    ]


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


def test_untraced_final_value_advice_matches_what_the_next_reply_can_do() -> None:
    args = ("7.9", "percent", ["Net sales 1,577"], [])
    _, free = fv_check(*args, page="Net sales 1,577")
    answer = ans("x", ["Net sales 1,577"])
    forced_answer = answer.model_copy(
        update={"final_value": FinalValue(value=Decimal("7.9"), unit="percent")}
    )
    _, forced = verify_answer(forced_answer, {0: "Net sales 1,577"}, "?", [], resubmit_only=True)
    assert free[0].endswith(
        "compute it with calculate from cited figures, then submit that result rounded"
    )
    assert forced == [
        "final value 7.9 is not a correct rounding of a number in a cited quote or of a "
        "calculate result: resubmit with a final value that is a correct rounding of a quoted "
        "number or of a calculate result you already ran"
    ]


def test_fractional_powers_of_positive_numbers_give_growth_rates() -> None:
    cagr = "(1.5/1.2)**(1/3)-1"
    first, second = evaluate_expression(cagr), evaluate_expression(cagr)
    assert first.result == second.result and first.operands == second.operands
    assert str(first.result)[:6] == "0.0772" and first.result.as_tuple().digits.__len__() <= 28
    assert evaluate_expression("2**0.5").result.quantize(Decimal("1e-6")) == Decimal("1.414214")
    from footnote.verify import _calc_is_sound

    assert _calc_is_sound(first)
    for bad in ("(-8)**(1/3)", "0**0.5", "2**10.5", "2**99"):
        with pytest.raises(ValueError):
            evaluate_expression(bad)
