from decimal import ROUND_DOWN, Decimal

from test_verify import ans, fv_answer, fv_check

from footnote.answers import Citation
from footnote.verify import Calc, evaluate_expression, verify_answer

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


def test_a_valid_chain_never_rescues_a_separately_fabricated_citation() -> None:
    calcs = chain("(6854+7367)/2", "24576/7110.5")
    quotes = [*CHAIN_QUOTES, "cost of sales was 24,576 million dollars"]  # true number, not on page
    a = fv_answer("See filing.", quotes, "3.46", "ratio")
    ok, problems = verify_answer(a, {0: CHAIN_PAGE}, "What?", calcs)
    assert not ok and len(problems) == 1
    assert problems[0].startswith("quote not found on page 0: 'cost of sales was 24,576")
    assert verify_answer(
        fv_answer("See filing.", CHAIN_QUOTES, "3.46", "ratio"), {0: CHAIN_PAGE}, "?", calcs
    )[0]


def test_a_value_that_is_not_a_correct_rounding_stays_rejected_with_chaining() -> None:
    page = "Operating income 1,000 and interest 541.2"
    first = evaluate_expression("1000/541.2")
    exact = format(first.result, "f")  # 28 digits, copied exactly into the next expression
    calcs = [first, evaluate_expression(f"{exact}*10")]
    value = calcs[1].result
    assert str(value) not in page and exact not in page  # no result appears in any quote
    right = value.quantize(Decimal(10) ** -15)
    cut = value.quantize(Decimal(10) ** -15, rounding=ROUND_DOWN)
    assert right != cut  # truncating instead of rounding gives a different last place
    assert fv_check(str(right), "ratio", [page], calcs, page=page)[0]
    assert not fv_check(str(cut), "ratio", [page], calcs, page=page)[0]
    assert fv_check("18.48", "ratio", [page], calcs, page=page)[0]
    # take away the earlier calculation and nothing supports the chained value
    assert not fv_check(str(right), "ratio", [page], calcs[1:], page=page)[0]
    assert not fv_check("18.48", "ratio", [page], calcs[1:], page=page)[0]


def test_the_recorded_aes_answer_with_its_real_figures() -> None:
    quotes = ["Total cost of sales (10,069) (8,430) (6,967)", "Total $ 1,055 $ 604"]
    page = " ... ".join(quotes)
    calcs = chain("(1055+604)/2", "10069/829.5")
    assert calcs[0].result == Decimal("829.5")
    assert calcs[1].result == Decimal("12.13863773357444243520192887")
    assert not fv_check("12.138637733574441", "other", quotes, calcs, page=page)[0]
    assert fv_check("12.138637733574442", "other", quotes, calcs, page=page)[0]
    assert fv_check("12.14", "other", quotes, calcs, page=page)[0]
    for value in ("12.138637733574442", "12.14"):
        assert not fv_check(value, "other", quotes, calcs[1:], page=page)[0]


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


def test_missing_cited_page_still_says_where_the_exact_quote_is() -> None:
    a = ans("x", ["Net sales 1,577"])
    a = a.model_copy(update={"citations": [Citation(page_index=9, quote="Net sales 1,577")]})
    ok, problems = verify_answer(a, {0: "Net sales 1,577", 1: "Other"}, "?", [])
    assert not ok and problems[0].startswith("cited page 9 is not in the filing")
    assert "this exact text is on page_index 0" in problems[0]
    ok, problems = verify_answer(a, {1: "Other"}, "?", [])
    assert problems[0] == "cited page 9 is not in the filing"
