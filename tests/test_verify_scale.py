from decimal import Decimal

from test_verify import fv_check

from footnote.verify import Calc, evaluate_expression

PAGE = "Revenue 280,714 less cost 31,563; shares 3,027,128,112"
QUOTES = ["Revenue 280,714 less cost 31,563", "shares 3,027,128,112"]


def calcs_for(second: str) -> list[Calc]:
    return [evaluate_expression("280714-31563"), evaluate_expression(second)]


def check(second: str, value: str) -> bool:
    return fv_check(value, "other", QUOTES, calcs_for(second), page=PAGE)[0]


def test_an_earlier_result_may_be_scaled_by_a_power_of_1000() -> None:
    calcs = calcs_for("249151000000/3027128112")
    assert calcs[0].result == Decimal(249151)
    assert fv_check("82.31", "other", QUOTES, calcs, page=PAGE)[0]
    assert check("249151000/3027128112", "0.0823")
    assert check("249151/3027128112", "0.0000823")
    assert check("-249151000000/3027128112", "82.31")  # either sign


def test_an_earlier_result_scaled_by_anything_else_is_not_traced() -> None:
    for second in (
        "24915100/3027128112",  # by 10^2
        "1744057/3027128112",  # by 7
        "2491510/3027128112",  # by 10
    ):
        ratio = str(evaluate_expression(second).result.quantize(Decimal("0.0001")))
        assert not check(second, ratio)


def test_a_later_result_scaled_is_not_traced() -> None:
    calcs = [evaluate_expression("249151000000/3027128112"), evaluate_expression("280714-31563")]
    assert not fv_check("82.31", "other", QUOTES, calcs, page=PAGE)[0]


def test_a_scaled_result_alone_still_needs_a_traced_operand_and_the_log() -> None:
    alone = calcs_for("249151000000/3027128112")[1:]
    assert not fv_check("82.31", "other", QUOTES, alone, page=PAGE)[0]
    only_constants = [evaluate_expression("280714-31563"), evaluate_expression("1000000*2")]
    assert not fv_check("2000000", "other", QUOTES, only_constants[1:], page=PAGE)[0]


def test_operand_scaling_and_final_unit_scaling_are_separate_rules() -> None:
    page = "Revenue 125 and cost 42"
    quotes = [page]

    def accepted(second: str, value: str, unit: str) -> bool:
        calcs = [evaluate_expression("125-42"), evaluate_expression(second)]
        return fv_check(value, unit, quotes, calcs, page=page)[0]

    assert accepted("83000/8", "10375", "ratio")  # operand 83 scaled by 1,000
    assert not accepted("83/8", "10375", "ratio")  # a ratio's final value takes no scale
    assert accepted("83/8", "10375", "usd")  # a usd final value may be scaled


def test_an_earlier_result_is_never_scaled_down_as_an_operand() -> None:
    calcs = [evaluate_expression("280714-31563"), evaluate_expression("249.151/3027128112")]
    assert calcs[1].operands[0] == Decimal("249.151")
    ratio = str(calcs[1].result.quantize(Decimal("0.000000001")))
    assert not fv_check(ratio, "other", QUOTES, calcs, page=PAGE)[0]
