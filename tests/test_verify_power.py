from decimal import Decimal, localcontext

import pytest
from test_verify import fv_check

from footnote.verify import evaluate_expression

CAGR = "(1.5/1.2)**(1/3)-1"
CAGR_VALUE = Decimal("0.077217345015941860879646783")
PAGE = "Sales were 1.2 in 2020 and 1.5 in 2023"
QUOTES = ["Sales were 1.2 in 2020", "1.5 in 2023"]
UNDERFLOW = "((((((2**-10)**10)**10)**10)**10)**10)**(9+1/2)"


def test_cagr_has_the_same_value_whatever_the_ambient_precision() -> None:
    plain = evaluate_expression(CAGR)
    with localcontext() as ctx:
        ctx.prec = 10
        low = evaluate_expression(CAGR)
    assert plain.result == low.result == evaluate_expression(CAGR).result
    assert plain.result == CAGR_VALUE


def test_the_check_accepts_a_calculation_made_under_another_ambient_context() -> None:
    with localcontext() as ctx:
        ctx.prec = 10
        calc = evaluate_expression("(" + CAGR + ")*100")
        ok, problems = fv_check("7.72", "percent", QUOTES, [calc], page=PAGE)
    assert ok, problems


def test_a_value_that_underflows_is_an_error_not_zero() -> None:
    with pytest.raises(ValueError):
        evaluate_expression(UNDERFLOW)
    with localcontext() as ctx:
        ctx.prec = 10
        with pytest.raises(ValueError):
            evaluate_expression(UNDERFLOW)
    ok, _ = fv_check("0", "ratio", ["2"], [], page="2")
    assert not ok  # no calculation was logged, and a tiny number never supports zero


def test_a_correct_rounding_of_a_growth_rate_verifies_and_a_wrong_one_does_not() -> None:
    calc = evaluate_expression("(" + CAGR + ")*100")
    assert fv_check("7.72", "percent", QUOTES, [calc], page=PAGE)[0]
    assert fv_check("7.7", "percent", QUOTES, [calc], page=PAGE)[0]
    for wrong in ("7.73", "7.71", "0.77"):
        assert not fv_check(wrong, "percent", QUOTES, [calc], page=PAGE)[0], wrong
