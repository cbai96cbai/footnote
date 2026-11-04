from decimal import Decimal as D

import pytest

from footnote.numbers import ParsedNumber, is_valid_rounding, parse_gold_number, parse_numbers


def one(text: str) -> ParsedNumber:
    found = parse_numbers(text)
    assert len(found) == 1, found
    return found[0]


def test_parse_number_financial_formats() -> None:
    assert one("(1,234)").value == D(-1234)
    assert parse_numbers("—") == []
    assert parse_numbers("-") == []
    b = one("$ 1.2 billion")
    assert (b.value, b.scale, b.decimals, b.percent) == (D("1.2"), D("1e9"), 1, False)
    p = one("65.4%")
    assert (p.value, p.percent, p.decimals) == (D("65.4"), True, 1)
    assert one("2.1x").value == D("2.1")
    d = one("$1577.00")
    assert (d.value, d.decimals) == (D("1577.00"), 2)


def test_percent_word_parses_like_symbol() -> None:
    assert one("65.4 percent") == one("65.4%")
    assert one("65.4 %") == one("65.4%")


def test_scale_words_and_separators() -> None:
    assert one("$3,500 thousand").scale == D("1e3")
    assert one("12 Million").scale == D("1e6")
    assert one("$1,234,567.8").value == D("1234567.8")
    assert one("12").scale == D(1)


def test_negatives() -> None:
    assert one("-5.5%").value == D("-5.5")
    assert one("−3").value == D(-3)  # unicode minus
    assert one("($1.2 million)").value == D("-1.2")
    assert one("$-4").value == D(-4)


def test_range_hyphen_is_not_a_minus() -> None:
    assert [n.value for n in parse_numbers("2017-2018")] == [D(2017), D(2018)]


def test_identifiers_are_not_numbers() -> None:
    assert parse_numbers("Form 10-K") == []
    assert parse_numbers("Q3 results") == []
    assert parse_numbers("FY2018") == []
    assert parse_numbers("3rd quarter") == []
    assert one("in 2018").value == D(2018)


def test_unmatched_open_paren_is_not_negative() -> None:
    assert one("(about 5 million").value == D(5)


def test_sentence_period_not_decimal() -> None:
    n = one("It was 65.")
    assert (n.value, n.decimals) == (D(65), 0)


def test_several_numbers() -> None:
    vals = [n.value for n in parse_numbers("up from $1.5 billion to $1.6 billion, or 6.7%")]
    assert vals == [D("1.5"), D("1.6"), D("6.7")]


def test_parse_gold_number_exactly_one() -> None:
    assert parse_gold_number("$1,577.00") is not None
    assert parse_gold_number("24.26") == one("24.26")
    assert parse_gold_number("1.5 and 1.6") is None
    assert parse_gold_number("no figure") is None
    assert parse_gold_number("FY2018 had none") is None
    assert parse_gold_number("") is None


def test_valid_rounding() -> None:
    assert is_valid_rounding(one("$1.6 billion"), D("1.577e9"))
    assert not is_valid_rounding(one("1.49"), D(1))
    assert is_valid_rounding(one("24.3"), D("24.26"))
    assert not is_valid_rounding(one("24.2"), D("24.26"))


@pytest.mark.parametrize(
    ("shown", "source", "ok"),
    [
        ("0.5", "0.45", True),  # tie rounds half up
        ("-0.5", "-0.45", True),  # ties go away from zero
        ("-0.4", "-0.45", False),
        ("1577", "1577.4", True),
        ("1578", "1577.4", False),
        ("2", "1.5e0", True),
        ("1.60", "1.595", True),
    ],
)
def test_rounding_ties_and_signs(shown: str, source: str, ok: bool) -> None:
    assert is_valid_rounding(one(shown), D(source)) is ok


def test_rounding_is_exact_not_float() -> None:
    assert is_valid_rounding(one("0.3"), D("0.25")) is True
    assert is_valid_rounding(one("1.1"), D("1.05")) is True
    assert is_valid_rounding(one("1.0"), D("1.05")) is False


def test_huge_values_do_not_overflow_the_context() -> None:
    assert is_valid_rounding(one("1 billion"), D("1e9")) is True
    assert is_valid_rounding(
        one("123456789012345678901234567890.5"), D("123456789012345678901234567890.46")
    )
