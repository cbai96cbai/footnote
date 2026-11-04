"""Numeric grading: compares the gold's value as written with the answer's final_value only."""

from decimal import Decimal

from footnote.answers import FinalValue
from footnote.numbers import ParsedNumber

_HUNDRED = Decimal(100)
_REL_TOL = Decimal("0.005")


def numeric_match(gold: ParsedNumber, final: FinalValue) -> bool:
    """Correct iff |answer - gold| <= max(half unit of gold's last digit, 0.5% of |gold|).

    The gold's scale word is ignored. Only percent<->ratio is converted (x100), chosen by the
    gold's % sign and the answer's explicit unit.
    """
    value = final.value
    if gold.percent and final.unit == "ratio":
        value *= _HUNDRED
    elif not gold.percent and final.unit == "percent":
        value /= _HUNDRED
    tolerance = max(Decimal(5) / Decimal(10) ** (gold.decimals + 1), _REL_TOL * abs(gold.value))
    return abs(value - gold.value) <= tolerance
