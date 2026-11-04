"""Reading numbers the way filings write them, and checking a shown number is a rounding.

Choices (each has a test): "FY2018", "Q3", "10-K" and "3rd" contain no number; a standalone
"2018" is one; "(1,234)" is accounting negative; a hyphen between digits is a range, not a minus.
"""

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, localcontext

_SCALES = {"thousand": Decimal("1e3"), "million": Decimal("1e6"), "billion": Decimal("1e9")}

_NUMBER = re.compile(
    r"""
    (?P<open>\()?\s*
    (?:(?<![\w.])(?P<neg>[-−]))?\s*
    \$?\s*
    (?:(?<![\w.])(?P<neg2>[-−]))?
    (?<![A-Za-z0-9_])
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    (?:\s*(?P<scale>thousand|million|billion)\b)?
    (?:(?P<pct>\s*%|\s+percent\b)|(?P<x>x\b))?
    (?(open)\s*\)|)
    (?![A-Za-z0-9_])
    (?!-[A-Za-z])
    """,
    re.VERBOSE | re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedNumber:
    value: Decimal  # as written, before any scale word
    decimals: int  # digits after the decimal point as written
    percent: bool
    scale: Decimal  # 1, 1e3, 1e6 or 1e9


def parse_numbers(text: str) -> list[ParsedNumber]:
    out: list[ParsedNumber] = []
    for m in _NUMBER.finditer(text):
        digits = m["num"].replace(",", "")
        value = Decimal(digits)
        if m["open"] or m["neg"] or m["neg2"]:
            value = -value
        scale = _SCALES[m["scale"].lower()] if m["scale"] else Decimal(1)
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        out.append(ParsedNumber(value, decimals, m["pct"] is not None, scale))
    return out


def parse_gold_number(text: str) -> ParsedNumber | None:
    """The number in a gold answer, only when the text holds exactly one."""
    found = parse_numbers(text)
    return found[0] if len(found) == 1 else None


def is_valid_rounding(shown: ParsedNumber, source: Decimal) -> bool:
    """True when `shown` is `source` rounded half-up to the shown scale and decimals."""
    with localcontext() as ctx:
        ctx.prec = 100
        rounded = (source / shown.scale).quantize(
            Decimal(1).scaleb(-shown.decimals), rounding=ROUND_HALF_UP
        )
        return rounded == shown.value
