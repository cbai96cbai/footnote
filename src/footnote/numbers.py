"""Reading numbers the way filings write them, and checking a shown number is a rounding.

Choices (each has a test): "FY2018", "Q3", "10-K" and "3rd" contain no number; a standalone
"2018" is one; "(1,234)" is accounting negative; a hyphen between digits is a range, not a minus.
"""

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, localcontext

_K, _M, _B = Decimal("1e3"), Decimal("1e6"), Decimal("1e9")
# words in any case; abbreviations exactly as listed (lowercase "m" alone is not a scale)
_SCALES = {
    "thousand": _K,
    "million": _M,
    "billion": _B,
    "k": _K,
    "K": _K,
    "mn": _M,
    "mm": _M,
    "M": _M,
    "bn": _B,
    "b": _B,
    "B": _B,
}

_NUMBER = re.compile(
    r"""
    (?P<open>\()?\s*
    (?:(?<![\w.])(?P<neg>[-−]))?\s*
    \$?\s*
    (?:(?<![\w.])(?P<neg2>[-−]))?
    (?<![A-Za-z0-9_])(?<!\d\.)
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)
    (?:\s*(?P<scale>(?i:thousand|million|billion)|(?-i:bn|mn|mm|b|B|M|k|K)))?
    (?:(?P<pct>\s*%|\s+percent\b)|(?(scale)(?!)|(?P<x>x\b)))?
    (?(open)\s*\)(?(pct)|(?P<pct2>\s*%)?)|)
    (?![A-Za-z0-9_])
    (?!-[A-Za-z])
    (?!\.\d)
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _scale(word: str) -> Decimal:
    return _SCALES.get(word) or _SCALES[word.lower()]


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
        scale = _scale(m["scale"]) if m["scale"] else Decimal(1)
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        out.append(ParsedNumber(value, decimals, (m["pct"] or m["pct2"]) is not None, scale))
    return out


_TOKEN = re.compile(r"\w+(?:[.,−-]\w+)*")
_IDENTIFIER = re.compile(r"[^\W\d]|\d+-[A-Za-z]|\d+(?:st|nd|rd|th)\b", re.IGNORECASE)


def unrecognized_tokens(text: str) -> list[str]:
    """Digit-bearing tokens the parser did not fully read (closed world: these fail).

    Identifiers ("FY2018", "Q3", "10-K", "3rd") are not numbers. Anything else with a digit
    that the parser skipped ("999e9", "1.2.3", "7A") is returned.
    """
    masked = list(text)
    for m in _NUMBER.finditer(text):
        masked[m.start() : m.end()] = " " * (m.end() - m.start())
    left = {i for i, ch in enumerate(masked) if ch.isdigit()}
    found: list[str] = []
    for tok in _TOKEN.finditer(text):
        if any(i in left for i in range(tok.start(), tok.end())):
            if not _IDENTIFIER.match(tok[0]) and tok[0] not in found:
                found.append(tok[0])
    return found


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
