"""The grounding check. It guarantees exactly two things, and nothing about the answer's prose.

1. Every quote the answer cites appears word for word on the cited page of the filing, and an
   answered (not declined) question cites at least one quote.
2. The answer's structured final value is a correct rounding of a number written in a cited quote,
   or of a logged calculation whose numbers come from cited quotes or from the exact results of
   earlier such calculations (a power may have a fractional exponent when its base is positive,
   as in a compound growth rate; plus small fixed constants such as 2, 12, 365 or 100, with at least
   one cited or earlier-result number in every calculation). A figure counts with either sign: a
   value shown in parentheses or with a minus sign is the same figure. A calculation operand,
   whether a quoted number or an earlier result, may be scaled up by a power of 1,000 whatever
   the final unit (tables state their scale in a header). Units of the final value:
   - usd, count and other: the final value may be a quoted number or a calculation result scaled
     by a power of 1,000 (thousands, millions, billions; tables rarely state their unit in the
     quoted line), either direction.
   - percent and ratio: no scale applies to the final value; it must match the quoted number as
     written. A percent may also be a calculation result times 100 (1018/1557 gives 65.4), never
     a quoted number times 100.
   - a quoted number written with a % sign supports only a percent value; a plain table cell such
     as "65.4" may support 65.4 percent.
   - a value of zero is supported only by a source that is zero, never by a tiny number that
     rounds to zero.
   A final value with an absurd size is reported as out of range.

Numbers in the free-text answer are not checked; the grader measures whether the answer is right.
"""

import ast
import re
from dataclasses import dataclass
from decimal import Decimal, DivisionByZero, InvalidOperation, Overflow

from footnote.answers import AgentAnswer, FinalValue
from footnote.corpus.normalize import map_hyphen_marks, normalize_text
from footnote.numbers import ParsedNumber, is_valid_rounding, parse_numbers

VERIFY_VERSION = "4"  # changes whenever a rule or a problem message of the check changes
_MAGNITUDES = (Decimal(1), Decimal("1e3"), Decimal("1e6"), Decimal("1e9"))
# usd, count and other values are stated in the question's requested scale: any power of 1000
# constants a calculation may use without a citation: averages, annualizing, day counts, units
_UNIT_CONSTANTS = frozenset(
    Decimal(n) for n in (*range(1, 13), 360, 365, 100, 1000, 1000000, 1000000000)
)
_SCALED_UNITS = ("usd", "count", "other")
_MAX_FINAL_EXPONENT = 30
_FINAL_MAGNITUDES = tuple(Decimal(10) ** e for e in range(-9, 10, 3))
MAX_QUOTE_CHARS = 400  # the site's excerpt cap; a longer quote could "support" a whole page
_MAX_POWER = 10
_MAX_EXPRESSION_CHARS = 200


@dataclass(frozen=True)
class Calc:
    expression: str
    operands: list[Decimal]
    result: Decimal


_LETTER_HYPHEN = re.compile(r"(?<=[^\W\d_])-(?=[^\W\d_])")


def _squash(s: str) -> str:
    # the normalizer joins "-\n" without a space, so a quoted "long-term" must meet "longterm";
    # hyphens between digits (ranges such as 2019-2020) stay
    return _LETTER_HYPHEN.sub("", normalize_text(s))


def quote_in_page(quote: str, page_text: str) -> bool:
    """The quote occurs at token boundaries: never inside a longer word or number."""
    q = _squash(quote)
    if not q:
        return False
    pattern = r"(?<!\w)(?<!\d[.,])" + re.escape(q) + r"(?!\w)(?![.,]\d)"
    return re.search(pattern, _squash(page_text)) is not None


def _literals(node: ast.AST, source: str, out: list[Decimal]) -> Decimal:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        # a float literal is read from its own text: every digit the model copied is kept
        text = ast.get_source_segment(source, node)
        value = Decimal(node.value if type(node.value) is int else text or repr(node.value))
        out.append(value)
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _literals(node.operand, source, out)
        return -inner if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.BinOp):
        left = _literals(node.left, source, out)
        right = _literals(node.right, source, out)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_POWER:
                raise ValueError("exponent must be at most 10 in size")
            if right == right.to_integral_value():
                return left ** int(right)
            if left <= 0:
                raise ValueError("a fractional exponent needs a positive base")
            return left**right
    raise ValueError("only numbers, + - * / ** and parentheses are allowed")


def evaluate_expression(expression: str) -> Calc:
    """Evaluate arithmetic safely; operands are the numeric literals in order of appearance."""
    text = expression.replace("×", "*").replace("÷", "/").replace("−", "-").replace(",", "")
    if len(text) > _MAX_EXPRESSION_CHARS:
        raise ValueError(f"expression longer than {_MAX_EXPRESSION_CHARS} characters")
    try:
        text = text.strip()
        tree = ast.parse(text, mode="eval")
        operands: list[Decimal] = []
        result = _literals(tree.body, text, operands)
    except (
        SyntaxError,
        RecursionError,
        MemoryError,
        Overflow,
        DivisionByZero,
        InvalidOperation,
        ZeroDivisionError,
    ) as exc:
        raise ValueError(f"cannot evaluate {expression!r}: {exc}") from exc
    if not operands:
        raise ValueError(f"no numbers in {expression!r}")
    return Calc(expression, operands, result)


def _calc_is_sound(c: Calc) -> bool:
    try:
        again = evaluate_expression(c.expression)
    except ValueError:
        return False
    return again.operands == c.operands and again.result == c.result


def _bases(n: ParsedNumber) -> list[Decimal]:
    """Values a quoted number can stand for: as written, and with its own scale word."""
    return [n.value] if n.scale == 1 else [n.value, n.value * n.scale]


def _rounds_to(shown: ParsedNumber, source: Decimal) -> bool:
    if shown.value == 0 and source != 0:
        return False  # a tiny nonzero number rounding to zero supports nothing
    return is_valid_rounding(shown, source) or is_valid_rounding(shown, -source)


def _supported_by_value(
    shown: ParsedNumber, base: Decimal, magnitudes: tuple[Decimal, ...]
) -> bool:
    return any(_rounds_to(shown, base * s) for s in magnitudes)


def _operand_in_quotes(op: Decimal, cited: list[ParsedNumber]) -> bool:
    return any(op in (b * s, -b * s) for n in cited for b in _bases(n) for s in _MAGNITUDES)


def _calc_results(calcs: list[Calc], cited: list[ParsedNumber]) -> list[Decimal]:
    """Results of the traced calculations, in log order. A number is traced when it is in a cited
    quote, is a unit constant, or equals (or negates) the exact result of an earlier traced one;
    a quoted number and an earlier result may each be scaled by a power of 1,000;
    a calculation needs at least one operand that is not a constant."""
    traced: list[Decimal] = []
    for c in calcs:
        if not _calc_is_sound(c):
            continue
        roots = [o for o in c.operands if _operand_in_quotes(o, cited) or _earlier(o, traced)]
        if roots and all(o in _UNIT_CONSTANTS or o in roots for o in c.operands):
            traced.append(c.result)
    return traced


def _earlier(op: Decimal, traced: list[Decimal]) -> bool:
    """The operand is an earlier traced result, either sign, scaled like a quoted operand."""
    return any(op in (r * s, -r * s) for r in traced for s in _MAGNITUDES)


def _number_supported(
    shown: ParsedNumber, unit: str, cited: list[ParsedNumber], results: list[Decimal]
) -> bool:
    scaled = unit in _SCALED_UNITS
    magnitudes = _FINAL_MAGNITUDES if scaled else (Decimal(1),)
    if any(_supported_by_value(shown, r, magnitudes) for r in results):
        return True
    if unit == "percent" and any(_rounds_to(shown, r * 100) for r in results):
        return True
    # a quoted number with a % sign supports only a percent value; percent and ratio match the
    # quoted number as written, so its scale word is not applied
    return any(
        _supported_by_value(shown, b, magnitudes)
        for n in cited
        if unit == "percent" or not n.percent
        for b in (_bases(n) if scaled else [n.value])
    )


def _final_as_shown(fv: FinalValue) -> ParsedNumber:
    exponent = fv.value.as_tuple().exponent
    decimals = max(0, -exponent) if isinstance(exponent, int) else 0
    return ParsedNumber(fv.value, decimals, False, Decimal(1))


def _elsewhere(quote: str, page_index: int, pages: dict[int, str]) -> str:
    found = [n for n, t in sorted(pages.items()) if n != page_index and quote_in_page(quote, t)]
    if not found:
        return ""
    return f"; this exact text is on page_index {', '.join(str(n) for n in found[:3])}"


def _not_found(quote: str, page_index: int, pages: dict[int, str]) -> str:
    message = f"quote not found on page {page_index}: {quote[:80]!r}"
    return message + _elsewhere(quote, page_index, pages)


def verify_answer(
    answer: AgentAnswer,
    pages: dict[int, str],
    question: str,
    calcs: list[Calc],
    resubmit_only: bool = False,
) -> tuple[bool, list[str]]:
    """`resubmit_only`: the reply after this check can only submit, so the advice must not ask
    for a new calculation."""
    problems: list[str] = []
    cited: list[ParsedNumber] = []
    problems.extend(
        f"calculation rejected: {c.expression[:60]!r}" for c in calcs if not _calc_is_sound(c)
    )
    for c in answer.citations:
        text = pages.get(c.page_index)
        too_long = len(c.quote) > MAX_QUOTE_CHARS
        if too_long:
            problems.append(
                f"quote too long ({len(c.quote)} > {MAX_QUOTE_CHARS} characters, about 60 words): "
                "cut it to the sentence or table row that holds the figure"
            )
        if text is None:
            problems.append(
                f"cited page {c.page_index} is not in the filing"
                + _elsewhere(c.quote, c.page_index, pages)
            )
        elif not quote_in_page(c.quote, text):
            problems.append(_not_found(c.quote, c.page_index, pages))
        elif not too_long:
            cited.extend(parse_numbers(map_hyphen_marks(c.quote)))
    if not answer.declined:
        if not answer.citations:
            problems.append("no citations")
        fv = answer.final_value
        if fv is not None:
            problems.extend(_final_value_problems(fv, cited, calcs, resubmit_only))
    return (not problems, problems)


def _final_value_problems(
    fv: FinalValue, cited: list[ParsedNumber], calcs: list[Calc], resubmit_only: bool
) -> list[str]:
    exponent = fv.value.as_tuple().exponent
    if (
        not fv.value.is_finite()
        or not isinstance(exponent, int)
        or abs(exponent) > _MAX_FINAL_EXPONENT
        or fv.value.adjusted() > _MAX_FINAL_EXPONENT
    ):
        return ["final value out of range"]
    try:
        supported = _number_supported(
            _final_as_shown(fv), fv.unit, cited, _calc_results(calcs, cited)
        )
    except InvalidOperation:
        return ["final value out of range"]
    if supported:
        return []
    fix = (
        "resubmit with a final value that is a correct rounding of a quoted number or of a "
        "calculate result you already ran"
        if resubmit_only
        else "compute it with calculate from cited figures, then submit that result rounded"
    )
    return [
        f"final value {fv.value} is not a correct rounding of a number in a cited quote or of a "
        f"calculate result: {fix}"
    ]
