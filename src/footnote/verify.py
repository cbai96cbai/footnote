"""The grounding check: every quote exists in the filing, every number traces to a quote.

What it proves: every quote exists in the cited filing page, and every number in the answer is a
correct rounding of a number in a cited quote, or of a logged calculation whose operands all appear
in cited quotes (or are unit constants). Numbers in the question are exempt only as displayed
numbers, never as operands; a digit the parser cannot fully read fails closed. It does not prove
that a number means what the answer says (units, periods); the grader measures that.

Documented limits: tables rarely state units in the quoted line, so a shown number with a scale
word may match a quoted number times 1e3, 1e6 or 1e9; signs are not compared (an answer says
"fell $1.6 billion" for a quoted "(1,577)").
"""

import ast
import re
from dataclasses import dataclass
from decimal import Decimal, DivisionByZero, InvalidOperation, Overflow

from footnote.answers import AgentAnswer, FinalValue
from footnote.corpus.normalize import normalize_text
from footnote.numbers import ParsedNumber, is_valid_rounding, parse_numbers, unrecognized_tokens

_MAGNITUDES = (Decimal(1), Decimal("1e3"), Decimal("1e6"), Decimal("1e9"))
# final_value is stated in the question's requested unit: any power of 1000 either way
_UNIT_CONSTANTS = (Decimal(100), Decimal(1000), Decimal("1e6"), Decimal("1e9"))
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


def _literals(node: ast.AST, out: list[Decimal]) -> Decimal:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = Decimal(repr(node.value))
        out.append(value)
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _literals(node.operand, out)
        return -inner if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.BinOp):
        left = _literals(node.left, out)
        right = _literals(node.right, out)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            if right != right.to_integral_value() or abs(right) > _MAX_POWER:
                raise ValueError("exponent must be a whole number of at most 10")
            return left ** int(right)
    raise ValueError("only numbers, + - * / ** and parentheses are allowed")


def evaluate_expression(expression: str) -> Calc:
    """Evaluate arithmetic safely; operands are the numeric literals in order of appearance."""
    text = expression.replace("×", "*").replace("÷", "/").replace("−", "-").replace(",", "")
    if len(text) > _MAX_EXPRESSION_CHARS:
        raise ValueError(f"expression longer than {_MAX_EXPRESSION_CHARS} characters")
    try:
        tree = ast.parse(text.strip(), mode="eval")
        operands: list[Decimal] = []
        result = _literals(tree.body, operands)
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
    return is_valid_rounding(shown, source) or is_valid_rounding(shown, -source)


def _supported_by_value(
    shown: ParsedNumber,
    base: Decimal,
    plain: bool,
    magnitudes: tuple[Decimal, ...] | None = None,
) -> bool:
    """`plain` marks a calculation result or a quoted number without a % sign."""
    if magnitudes is None:
        magnitudes = _MAGNITUDES if shown.scale != 1 else _MAGNITUDES[:1]
    if any(_rounds_to(shown, base * s) for s in magnitudes):
        return True
    return shown.percent and plain and _rounds_to(shown, base * 100)


def _in_question(shown: ParsedNumber, question: list[ParsedNumber]) -> bool:
    return any(shown.value == q.value and shown.scale == q.scale for q in question)


def _operand_cited(op: Decimal, cited: list[ParsedNumber]) -> bool:
    """An operand is a number from a cited quote or a unit constant; the question never counts."""
    if op in _UNIT_CONSTANTS:
        return True
    return any(op in (b * s, -b * s) for n in cited for b in _bases(n) for s in _MAGNITUDES)


def _calc_results(calcs: list[Calc], cited: list[ParsedNumber]) -> list[Decimal]:
    return [
        c.result
        for c in calcs
        if _calc_is_sound(c) and all(_operand_cited(o, cited) for o in c.operands)
    ]


def _number_supported(
    shown: ParsedNumber,
    cited: list[ParsedNumber],
    results: list[Decimal],
    question: list[ParsedNumber],
    magnitudes: tuple[Decimal, ...] | None = None,
) -> bool:
    if _in_question(shown, question):
        return True
    if any(_supported_by_value(shown, r, True, magnitudes) for r in results):
        return True
    return any(
        _supported_by_value(shown, b, not n.percent, magnitudes) for n in cited for b in _bases(n)
    )


def _final_as_shown(fv: FinalValue) -> ParsedNumber:
    exponent = fv.value.as_tuple().exponent
    decimals = max(0, -exponent) if isinstance(exponent, int) else 0
    return ParsedNumber(fv.value, decimals, fv.unit == "percent", Decimal(1))


def verify_answer(
    answer: AgentAnswer, pages: dict[int, str], question: str, calcs: list[Calc]
) -> tuple[bool, list[str]]:
    problems: list[str] = []
    cited: list[ParsedNumber] = []
    problems.extend(
        f"calculation rejected: {c.expression[:60]!r}" for c in calcs if not _calc_is_sound(c)
    )
    for c in answer.citations:
        text = pages.get(c.page_index)
        if len(c.quote) > MAX_QUOTE_CHARS:
            problems.append(f"quote too long ({len(c.quote)} > {MAX_QUOTE_CHARS} characters)")
        elif text is None:
            problems.append(f"cited page {c.page_index} is not in the filing")
        elif not quote_in_page(c.quote, text):
            problems.append(f"quote not found on page {c.page_index}: {c.quote[:80]!r}")
        else:
            cited.extend(parse_numbers(c.quote))
    if not answer.declined:
        q_nums = parse_numbers(question)
        results = _calc_results(calcs, cited)
        problems.extend(
            f"unrecognized number format: {t}" for t in unrecognized_tokens(answer.answer)
        )
        for shown in parse_numbers(answer.answer):
            if not _number_supported(shown, cited, results, q_nums):
                problems.append(
                    f"number not supported by a cited quote or calculation: {shown.value}"
                )
        fv = answer.final_value
        if fv is not None and not _number_supported(
            _final_as_shown(fv), cited, results, q_nums, _FINAL_MAGNITUDES
        ):
            problems.append(
                f"final value not supported by a cited quote or calculation: {fv.value}"
            )
    return (not problems, problems)
