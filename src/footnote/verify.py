"""The grounding check: every quote exists in the filing, every number traces to a quote.

What it proves: every quote exists in the cited filing page, and every number in the answer is a
correct rounding of a number in a cited quote, or of a logged calculation whose operands all appear
in cited quotes (numbers in the question are exempt). It does not prove that a number means what
the answer says (units, periods); the grader measures that.

Documented limits: tables rarely state units in the quoted line, so a shown number with a scale
word may match a quoted number times 1e3, 1e6 or 1e9; signs are not compared (an answer says
"fell $1.6 billion" for a quoted "(1,577)").
"""

from dataclasses import dataclass
from decimal import Decimal

from footnote.answers import AgentAnswer
from footnote.corpus.normalize import normalize_text
from footnote.numbers import ParsedNumber, is_valid_rounding, parse_numbers

_MAGNITUDES = (Decimal(1), Decimal("1e3"), Decimal("1e6"), Decimal("1e9"))


@dataclass(frozen=True)
class Calc:
    expression: str
    operands: list[Decimal]
    result: Decimal


def _squash(s: str) -> str:
    # the normalizer joins "-\n" without a space, so a quoted "long-term" must meet "longterm"
    return normalize_text(s).replace("-", "")


def quote_in_page(quote: str, page_text: str) -> bool:
    q = _squash(quote)
    return bool(q) and q in _squash(page_text)


def _bases(n: ParsedNumber) -> list[Decimal]:
    """Values a quoted number can stand for: as written, and with its own scale word."""
    return [n.value] if n.scale == 1 else [n.value, n.value * n.scale]


def _rounds_to(shown: ParsedNumber, source: Decimal) -> bool:
    return is_valid_rounding(shown, source) or is_valid_rounding(shown, -source)


def _supported_by_value(shown: ParsedNumber, base: Decimal, plain: bool) -> bool:
    """`plain` marks a calculation result or a quoted number without a % sign."""
    magnitudes = _MAGNITUDES if shown.scale != 1 else _MAGNITUDES[:1]
    if any(_rounds_to(shown, base * s) for s in magnitudes):
        return True
    return shown.percent and plain and _rounds_to(shown, base * 100)


def _in_question(shown: ParsedNumber, question: list[ParsedNumber]) -> bool:
    return any(shown.value == q.value and shown.scale == q.scale for q in question)


def _operand_cited(op: Decimal, cited: list[ParsedNumber], question: list[ParsedNumber]) -> bool:
    pool = cited + question
    return any(op in (b * s, -b * s) for n in pool for b in _bases(n) for s in _MAGNITUDES)


def _calc_results(
    calcs: list[Calc], cited: list[ParsedNumber], q: list[ParsedNumber]
) -> list[Decimal]:
    return [c.result for c in calcs if all(_operand_cited(o, cited, q) for o in c.operands)]


def _number_supported(
    shown: ParsedNumber,
    cited: list[ParsedNumber],
    results: list[Decimal],
    question: list[ParsedNumber],
) -> bool:
    if _in_question(shown, question):
        return True
    if any(_supported_by_value(shown, r, True) for r in results):
        return True
    return any(_supported_by_value(shown, b, not n.percent) for n in cited for b in _bases(n))


def verify_answer(
    answer: AgentAnswer, pages: dict[int, str], question: str, calcs: list[Calc]
) -> tuple[bool, list[str]]:
    problems: list[str] = []
    cited: list[ParsedNumber] = []
    for c in answer.citations:
        text = pages.get(c.page_index)
        if text is None:
            problems.append(f"cited page {c.page_index} is not in the filing")
        elif not quote_in_page(c.quote, text):
            problems.append(f"quote not found on page {c.page_index}: {c.quote[:80]!r}")
        else:
            cited.extend(parse_numbers(c.quote))
    if not answer.declined:
        q_nums = parse_numbers(question)
        results = _calc_results(calcs, cited, q_nums)
        for shown in parse_numbers(answer.answer):
            if not _number_supported(shown, cited, results, q_nums):
                problems.append(
                    f"number not supported by a cited quote or calculation: {shown.value}"
                )
    return (not problems, problems)
