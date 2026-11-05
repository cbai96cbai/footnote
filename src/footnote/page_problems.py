"""How a quote is matched to a page, and the one place that writes and reads the page problems.

`not_found` and `missing_page` write the messages `verify_answer` reports for a quote that is not
on its cited page; `parse_page_problem` reads them back. They live together so the format has one
owner: change a message here and the parser (and its tests) change with it.
"""

import re
from dataclasses import dataclass

from footnote.corpus.normalize import normalize_text

_LETTER_HYPHEN = re.compile(r"(?<=[^\W\d_])-(?=[^\W\d_])")
_MAX_HINTS = 3
_HINT = "; this exact text is on page_index "
_HINT_AT_END = re.compile(re.escape(_HINT) + r"(-?\d+(?:, -?\d+)*)$")
_NOT_FOUND = re.compile(r"quote not found on page (-?\d+): ")
_NO_PAGE = re.compile(r"cited page (-?\d+) is not in the filing")


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


def _elsewhere(quote: str, page_index: int, pages: dict[int, str]) -> str:
    found = [n for n, t in sorted(pages.items()) if n != page_index and quote_in_page(quote, t)]
    if not found:
        return ""
    return _HINT + ", ".join(str(n) for n in found[:_MAX_HINTS])


def not_found(quote: str, page_index: int, pages: dict[int, str]) -> str:
    message = f"quote not found on page {page_index}: {quote[:80]!r}"
    return message + _elsewhere(quote, page_index, pages)


def missing_page(quote: str, page_index: int, pages: dict[int, str]) -> str:
    return f"cited page {page_index} is not in the filing" + _elsewhere(quote, page_index, pages)


@dataclass(frozen=True)
class PageProblem:
    """A quote that was not confirmed on its cited page (zero-based page indexes).

    `elsewhere` lists up to three other pages that hold the exact text; `page_exists` is False
    when the cited page is not in the filing at all."""

    cited: int
    elsewhere: tuple[int, ...]
    page_exists: bool


def parse_page_problem(message: str) -> PageProblem | None:
    """The page problem a verification message reports; None for any other kind of problem."""
    hint = _HINT_AT_END.search(message)
    elsewhere = tuple(int(n) for n in hint.group(1).split(", ")) if hint else ()
    body = message[: hint.start()] if hint else message
    if m := _NOT_FOUND.match(body):
        return PageProblem(int(m.group(1)), elsewhere, True)
    if (m := _NO_PAGE.fullmatch(body)) is not None:
        return PageProblem(int(m.group(1)), elsewhere, False)
    return None
