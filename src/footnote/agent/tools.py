"""The agent's tools over one filing: search, read a page, calculate."""

from dataclasses import dataclass
from typing import Any, Protocol

from footnote.llm.tokens import count_text_tokens
from footnote.retrieval.index import ScoredPassage
from footnote.verify import Calc, evaluate_expression

READ_PAGE_MAX_TOKENS = 3_000
DEFAULT_K = 5
MAX_K = 8
SUMMARY_CHARS = 200


class SearchIndex(Protocol):
    def search(self, query: str, k: int = 8) -> list[ScoredPassage]: ...


class ToolError(Exception):
    """The call was malformed or impossible; the message goes back to the model."""


@dataclass(frozen=True)
class ToolOutput:
    content: str
    summary: str
    pages_seen: list[int]


def cap(text: str, limit: int = SUMMARY_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def cut_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    if count_text_tokens(text) <= max_tokens:
        return text, False
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_text_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo], True


def _int_arg(args: dict[str, Any], name: str) -> int:
    value = args.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"{name} must be an integer")
    return value


class Toolbox:
    def __init__(self, pages: dict[int, str], index: SearchIndex) -> None:
        self.pages = pages
        self.index = index
        self.calcs: list[Calc] = []

    def run(self, name: str, args: dict[str, Any]) -> ToolOutput:
        if name == "search_filing":
            return self._search(args)
        if name == "read_page":
            return self._read(args)
        if name == "calculate":
            return self._calculate(args)
        raise ToolError(f"unknown tool {name!r}")

    def _search(self, args: dict[str, Any]) -> ToolOutput:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")
        k = _int_arg(args, "k") if "k" in args else DEFAULT_K
        if not 1 <= k <= MAX_K:
            raise ToolError(f"k must be between 1 and {MAX_K}")
        hits = self.index.search(query, k)
        pages = [h.passage.page_index for h in hits]
        body = "\n\n".join(f"[page_index {h.passage.page_index}]\n{h.passage.text}" for h in hits)
        return ToolOutput(
            body or "no passages found",
            cap(f"search {query!r}: {len(hits)} passages, page_index {sorted(set(pages))}"),
            pages,
        )

    def _read(self, args: dict[str, Any]) -> ToolOutput:
        n = _int_arg(args, "page_index")
        if n not in self.pages:
            raise ToolError(f"page_index {n} is out of range (0 to {len(self.pages) - 1})")
        text, cut = cut_to_tokens(self.pages[n], READ_PAGE_MAX_TOKENS)
        note = "\n[page cut at 3,000 tokens]" if cut else ""
        return ToolOutput(
            f"[page_index {n}]\n{text}{note}",
            f"read page_index {n}: {len(self.pages[n])} characters" + (", cut" if cut else ""),
            [n],
        )

    def _calculate(self, args: dict[str, Any]) -> ToolOutput:
        expression = args.get("expression")
        if not isinstance(expression, str):
            raise ToolError("expression must be a string")
        try:
            calc = evaluate_expression(expression)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        self.calcs.append(calc)
        line = f"{expression} = {format(calc.result, 'f')}"
        return ToolOutput(line, cap(line), [])
