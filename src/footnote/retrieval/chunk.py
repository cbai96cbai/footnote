"""Split pages into overlapping passages that never cross a page."""

from collections.abc import Sequence
from dataclasses import dataclass

from footnote.corpus.parse import Page
from footnote.llm.tokens import count_text_tokens

MAX_TOKENS = 350
OVERLAP_TOKENS = 50
_SLICE_CHARS = 200  # fallback for a single "word" longer than the limit (no whitespace to break on)


@dataclass(frozen=True)
class Passage:
    id: str  # "{doc_name}:p{page}:{n}"
    doc_name: str
    page_index: int
    text: str


def _words(text: str, max_tokens: int) -> list[str]:
    out: list[str] = []
    for w in text.split():
        if count_text_tokens(" " + w) <= max_tokens:
            out.append(w)
        else:
            out.extend(w[i : i + _SLICE_CHARS] for i in range(0, len(w), _SLICE_CHARS))
    return out


def _fit(words: list[str], start: int, end: int, max_tokens: int) -> int:
    """Largest end' <= end whose joined text really is within the limit (at least one word)."""
    while end > start + 1 and count_text_tokens(" ".join(words[start:end])) > max_tokens:
        end -= 1
    return end


def _split_page(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    words = _words(text, max_tokens)
    if not words:
        return []
    cost = [count_text_tokens(" " + w) for w in words]
    pieces: list[str] = []
    start = 0
    while True:
        end, used = start, 0
        while end < len(words) and used + cost[end] <= max_tokens:
            used += cost[end]
            end += 1
        end = _fit(words, start, max(end, start + 1), max_tokens)
        pieces.append(" ".join(words[start:end]))
        if end >= len(words):
            return pieces
        back, used = end, 0
        while back > start + 1 and used + cost[back - 1] <= overlap_tokens:
            back -= 1
            used += cost[back]
        start = back


def chunk_pages(
    pages: Sequence[Page], max_tokens: int = MAX_TOKENS, overlap_tokens: int = OVERLAP_TOKENS
) -> list[Passage]:
    if max_tokens < 1 or not 0 <= overlap_tokens < max_tokens:
        raise ValueError("need max_tokens >= 1 and 0 <= overlap_tokens < max_tokens")
    out: list[Passage] = []
    for page in pages:
        for n, text in enumerate(_split_page(page.text, max_tokens, overlap_tokens)):
            out.append(
                Passage(
                    f"{page.doc_name}:p{page.page_index}:{n}", page.doc_name, page.page_index, text
                )
            )
    return out
