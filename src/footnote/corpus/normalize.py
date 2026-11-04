"""Text normalization shared by evidence alignment and quote matching."""

import re
import unicodedata

_HYPHEN_BREAK = re.compile(r"-\r?\n")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """NFKC, join `-` + line break without a space, collapse whitespace, strip, casefold."""
    s = unicodedata.normalize("NFKC", s)
    s = _HYPHEN_BREAK.sub("", s)
    return _WHITESPACE.sub(" ", s).strip().casefold()


_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def word_tokens(s: str) -> set[str]:
    return set(_TOKEN.findall(normalize_text(s)))


def word_coverage(gold: str, page: str) -> float:
    """Share of the distinct words/numbers in `gold` that appear on `page`, ignoring order."""
    want = word_tokens(gold)
    return len(want & word_tokens(page)) / len(want) if want else 0.0
