"""Text normalization shared by evidence alignment and quote matching."""

import re
import unicodedata

# pdfium writes some hyphens as the control character U+0002 (on 1,730 of 12,013 cached pages);
# the soft hyphen is an optional hyphen. Both stand for "-".
_HYPHEN_MARKS = str.maketrans({"\x02": "-", "\u00ad": "-"})
_HYPHEN_BREAK = re.compile(r"-\r?\n")
_WHITESPACE = re.compile(r"\s+")


def map_hyphen_marks(s: str) -> str:
    """Write the extractor's hyphen marks (U+0002, soft hyphen) as a plain hyphen."""
    return s.translate(_HYPHEN_MARKS)


def normalize_text(s: str) -> str:
    """NFKC, hyphen marks to "-", join `-` + line break without a space, collapse whitespace,
    strip, casefold."""
    s = map_hyphen_marks(unicodedata.normalize("NFKC", s))
    s = _HYPHEN_BREAK.sub("", s)
    return _WHITESPACE.sub(" ", s).strip().casefold()


_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def word_tokens(s: str) -> set[str]:
    return set(_TOKEN.findall(normalize_text(s)))


def word_coverage(gold: str, page: str) -> float:
    """Share of the distinct words/numbers in `gold` that appear on `page`, ignoring order."""
    want = word_tokens(gold)
    return len(want & word_tokens(page)) / len(want) if want else 0.0
