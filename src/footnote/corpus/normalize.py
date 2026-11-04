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
