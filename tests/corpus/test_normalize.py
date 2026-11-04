from footnote.corpus.normalize import normalize_text


def test_normalize_joins_hyphen_line_break() -> None:
    assert normalize_text("net reve-\nnue grew") == "net revenue grew"


def test_normalize_keeps_hyphen_without_line_break() -> None:
    assert normalize_text("year-over-year") == "year-over-year"


def test_normalize_collapses_whitespace_and_strips() -> None:
    assert normalize_text("  a \t b\n\n c  ") == "a b c"


def test_normalize_nfkc_and_casefold() -> None:
    assert normalize_text("ＡＢＣ ﬁnal Straße") == "abc final strasse"


def test_normalize_hyphen_before_crlf() -> None:
    assert normalize_text("co-\r\noperate") == "cooperate"
