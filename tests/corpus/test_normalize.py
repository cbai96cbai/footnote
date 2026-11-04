from footnote.corpus.normalize import normalize_text, word_coverage, word_tokens


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


def test_word_coverage_full_for_reordered_table_row() -> None:
    gold = "Cash and cash equivalents\n$\n7,776 \n$\n5,197"
    page = "Cash and cash equivalents $ 5,197 $ 7,776"
    assert word_coverage(gold, page) == 1.0


def test_word_coverage_partial_when_number_missing() -> None:
    assert word_coverage("revenue 1,234 and 5.6 growth", "revenue 1,234 growth") == 0.6


def test_word_coverage_keeps_number_formats_whole() -> None:
    assert word_tokens("Sales $1,234.5 up 3.2%") == {"sales", "1,234.5", "up", "3.2"}


def test_word_coverage_empty_gold_is_zero() -> None:
    assert word_coverage("$ - %", "anything") == 0.0
