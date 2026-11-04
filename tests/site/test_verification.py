from pathlib import Path
from typing import Any

from support import publish, write_run

from footnote.site.build import build_site

QUOTE = "Net sales were 1,234 million dollars"


def _page(tmp_path: Path, inject: dict[str, Any], **kw: Any) -> str:
    root = tmp_path / "results"
    write_run(root, "agent-dev-1", "agent", "dev", ["q1"], quote=QUOTE, **kw)
    publish(root, ["agent-dev-1"])
    out = tmp_path / "out"
    build_site(root, out, **inject)
    return (out / "q" / "q1.html").read_text()


def test_confirmed_quote_is_marked(tmp_path: Path, inject: dict[str, Any]) -> None:
    html = _page(tmp_path, inject)
    assert f"<mark>{QUOTE}</mark>" in html
    assert "Quote check: passed" in html


def test_failed_quote_is_never_shown_as_filing_text(tmp_path: Path, inject: dict[str, Any]) -> None:
    html = _page(
        tmp_path,
        inject,
        verify_ok=False,
        problems=["quote not found on page 4: 'Net sales'"],
    )
    assert QUOTE not in html and "<mark>" not in html
    assert "Quote not found in the filing" in html
    assert "quote not found on page 4" in html  # the check's own message
    assert "of the benchmark PDF" not in html


def test_unchecked_citation_is_not_shown_as_filing_text(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    html = _page(tmp_path, inject, verify_ok=None)
    assert QUOTE not in html and "<mark>" not in html
    assert "not checked" in html.lower()


def test_quotes_found_but_final_value_untraced_shows_quotes_and_a_note(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    html = _page(
        tmp_path,
        inject,
        verify_ok=False,
        problems=["final value not supported by a cited quote or calculation: 12.5"],
    )
    assert f"<mark>{QUOTE}</mark>" in html  # the quotes were found on the page
    assert "of the benchmark PDF" in html
    assert "Quote not found" not in html and "Citation check failed" not in html
    assert "could not be traced to those quotes" in html
    assert "12.5" in html  # the check's own message


def test_a_missing_quote_still_hides_all_quotes_even_with_a_final_value_problem(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    html = _page(
        tmp_path,
        inject,
        verify_ok=False,
        problems=["quote not found on page 4: 'x'", "final value not supported: 1"],
    )
    assert QUOTE not in html and "<mark>" not in html
    assert "Quote not found in the filing" in html
