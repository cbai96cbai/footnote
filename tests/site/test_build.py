import re
from pathlib import Path
from typing import Any

import pytest
from support import write_run

from footnote.benchmark import load_questions
from footnote.corpus.parse import load_pages
from footnote.site.build import build_site
from footnote.site.filters import EXCERPT_LIMIT, excerpt


def build(results: Path, out: Path, inject: dict[str, Any]) -> None:
    build_site(results, out, **inject)


def test_every_document_string_through_excerpt_filter(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    out = tmp_path / "out"
    build(results, out, inject)
    html = (out / "q" / "q1.html").read_text()
    assert "word " * 100 not in html  # nothing near the 1500-character originals
    for text in re.findall(r"<(?:mark|blockquote|p)[^>]*>([^<]*)<", html):
        assert len(text) <= EXCERPT_LIMIT + 5
    assert len(excerpt("x" * 1000)) <= EXCERPT_LIMIT + 1
    assert excerpt("short") == "short"
    assert len(excerpt("y" * 1000, 200)) <= 201


def test_trace_has_no_raw_page_text(results: Path, tmp_path: Path, inject: dict[str, Any]) -> None:
    out = tmp_path / "out"
    build(results, out, inject)
    html = (out / "q" / "q1.html").read_text()
    assert "read_page" in html and "found 3 pages" in html
    trace = html.split('<ol class="trace">')[1].split("</ol>")[0]
    assert "word " * 41 not in trace  # step summaries are cut at 200 characters


def test_featured_excludes_health_care(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    out = tmp_path / "out"
    build(results, out, inject)
    index = (out / "index.html").read_text()
    assert 'href="q/q1.html"' in index
    assert 'href="q/q3.html"' not in index


def test_build_is_deterministic(results: Path, tmp_path: Path, inject: dict[str, Any]) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    build(results, a, inject)
    build(results, b, inject)
    files = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    assert files and files == sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    for f in files:
        assert (a / f).read_bytes() == (b / f).read_bytes(), f


def test_source_link_falls_back_to_edgar(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    out = tmp_path / "out"
    build(results, out, inject)
    acme = (out / "q" / "q1.html").read_text()
    assert "https://a.test/x" not in acme
    assert "https://www.sec.gov/cgi-bin/browse-edgar?company=Acme" in acme
    assert "type=10-K" in acme
    assert "page 5 of the benchmark PDF" in acme
    assert "https://p.test/x" in (out / "q" / "q3.html").read_text()


def test_pages_exist_and_labels(results: Path, tmp_path: Path, inject: dict[str, Any]) -> None:
    out = tmp_path / "out"
    build(results, out, inject)
    for name in ("index.html", "failures.html", "about.html", "style.css"):
        assert (out / name).exists()
    index = (out / "index.html").read_text()
    assert "without documents" in index.lower()
    assert "development runs (exploratory)" in index.lower()
    assert "historical context" in index.lower() and "85%" in index
    assert "<svg" in index
    about = (out / "about.html").read_text()
    assert "CC BY-NC 4.0" in about and "2311.11944" in about
    assert "AI-reference estimates" in about
    assert "page_index + 1" in about or "page N of the benchmark PDF" in about
    assert "Picked the wrong year." in (out / "failures.html").read_text()


@pytest.mark.corpus
def test_no_long_page_substring_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOOTNOTE_CACHE")  # the real page cache, not the per-test one
    questions = load_questions()[:3]
    root = tmp_path / "results"
    pages = {q.doc_name: load_pages(q.doc_name) for q in questions}
    for q in questions:
        page = max(pages[q.doc_name], key=lambda p: len(p.text))
        quote = " ".join(page.text.split())[:80]
        write_run(root, "agent-dev-1", "agent", "dev", [q], quote=quote, page_index=page.page_index)
    out = tmp_path / "out"
    build_site(root, out)  # real questions, reference answers, documents and committed content
    html = "\n".join(p.read_text() for p in out.rglob("*.html"))
    assert all(f"q/{q.id}.html" in "".join(str(p) for p in out.rglob("*.html")) for q in questions)
    for doc_pages in pages.values():
        for page in doc_pages:
            t = " ".join(page.text.split())
            for i in range(0, max(len(t) - 200, 0), 25):
                assert t[i : i + 200] not in html
