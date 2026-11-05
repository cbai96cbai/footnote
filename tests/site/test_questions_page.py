import re
from pathlib import Path
from typing import Any

from support import publish, write_run

from footnote.site.build import build_site


def _html(results: Path, tmp_path: Path, inject: dict[str, Any], page: str) -> str:
    out = tmp_path / "out"
    build_site(results, out, **inject)
    return (out / page).read_text()


def _row(html: str, qid: str) -> str:
    rows = re.findall(r"<tr>.*?</tr>", html, re.S)
    return next(r for r in rows if f"q/{qid}.html" in r)


def test_one_row_per_question_with_company_filing_question_and_badges(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    html = _html(results, tmp_path, inject, "questions.html")
    row = _row(html, "q1")
    assert "Acme" in row and "Acme 10-K 2022" in row
    assert 'href="q/q1.html"' in row and "What was Acme revenue?" in row
    assert re.findall(r'class="badge (\w+)"', row) == ["correct", "correct"]
    assert re.findall(r'class="badge (\w+)"', _row(html, "q2")) == ["incorrect", "incorrect"]
    assert html.count("<tr>") == 3  # header and two non-Health-Care questions


def test_declined_badge_reads_as_its_verdict(tmp_path: Path, inject: dict[str, Any]) -> None:
    root = tmp_path / "results"
    write_run(root, "agent-dev-1", "agent", "dev", ["q1", "q2"], labels=["declined", "correct"])
    publish(root, ["agent-dev-1"])
    html = _html(root, tmp_path, inject, "questions.html")
    assert "declined</span>" in _row(html, "q1") and "correct</span>" in _row(html, "q2")


def test_the_front_page_and_nav_link_to_the_questions_page(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    index = _html(results, tmp_path, inject, "index.html")
    assert index.count('href="questions.html"') >= 2  # nav and a link in the text
