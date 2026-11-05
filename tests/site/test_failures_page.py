from pathlib import Path
from typing import Any

from support import publish, write_run

from footnote.site.build import build_site


def _failures(root: Path, tmp_path: Path, inject: dict[str, Any]) -> str:
    out = tmp_path / "out"
    build_site(root, out, **inject)
    return (out / "failures.html").read_text()


def test_typed_failures_of_each_run_link_to_their_question_pages(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    html = _failures(results, tmp_path, inject)
    assert "No failure write-ups" not in html
    assert "Footnote agent · development run (v1)" in html
    assert 'href="q/q2.html"' in html and "What was Acme income?" in html
    assert "retrieval miss" in html  # agent: the gold pages were not read
    assert "wrong number" in html  # no-docs: graded by code and wrong
    assert "none of the gold evidence pages was read" in html
    assert "q/q3.html" not in html and "Pfizer" not in html


def test_curated_write_ups_appear_above_the_typed_table(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    html = _failures(results, tmp_path, inject)
    assert html.index("Picked the wrong year.") < html.index("<table")


def test_a_run_without_failures_says_so(tmp_path: Path, inject: dict[str, Any]) -> None:
    root = tmp_path / "results"
    write_run(root, "agent-dev-1", "agent", "dev", ["q1"], labels=["correct"])
    publish(root, ["agent-dev-1"])
    inject["failures"] = []
    html = _failures(root, tmp_path, inject)
    assert "No failed questions in this run." in html and "No failure write-ups" not in html


def test_no_published_runs_says_nothing_has_been_published(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    root = tmp_path / "results"
    publish(root, [])
    inject["failures"] = []
    assert "No results have been published yet." in _failures(root, tmp_path, inject)
