from pathlib import Path
from typing import Any

from crawl import crawl, links_of
from support import publish, write_run

from footnote.site.build import build_site


def _built(results: Path, tmp_path: Path, inject: dict[str, Any]) -> Path:
    out = tmp_path / "out"
    build_site(results, out, **inject)
    return out


def _check(out: Path) -> None:
    reached, broken = crawl(out)
    questions = {f"q/{p.name}" for p in (out / "q").glob("*.html")}
    assert questions, "the fixture should build question pages"
    assert questions <= reached, f"not reachable from index.html: {sorted(questions - reached)}"
    assert broken == []
    for name in ("index.html", "questions.html", "failures.html", "about.html"):
        assert name in reached, name


def test_every_question_page_is_reachable_and_no_link_is_broken(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    _check(_built(results, tmp_path, inject))


def test_reachability_holds_for_test_and_development_runs_together(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    root = tmp_path / "results"
    write_run(root, "agent-test-1", "agent", "test", ["q1"])
    write_run(root, "nodocs-test-1", "no-docs", "test", ["q1"])
    write_run(root, "agent-dev-1", "agent", "dev", ["q2"])
    publish(root, ["agent-test-1", "nodocs-test-1", "agent-dev-1"])
    _check(_built(root, tmp_path, inject))


def test_every_nav_item_leads_to_a_page_with_content(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    out = _built(results, tmp_path, inject)
    nav = (out / "index.html").read_text().split("<nav>")[1].split("</nav>")[0]
    targets = links_of(nav)
    assert len(targets) == 4
    for href in targets:
        html = (out / href).read_text()
        body = html.split("<main")[1].split("</main>")[0]
        assert "<h1>" in body and len(body) > 400, href


def test_a_site_without_results_still_links_every_page(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    root = tmp_path / "results"
    publish(root, [])
    out = _built(root, tmp_path, inject)
    reached, broken = crawl(out)
    assert broken == [] and {"questions.html", "failures.html", "about.html"} <= reached
