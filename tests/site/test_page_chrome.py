import re
from pathlib import Path
from typing import Any

from crawl import links_of
from support import publish, write_run

from footnote.site.build import build_site


def _out(results: Path, tmp_path: Path, inject: dict[str, Any]) -> Path:
    out = tmp_path / "out"
    build_site(results, out, **inject)
    return out


def test_every_page_has_an_inline_favicon_and_no_external_request(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    out = _out(results, tmp_path, inject)
    for page in ("index.html", "q/q1.html", "404.html"):
        html = (out / page).read_text()
        assert '<link rel="icon" href="data:image/svg+xml,' in html, page


def test_a_404_page_is_built_with_links_that_work_from_any_address(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    html = (_out(results, tmp_path, inject) / "404.html").read_text()
    assert "Page not found" in html
    assert 'href="/footnote/style.css"' in html and 'href="/footnote/index.html"' in html
    hrefs = links_of(html)
    # site links carry the base path; absolute links to other sites work from any address
    assert all(h.startswith(("/footnote/", "data:", "https://")) for h in hrefs)
    assert any(h.startswith("/footnote/") for h in hrefs)
    other = tmp_path / "other"
    build_site(results, other, **inject, base_path="/site/")
    assert 'href="/site/style.css"' in (other / "404.html").read_text()
    index = (_out(results, tmp_path / "x", inject) / "index.html").read_text()
    assert "/footnote/" not in index  # every other page keeps relative links


def test_historical_context_is_a_small_labelled_table(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    html = (_out(results, tmp_path, inject) / "index.html").read_text()
    flat = " ".join(html.split())
    assert "results from the FinanceBench paper's own setups" in flat
    table = flat.split("Historical context")[1]
    assert "<th>Setup</th>" in table and '<th class="num">Accuracy</th>' in table
    assert "<td>Evidence pages given</td>" in table and '<td class="num">85%</td>' in table


def test_front_page_states_the_result_above_the_chart(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    root = tmp_path / "results"
    write_run(
        root, "agent-test-1", "agent", "test", ["q1", "q2", "q3"], summary_extra={"accuracy": 1.0}
    )
    write_run(
        root,
        "nodocs-test-1",
        "no-docs",
        "test",
        ["q1", "q2", "q3"],
        summary_extra={"accuracy": 0.0},
    )
    publish(root, ["agent-test-1", "nodocs-test-1"])
    html = (_out(root, tmp_path, inject) / "index.html").read_text()
    sentence = '<p class="headline">On the 3 test questions, the agent scores highest (100%)'
    assert sentence in html and html.index(sentence) < html.index("<svg")


def test_no_headline_without_results(tmp_path: Path, inject: dict[str, Any]) -> None:
    root = tmp_path / "results"
    publish(root, [])
    assert 'class="headline"' not in (_out(root, tmp_path, inject) / "index.html").read_text()


def test_the_form_is_hyphenated_wherever_it_is_shown(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    out = _out(results, tmp_path, inject)
    for page in out.rglob("*.html"):
        assert not re.search(r"(?<![_\w])10K", page.read_text()), page.name
