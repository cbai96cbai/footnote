"""Build the static site from result folders."""

import shutil
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from footnote.benchmark import AgentQuestion, DocInfo, doc_info, load_questions
from footnote.eval.gold import Gold, load_gold
from footnote.site import filters, wording
from footnote.site.chart import Bar, accuracy_chart
from footnote.site.headline import headline
from footnote.site.links import load_link_status
from footnote.site.output import MARKER, replace_output
from footnote.site.pages import failure_tables, question_pages, question_rows
from footnote.site.paired import paired_differences
from footnote.site.runs import Run, load_runs, split_runs

__all__ = ["MARKER", "build_site"]
HERE = Path(__file__).parent
# where the site is served (GitHub Pages project site); only 404.html needs it, because the
# host serves that one page for any missing path and relative links would not resolve
BASE_PATH = "/footnote/"
PAPER = [
    ("Shared vector store", 19),
    ("One vector store per filing", 50),
    ("Long context", 79),
    ("Evidence pages given", 85),
]


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(HERE / "templates"),
        autoescape=select_autoescape(["html"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["excerpt"] = filters.excerpt
    env.filters["step_text"] = wording.step_text
    env.filters["problem_notes"] = wording.problem_notes
    env.filters["shown_steps"] = wording.shown_steps
    env.filters["final_value_line"] = wording.final_value_line
    env.filters["decline_headline"] = wording.decline_headline
    env.filters["filing"] = wording.filing
    env.filters["form"] = wording.form
    env.filters["pack_note"] = wording.pack_note
    env.filters["own_note"] = wording.own_note
    env.filters["final_value_only"] = filters.final_value_only
    env.filters["difference"] = filters.difference
    env.filters["pct"] = filters.pct
    env.filters["usd"] = filters.usd
    env.filters["seconds"] = filters.seconds
    return env


def _yaml_list(name: str) -> list[Any]:
    return yaml.safe_load((HERE / name).read_text(encoding="utf-8")) or []


def _earlier(runs: list[Run]) -> set[str]:
    """Run ids of a system's versions that a later published version replaces."""
    last = {r.system: r.run_id for r in runs}
    return {r.run_id for r in runs if r.split != "test" and last[r.system] != r.run_id}


def _chart(runs: list[Run], title: str, earlier: set[str]) -> Any:
    bars = []
    for r in runs:
        ci = r.summary.get("ci95")
        label = r.display_label + (" (earlier version)" if r.run_id in earlier else "")
        bars.append(Bar(label, r.summary.get("accuracy"), *(ci if ci else (None, None))))
    return accuracy_chart(bars, title)


def build_site(
    results_dir: Path,
    out_dir: Path,
    *,
    questions: list[AgentQuestion] | None = None,
    gold: dict[str, Gold] | None = None,
    docs: dict[str, DocInfo] | None = None,
    link_status: dict[str, Any] | None = None,
    featured: list[str] | None = None,
    failures: list[dict[str, str]] | None = None,
    base_path: str = BASE_PATH,
) -> None:
    by_id = {q.id: q for q in (questions if questions is not None else load_questions())}
    gold = gold if gold is not None else load_gold()
    docs = docs if docs is not None else doc_info()
    link_status = link_status if link_status is not None else load_link_status()
    featured_ids = featured if featured is not None else _yaml_list("featured.yaml")
    failure_list = failures if failures is not None else _yaml_list("failures.yaml")

    runs = load_runs(results_dir)
    test, dev = split_runs(runs)
    published = question_pages(runs, by_id, gold, docs, link_status)
    pages = published.pages
    site = {
        "test": test,
        "dev": dev,
        "paired": paired_differences(test, by_id),
        "pages": pages,
        "rows": question_rows(pages),
        "hidden": published.hidden,
        "featured": [pages[i] for i in featured_ids if i in pages],
        "curated": [{**f, "page": pages[f["id"]]} for f in failure_list if f.get("id") in pages],
        "failure_tables": failure_tables(runs, pages, gold),
        "has_runs": bool(runs),
        "headline": headline(runs, lambda q: by_id[q].doc_name if q in by_id else q),
    }
    replace_output(out_dir, lambda target: _write_pages(target, site, base_path))


def _write_pages(out_dir: Path, site: dict[str, Any], base_path: str) -> None:
    env = _env()
    shutil.copy(HERE / "style.css", out_dir / "style.css")
    (out_dir / "q").mkdir(exist_ok=True)

    def write(rel: str, template: str, root: str, **ctx: Any) -> None:
        html = env.get_template(template).render(root=root, **ctx)
        (out_dir / rel).write_text(html, encoding="utf-8")

    test, dev = site["test"], site["dev"]
    write(
        "index.html",
        "index.html",
        "",
        test=test,
        dev=dev,
        paired=site["paired"],
        headline=site["headline"],
        hidden=site["hidden"],
        earlier=_earlier(dev),
        test_chart=_chart(test, "Accuracy on the test split", set()) if test else None,
        dev_chart=_chart(dev, "Accuracy on development runs", _earlier(dev)) if dev else None,
        paper=PAPER,
        featured=site["featured"],
    )
    for i, page in site["pages"].items():
        write(f"q/{i}.html", "question.html", "../", **page)
    write("questions.html", "questions.html", "", rows=site["rows"], hidden=site["hidden"])
    write(
        "failures.html",
        "failures.html",
        "",
        items=site["curated"],
        tables=site["failure_tables"],
        has_runs=site["has_runs"],
    )
    write("about.html", "about.html", "", has_test=bool(test))
    write("404.html", "not_found.html", base_path)
