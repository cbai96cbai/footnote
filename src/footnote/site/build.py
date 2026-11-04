"""Build the static site from result folders."""

import shutil
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from footnote.benchmark import AgentQuestion, DocInfo, doc_info, load_questions
from footnote.eval.gold import Gold, load_gold
from footnote.site import filters
from footnote.site.chart import Bar, accuracy_chart
from footnote.site.links import edgar_fallback, load_link_status
from footnote.site.output import MARKER, replace_output
from footnote.site.runs import Run, load_runs, split_runs

__all__ = ["MARKER", "build_site"]
HERE = Path(__file__).parent
HIDDEN_SECTOR = "Health Care"
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
    env.filters["step_arguments"] = filters.step_arguments
    env.filters["pct"] = filters.pct
    env.filters["usd"] = filters.usd
    env.filters["seconds"] = filters.seconds
    return env


def _yaml_list(name: str) -> list[Any]:
    return yaml.safe_load((HERE / name).read_text(encoding="utf-8")) or []


def _chart(runs: list[Run], title: str) -> Any:
    bars = []
    for r in runs:
        ci = r.summary.get("ci95")
        bars.append(Bar(r.label, r.summary.get("accuracy"), *(ci if ci else (None, None))))
    return accuracy_chart(bars, title)


def _question_page(
    q: AgentQuestion, runs: list[Run], gold: Gold, doc: DocInfo, link_status: dict[str, Any]
) -> dict[str, Any]:
    entries = []
    for r in runs:
        p, v = r.predictions.get(q.id), r.verdicts.get(q.id)
        if p is not None and v is not None:
            entries.append({"run": r, "label": r.label, "pred": p, "verdict": v})
    return {
        "q": q,
        "gold": gold,
        "doc": doc,
        "source": doc.doc_link,
        "edgar": edgar_fallback(doc, link_status),
        "entries": entries,
    }


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
) -> None:
    by_id = {q.id: q for q in (questions if questions is not None else load_questions())}
    gold = gold if gold is not None else load_gold()
    docs = docs if docs is not None else doc_info()
    link_status = link_status if link_status is not None else load_link_status()
    featured_ids = featured if featured is not None else _yaml_list("featured.yaml")
    failure_list = failures if failures is not None else _yaml_list("failures.yaml")

    runs = load_runs(results_dir)
    test, dev = split_runs(runs)
    shown = test or dev
    ids = sorted({i for r in shown for i in r.predictions if i in by_id and i in gold})
    pages = {
        i: _question_page(by_id[i], shown, gold[i], docs[by_id[i].doc_name], link_status)
        for i in ids
    }

    # a Health Care company stays in the scores and on its own page, never in a curated example list
    def curated(ids: list[str]) -> list[str]:
        return [i for i in ids if i in pages and pages[i]["doc"].gics_sector != HIDDEN_SECTOR]

    featured_pages = [pages[i] for i in curated(featured_ids)]
    kept = set(curated([f.get("id", "") for f in failure_list]))
    failure_items = [{**f, "page": pages[f["id"]]} for f in failure_list if f.get("id") in kept]

    replace_output(
        out_dir,
        lambda target: _write_pages(target, test, dev, pages, featured_pages, failure_items),
    )


def _write_pages(
    out_dir: Path,
    test: list[Run],
    dev: list[Run],
    pages: dict[str, dict[str, Any]],
    featured_pages: list[dict[str, Any]],
    failure_items: list[dict[str, Any]],
) -> None:
    env = _env()
    shutil.copy(HERE / "style.css", out_dir / "style.css")
    (out_dir / "q").mkdir(exist_ok=True)

    def write(rel: str, template: str, root: str, **ctx: Any) -> None:
        html = env.get_template(template).render(root=root, **ctx)
        (out_dir / rel).write_text(html, encoding="utf-8")

    write(
        "index.html",
        "index.html",
        "",
        test=test,
        dev=dev,
        test_chart=_chart(test, "Accuracy on the test split") if test else None,
        dev_chart=_chart(dev, "Accuracy on development runs") if dev else None,
        paper=PAPER,
        featured=featured_pages,
    )
    for i, page in pages.items():
        write(f"q/{i}.html", "question.html", "../", **page)
    write("failures.html", "failures.html", "", items=failure_items)
    write("about.html", "about.html", "")
