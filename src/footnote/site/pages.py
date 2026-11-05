"""What each page shows, worked out from the published runs and the benchmark."""

from dataclasses import dataclass
from typing import Any

from footnote.benchmark import AgentQuestion, DocInfo
from footnote.eval.failures import REASONS, classify_run
from footnote.eval.gold import Gold
from footnote.site.links import edgar_fallback
from footnote.site.runs import Run, run_tags

HIDDEN_SECTOR = "Health Care"


@dataclass(frozen=True)
class Published:
    """The question pages to build (never a Health Care company's) and how many were left out."""

    pages: dict[str, dict[str, Any]]
    hidden: int


def _question_page(
    q: AgentQuestion, runs: list[Run], gold: Gold, doc: DocInfo, link_status: dict[str, Any]
) -> dict[str, Any]:
    entries = []
    for r in runs:
        p, v = r.predictions.get(q.id), r.verdicts.get(q.id)
        if p is not None and v is not None:
            entries.append({"run": r, "label": r.page_label, "pred": p, "verdict": v})
    return {
        "q": q,
        "gold": gold,
        "doc": doc,
        "source": doc.doc_link,
        "edgar": edgar_fallback(doc, link_status),
        "entries": entries,
    }


def question_pages(
    runs: list[Run],
    by_id: dict[str, AgentQuestion],
    gold: dict[str, Gold],
    docs: dict[str, DocInfo],
    link_status: dict[str, Any],
) -> Published:
    """A page for every question any published run answered, except Health Care companies'
    (their answers stay in the scores)."""
    ids = sorted({i for r in runs for i in r.predictions if i in by_id and i in gold})
    pages: dict[str, dict[str, Any]] = {}
    hidden = 0
    for i in ids:
        doc = docs[by_id[i].doc_name]
        page = _question_page(by_id[i], runs, gold[i], doc, link_status)
        if not page["entries"]:
            continue
        if doc.gics_sector == HIDDEN_SECTOR:
            hidden += 1
        else:
            pages[i] = page
    return Published(pages, hidden)


def question_rows(pages: dict[str, dict[str, Any]], runs: list[Run]) -> list[dict[str, Any]]:
    """One row per question for the Questions page: where it is from and each run's verdict."""
    tags = run_tags(runs)
    rows = [
        {
            "page": page,
            "badges": [
                {"tag": tags[e["run"].run_id], "label": e["verdict"].label, "title": e["label"]}
                for e in page["entries"]
            ],
        }
        for page in pages.values()
    ]
    return sorted(
        rows,
        key=lambda r: (r["page"]["doc"].company, r["page"]["doc"].doc_period, r["page"]["q"].id),
    )


def failure_tables(
    runs: list[Run], pages: dict[str, dict[str, Any]], gold: dict[str, Gold]
) -> list[dict[str, Any]]:
    """The typed failures of each published run, as `footnote failures` types them."""
    tables = []
    for r in runs:
        types = classify_run(r.predictions, r.verdicts, gold)
        rows = [
            {"page": pages[q], "type": t.replace("_", " "), "reason": REASONS[t]}
            for q, t in sorted(types.items())
            if q in pages
        ]
        tables.append({"label": r.page_label, "rows": rows})
    return tables
