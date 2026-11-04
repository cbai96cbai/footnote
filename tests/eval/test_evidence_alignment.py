import json
from pathlib import Path
from typing import Any

import pytest

from footnote.corpus.normalize import normalize_text
from footnote.corpus.parse import PARSER_VERSION, load_pages
from footnote.eval.gold import load_gold

RESULT_PATH = Path(__file__).resolve().parents[2] / "results" / "corpus_quality.json"
MIN_RATE = 0.90


@pytest.mark.corpus
def test_evidence_alignment() -> None:
    normalized: dict[str, list[str]] = {}
    total = 0
    failures: list[dict[str, Any]] = []
    for qid, gold in sorted(load_gold().items()):
        for ev in gold.evidence:
            total += 1
            pages = normalized.setdefault(
                ev.doc_name, [normalize_text(p.text) for p in load_pages(ev.doc_name)]
            )
            needle = normalize_text(ev.text)
            if 0 <= ev.page_index < len(pages) and needle in pages[ev.page_index]:
                continue
            near = [
                i
                for i in (ev.page_index - 1, ev.page_index + 1)
                if 0 <= i < len(pages) and needle in pages[i]
            ]
            anywhere = [i for i, p in enumerate(pages) if needle in p]
            failures.append(
                {
                    "question_id": qid,
                    "doc_name": ev.doc_name,
                    "page_index": ev.page_index,
                    "page_in_range": 0 <= ev.page_index < len(pages),
                    "found_on_neighbouring_page": near,
                    "found_anywhere_in_doc": anywhere,
                }
            )
    matched = total - len(failures)
    rate = matched / total
    RESULT_PATH.parent.mkdir(exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(
            {
                "parser_version": PARSER_VERSION,
                "rate": round(rate, 4),
                "evidence_items": total,
                "matched": matched,
                "failed": len(failures),
                "failed_found_on_neighbouring_page": sum(
                    1 for f in failures if f["found_on_neighbouring_page"]
                ),
                "failed_found_elsewhere_in_doc": sum(
                    1
                    for f in failures
                    if f["found_anywhere_in_doc"] and not f["found_on_neighbouring_page"]
                ),
                "failed_not_found_in_doc": sum(
                    1 for f in failures if not f["found_anywhere_in_doc"]
                ),
                "failures": failures,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    assert rate >= MIN_RATE, f"alignment {rate:.1%} below {MIN_RATE:.0%}; see {RESULT_PATH}"
