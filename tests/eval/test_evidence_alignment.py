import json
from pathlib import Path
from typing import Any

import pytest

from footnote.corpus.normalize import normalize_text, word_coverage
from footnote.corpus.parse import PARSER_VERSION, load_pages
from footnote.eval.gold import load_gold

RESULT_PATH = Path(__file__).resolve().parents[2] / "results" / "corpus_quality.json"
COVERAGE_THRESHOLD = 0.90
MIN_RATE = 0.90

# One-line cause per low-coverage item, from inspecting it: (question id, doc, page) -> cause.
# Inspected: the gold text has table row labels run together without spaces ("costofgoodssold"),
# which our text layer spaces normally; neighbouring pages score at most 0.23, so the page is right.
RUN_TOGETHER = "gold table labels are run together without spaces; right page (neighbours <= 0.23)"
CAUSES: dict[tuple[str, str, int], str] = {
    ("financebench_id_02608", "BESTBUY_2017_10K", 55): RUN_TOGETHER,
    ("financebench_id_02981", "CORNING_2021_10K", 64): RUN_TOGETHER,
    ("financebench_id_03838", "BLOCK_2020_10K", 85): RUN_TOGETHER,
    ("financebench_id_03849", "MGMRESORTS_2020_10K", 66): RUN_TOGETHER,
    ("financebench_id_04171", "MGMRESORTS_2018_10K", 56): RUN_TOGETHER,
    ("financebench_id_04417", "BESTBUY_2019_10K", 51): RUN_TOGETHER,
    ("financebench_id_04660", "BLOCK_2016_10K", 67): RUN_TOGETHER,
    ("financebench_id_04784", "WALMART_2019_10K", 47): RUN_TOGETHER,
    ("financebench_id_05915", "CVSHEALTH_2018_10K", 301): RUN_TOGETHER,
    ("financebench_id_05915", "CVSHEALTH_2018_10K", 303): RUN_TOGETHER,
    ("financebench_id_07661", "BLOCK_2020_10K", 89): RUN_TOGETHER,
    ("financebench_id_10130", "CORNING_2020_10K", 69): RUN_TOGETHER,
    ("financebench_id_10130", "CORNING_2020_10K", 71): RUN_TOGETHER,
}


@pytest.mark.corpus
def test_evidence_alignment() -> None:
    normalized: dict[str, list[str]] = {}
    texts: dict[str, list[str]] = {}
    total = exact = 0
    low: list[dict[str, Any]] = []
    for qid, gold in sorted(load_gold().items()):
        for ev in gold.evidence:
            total += 1
            if ev.doc_name not in texts:
                texts[ev.doc_name] = [p.text for p in load_pages(ev.doc_name)]
                normalized[ev.doc_name] = [normalize_text(t) for t in texts[ev.doc_name]]
            in_range = 0 <= ev.page_index < len(texts[ev.doc_name])
            if in_range and normalize_text(ev.text) in normalized[ev.doc_name][ev.page_index]:
                exact += 1
            cov = word_coverage(ev.text, texts[ev.doc_name][ev.page_index]) if in_range else 0.0
            if cov < COVERAGE_THRESHOLD:
                low.append(
                    {
                        "question_id": qid,
                        "doc_name": ev.doc_name,
                        "page_index": ev.page_index,
                        "coverage": round(cov, 3),
                        "cause": CAUSES.get((qid, ev.doc_name, ev.page_index), "uninspected"),
                    }
                )
    aligned = total - len(low)
    rate = aligned / total
    RESULT_PATH.parent.mkdir(exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(
            {
                "parser_version": PARSER_VERSION,
                "evidence_items": total,
                "word_coverage_threshold": COVERAGE_THRESHOLD,
                "word_coverage_rate": round(rate, 4),
                "aligned_by_word_coverage": aligned,
                "exact_substring_rate": round(exact / total, 4),
                "exact_substring_matches": exact,
                "below_threshold": low,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    assert rate >= MIN_RATE, f"word-coverage alignment {rate:.1%} below {MIN_RATE:.0%}"
