"""Company-grouped dev/test split, frozen in data/split.json.

Rule: companies sorted by sha256("footnote-split-v1:" + company) go to test until test holds at
least 50 questions; the rest is dev. Gate: the first 10 dev questions of each question type by
sha256(id).
"""

import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, NamedTuple

from footnote._financebench import FILENAME, REPO, REVISION

TEST_MIN = 50
GATE_PER_TYPE = 10
SPLIT_PATH = Path(__file__).resolve().parents[3] / "data" / "split.json"

SplitName = Literal["dev", "test", "gate"]


class Item(NamedTuple):
    id: str
    company: str
    question_type: str


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def compute_split(items: Sequence[Item]) -> dict[str, list[str]]:
    by_company: dict[str, list[str]] = defaultdict(list)
    for it in items:
        by_company[it.company].append(it.id)
    test_companies: list[str] = []
    test_ids: list[str] = []
    for company in sorted(by_company, key=lambda c: _h("footnote-split-v1:" + c)):
        if len(test_ids) >= TEST_MIN:
            break
        test_companies.append(company)
        test_ids.extend(by_company[company])
    test_set = set(test_ids)
    dev = [it for it in items if it.id not in test_set]
    gate: list[str] = []
    for qtype in sorted({it.question_type for it in dev}):
        of_type = sorted((it.id for it in dev if it.question_type == qtype), key=_h)
        gate.extend(of_type[:GATE_PER_TYPE])
    return {
        "dev": sorted(it.id for it in dev),
        "test": sorted(test_ids),
        "gate": sorted(gate),
        "test_companies": sorted(test_companies),
        "dev_companies": sorted({it.company for it in dev}),
    }


def load_split() -> dict[SplitName, list[str]]:
    data = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    return {"dev": data["dev"], "test": data["test"], "gate": data["gate"]}


def _write() -> None:
    from footnote.benchmark import load_questions

    items = [Item(q.id, q.company, q.question_type) for q in load_questions()]
    split = compute_split(items)
    if not 45 <= len(split["test"]) <= 60:
        sys.exit(f"test split has {len(split['test'])} questions, outside 45-60; not written")
    SPLIT_PATH.write_text(json.dumps(split, indent=1) + "\n", encoding="utf-8")
    source = {
        "dataset": REPO,
        "file": FILENAME,
        "revision": REVISION,
        "license": "CC BY-NC 4.0",
        "document_metadata": "gics_sector, doc_type, doc_period, doc_link are columns of this file",
        "pdfs": "https://raw.githubusercontent.com/patronus-ai/financebench/"
        "cc39aeb4afdf33909ee1412188bf89035950c2eb/pdfs/{doc_name}.pdf",
    }
    SPLIT_PATH.with_name("benchmark_source.json").write_text(
        json.dumps(source, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--write"]:
        _write()
    else:
        sys.exit("usage: python -m footnote.eval.split --write")
