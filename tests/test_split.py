import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from footnote.benchmark import load_questions
from footnote.eval.split import Item, compute_split, load_split

ROOT = Path(__file__).resolve().parents[1]
RAW = json.loads((ROOT / "data" / "split.json").read_text())


def _toy() -> list[Item]:
    types = ["metrics-generated", "domain-relevant", "novel-generated"]
    return [Item(f"q{c}_{i}", f"co{c}", types[i % 3]) for c in range(20) for i in range(6)]


def test_split_grouped_by_company() -> None:
    s = compute_split(_toy())
    comp = lambda ids: {i.split("_")[0][1:] for i in ids}  # noqa: E731
    assert not comp(s["dev"]) & comp(s["test"])
    assert set(s["dev"]) | set(s["test"]) == {i.id for i in _toy()}
    assert not set(s["dev"]) & set(s["test"])
    # the committed file: no company on both sides
    assert not set(RAW["dev_companies"]) & set(RAW["test_companies"])


def test_split_rule_uses_company_hash_order() -> None:
    items = _toy()
    s = compute_split(items)
    order = sorted(
        {i.company for i in items},
        key=lambda c: hashlib.sha256(("footnote-split-v1:" + c).encode()).hexdigest(),
    )
    n = len(s["test_companies"])
    assert s["test_companies"] == sorted(order[:n])
    assert len(s["test"]) >= 50
    assert n == 9  # 9 companies * 6 = 54 >= 50, 8 * 6 = 48 < 50


def test_split_sizes() -> None:
    s = load_split()
    assert set(s) == {"dev", "test", "gate"}
    assert 45 <= len(s["test"]) <= 60
    assert len(s["dev"]) + len(s["test"]) == 150
    assert not set(s["dev"]) & set(s["test"])
    assert set(s["gate"]) <= set(s["dev"])
    assert len(s["gate"]) == 30


@pytest.mark.corpus
def test_split_file_matches_rule() -> None:
    qs = load_questions()
    s = compute_split([Item(q.id, q.company, q.question_type) for q in qs])
    assert s == RAW
    types = {q.id: q.question_type for q in qs}
    assert set(Counter(types[i] for i in s["gate"]).values()) == {10}
