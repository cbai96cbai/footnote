import dataclasses
import json
from pathlib import Path

import pytest

from footnote.benchmark import AgentQuestion, load_questions

ROOT = Path(__file__).resolve().parents[1]


def test_agent_question_has_no_answer_fields() -> None:
    names = {f.name for f in dataclasses.fields(AgentQuestion)}
    assert names == {"id", "company", "doc_name", "question_type", "question"}


def test_benchmark_module_exposes_no_gold() -> None:
    import footnote.benchmark as b

    public = {n for n in dir(b) if not n.startswith("_")}
    assert not {"load_gold", "Gold", "Evidence", "read_rows"} & public


def test_benchmark_source_pins_revision() -> None:
    src = json.loads((ROOT / "data" / "benchmark_source.json").read_text())
    assert src["revision"] == "e04404e3a97f69f79c14d42f24981a1c9c3bcd18"
    assert src["file"] == "financebench_merged.jsonl"


@pytest.mark.corpus
def test_load_questions_150() -> None:
    qs = load_questions()
    assert len(qs) == 150
    assert len({q.id for q in qs}) == 150
    assert len({q.company for q in qs}) == 32
    assert len({q.doc_name for q in qs}) == 84
