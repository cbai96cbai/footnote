import json

import pytest
from test_baselines import GOOD, FakeClient, Q, res

from footnote.baselines import plain_retrieval
from footnote.budget.limits import LIMITS
from footnote.retrieval.chunk import Passage
from footnote.retrieval.index import ScoredPassage


class FakeIndex:
    def __init__(self) -> None:
        self.asked: list[tuple[str, int]] = []
        pages = [5, 5, 9, 2, 9, 7, 1, 3, 4, 6]  # ten ranked hits, repeated pages
        self.hits = [
            ScoredPassage(Passage(f"D:p{pg}:{i}", "D", pg, f"text {i}"), 1.0 / (i + 1))
            for i, pg in enumerate(pages)
        ]

    def search(self, query: str, k: int = 8) -> list[ScoredPassage]:
        self.asked.append((query, k))
        return self.hits[:k]


def test_plain_retrieval_uses_top_8(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = FakeIndex()
    monkeypatch.setattr(plain_retrieval, "open_index", lambda doc: idx)
    client = FakeClient([res()])
    pred = plain_retrieval.run(Q, client)  # type: ignore[arg-type]
    assert idx.asked == [(Q.question, 8)]
    assert len(client.calls) == 1
    user = client.calls[0][1]["content"]
    assert user.count("[page ") == 8 and "text 7" in user and "text 8" not in user
    assert user.index("text 0") < user.index("text 1") < user.index("text 7")
    assert pred.pages_read == [5, 9, 2, 7, 1, 3]
    assert pred.status == "answered" and pred.verify_ok is None and pred.verify_problems == []
    assert pred.system == "plain-retrieval"


def test_plain_retrieval_retry_and_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plain_retrieval, "open_index", lambda doc: FakeIndex())
    c = FakeClient([res(None, "length"), res(None, "stop", "bad"), res()])
    p = plain_retrieval.run(Q, c)  # type: ignore[arg-type]
    assert len(c.calls) == 2 and p.status == "error" and p.usage.input_tokens == 200
    assert plain_retrieval.SPEC.config["limits"]["max_input_tokens"] == (
        LIMITS["plain_retrieval"].max_input_tokens
    )
    assert plain_retrieval.SPEC.config["retrieval"]["k"] == 8
    assert plain_retrieval.SPEC.max_cost_per_question > 0


def test_plain_retrieval_prompt_has_no_gold_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plain_retrieval, "open_index", lambda doc: FakeIndex())
    client = FakeClient([res(GOOD)])
    plain_retrieval.run(Q, client)  # type: ignore[arg-type]
    text = json.dumps(client.calls[0]).lower()
    for banned in ("justification", "evidence", "gold", "financebench_id"):
        assert banned not in text


def test_passages_with_extractor_hyphen_marks_are_shown_with_a_hyphen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = FakeIndex()
    idx.hits[0] = ScoredPassage(
        Passage("D:p5:0", "D", 5, "forward\x02looking\u00adstatements"), 1.0
    )
    monkeypatch.setattr(plain_retrieval, "open_index", lambda doc: idx)
    client = FakeClient([res()])
    plain_retrieval.run(Q, client)  # type: ignore[arg-type]
    user = client.calls[0][1]["content"]
    assert "forward-looking-statements" in user and "\x02" not in user and "\u00ad" not in user
