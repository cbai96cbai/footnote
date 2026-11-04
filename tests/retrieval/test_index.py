import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from footnote.corpus import fetch
from footnote.retrieval import index as index_mod
from footnote.retrieval.index import DocIndex, embedding_cache_key, rrf


class FakeEmbedder:
    """Bag-of-letters vectors: deterministic, no model, no network."""

    model_name = "fake"
    library_version = "0"

    def __init__(self) -> None:
        self.calls = 0

    def _vec(self, text: str) -> NDArray[np.float32]:
        v = np.zeros(26, dtype=np.float32)
        for ch in text.lower():
            if "a" <= ch <= "z":
                v[ord(ch) - 97] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts: list[str]) -> NDArray[np.float32]:
        self.calls += 1
        return np.array([self._vec(t) for t in texts], dtype=np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self._vec(text)


@pytest.fixture
def corpus(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_pdf: Path) -> None:
    digest = hashlib.sha256(sample_pdf.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"SAMPLE": digest}))
    monkeypatch.setattr(fetch, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(fetch, "_download_bytes", lambda url: sample_pdf.read_bytes())


def test_rrf_scores_exact() -> None:
    fused = dict(rrf([["a", "b"], ["c", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62)
    assert [item for item, _ in rrf([["a", "b"], ["c", "a"]], k=60)][0] == "a"


def test_keyword_passage_ranks_first(corpus: None) -> None:
    idx = DocIndex.for_doc("SAMPLE", FakeEmbedder())
    hits = idx.search("beta", k=3)
    assert hits[0].passage.page_index == 1
    assert "beta" in hits[0].passage.text
    assert hits[0].score > hits[-1].score or len(hits) == 1


def test_embedding_cache_reused_and_rebuilt_on_key_change(corpus: None) -> None:
    emb = FakeEmbedder()
    DocIndex.for_doc("SAMPLE", emb)
    DocIndex.for_doc("SAMPLE", emb)
    assert emb.calls == 1
    DocIndex.for_doc("SAMPLE", emb, max_tokens=200)
    assert emb.calls == 2
    other = FakeEmbedder()
    other.library_version = "1"
    DocIndex.for_doc("SAMPLE", other)
    assert other.calls == 1


def test_embedding_cache_key_changes_with_settings() -> None:
    base = embedding_cache_key("d", 350, 50, "m", "1")
    assert base == embedding_cache_key("d", 350, 50, "m", "1")
    variants = [
        embedding_cache_key("e", 350, 50, "m", "1"),
        embedding_cache_key("d", 300, 50, "m", "1"),
        embedding_cache_key("d", 350, 40, "m", "1"),
        embedding_cache_key("d", 350, 50, "n", "1"),
        embedding_cache_key("d", 350, 50, "m", "2"),
    ]
    assert len({base, *variants}) == 6
    original = index_mod.PARSER_VERSION
    try:
        index_mod.PARSER_VERSION = "other"
        assert embedding_cache_key("d", 350, 50, "m", "1") != base
    finally:
        index_mod.PARSER_VERSION = original
