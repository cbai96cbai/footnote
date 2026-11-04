import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from footnote.corpus import fetch
from footnote.retrieval import index as index_mod
from footnote.retrieval.chunk import Passage
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


def _meta(doc: str = "SAMPLE") -> Path:
    from footnote.paths import cache_dir

    return cache_dir() / "embeddings" / f"{doc}.json"


def _make_legacy(ids: list[str] | None = None) -> None:
    meta = _meta()
    info = json.loads(meta.read_text())
    old_key = index_mod._legacy_key(
        json.loads(Path(fetch.MANIFEST_PATH).read_text())["SAMPLE"], 350, 50, "fake", "0"
    )
    meta.write_text(json.dumps({"key": old_key, "ids": ids or info["ids"]}))


def test_chunker_version_change_rebuilds(corpus: None, monkeypatch: pytest.MonkeyPatch) -> None:
    emb = FakeEmbedder()
    DocIndex.for_doc("SAMPLE", emb)
    monkeypatch.setattr(index_mod, "CHUNKER_VERSION", "2")
    DocIndex.for_doc("SAMPLE", emb)
    assert emb.calls == 2
    DocIndex.for_doc("SAMPLE", emb)
    assert emb.calls == 2


def test_passage_text_change_with_same_ids_rebuilds(
    corpus: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    emb = FakeEmbedder()
    DocIndex.for_doc("SAMPLE", emb)
    real = index_mod.chunk_pages

    def altered(*args: Any, **kwargs: Any) -> list[Passage]:
        return [replace(p, text=p.text + " extra") for p in real(*args, **kwargs)]

    monkeypatch.setattr(index_mod, "chunk_pages", altered)
    DocIndex.for_doc("SAMPLE", emb)
    assert emb.calls == 2


def test_legacy_cache_accepted_and_upgraded(corpus: None) -> None:
    emb = FakeEmbedder()
    DocIndex.for_doc("SAMPLE", emb)
    _make_legacy()
    DocIndex.for_doc("SAMPLE", emb)
    assert emb.calls == 1  # accepted, not rebuilt
    upgraded = json.loads(_meta().read_text())
    assert upgraded["chunker_version"] == "1" and upgraded["text_digest"]
    DocIndex.for_doc("SAMPLE", emb)
    assert emb.calls == 1


def test_legacy_cache_with_different_ids_rebuilds(corpus: None) -> None:
    emb = FakeEmbedder()
    DocIndex.for_doc("SAMPLE", emb)
    _make_legacy(ids=["other"])
    DocIndex.for_doc("SAMPLE", emb)
    assert emb.calls == 2
