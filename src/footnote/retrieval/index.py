"""One filing's search index: BM25 and embedding cosine, fused by reciprocal rank."""

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi

from footnote.corpus.fetch import load_manifest
from footnote.corpus.parse import PARSER_VERSION, load_pages
from footnote.paths import cache_dir
from footnote.retrieval.chunk import MAX_TOKENS, OVERLAP_TOKENS, Passage, chunk_pages
from footnote.retrieval.embed import Embedder, FastEmbedder

RRF_K = 60
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class ScoredPassage:
    passage: Passage
    score: float  # the fused rank score


def rrf[T: Hashable](rankings: Sequence[Sequence[T]], k: int = RRF_K) -> list[tuple[T, float]]:
    """Reciprocal rank fusion: each ranking gives an item 1/(k + rank), rank starting at 1."""
    scores: dict[T, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def embedding_cache_key(
    pdf_digest: str, max_tokens: int, overlap_tokens: int, model_name: str, library_version: str
) -> str:
    parts = [
        pdf_digest,
        PARSER_VERSION,
        str(max_tokens),
        str(overlap_tokens),
        model_name,
        library_version,
    ]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def _cache_files(doc_name: str, key: str) -> tuple[Path, Path]:
    base = cache_dir() / "embeddings" / f"{doc_name}.{key[:16]}"
    return base.with_suffix(".npy"), base.with_suffix(".json")


def _load_cached(npy: Path, meta: Path, key: str, ids: list[str]) -> NDArray[np.float32] | None:
    if not (npy.exists() and meta.exists()):
        return None
    info = json.loads(meta.read_text(encoding="utf-8"))
    if info.get("key") != key or info.get("ids") != ids:
        return None
    vectors: NDArray[np.float32] = np.load(npy)
    return vectors if vectors.shape[0] == len(ids) else None


def _store(npy: Path, meta: Path, key: str, ids: list[str], vectors: NDArray[np.float32]) -> None:
    npy.parent.mkdir(parents=True, exist_ok=True)
    tmp = npy.with_name(npy.name + ".part")
    with tmp.open("wb") as f:
        np.save(f, vectors)
    tmp.replace(npy)
    meta.write_text(json.dumps({"key": key, "ids": ids}), encoding="utf-8")


class DocIndex:
    def __init__(
        self, passages: list[Passage], vectors: NDArray[np.float32], embedder: Embedder
    ) -> None:
        self.passages = passages
        self._vectors = vectors
        self._embedder = embedder
        self._bm25 = BM25Okapi([_tokens(p.text) or [""] for p in passages]) if passages else None

    @classmethod
    def for_doc(
        cls,
        doc_name: str,
        embedder: Embedder | None = None,
        max_tokens: int = MAX_TOKENS,
        overlap_tokens: int = OVERLAP_TOKENS,
    ) -> "DocIndex":
        emb = embedder or FastEmbedder()
        passages = chunk_pages(load_pages(doc_name), max_tokens, overlap_tokens)
        ids = [p.id for p in passages]
        key = embedding_cache_key(
            load_manifest()[doc_name],
            max_tokens,
            overlap_tokens,
            emb.model_name,
            emb.library_version,
        )
        npy, meta = _cache_files(doc_name, key)
        vectors = _load_cached(npy, meta, key, ids)
        if vectors is None:
            vectors = (
                emb.embed_passages([p.text for p in passages])
                if passages
                else np.zeros((0, 1), dtype=np.float32)
            )
            _store(npy, meta, key, ids, vectors)
        return cls(passages, vectors, emb)

    def search(self, query: str, k: int = 8) -> list[ScoredPassage]:
        if not self.passages or self._bm25 is None:
            return []
        q = _tokens(query)
        bm25 = self._bm25.get_scores(q) if q else np.zeros(len(self.passages))
        by_bm25 = [int(i) for i in np.argsort(-bm25, kind="stable") if bm25[i] > 0]
        cosine = self._vectors @ self._embedder.embed_query(query)
        by_cosine = [int(i) for i in np.argsort(-cosine, kind="stable")]
        fused = rrf([by_bm25, by_cosine], RRF_K)[:k]
        return [ScoredPassage(self.passages[i], score) for i, score in fused]
