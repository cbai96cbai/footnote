"""One filing's search index: BM25 and embedding cosine, fused by reciprocal rank."""

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi

from footnote.corpus.fetch import load_manifest
from footnote.corpus.parse import PARSER_VERSION, load_pages
from footnote.paths import cache_dir
from footnote.retrieval.chunk import (
    CHUNKER_VERSION,
    MAX_TOKENS,
    OVERLAP_TOKENS,
    Passage,
    chunk_pages,
)
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


def _digest(parts: list[str]) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def embedding_cache_key(
    pdf_digest: str,
    max_tokens: int,
    overlap_tokens: int,
    model_name: str,
    library_version: str,
    chunker_version: str | None = None,
) -> str:
    parts = [
        pdf_digest,
        PARSER_VERSION,
        str(max_tokens),
        str(overlap_tokens),
        model_name,
        library_version,
    ]
    if chunker_version is None:  # the key as written before the chunker version existed
        chunker_version = CHUNKER_VERSION
    return _digest([*parts, chunker_version])


def _legacy_key(*args: object) -> str:
    """The pre-chunker-version key; only used to find caches built before that field existed."""
    pdf_digest, max_tokens, overlap_tokens, model_name, library_version = args
    return _digest(
        [
            str(pdf_digest),
            PARSER_VERSION,
            str(max_tokens),
            str(overlap_tokens),
            str(model_name),
            str(library_version),
        ]
    )


def _text_digest(passages: list[Passage]) -> str:
    return _digest([p.text for p in passages])


def _cache_files(doc_name: str) -> tuple[Path, Path]:
    """One slot per filing; the key inside the meta says what the vectors were built from."""
    base = cache_dir() / "embeddings"
    return base / f"{doc_name}.npy", base / f"{doc_name}.json"


def _read(npy: Path, meta: Path) -> tuple[dict[str, Any], NDArray[np.float32]] | None:
    if not (npy.exists() and meta.exists()):
        return None
    info: dict[str, Any] = json.loads(meta.read_text(encoding="utf-8"))
    vectors: NDArray[np.float32] = np.load(npy)
    return info, vectors


def _store(
    npy: Path, meta: Path, key: str, passages: list[Passage], vectors: NDArray[np.float32]
) -> None:
    npy.parent.mkdir(parents=True, exist_ok=True)
    tmp = npy.with_name(npy.name + ".part")
    with tmp.open("wb") as f:
        np.save(f, vectors)
    tmp.replace(npy)
    info = {
        "key": key,
        "ids": [p.id for p in passages],
        "chunker_version": CHUNKER_VERSION,
        "text_digest": _text_digest(passages),
    }
    meta.write_text(json.dumps(info), encoding="utf-8")


def _load_cached(
    doc_name: str, key: str, legacy_key: str, passages: list[Passage]
) -> NDArray[np.float32] | None:
    ids = [p.id for p in passages]
    npy, meta = _cache_files(doc_name)
    hit = _read(npy, meta)
    if hit is None:
        return None
    info, vectors = hit
    if info.get("ids") != ids or vectors.shape[0] != len(ids):
        return None
    if info.get("key") == key:
        if info.get("chunker_version") == CHUNKER_VERSION and info.get(
            "text_digest"
        ) == _text_digest(passages):
            return vectors
        return None
    # written before chunker versions existed: counts as chunker version "1" when the passage
    # ids still match, and is upgraded in place with the text digest
    if (
        CHUNKER_VERSION == "1"
        and info.get("key") == legacy_key
        and "chunker_version" not in info
        and "text_digest" not in info
    ):
        _store(npy, meta, key, passages, vectors)
        return vectors
    return None


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
        settings = (
            load_manifest()[doc_name],
            max_tokens,
            overlap_tokens,
            emb.model_name,
            emb.library_version,
        )
        key = embedding_cache_key(*settings)
        vectors = _load_cached(doc_name, key, _legacy_key(*settings), passages)
        if vectors is None:
            vectors = (
                emb.embed_passages([p.text for p in passages])
                if passages
                else np.zeros((0, 1), dtype=np.float32)
            )
            npy, meta = _cache_files(doc_name)
            _store(npy, meta, key, passages, vectors)
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
