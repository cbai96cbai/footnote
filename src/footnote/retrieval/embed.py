"""The dense embedder (fastembed, BGE small) behind a small interface so tests can swap it."""

from importlib.metadata import version
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

MODEL_NAME = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    model_name: str
    library_version: str

    def embed_passages(self, texts: list[str]) -> NDArray[np.float32]: ...

    def embed_query(self, text: str) -> NDArray[np.float32]: ...


def _unit(m: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return (m / np.where(norms == 0, 1, norms)).astype(np.float32)


class FastEmbedder:
    model_name = MODEL_NAME
    library_version = version("fastembed")

    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(MODEL_NAME)

    def embed_passages(self, texts: list[str]) -> NDArray[np.float32]:
        rows = list(self._model.passage_embed(texts))
        return _unit(np.array(rows, dtype=np.float32).reshape(len(texts), -1))

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return _unit(np.array(next(iter(self._model.query_embed(text))), dtype=np.float32))
