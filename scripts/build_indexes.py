"""Build (and cache) the search index for every filing; prints time per document and in total."""

import time

from footnote.corpus.fetch import load_manifest
from footnote.retrieval.embed import FastEmbedder
from footnote.retrieval.index import DocIndex


def main() -> None:
    embedder = FastEmbedder()
    start = time.perf_counter()
    passages = 0
    for doc_name in sorted(load_manifest()):
        t = time.perf_counter()
        index = DocIndex.for_doc(doc_name, embedder)
        passages += len(index.passages)
        print(
            f"{doc_name}: {len(index.passages)} passages, {time.perf_counter() - t:.1f}s",
            flush=True,
        )
    print(
        f"total: {len(load_manifest())} documents, {passages} passages, "
        f"{time.perf_counter() - start:.1f}s"
    )


if __name__ == "__main__":
    main()
