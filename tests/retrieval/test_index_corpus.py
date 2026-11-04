import pytest

from footnote.corpus.fetch import load_manifest
from footnote.retrieval.index import DocIndex


@pytest.mark.corpus
def test_real_filing_keyword_query_finds_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOOTNOTE_CACHE")  # use the real corpus cache
    doc = sorted(load_manifest())[0]
    hits = DocIndex.for_doc(doc).search("total revenue", k=5)
    assert hits and any("revenue" in h.passage.text.lower() for h in hits)
