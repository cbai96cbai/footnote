import hashlib
import json
from pathlib import Path

import pytest

from footnote.corpus import fetch, parse


@pytest.fixture
def corpus(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_pdf: Path) -> Path:
    digest = hashlib.sha256(sample_pdf.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"SAMPLE": digest}))
    monkeypatch.setattr(fetch, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(fetch, "_download_bytes", lambda url: sample_pdf.read_bytes())
    return sample_pdf


def test_parse_page_index_zero_based(corpus: Path) -> None:
    pages = parse.load_pages("SAMPLE")
    assert [p.page_index for p in pages] == [0, 1, 2]
    assert "alpha" in pages[0].text and "beta" in pages[1].text
    assert all(p.doc_name == "SAMPLE" for p in pages)


def test_page_cache_reused(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = parse.load_pages("SAMPLE")
    monkeypatch.setattr(parse, "_extract", lambda path: pytest.fail("re-parsed"))
    assert parse.load_pages("SAMPLE") == first


def test_page_cache_invalidated_by_parser_version(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parse.load_pages("SAMPLE")
    monkeypatch.setattr(parse, "PARSER_VERSION", parse.PARSER_VERSION + "-next")
    calls: list[Path] = []
    real = parse._extract

    def spy(path: Path) -> list[str]:
        calls.append(path)
        return real(path)

    monkeypatch.setattr(parse, "_extract", spy)
    parse.load_pages("SAMPLE")
    assert len(calls) == 1
