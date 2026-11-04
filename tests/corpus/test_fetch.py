import hashlib
import json
from pathlib import Path

import pytest

from footnote.corpus import fetch


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pdf: Path, digest: str) -> list[str]:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"SAMPLE": digest}))
    monkeypatch.setattr(fetch, "MANIFEST_PATH", manifest)
    calls: list[str] = []

    def fake(url: str) -> bytes:
        calls.append(url)
        return pdf.read_bytes()

    monkeypatch.setattr(fetch, "_download_bytes", fake)
    return calls


def test_fetch_rejects_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_pdf: Path
) -> None:
    _setup(monkeypatch, tmp_path, sample_pdf, "0" * 64)
    with pytest.raises(fetch.CorpusIntegrityError):
        fetch.fetch_pdf("SAMPLE")
    assert not (fetch.pdf_dir() / "SAMPLE.pdf").exists()


def test_fetch_downloads_once_and_verifies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_pdf: Path
) -> None:
    calls = _setup(monkeypatch, tmp_path, sample_pdf, _digest(sample_pdf))
    first = fetch.fetch_pdf("SAMPLE")
    assert fetch.fetch_pdf("SAMPLE") == first
    assert len(calls) == 1
    assert calls[0].endswith("/pdfs/SAMPLE.pdf")


def test_fetch_rejects_tampered_cached_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_pdf: Path
) -> None:
    _setup(monkeypatch, tmp_path, sample_pdf, _digest(sample_pdf))
    path = fetch.fetch_pdf("SAMPLE")
    path.write_bytes(b"tampered")
    with pytest.raises(fetch.CorpusIntegrityError):
        fetch.fetch_pdf("SAMPLE")


def test_fetch_unknown_document(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_pdf: Path
) -> None:
    _setup(monkeypatch, tmp_path, sample_pdf, _digest(sample_pdf))
    with pytest.raises(fetch.CorpusIntegrityError):
        fetch.fetch_pdf("OTHER")


def test_download_retries_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def flaky(url: str) -> bytes:
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("boom")
        return b"ok"

    sleeps: list[float] = []
    monkeypatch.setattr(fetch, "_get", flaky)
    monkeypatch.setattr(fetch.time, "sleep", sleeps.append)
    assert fetch._download_bytes("u") == b"ok"
    assert sleeps == [1.0, 2.0]
