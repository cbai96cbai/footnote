"""Download filings PDFs from the pinned FinanceBench commit and verify them against the manifest.

python -m footnote.corpus.fetch --build-manifest
"""

import argparse
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from footnote.benchmark import doc_info
from footnote.paths import cache_dir

PDF_URL = (
    "https://raw.githubusercontent.com/patronus-ai/financebench/"
    "cc39aeb4afdf33909ee1412188bf89035950c2eb/pdfs/{doc_name}.pdf"
)
MANIFEST_PATH = Path(__file__).resolve().parents[3] / "data" / "corpus_manifest.json"
ATTEMPTS = 4
WORKERS = 4


class CorpusIntegrityError(Exception):
    pass


def pdf_dir() -> Path:
    return cache_dir() / "pdfs"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> dict[str, str]:
    data: dict[str, str] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return data


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:
        data: bytes = resp.read()
    return data


def _download_bytes(url: str) -> bytes:
    for attempt in range(ATTEMPTS):
        try:
            return _get(url)
        except OSError:
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(2.0**attempt)
    raise AssertionError("unreachable")


def _store(doc_name: str) -> Path:
    """Download into the cache without verifying (used to build the manifest)."""
    path = pdf_dir() / f"{doc_name}.pdf"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.part")
        tmp.write_bytes(_download_bytes(PDF_URL.format(doc_name=doc_name)))
        tmp.replace(path)
    return path


def fetch_pdf(doc_name: str) -> Path:
    """Path of the cached PDF, downloading it if needed; its SHA-256 must match the manifest."""
    expected = load_manifest().get(doc_name)
    if expected is None:
        raise CorpusIntegrityError(f"{doc_name} is not in the corpus manifest")
    path = _store(doc_name)
    actual = sha256_file(path)
    if actual != expected:
        path.unlink()
        raise CorpusIntegrityError(
            f"{doc_name}: sha256 {actual} does not match manifest {expected}; removed from cache"
        )
    return path


def build_manifest() -> dict[str, str]:
    names = sorted(doc_info())
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        paths = list(ex.map(_store, names))
    manifest = {n: sha256_file(p) for n, p in zip(names, paths, strict=True)}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-manifest", action="store_true", required=True)
    ap.parse_args()
    print(f"manifest written: {len(build_manifest())} documents -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
