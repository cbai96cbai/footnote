"""Page text from the verified PDFs (pypdfium2), cached as JSONL keyed by PDF digest and parser."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pypdfium2 as pdfium

from footnote.corpus.fetch import fetch_pdf, sha256_file
from footnote.paths import cache_dir

PARSER_VERSION = "pdfium-1"


@dataclass(frozen=True)
class Page:
    doc_name: str
    page_index: int  # 0-based, like FinanceBench's evidence_page_num
    text: str


def _extract(path: Path) -> list[str]:
    pdf = pdfium.PdfDocument(str(path))
    try:
        out = []
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            out.append(textpage.get_text_bounded())
            textpage.close()
            page.close()
        return out
    finally:
        pdf.close()


def _cache_path(doc_name: str, digest: str) -> Path:
    return cache_dir() / "pages" / f"{doc_name}.{digest[:16]}.{PARSER_VERSION}.jsonl"


def load_pages(doc_name: str) -> list[Page]:
    path = fetch_pdf(doc_name)
    cache = _cache_path(doc_name, sha256_file(path))
    if cache.exists():
        with cache.open(encoding="utf-8") as f:
            return [Page(**json.loads(line)) for line in f if line.strip()]
    pages = [Page(doc_name, i, t) for i, t in enumerate(_extract(path))]
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".part")
    tmp.write_text("".join(json.dumps(asdict(p)) + "\n" for p in pages), encoding="utf-8")
    tmp.replace(cache)
    return pages
