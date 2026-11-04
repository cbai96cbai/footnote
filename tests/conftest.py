from pathlib import Path

import pytest
from fpdf import FPDF


def make_pdf(path: Path, pages: list[str]) -> Path:
    pdf = FPDF()
    for text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, text)
    pdf.output(str(path))
    return path


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "cache"
    monkeypatch.setenv("FOOTNOTE_CACHE", str(cache))
    return cache


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    return make_pdf(
        tmp_path / "SAMPLE.pdf", ["First page alpha", "Second page beta", "Third gamma"]
    )
