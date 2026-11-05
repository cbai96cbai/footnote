import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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
def offline(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Tests run offline unless marked live or corpus: any network connection fails the test."""
    if request.node.get_closest_marker("live") or request.node.get_closest_marker("corpus"):
        yield
        return
    attempts: list[object] = []
    real_connect = socket.socket.connect

    def connect(self: socket.socket, address: Any) -> None:
        if self.family == socket.AF_UNIX:
            return real_connect(self, address)
        attempts.append(address)
        raise OSError(f"network access in a test not marked live or corpus: {address}")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect)
    yield
    assert not attempts, f"the test tried to reach the network: {attempts}"


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
