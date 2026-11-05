import socket
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fpdf import FPDF

from footnote.llm.tokens import count_text_tokens


def make_pdf(path: Path, pages: list[str]) -> Path:
    pdf = FPDF()
    for text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, text)
    pdf.output(str(path))
    return path


@pytest.fixture(scope="session", autouse=True)
def tokenizer() -> None:
    """Setup, before any test runs offline: tiktoken fetches its data file once, then caches it."""
    count_text_tokens("")


@pytest.fixture(autouse=True)
def offline(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Tests run offline unless marked live or corpus: any network connection fails the test."""
    if request.node.get_closest_marker("live") or request.node.get_closest_marker("corpus"):
        yield
        return
    attempts: list[object] = []

    def refuse(real: Callable[[socket.socket, Any], Any]) -> Callable[[socket.socket, Any], Any]:
        def call(self: socket.socket, address: Any) -> Any:
            if self.family == socket.AF_UNIX:
                return real(self, address)
            attempts.append(address)
            raise OSError(f"network access in a test not marked live or corpus: {address}")

        return call

    monkeypatch.setattr(socket.socket, "connect", refuse(socket.socket.connect))
    monkeypatch.setattr(socket.socket, "connect_ex", refuse(socket.socket.connect_ex))
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
