import tomllib
from pathlib import Path
from typing import Any

import pytest

from footnote.site.build import build_site

ROOT = Path(__file__).resolve().parents[2]
# the check proves quotes are on the cited page and the final value comes from them; no more
FORBIDDEN = ("every number", "each number", "every claim", "each claim")


@pytest.fixture
def public_text(results: Path, tmp_path: Path, inject: dict[str, Any]) -> dict[str, str]:
    out = tmp_path / "out"
    build_site(results, out, **inject)
    return {
        "README.md": (ROOT / "README.md").read_text(),
        "pyproject description": tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
            "description"
        ],
        "index.html": (out / "index.html").read_text(),
        "about.html": (out / "about.html").read_text(),
    }


def test_no_overclaiming_phrases(public_text: dict[str, str]) -> None:
    for name, text in public_text.items():
        flat = " ".join(text.lower().split())
        for phrase in FORBIDDEN:
            assert phrase not in flat, f"{name} says {phrase!r}"


def test_grading_scope_is_stated_exactly(public_text: dict[str, str]) -> None:
    about = " ".join(public_text["about.html"].split())
    assert "reference answer is a single number" in about
    assert "structured final value" in about
    assert "Numeric questions are matched" not in about


def test_paper_comparison_names_what_differs(public_text: dict[str, str]) -> None:
    flat = " ".join(public_text["index.html"].split())
    assert "question subset" not in flat
    assert "human grading" in flat and "retrieval setups" in flat


def test_readme_boundary_claim_is_exact_and_results_table_has_paired_column() -> None:
    readme = (ROOT / "README.md").read_text()
    assert "statically checks imports" in readme
    assert "Agent minus this system" in readme
