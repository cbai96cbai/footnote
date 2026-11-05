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
        "question.html": (out / "q" / "q1.html").read_text(),
        "failures.html": (out / "failures.html").read_text(),
        "base.html": (ROOT / "src/footnote/site/templates/base.html").read_text(),
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


def test_about_states_how_checks_and_failure_types_differ_by_system(
    public_text: dict[str, str],
) -> None:
    about = " ".join(public_text["about.html"].split())
    assert "The no-documents system is not checked" in about
    assert "checked without a calculator" in about and "cannot be traced" in about
    assert "applies only to the agent" in about
    assert "Failure types therefore differ by system" in about


OPERANDS = (
    "A calculation operand, whether a quoted number or an earlier result, may be scaled up by a "
    "power of 1,000 whatever the final unit, because tables state their scale in a header"
)
NO_NUMBER = (
    "An answer may give no final number (for example when the filing says there were none); "
    "then only its quotes are checked."
)


def test_scale_and_no_final_number_sentences_are_stated_in_each_scope_statement(
    public_text: dict[str, str],
) -> None:
    for name in ("README.md", "about.html"):
        flat = " ".join(public_text[name].split())
        assert NO_NUMBER in flat, name
    readme = " ".join(public_text["README.md"].split())
    assert readme.count(OPERANDS) == 2 and readme.count(NO_NUMBER) == 2
    assert (
        readme.count("percents and ratios take no scale")
        + readme.count("percent and ratio take no scale")
        == 2
    )
    about = " ".join(public_text["about.html"].split())
    assert OPERANDS in about and "percent and ratio take no scale" in about
    assert "The final value may be scaled by a power of 1,000" in about
    assert "Numbers in the explanation text are not checked" in about
