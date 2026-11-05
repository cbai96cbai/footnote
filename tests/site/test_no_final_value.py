from decimal import Decimal
from pathlib import Path
from typing import Any

from support import publish, write_run

from footnote.answers import FinalValue
from footnote.site.build import build_site

NOTE = "No final number was given; only the quotes were checked."
QUOTE = "Net sales were 1,234 million dollars"


def _html(tmp_path: Path, inject: dict[str, Any], page: str = "q/q1.html", **kw: Any) -> str:
    root = tmp_path / "results"
    write_run(root, "agent-dev-1", "agent", "dev", ["q1"], quote=QUOTE, **kw)
    publish(root, ["agent-dev-1"])
    out = tmp_path / "out"
    build_site(root, out, **inject)
    return (out / page).read_text()


def test_an_answer_without_a_final_value_says_only_the_quotes_were_checked(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    assert NOTE in _html(tmp_path, inject)


def test_the_note_is_absent_when_a_final_value_was_given(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    html = _html(tmp_path, inject, final_value=FinalValue(value=Decimal(1234), unit="usd"))
    assert NOTE not in html


def test_the_note_is_absent_for_a_declined_prediction(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    assert NOTE not in _html(tmp_path, inject, status="declined")


def test_the_results_page_states_the_count_under_the_table(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    html = _html(tmp_path, inject, "index.html", summary_extra={"answered_without_final_value": 7})
    assert "Answered without a final number: Footnote agent v1 7." in html
    older = _html(tmp_path / "again", inject, "index.html")  # a summary written before the count
    assert "Answered without a final number: Footnote agent v1 1." in older  # from predictions
