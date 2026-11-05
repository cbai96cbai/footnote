import re
from pathlib import Path
from typing import Any

from support import publish, write_run

from footnote.site.build import build_site

HEADERS = [
    "System",
    "Accuracy (95% interval)",
    "Answered",
    "Passed code check",
    "Cost per question",
    "Time per question",
]


def _index(root: Path, tmp_path: Path, inject: dict[str, Any]) -> str:
    out = tmp_path / "out"
    build_site(root, out, **inject)
    return (out / "index.html").read_text()


def _tables(html: str) -> list[str]:
    return re.findall(r"<table.*?</table>", html.split("Historical context")[0], re.S)


def _headers(table: str) -> list[str]:
    return [" ".join(h.split()) for h in re.findall(r"<th(?:\s[^>]*)?>(.*?)</th>", table, re.S)]


def test_development_table_has_six_columns_with_the_listed_headers(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    (table,) = _tables(_index(results, tmp_path, inject))
    assert _headers(table) == HEADERS
    for row in re.findall(r"<tr>(.*?)</tr>", table, re.S)[1:]:
        assert len(re.findall(r"<td", row)) == 6


def test_version_split_and_run_id_move_to_one_caption_line(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    html = _index(results, tmp_path, inject)
    (table,) = _tables(html)
    assert "agent-dev-1" not in table and "Split" not in table and "Version" not in table
    assert "Runs shown:" in html
    assert "Footnote agent v1: version v1, dev split, run agent-dev-1" in html


def test_test_split_has_a_main_table_and_a_separate_paired_table(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    root = tmp_path / "results"
    write_run(root, "agent-test-1", "agent", "test", ["q1", "q2", "q3"], labels=["correct"] * 3)
    write_run(
        root, "nodocs-test-1", "no-docs", "test", ["q1", "q2", "q3"], labels=["incorrect"] * 3
    )
    publish(root, ["agent-test-1", "nodocs-test-1"])
    main, paired = _tables(_index(root, tmp_path, inject))
    assert _headers(main) == HEADERS
    assert _headers(paired) == ["System", "Agent minus this system (paired 95% interval)"]
    assert "+100.0 pts (+100.0 to +100.0)" in paired and "+100.0" not in main


def test_cost_and_time_cells_are_shown(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    (table,) = _tables(_index(results, tmp_path, inject))
    assert "$0.0100" in table and "2.5 s" in table


def test_each_row_shows_its_own_pass_rate_or_not_available(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    root = tmp_path / "results"
    write_run(root, "agent-dev-1", "agent", "dev", ["q1"], summary_extra={"verified_rate": 0.5})
    write_run(root, "nodocs-dev-1", "no-docs", "dev", ["q1"])
    publish(root, ["agent-dev-1", "nodocs-dev-1"])
    (table,) = _tables(_index(root, tmp_path, inject))
    rows = re.findall(r"<tr>(.*?)</tr>", table, re.S)[1:]
    cells = [re.findall(r"<td[^>]*>(.*?)</td>", r, re.S) for r in rows]
    assert [c[3] for c in cells] == ["50.0%", "n/a"]


def test_the_code_check_note_says_what_is_counted_and_why_baselines_fail_it(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    flat = " ".join(_index(results, tmp_path, inject).split())
    assert "Passed code check is the share of answered questions" in flat
    assert "found on their cited pages" in flat and "final number was traced" in flat
    assert "baselines log no calculations" in flat and "cannot be traced" in flat
