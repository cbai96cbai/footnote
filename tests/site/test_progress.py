from pathlib import Path
from typing import Any

import pytest
from support import publish, write_run

from footnote.site.build import build_site
from footnote.site.runs import load_runs


def test_several_dev_runs_per_system_are_kept_in_order(tmp_path: Path) -> None:
    root = tmp_path / "results"
    write_run(root, "agent-dev-1", "agent", "dev", ["q1"], version="agent-1")
    write_run(root, "agent-dev-2", "agent", "dev", ["q1"], version="agent-2")
    write_run(root, "nodocs-dev-1", "no-docs", "dev", ["q1"], version="nodocs-1")
    publish(root, ["agent-dev-1", "nodocs-dev-1", "agent-dev-2"])
    assert [r.run_id for r in load_runs(root)] == ["agent-dev-1", "agent-dev-2", "nodocs-dev-1"]


def test_same_system_version_and_split_twice_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "results"
    write_run(root, "agent-dev-1", "agent", "dev", ["q1"], version="agent-1")
    write_run(root, "agent-dev-1b", "agent", "dev", ["q1"], version="agent-1")
    publish(root, ["agent-dev-1", "agent-dev-1b"])
    with pytest.raises(ValueError, match="agent-dev-1b"):
        load_runs(root)


def test_two_test_runs_for_one_system_are_refused(tmp_path: Path) -> None:
    root = tmp_path / "results"
    write_run(root, "agent-test-1", "agent", "test", ["q1"], version="agent-1")
    write_run(root, "agent-test-2", "agent", "test", ["q1"], version="agent-2")
    publish(root, ["agent-test-1", "agent-test-2"])
    with pytest.raises(ValueError, match="agent-test-2"):
        load_runs(root)


def test_index_shows_each_development_version(tmp_path: Path, inject: dict[str, Any]) -> None:
    root = tmp_path / "results"
    write_run(root, "agent-dev-1", "agent", "dev", ["q1", "q2"], version="agent-1")
    write_run(root, "agent-dev-2", "agent", "dev", ["q1", "q2"], version="agent-2")
    publish(root, ["agent-dev-1", "agent-dev-2"])
    out = tmp_path / "out"
    build_site(root, out, **inject)
    index = (out / "index.html").read_text()
    assert "Exploratory development runs" in index
    for v in ("agent-1", "agent-2"):
        assert f"Footnote agent {v}</text>" in index  # chart label
        assert f"<td>{v}</td>" in index  # table row


def _test_split(root: Path, agent: list[str], base: list[str]) -> None:
    write_run(root, "agent-test-1", "agent", "test", ["q1", "q2", "q3"], labels=agent)
    write_run(root, "nodocs-test-1", "no-docs", "test", ["q1", "q2", "q3"], labels=base)
    publish(root, ["agent-test-1", "nodocs-test-1"])


def test_test_table_has_paired_difference_with_interval(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    root = tmp_path / "results"
    _test_split(root, ["correct"] * 3, ["incorrect"] * 3)
    out = tmp_path / "out"
    build_site(root, out, **inject)
    index = (out / "index.html").read_text()
    assert "Agent minus this system (paired 95% interval)" in index
    assert "+100.0 pts (+100.0 to +100.0)" in index


def test_paired_difference_uses_only_questions_graded_in_both(
    tmp_path: Path, inject: dict[str, Any]
) -> None:
    root = tmp_path / "results"
    _test_split(root, ["correct", "correct", "correct"], ["incorrect", "correct", "incorrect"])
    out = tmp_path / "out"
    build_site(root, out, **inject)
    index = (out / "index.html").read_text()
    assert "+66.7 pts" in index  # (3 - 1) / 3


def test_no_agent_test_run_means_no_paired_column(tmp_path: Path, inject: dict[str, Any]) -> None:
    root = tmp_path / "results"
    write_run(root, "nodocs-test-1", "no-docs", "test", ["q1", "q2", "q3"])
    publish(root, ["nodocs-test-1"])
    out = tmp_path / "out"
    build_site(root, out, **inject)
    assert "paired" not in (out / "index.html").read_text()
