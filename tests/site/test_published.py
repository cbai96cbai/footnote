from pathlib import Path
from typing import Any

import pytest
from support import publish, write_run

from footnote.site.build import MARKER, build_site
from footnote.site.runs import load_runs


def test_later_unlisted_run_never_replaces_listed(tmp_path: Path) -> None:
    root = tmp_path / "results"
    write_run(root, "agent-test-1", "agent", "test", ["q1", "q2"])
    write_run(root, "agent-test-2", "agent", "test", ["q1"])
    publish(root, ["agent-test-1"])
    assert [r.run_id for r in load_runs(root)] == ["agent-test-1"]


def test_incomplete_or_missing_listed_run_refused(tmp_path: Path) -> None:
    root = tmp_path / "results"
    write_run(root, "agent-test-1", "agent", "test", ["q1"])
    summary = root / "agent-test-1" / "summary.json"
    summary.write_text(summary.read_text().replace('"complete": true', '"complete": false'))
    publish(root, ["agent-test-1"])
    with pytest.raises(ValueError, match="agent-test-1"):
        load_runs(root)
    publish(root, ["ghost-run"])
    with pytest.raises(ValueError, match="ghost-run"):
        load_runs(root)


def test_absent_or_empty_published_is_empty_state(tmp_path: Path, inject: dict[str, Any]) -> None:
    root = tmp_path / "results"
    write_run(root, "agent-test-1", "agent", "test", ["q1"])
    assert load_runs(root) == []
    publish(root, [])
    assert load_runs(root) == []
    out = tmp_path / "out"
    build_site(root, out, **inject)
    assert (out / "index.html").exists() and not list((out / "q").glob("*.html"))


def test_rebuild_drops_pages_of_removed_runs(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    out = tmp_path / "out"
    build_site(results, out, **inject)
    assert (out / "q" / "q3.html").exists()
    write_run(results, "agent-dev-2", "agent", "dev", ["q1"])
    write_run(results, "nodocs-dev-2", "no-docs", "dev", ["q1"])
    publish(results, ["agent-dev-2", "nodocs-dev-2"])
    build_site(results, out, **inject)
    assert (out / "q" / "q1.html").exists()
    assert not (out / "q" / "q3.html").exists()
    assert (out / MARKER).exists()


def test_refuses_to_replace_unrelated_directory(
    results: Path, tmp_path: Path, inject: dict[str, Any]
) -> None:
    out = tmp_path / "precious"
    out.mkdir()
    (out / "notes.txt").write_text("keep")
    with pytest.raises(ValueError, match="previous build"):
        build_site(results, out, **inject)
    assert (out / "notes.txt").read_text() == "keep"
