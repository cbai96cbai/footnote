import json
from pathlib import Path

import pytest

from footnote.eval.registry import TestRunRefused, complete_test_run, start_test_run


def test_completed_test_run_refused(tmp_path: Path) -> None:
    start_test_run(tmp_path, "agent", "h1", "r1")
    complete_test_run(tmp_path, "agent", "r1")
    with pytest.raises(TestRunRefused):
        start_test_run(tmp_path, "agent", "h2", "r2")  # same family, other version
    with pytest.raises(TestRunRefused):
        start_test_run(tmp_path, "agent", "h2", "r2", rerun=True)  # no reason
    assert start_test_run(tmp_path, "agent", "h2", "r2", rerun=True, reason="judge fixed") == (
        "judge fixed"
    )
    reg = json.loads((tmp_path / "test_runs.json").read_text())
    assert reg["agent"]["run_id"] == "r2" and reg["agent"]["state"] == "started"
    assert reg["agent"]["previous"][0]["run_id"] == "r1"
    start_test_run(tmp_path, "other", "h", "r3")  # another family is unaffected


def test_started_test_run_resumable(tmp_path: Path) -> None:
    start_test_run(tmp_path, "agent", "h1", "r1")
    start_test_run(tmp_path, "agent", "h1", "r1")
    with pytest.raises(TestRunRefused):
        start_test_run(tmp_path, "agent", "h1", "r9")
    complete_test_run(tmp_path, "agent", "r1")
    assert json.loads((tmp_path / "test_runs.json").read_text())["agent"]["state"] == "completed"
