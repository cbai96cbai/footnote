import json
from pathlib import Path
from typing import Any

import pytest
from test_run import MAX_Q, QS, prediction, setup

from footnote.eval import summary as summary_mod
from footnote.eval.identity import FIELDS, RunIdentityMismatch
from footnote.eval.run import run_eval
from footnote.systems import register_system


def _kw(tmp_path: Path, family: str = "fake") -> dict[str, Any]:
    return setup(tmp_path, lambda q, c: prediction(q, "declined"), family)


def test_run_json_records_identity_and_summary_reads_it(tmp_path: Path) -> None:
    kw = _kw(tmp_path)
    summary = run_eval("fake", "dev", "r1", **kw)
    ident = json.loads((tmp_path / "results" / "r1" / "run.json").read_text())
    assert set(ident) == set(FIELDS) and ident["family"] == "fake" and ident["split"] == "dev"
    assert summary["rubric_hash"] == ident["rubric_hash"]


def test_exact_resume_still_works(tmp_path: Path) -> None:
    kw = _kw(tmp_path)
    run_eval("fake", "dev", "r1", **kw)
    assert run_eval("fake", "dev", "r1", **kw)["n"] == len(QS)


def test_other_family_with_same_run_id_refused(tmp_path: Path) -> None:
    kw = _kw(tmp_path)
    run_eval("fake", "test", "r1", **kw)
    register_system("other", lambda q, c: prediction(q, "declined"), MAX_Q, {"version": "1"})
    with pytest.raises(RunIdentityMismatch, match="family"):
        run_eval("other", "test", "r1", **kw)
    assert "other" not in json.loads((tmp_path / "results" / "test_runs.json").read_text())


def test_other_split_with_same_run_id_refused(tmp_path: Path) -> None:
    kw = _kw(tmp_path)
    run_eval("fake", "dev", "r1", **kw)
    with pytest.raises(RunIdentityMismatch, match="split"):
        run_eval("fake", "gate", "r1", **kw)


def test_changed_rubric_on_resume_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kw = _kw(tmp_path)
    run_eval("fake", "dev", "r1", **kw)
    monkeypatch.setattr(summary_mod, "rubric_hash", lambda: "changed")
    monkeypatch.setattr("footnote.eval.run.rubric_hash", lambda: "changed")
    with pytest.raises(RunIdentityMismatch, match="rubric_hash"):
        run_eval("fake", "dev", "r1", **kw)
