"""Read result folders into what the pages show."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from footnote.eval.grade import Verdict
from footnote.eval.records import read_predictions, read_verdicts
from footnote.prediction import Prediction

LABELS = {
    "agent": "Footnote agent",
    "plain-retrieval": "Plain retrieval",
    "long-context": "Long context",
    "no-docs": "Without documents",
}
ORDER = list(LABELS)


@dataclass(frozen=True)
class Run:
    run_id: str
    system: str
    split: str
    summary: dict[str, Any]
    predictions: dict[str, Prediction]
    verdicts: dict[str, Verdict]

    @property
    def label(self) -> str:
        return LABELS.get(self.system, self.system)


def _order(system: str) -> tuple[int, str]:
    return (ORDER.index(system) if system in ORDER else len(ORDER), system)


def _published_ids(results_dir: Path) -> list[str]:
    f = results_dir / "published.json"
    if not f.is_file():
        return []
    ids = json.loads(f.read_text(encoding="utf-8"))
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ValueError(f"{f} must be a list of run ids")
    return ids


def load_runs(results_dir: Path) -> list[Run]:
    """The runs listed in published.json, in display order; none if it is absent or empty."""
    runs: dict[tuple[str, str], Run] = {}
    for run_id in _published_ids(results_dir):
        summary_file = results_dir / run_id / "summary.json"
        if not summary_file.is_file():
            raise ValueError(f"published run {run_id} has no summary.json in {results_dir}")
        s = json.loads(summary_file.read_text(encoding="utf-8"))
        if s.get("complete") is not True:
            raise ValueError(
                f"published run {run_id} is not complete (summary.json complete: false)"
            )
        d = results_dir / run_id
        run = Run(run_id, s["system"], s["split"], s, read_predictions(d), read_verdicts(d))
        key = (run.system, run.split)
        if key in runs:
            raise ValueError(
                f"published.json lists two runs for {run.system}/{run.split}: "
                f"{runs[key].run_id} and {run_id}"
            )
        runs[key] = run
    return sorted(runs.values(), key=lambda r: (_order(r.system), r.split))


def split_runs(runs: list[Run]) -> tuple[list[Run], list[Run]]:
    """(test-split runs, development and gate runs)."""
    return [r for r in runs if r.split == "test"], [r for r in runs if r.split != "test"]
