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


def load_runs(results_dir: Path) -> list[Run]:
    """Runs with a summary, the latest run id of each system and split, in display order."""
    latest: dict[tuple[str, str], Run] = {}
    for d in sorted(
        p
        for p in (results_dir.iterdir() if results_dir.is_dir() else [])
        if (p / "summary.json").is_file()
    ):
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        run = Run(d.name, s["system"], s["split"], s, read_predictions(d), read_verdicts(d))
        latest[(run.system, run.split)] = run
    return sorted(latest.values(), key=lambda r: (_order(r.system), r.split))


def split_runs(runs: list[Run]) -> tuple[list[Run], list[Run]]:
    """(test-split runs, development and gate runs)."""
    return [r for r in runs if r.split == "test"], [r for r in runs if r.split != "test"]
