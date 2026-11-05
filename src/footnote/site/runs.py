"""Read result folders into what the pages show."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from footnote.eval.grade import Verdict
from footnote.eval.records import read_predictions, read_verdicts
from footnote.prediction import Prediction, Step

LABELS = {
    "agent": "Footnote agent",
    "plain-retrieval": "Plain retrieval",
    "long-context": "Long context",
    "no-docs": "Without documents",
}
ORDER = list(LABELS)
SYSTEM_WORDS = {
    "agent": "agent",
    "plain-retrieval": "plain retrieval",
    "long-context": "long context",
    "no-docs": "no documents",
}


@dataclass(frozen=True)
class Run:
    run_id: str
    system: str
    split: str
    version: str
    summary: dict[str, Any]
    predictions: dict[str, Prediction]
    verdicts: dict[str, Verdict]

    @property
    def label(self) -> str:
        return LABELS.get(self.system, self.system)

    @property
    def display_label(self) -> str:
        """The label, with the version for exploratory runs (several may exist per system)."""
        return self.label if self.split == "test" else f"{self.label} {self.version}"

    @property
    def version_name(self) -> str:
        """Short caption form, e.g. "long context v2"."""
        return f"{SYSTEM_WORDS.get(self.system, self.system)} v{self.version}"

    @property
    def page_label(self) -> str:
        """Names the run on a question page, e.g. "Footnote agent · development run (agent-2)"."""
        if self.split == "test":
            return f"{self.label} · test run"
        kind = "development" if self.split == "dev" else self.split
        return f"{self.label} · {kind} run" + (f" ({self.version})" if self.version else "")

    @property
    def truncated_questions(self) -> list[str]:
        """Questions whose filing was longer than the window and was cut, by the pack record."""
        return sorted(q for q, p in self.predictions.items() if pack_truncation(p.trace))

    @property
    def answered_without_final_value(self) -> int:
        """Answers that gave no final number: the summary's count, else counted from predictions."""
        recorded = self.summary.get("answered_without_final_value")
        return recorded if recorded is not None else _count_without_final_value(self)


def _count_without_final_value(r: Run) -> int:
    return sum(
        p.status == "answered" and p.answer is not None and p.answer.final_value is None
        for p in r.predictions.values()
    )


def pack_truncation(trace: list[Step]) -> tuple[int, int] | None:
    """(pages given, pages in the filing) when the recorded pack step cut the filing short."""
    for s in trace:
        if s.tool == "pack" and s.arguments.get("truncated") is True:
            return int(s.arguments["pages_included"]), int(s.arguments["pages_in_filing"])
    return None


def _order(system: str) -> tuple[int, str]:
    return (ORDER.index(system) if system in ORDER else len(ORDER), system)


SHORT = {
    "agent": "Agent",
    "plain-retrieval": "Plain",
    "long-context": "Long context",
    "no-docs": "No docs",
}


def run_tags(runs: list[Run]) -> dict[str, str]:
    """A short name per run id for badges: the system, plus the version or split when a system
    has more than one published run."""
    count: dict[str, int] = {}
    for r in runs:
        count[r.system] = count.get(r.system, 0) + 1
    tags = {}
    for r in runs:
        tag = SHORT.get(r.system, r.label)
        if count[r.system] > 1:
            tag += " (test)" if r.split == "test" else f" ({r.version or r.split})"
        tags[r.run_id] = tag
    return tags


def _published_ids(results_dir: Path) -> list[str]:
    f = results_dir / "published.json"
    if not f.is_file():
        return []
    ids = json.loads(f.read_text(encoding="utf-8"))
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ValueError(f"{f} must be a list of run ids")
    return ids


def load_runs(results_dir: Path) -> list[Run]:
    """The runs listed in published.json, in display order; none if it is absent or empty.

    A test run is unique per system. Other runs are unique per (system, version, split) and
    keep their published order within a system.
    """
    runs: list[Run] = []
    seen: dict[tuple[str, ...], str] = {}
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
        version = str(s.get("version", ""))
        run = Run(
            run_id, s["system"], s["split"], version, s, read_predictions(d), read_verdicts(d)
        )
        key = (run.system, run.split) if run.split == "test" else (run.system, run.split, version)
        if key in seen:
            raise ValueError(
                f"published.json lists two runs for {' / '.join(key)}: {seen[key]} and {run_id}"
            )
        seen[key] = run_id
        runs.append(run)
    # sorted() is stable, so published order is kept within a system and split
    return sorted(runs, key=lambda r: (_order(r.system), r.split))


def split_runs(runs: list[Run]) -> tuple[list[Run], list[Run]]:
    """(test-split runs, development and gate runs)."""
    return [r for r in runs if r.split == "test"], [r for r in runs if r.split != "test"]
