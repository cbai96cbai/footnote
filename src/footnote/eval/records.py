"""Run files: predictions.jsonl and verdicts.jsonl, appended one line at a time."""

import json
import logging
from pathlib import Path
from typing import Any

from footnote.eval.grade import Verdict
from footnote.prediction import (
    Prediction,
    prediction_from_dict,
    prediction_to_dict,
    usage_from_dict,
    usage_to_dict,
)

log = logging.getLogger(__name__)


def verdict_to_dict(v: Verdict) -> dict[str, Any]:
    return {
        "question_id": v.question_id,
        "label": v.label,
        "method": v.method,
        "rationale": v.rationale,
        "usage": usage_to_dict(v.usage),
    }


def verdict_from_dict(d: dict[str, Any]) -> Verdict:
    return Verdict(
        d["question_id"], d["label"], d["method"], d["rationale"], usage_from_dict(d["usage"])
    )


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def _read(path: Path) -> list[dict[str, Any]]:
    """Complete records only; a torn last line (crash mid-write) is dropped and cut off the file."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    *terminated, tail = raw.split(b"\n")
    keep = len(raw) - len(tail)  # bytes up to the end of the last terminated line
    if tail.strip():
        log.warning("%s: dropping a torn last line (not fully written)", path)
    records: list[dict[str, Any]] = []
    for i, line in enumerate(terminated):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            if any(x.strip() for x in terminated[i + 1 :]) or tail.strip():
                raise
            log.warning("%s: dropping an unparseable last line", path)
            keep = sum(len(x) + 1 for x in terminated[:i])
    if keep < len(raw):
        with path.open("r+b") as fh:
            fh.truncate(keep)
    return records


def append_prediction(run_dir: Path, p: Prediction) -> None:
    _append(run_dir / "predictions.jsonl", prediction_to_dict(p))


def append_verdict(run_dir: Path, v: Verdict) -> None:
    _append(run_dir / "verdicts.jsonl", verdict_to_dict(v))


def read_predictions(run_dir: Path) -> dict[str, Prediction]:
    return {d["question_id"]: prediction_from_dict(d) for d in _read(run_dir / "predictions.jsonl")}


def read_verdicts(run_dir: Path) -> dict[str, Verdict]:
    return {d["question_id"]: verdict_from_dict(d) for d in _read(run_dir / "verdicts.jsonl")}
