"""run.json: what a run folder was started as. A resume must match it exactly."""

import fcntl
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

FIELDS = ("family", "version", "config_hash", "split", "rubric_hash")


class RunIdentityMismatch(Exception):
    """The run folder was started as something else."""


class RunLocked(Exception):
    """Another invocation is running this run id."""


@contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    """One invocation per run at a time: an exclusive non-blocking lock held for the whole run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".lock").open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunLocked(f"{run_dir.name} is already running in another process") from None
        yield  # the lock is released when the file closes


def read_identity(run_dir: Path) -> dict[str, str] | None:
    p = run_dir / "run.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def verify_identity(run_dir: Path, identity: dict[str, str]) -> None:
    """Refuse when the folder's recorded identity differs in any field. Writes nothing."""
    stored = read_identity(run_dir)
    if stored is None:
        if (run_dir / "predictions.jsonl").exists() or (run_dir / "verdicts.jsonl").exists():
            raise RunIdentityMismatch(f"{run_dir.name} predates run identity; use a new run id")
        return
    diff = [f for f in FIELDS if stored.get(f) != identity[f]]
    if diff:
        detail = "; ".join(f"{f}: started as {stored.get(f)!r}, now {identity[f]!r}" for f in diff)
        raise RunIdentityMismatch(f"{run_dir.name} was started differently ({detail})")


def write_identity(run_dir: Path, identity: dict[str, str]) -> None:
    """Record the identity on a run's first start; later starts already passed verify."""
    if (run_dir / "run.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / "run.json.tmp"
    tmp.write_text(json.dumps(identity, indent=1) + "\n", encoding="utf-8")
    tmp.replace(run_dir / "run.json")
