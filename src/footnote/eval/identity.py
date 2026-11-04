"""run.json: what a run folder was started as. A resume must match it exactly."""

import json
from pathlib import Path

FIELDS = ("family", "version", "config_hash", "split", "rubric_hash")


class RunIdentityMismatch(Exception):
    """The run folder was started as something else."""


def read_identity(run_dir: Path) -> dict[str, str] | None:
    p = run_dir / "run.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def verify_identity(run_dir: Path, identity: dict[str, str]) -> None:
    """Refuse when the folder's recorded identity differs in any field. Writes nothing."""
    stored = read_identity(run_dir)
    if stored is None:
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
