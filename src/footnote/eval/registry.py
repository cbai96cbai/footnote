"""One test run per system family: results/test_runs.json."""

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class TestRunRefused(Exception):
    __test__ = False  # not a pytest class


def _path(results_root: Path) -> Path:
    return results_root / "test_runs.json"


def _load(results_root: Path) -> dict[str, Any]:
    p = _path(results_root)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save(results_root: Path, reg: dict[str, Any]) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    tmp = results_root / f"test_runs.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n", "utf-8")
    tmp.replace(_path(results_root))  # atomic: a reader never sees half a registry


@contextmanager
def _locked(results_root: Path) -> Iterator[None]:
    """Exclusive lock over a read-check-write, on a side file the rename never replaces."""
    results_root.mkdir(parents=True, exist_ok=True)
    with (results_root / "test_runs.lock").open("a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def start_test_run(
    results_root: Path,
    family: str,
    config_hash: str,
    run_id: str,
    *,
    rerun: bool = False,
    reason: str | None = None,
) -> str | None:
    """Record a test run as started, or refuse it. Returns the rerun reason when one applies."""
    if rerun and not (reason and reason.strip()):
        raise TestRunRefused("--rerun-test needs --reason TEXT")
    with _locked(results_root):
        return _start(results_root, family, config_hash, run_id, rerun, reason)


def _start(
    results_root: Path, family: str, config_hash: str, run_id: str, rerun: bool, reason: str | None
) -> str | None:
    reg = _load(results_root)
    prior = reg.get(family)
    resuming = bool(
        prior
        and prior["state"] == "started"
        and prior["run_id"] == run_id
        and prior["config_hash"] == config_hash
    )
    if prior and not resuming and not rerun:
        what = (
            f"run {run_id} started with a different configuration"
            if prior["state"] == "started" and prior["run_id"] == run_id
            else f"test run {prior['run_id']} is {prior['state']}"
        )
        raise TestRunRefused(
            f"{family}: {what}; a family's test run happens once (use --rerun-test --reason TEXT)"
        )
    entry: dict[str, Any] = {"config_hash": config_hash, "run_id": run_id, "state": "started"}
    if prior and not resuming:
        entry["previous"] = [
            *prior.get("previous", []),
            {k: v for k, v in prior.items() if k != "previous"},
        ]
        entry["rerun_reason"] = reason
    elif prior:
        entry = dict(prior)
    reg[family] = entry
    _save(results_root, reg)
    return entry.get("rerun_reason")


def complete_test_run(results_root: Path, family: str, run_id: str) -> None:
    with _locked(results_root):
        reg = _load(results_root)
        entry = reg[family]
        if entry["run_id"] != run_id:
            raise ValueError(f"{family}: registry holds run {entry['run_id']}, not {run_id}")
        entry["state"] = "completed"
        _save(results_root, reg)
