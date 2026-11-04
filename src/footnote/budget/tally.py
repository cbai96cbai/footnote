"""Append-only spend tally with a hard cap, safe across processes."""

import fcntl
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import IO

DEFAULT_PATH = Path("~/.footnote/spend.jsonl")
DEFAULT_CAP = Decimal("25")


class BudgetExceeded(Exception):
    """Admitting this run would put the total over the cap."""


class Tally:
    """admit reserves a run's maximum; settle replaces the reservation with the actual cost.

    A run that never settles (a crash) stays charged its maximum.
    """

    def __init__(self, path: Path | None = None, cap: Decimal = DEFAULT_CAP) -> None:
        self.path = (path or DEFAULT_PATH).expanduser()
        self.cap = cap

    @contextmanager
    def _locked(self) -> Iterator[IO[str]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.seek(0)
                yield fh
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    @staticmethod
    def _state(fh: IO[str]) -> tuple[Decimal, dict[str, list[Decimal]]]:
        settled = Decimal(0)
        open_admits: dict[str, list[Decimal]] = {}
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            amount = Decimal(rec["amount"])
            if rec["event"] == "admit":
                open_admits.setdefault(rec["run_id"], []).append(amount)
            else:
                settled += amount
                # settle closes the most recent admission; an earlier crashed one stays charged
                open_admits[rec["run_id"]].pop()
        return settled, open_admits

    @classmethod
    def _total(cls, fh: IO[str]) -> Decimal:
        settled, open_admits = cls._state(fh)
        return settled + sum((a for amounts in open_admits.values() for a in amounts), Decimal(0))

    @staticmethod
    def _append(fh: IO[str], event: str, run_id: str, amount: Decimal) -> None:
        rec = {
            "event": event,
            "run_id": run_id,
            "amount": str(amount),
            "ts": datetime.now(UTC).isoformat(),
        }
        fh.seek(0, 2)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    def total(self) -> Decimal:
        with self._locked() as fh:
            return self._total(fh)

    def admit(self, run_id: str, max_cost: Decimal) -> None:
        with self._locked() as fh:
            total = self._total(fh)
            if total + max_cost > self.cap:
                raise BudgetExceeded(
                    f"run {run_id}: {total} spent or reserved + {max_cost} maximum exceeds cap "
                    f"{self.cap}"
                )
            self._append(fh, "admit", run_id, max_cost)

    def settle(self, run_id: str, actual: Decimal) -> None:
        with self._locked() as fh:
            if not self._state(fh)[1].get(run_id):
                raise ValueError(f"settle for run {run_id} without an open admission")
            self._append(fh, "settle", run_id, actual)


def settle_if_complete(tally: Tally, run_id: str, actual: Decimal, completed: bool) -> bool:
    """Settle only after a clean finish with every call's usage accumulated.

    Any other exit (an interrupt, an error escaping after a request may have been sent without
    usage coming back) leaves the admission open, so its maximum stays charged.
    """
    if not completed:
        return False
    tally.settle(run_id, actual)
    return True
