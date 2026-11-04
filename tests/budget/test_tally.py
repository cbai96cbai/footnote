import threading
from decimal import Decimal
from pathlib import Path

import pytest

from footnote.budget.tally import BudgetExceeded, Tally


def test_tally_refuses_over_cap(tmp_path: Path) -> None:
    t = Tally(tmp_path / "spend.jsonl", cap=Decimal("1"))
    t.admit("a", Decimal("0.6"))
    with pytest.raises(BudgetExceeded):
        t.admit("b", Decimal("0.5"))
    t.admit("b", Decimal("0.4"))  # exactly at the cap is allowed
    assert t.total() == Decimal("1.0")


def test_tally_crash_keeps_max(tmp_path: Path) -> None:
    path = tmp_path / "spend.jsonl"
    Tally(path).admit("crashed", Decimal("3"))
    assert Tally(path).total() == Decimal("3")  # a new process sees the unsettled maximum


def test_tally_settle_uses_actual(tmp_path: Path) -> None:
    t = Tally(tmp_path / "spend.jsonl")
    t.admit("a", Decimal("3"))
    t.settle("a", Decimal("0.75"))
    assert t.total() == Decimal("0.75")


def test_settle_closes_latest_admission_leaving_crashed_one(tmp_path: Path) -> None:
    t = Tally(tmp_path / "spend.jsonl")
    t.admit("r", Decimal("3"))  # crashed
    t.admit("r", Decimal("2"))  # resumed under the same id
    t.settle("r", Decimal("1"))
    assert t.total() == Decimal("4")


def test_settle_without_admission_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Tally(tmp_path / "spend.jsonl").settle("ghost", Decimal("1"))


def test_tally_records_strings_and_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "spend.jsonl"
    Tally(path).admit("a", Decimal("0.010"))
    line = path.read_text().splitlines()[0]
    assert '"amount": "0.010"' in line and '"ts"' in line and '"run_id": "a"' in line


def test_tally_locked_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "spend.jsonl"
    results: list[bool] = []

    def worker(i: int) -> None:
        try:
            Tally(path, cap=Decimal("25")).admit(f"r{i}", Decimal("3"))
            results.append(True)
        except BudgetExceeded:
            results.append(False)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert results.count(True) == 8  # 8 x 3 = 24 <= 25 < 27
    assert Tally(path).total() == Decimal("24")
