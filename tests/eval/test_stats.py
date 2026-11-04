import pytest

from footnote.eval.stats import cluster_bootstrap, paired_cluster_bootstrap
from footnote.systems import register_system


def test_cluster_bootstrap_deterministic() -> None:
    o = {"A": [True, False, True], "B": [False, False], "C": [True], "D": [True, True, False]}
    first = cluster_bootstrap(o, iters=2000)
    assert first == cluster_bootstrap(o, iters=2000)
    acc, lo, hi = first
    assert acc == pytest.approx(5 / 9) and lo <= acc <= hi


def test_cluster_bootstrap_identical_clusters_zero_width() -> None:
    o = {c: [True, False] for c in "ABCDE"}
    assert cluster_bootstrap(o, iters=500) == (0.5, 0.5, 0.5)


def test_paired_bootstrap_same_resample() -> None:
    a = {c: [True, True] for c in "ABCD"}
    b = {c: [False, True] for c in "ABCD"}
    assert paired_cluster_bootstrap(a, b, iters=500) == (0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        paired_cluster_bootstrap(a, {"A": [True, True]}, iters=10)


def test_config_hash_changes_with_prompt_text() -> None:
    def runner(q: object, c: object) -> None: ...

    from decimal import Decimal

    a = register_system("t", runner, Decimal(1), {"version": "1", "prompt": "be brief"})  # type: ignore[arg-type]
    b = register_system("t", runner, Decimal(1), {"version": "1", "prompt": "be thorough"})  # type: ignore[arg-type]
    c = register_system("t", runner, Decimal(1), {"prompt": "be brief", "version": "1"})  # type: ignore[arg-type]
    assert a.config_hash != b.config_hash and a.config_hash == c.config_hash
