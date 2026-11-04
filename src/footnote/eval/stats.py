"""Company-clustered bootstrap: questions of one company are not independent."""

import numpy as np

Interval = tuple[float, float, float]


def _counts(outcomes: dict[str, list[bool]], companies: list[str]) -> tuple[np.ndarray, np.ndarray]:
    correct = np.array([sum(outcomes[c]) for c in companies], dtype=float)
    n = np.array([len(outcomes[c]) for c in companies], dtype=float)
    return correct, n


def _percentile(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def cluster_bootstrap(
    outcomes: dict[str, list[bool]], iters: int = 10000, seed: int = 7
) -> Interval:
    """(accuracy, lo, hi): 95% percentile interval over resampled companies."""
    companies = sorted(c for c in outcomes if outcomes[c])
    if not companies:
        raise ValueError("no outcomes to resample")
    correct, n = _counts(outcomes, companies)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(companies), size=(iters, len(companies)))
    acc = correct[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = _percentile(acc)
    return float(correct.sum() / n.sum()), lo, hi


def paired_cluster_bootstrap(
    a: dict[str, list[bool]], b: dict[str, list[bool]], iters: int = 10000, seed: int = 7
) -> Interval:
    """(accuracy(a) - accuracy(b), lo, hi) on the same resampled companies.

    Both systems must have been scored on the same questions of each company.
    """
    companies = sorted(c for c in a if a[c])
    if companies != sorted(c for c in b if b[c]):
        raise ValueError("the two runs cover different companies")
    if any(len(a[c]) != len(b[c]) for c in companies):
        raise ValueError("the two runs cover different questions of a company")
    if not companies:
        raise ValueError("no outcomes to resample")
    ca, n = _counts(a, companies)
    cb, _ = _counts(b, companies)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(companies), size=(iters, len(companies)))
    total = n[idx].sum(axis=1)
    diff = (ca[idx].sum(axis=1) - cb[idx].sum(axis=1)) / total
    lo, hi = _percentile(diff)
    return float((ca.sum() - cb.sum()) / n.sum()), lo, hi
