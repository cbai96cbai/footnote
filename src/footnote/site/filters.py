"""Jinja filters. Every string that comes from a filing or the benchmark goes through `excerpt`."""

import json

EXCERPT_LIMIT = 400
STEP_LIMIT = 200


def excerpt(text: object, limit: int = EXCERPT_LIMIT) -> str:
    """Whitespace collapsed; never longer than `limit` characters, ellipsis included."""
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def difference(d: tuple[float, float, float] | None) -> str:
    """A paired difference in percentage points with its 95% interval; a dash when absent."""
    if d is None:
        return "\u2014"
    mid, lo, hi = (x * 100 for x in d)
    return f"{mid:+.1f} pts ({lo:+.1f} to {hi:+.1f})"


def usd(x: float | None) -> str:
    return "n/a" if x is None else f"${x:.4f}"


def seconds(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1f} s"


def step_arguments(arguments: object) -> str:
    return excerpt(json.dumps(arguments, sort_keys=True, default=str), STEP_LIMIT)


def quote_missing(problems: list[str]) -> bool:
    """True when the check reported a quote that is not on its cited page."""
    return any(p.startswith("quote not found") for p in problems)


def final_value_only(problems: list[str]) -> bool:
    """True when every problem is about the final value, so the quotes themselves were found."""
    return bool(problems) and all(p.startswith("final value") for p in problems)
