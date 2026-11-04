"""Jinja filters. Every string that comes from a filing or the benchmark goes through `excerpt`."""

import json

EXCERPT_LIMIT = 400
STEP_LIMIT = 200


def excerpt(text: object, limit: int = EXCERPT_LIMIT) -> str:
    """Whitespace collapsed, cut at `limit` characters, with an ellipsis when cut."""
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[:limit].rstrip() + "…"


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def usd(x: float | None) -> str:
    return "n/a" if x is None else f"${x:.4f}"


def seconds(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1f} s"


def step_arguments(arguments: object) -> str:
    return excerpt(json.dumps(arguments, sort_keys=True, default=str), STEP_LIMIT)
