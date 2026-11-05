"""Jinja filters. Every string that comes from a filing or the benchmark goes through `excerpt`."""

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
    """A step's arguments as a short readable phrase, e.g. `"net sales", page 5`."""
    if not isinstance(arguments, dict):
        return excerpt(arguments, STEP_LIMIT)
    parts = []
    for key, value in sorted(arguments.items()):
        if key == "page_index" and isinstance(value, int):
            parts.append(f"page {value + 1}")
        elif key == "query":
            parts.append(f'"{value}"')
        elif key == "expression":
            parts.append(str(value))
        else:
            shown = value if isinstance(value, str | int | float) else f"({len(value)} items)"
            parts.append(f"{key.replace('_', ' ')} {shown}")
    return excerpt(", ".join(parts), STEP_LIMIT)


def final_value_only(problems: list[str]) -> bool:
    """True when every problem is about the final value, so the quotes themselves were found."""
    return bool(problems) and all(p.startswith("final value") for p in problems)
