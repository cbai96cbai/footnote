"""Inline SVG bars with interval whiskers. Colours come from CSS classes (light and dark)."""

from dataclasses import dataclass
from html import escape

from markupsafe import Markup

W, LABEL_W, BAR_X, BAR_W, ROW_H = 640, 190, 200, 380, 34


@dataclass(frozen=True)
class Bar:
    label: str
    value: float | None
    lo: float | None
    hi: float | None


def _x(v: float) -> float:
    return BAR_X + max(0.0, min(1.0, v)) * BAR_W


def accuracy_chart(bars: list[Bar], title: str) -> Markup:
    height = ROW_H * len(bars) + 28
    parts = [
        f'<svg class="chart" viewBox="0 0 {W} {height}" role="img" aria-label="{escape(title)}">',
        f"<title>{escape(title)}</title>",
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1):
        x = _x(tick)
        parts.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{height - 22}"/>')
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{height - 6}" text-anchor="middle">'
            f"{int(tick * 100)}%</text>"
        )
    for i, b in enumerate(bars):
        y = i * ROW_H + 6
        mid = y + ROW_H / 2 - 6
        parts.append(
            f'<text class="lbl" x="{LABEL_W}" y="{mid + 4:.1f}" text-anchor="end">'
            f"{escape(b.label)}</text>"
        )
        if b.value is None:
            parts.append(f'<text class="tick" x="{BAR_X + 4}" y="{mid + 4:.1f}">not graded</text>')
            continue
        parts.append(
            f'<rect class="bar" x="{BAR_X}" y="{mid - 9:.1f}" '
            f'width="{_x(b.value) - BAR_X:.1f}" height="18"/>'
        )
        if b.lo is not None and b.hi is not None:
            lo, hi = _x(b.lo), _x(b.hi)
            parts.append(
                f'<path class="whisker" d="M{lo:.1f} {mid - 6:.1f}V{mid + 6:.1f}M{lo:.1f} {mid:.1f}'
                f'H{hi:.1f}M{hi:.1f} {mid - 6:.1f}V{mid + 6:.1f}"/>'
            )
    parts.append("</svg>")
    return Markup("".join(parts))
