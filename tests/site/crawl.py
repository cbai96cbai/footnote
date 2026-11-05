"""Follow the links of a built site the way a visitor does, starting at index.html."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("a", "link"):
            self.hrefs += [v for k, v in attrs if k == "href" and v]


def links_of(html: str) -> list[str]:
    parser = _Links()
    parser.feed(html)
    return parser.hrefs


def crawl(out: Path) -> tuple[set[str], list[tuple[str, str]]]:
    """(html pages reached from index.html as paths under `out`, (page, link) pairs that lead to
    a missing file). External and data links are skipped."""
    reached: set[str] = set()
    broken: list[tuple[str, str]] = []
    todo = ["index.html"]
    while todo:
        page = todo.pop()
        if page in reached:
            continue
        reached.add(page)
        base = (out / page).parent
        for href in links_of((out / page).read_text(encoding="utf-8")):
            parts = urlsplit(href)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            path = unquote(parts.path)
            target = (out / path.lstrip("/")) if path.startswith("/") else (base / path)
            target = target.resolve()
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                broken.append((page, href))
            elif target.suffix == ".html":
                todo.append(str(target.relative_to(out.resolve())))
    return reached, broken
