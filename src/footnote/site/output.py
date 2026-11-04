"""Replace the site output directory as a whole, never wiping a folder the build did not make."""

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

MARKER = ".footnote-site"


def replace_output(out_dir: Path, render: Callable[[Path], None]) -> None:
    """Render into a fresh sibling directory, then swap it in for `out_dir`."""
    if out_dir.exists() and (not out_dir.is_dir() or _unrelated(out_dir)):
        raise ValueError(
            f"{out_dir} exists and does not look like a previous build "
            f"(no {MARKER} file); refusing to replace it"
        )
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}-", dir=out_dir.parent))
    try:
        render(tmp)
        (tmp / MARKER).write_text("written by footnote site\n", encoding="utf-8")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp.rename(out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _unrelated(d: Path) -> bool:
    return not (d / MARKER).is_file() and any(d.iterdir())
